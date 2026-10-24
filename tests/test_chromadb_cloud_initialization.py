"""Regression test suite for ChromaDB initialization, path normalization, and client reuse.

Verifies the fix for ChromaDB KeyError on Streamlit Community Cloud:
- Normalization of relative, home-expanded, and absolute persistence paths.
- Reusing a single Chroma PersistentClient instance per canonical path.
- Thread-safe concurrent client retrieval without KeyError or race conditions.
- Collection creation, chunk indexing, dense similarity retrieval, and retention of existing data.
- Handling uncreatable/unwritable directories with actionable diagnostics rather than crashes.
- Streamlit cached resource integration ensuring reruns do not recreate clients.
"""

import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from typing import List

import chromadb
from streamlit.testing.v1 import AppTest

from src.document_ingestion import TextChunk
from src.vector_store import (
    LocalDeterministicEmbeddingFunction,
    SearchResult,
    VectorStoreService,
    get_vector_store,
    normalize_persist_directory,
)


class TestChromaDBPathNormalization(unittest.TestCase):
    """Test suite for persistence directory path normalization."""

    def test_relative_and_absolute_paths_resolve_identically(self):
        """Verify that relative, unnormalized, and absolute paths map to the same canonical path."""
        p_rel = "./chroma_db"
        p_bare = "chroma_db"
        p_slash = "./chroma_db/"
        p_abs = str(Path(p_rel).resolve())

        norm_rel = normalize_persist_directory(p_rel)
        norm_bare = normalize_persist_directory(p_bare)
        norm_slash = normalize_persist_directory(p_slash)
        norm_abs = normalize_persist_directory(p_abs)

        self.assertEqual(norm_rel, norm_abs)
        self.assertEqual(norm_bare, norm_abs)
        self.assertEqual(norm_slash, norm_abs)
        self.assertTrue(Path(norm_rel).is_absolute())

    def test_env_var_override(self):
        """Verify that SYNAPSE_CHROMA_DIR overrides default persistence directory."""
        custom_dir = str(Path("./custom_chroma_path").resolve())
        old_val = os.environ.get("SYNAPSE_CHROMA_DIR")
        try:
            os.environ["SYNAPSE_CHROMA_DIR"] = custom_dir
            normalized = normalize_persist_directory(None)
            self.assertEqual(normalized, custom_dir)
        finally:
            if old_val is not None:
                os.environ["SYNAPSE_CHROMA_DIR"] = old_val
            else:
                os.environ.pop("SYNAPSE_CHROMA_DIR", None)


class TestChromaDBSingleClientReuse(unittest.TestCase):
    """Test suite for single ChromaDB client reuse and concurrency safety."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="synapse_test_chroma_")
        self.embed_fn = LocalDeterministicEmbeddingFunction()

    def tearDown(self):
        VectorStoreService.close_client(self.test_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multiple_instances_reuse_same_client(self):
        """Verify that multiple VectorStoreService instances for the same path share one Client."""
        # 1. Initialize using relative path
        rel_path = os.path.relpath(self.test_dir)
        svc1 = VectorStoreService(
            persist_directory=rel_path,
            collection_name="test_col_1",
            embedding_function=self.embed_fn,
        )

        # 2. Initialize using absolute path
        abs_path = str(Path(self.test_dir).resolve())
        svc2 = VectorStoreService(
            persist_directory=abs_path,
            collection_name="test_col_2",
            embedding_function=self.embed_fn,
        )

        # 3. Assert that both services share the exact same underlying Chroma PersistentClient
        self.assertIs(
            svc1.client,
            svc2.client,
            "VectorStoreService failed to reuse the same PersistentClient for equivalent paths.",
        )
        self.assertEqual(svc1.persist_directory, svc2.persist_directory)

    def test_concurrent_initialization_no_keyerror(self):
        """Verify that concurrent threads instantiating the vector store do not hit KeyError race conditions."""
        results: List[VectorStoreService] = []
        errors: List[Exception] = []
        threads: List[threading.Thread] = []

        def worker():
            try:
                svc = VectorStoreService(
                    persist_directory=self.test_dir,
                    collection_name="concurrent_test",
                    embedding_function=self.embed_fn,
                )
                results.append(svc)
            except Exception as e:
                errors.append(e)

        # Launch 10 concurrent threads simultaneously
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent initializations raised errors: {errors}")
        self.assertEqual(len(results), 10)

        # Verify all threads received the same single client instance
        first_client = results[0].client
        for idx, svc in enumerate(results[1:], start=2):
            self.assertIs(
                svc.client,
                first_client,
                f"Thread {idx} did not receive the shared cached client instance.",
            )

    def test_close_and_reopen_client(self):
        """Verify that closing a client removes it from cache and reopening creates a fresh valid client."""
        svc1 = VectorStoreService(
            persist_directory=self.test_dir,
            collection_name="close_reopen_test",
            embedding_function=self.embed_fn,
        )
        client1 = svc1.client

        # Safely close client
        VectorStoreService.close_client(self.test_dir)
        self.assertIsNone(VectorStoreService.get_cached_client(self.test_dir))

        # Reopen
        svc2 = VectorStoreService(
            persist_directory=self.test_dir,
            collection_name="close_reopen_test",
            embedding_function=self.embed_fn,
        )
        client2 = svc2.client

        self.assertIsNot(client1, client2)
        self.assertFalse(getattr(client2, "_closed", False))


class TestChromaDBDataRetentionAndOperations(unittest.TestCase):
    """Test suite ensuring data is not wiped or lost on startup/rerun."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="synapse_persist_test_")
        self.embed_fn = LocalDeterministicEmbeddingFunction()

    def tearDown(self):
        VectorStoreService.close_client(self.test_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_indexing_retrieval_and_data_persistence(self):
        """Verify that chunks indexed in session 1 are preserved and readable when reopened in session 2."""
        # Session 1: Index chunks
        svc1 = VectorStoreService(
            persist_directory=self.test_dir,
            collection_name="persistence_check",
            embedding_function=self.embed_fn,
        )
        chunks = [
            TextChunk(
                chunk_id="chunk_quantum_001",
                text="Quantum entanglement allows instantaneous correlation between particles.",
                source_filename="quantum_physics.pdf",
                page_number=1,
                chunk_index=0,
                char_count=73,
            ),
            TextChunk(
                chunk_id="chunk_quantum_002",
                text="Superposition allows a quantum system to exist in multiple states simultaneously.",
                source_filename="quantum_physics.pdf",
                page_number=2,
                chunk_index=1,
                char_count=82,
            ),
        ]
        rep = svc1.index_chunks(chunks)
        self.assertEqual(rep["indexed_count"], 2)
        self.assertEqual(svc1.collection.count(), 2)

        # Close client 1
        VectorStoreService.close_client(self.test_dir)

        # Session 2: Open vector store on existing directory (simulates app restart)
        svc2 = VectorStoreService(
            persist_directory=self.test_dir,
            collection_name="persistence_check",
            embedding_function=self.embed_fn,
        )
        # CRITICAL ASSERTION: Existing data must NOT be cleared or wiped on startup
        self.assertEqual(svc2.collection.count(), 2)

        # Perform semantic query to verify index integrity
        search_results = svc2.search(query="entanglement particles", top_k=2)
        self.assertTrue(len(search_results) > 0)
        self.assertEqual(search_results[0].chunk_id, "chunk_quantum_001")
        self.assertIn("entanglement", search_results[0].text.lower())
        self.assertEqual(search_results[0].source_filename, "quantum_physics.pdf")
        self.assertEqual(search_results[0].page_number, 1)

    def test_actionable_error_on_uncreatable_directory(self):
        """Verify that attempting to initialize with an invalid path surfaces a descriptive RuntimeError."""
        # Using a file as parent directory to make mkdir fail
        bad_file = os.path.join(self.test_dir, "regular_file.txt")
        with open(bad_file, "w") as f:
            f.write("not a directory")

        impossible_path = os.path.join(bad_file, "sub_chroma_db")
        with self.assertRaises(RuntimeError) as ctx:
            VectorStoreService(persist_directory=impossible_path)

        err_msg = str(ctx.exception)
        self.assertIn("Failed to create canonical ChromaDB persistence directory", err_msg)


class TestStreamlitAppVectorStoreIntegration(unittest.TestCase):
    """End-to-end integration tests using Streamlit AppTest."""

    def test_app_startup_and_rerun_reuses_vector_store(self):
        """Verify Streamlit app starts up cleanly without ChromaDB KeyError and reuses client across reruns."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=20)
        at.run()

        # Enter workspace
        at.button(key="btn_enter_workspace_primary").click().run()
        self.assertFalse(at.exception, f"App crashed during startup: {at.exception}")

        # Check session state has valid vector store
        vstore1 = at.session_state.get("vector_store")
        self.assertIsNotNone(vstore1)
        self.assertIsInstance(vstore1, VectorStoreService)
        self.assertIsNone(at.session_state.get("vector_store_error"))

        # Navigate to Search & Explore page
        at.sidebar.radio(key="main_workspace_nav").set_value("🔍 Search & Explore").run()
        self.assertFalse(at.exception, f"App crashed navigating to Search & Explore: {at.exception}")

        # Trigger another rerun by toggling theme
        at.sidebar.button(key="sidebar_theme_toggle_btn").click().run()
        self.assertFalse(at.exception, f"App crashed during theme toggle rerun: {at.exception}")

        vstore2 = at.session_state.get("vector_store")
        self.assertIsNotNone(vstore2)
        # Verify vector store client is identical across reruns
        self.assertIs(vstore1.client, vstore2.client)


if __name__ == "__main__":
    unittest.main()
