"""Tests for SynapseAI Task 1: Project Structure, Imports, and Placeholder Integrity."""

import importlib
from pathlib import Path
import py_compile
import unittest


class TestSynapseAIProjectStructure(unittest.TestCase):
    """Validate repository layout, file presence, and module imports."""

    @classmethod
    def setUpClass(cls):
        # Resolve project root relative to this test file
        cls.project_root = Path(__file__).resolve().parent.parent

    def test_core_files_exist(self):
        """Ensure all required foundation files exist at the root."""
        expected_files = [
            "app.py",
            "requirements.txt",
            "README.md",
            ".gitignore",
            ".env.example",
            "data/.gitkeep",
        ]
        for rel_path in expected_files:
            file_path = self.project_root / rel_path
            self.assertTrue(
                file_path.exists(),
                f"Required project file missing: {rel_path}"
            )

    def test_directories_exist(self):
        """Ensure all essential project directories exist."""
        expected_dirs = ["src", "tests", "data"]
        for d in expected_dirs:
            dir_path = self.project_root / d
            self.assertTrue(
                dir_path.is_dir(),
                f"Required directory missing: {d}"
            )

    def test_src_modules_importable(self):
        """Verify that all modular src components can be imported cleanly."""
        modules_to_test = [
            "src",
            "src.document_ingestion",
            "src.entity_extraction",
            "src.knowledge_graph",
            "src.vector_store",
            "src.query_router",
            "src.answer_generator",
        ]
        for mod_name in modules_to_test:
            try:
                mod = importlib.import_module(mod_name)
                self.assertIsNotNone(mod, f"Imported module {mod_name} is None")
            except Exception as e:
                self.fail(f"Failed to import {mod_name}: {e}")

    def test_placeholder_classes_defined(self):
        """Ensure placeholder classes are properly declared with stub behavior."""
        from src.document_ingestion import DocumentIngestionService
        from src.entity_extraction import EntityExtractor
        from src.knowledge_graph import KnowledgeGraphBuilder
        from src.vector_store import VectorStoreService
        from src.query_router import QueryRouter, QueryIntent
        from src.answer_generator import AnswerGenerator

        # Instantiations should succeed
        doc_svc = DocumentIngestionService()
        self.assertIn(".pdf", doc_svc.supported_extensions)

        extractor = EntityExtractor()
        extracted = extractor.extract_entities("SynapseAI utilizes Knowledge Graph technology.")
        self.assertIsInstance(extracted, list)

        kg_builder = KnowledgeGraphBuilder()
        vis_data = kg_builder.get_visualization_data()
        self.assertIn("nodes", vis_data)
        self.assertIn("edges", vis_data)

        from src.vector_store import LocalDeterministicEmbeddingFunction
        vec_svc = VectorStoreService(
            persist_directory="./tests_temp_chroma",
            embedding_function=LocalDeterministicEmbeddingFunction(),
        )
        search_res = vec_svc.similarity_search("test query")
        self.assertIsInstance(search_res, list)

        router = QueryRouter()
        routing = router.route_query("What is knowledge graph?")
        self.assertEqual(QueryIntent.FACTUAL.value, "factual")
        self.assertIn("intent", routing)
        self.assertIn("plan", routing)

        ans_gen = AnswerGenerator(use_llm_if_available=False)
        ans = ans_gen.generate_answer("query", [], [])
        self.assertTrue(ans.is_abstained)
        self.assertEqual(len(ans.citations), 0)

    def test_app_py_compiles(self):
        """Ensure app.py has valid Python syntax and compiles without error."""
        app_path = self.project_root / "app.py"
        self.assertTrue(app_path.exists(), "app.py does not exist")
        # Compile to verify syntax
        compiled = py_compile.compile(str(app_path), doraise=True)
        self.assertIsNotNone(compiled)


if __name__ == "__main__":
    unittest.main()
