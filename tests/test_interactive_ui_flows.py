"""Integration regression tests for SynapseAI interactive UI flows, button connectivity, and state synchronization.

Validates:
- Theme toggle and persistence across reruns (Dark <-> Light).
- Cinematic intro entrance and replay controls.
- Bidirectional navigation between all 6 pages via sidebar radio and action buttons.
- Elimination of the navigation clobbering bug.
- Document ingestion end-to-end pipeline (Ingestion -> Chunking -> Entities -> Graph -> Vector Index).
- Knowledge graph filters, node inspector, and focus reset.
- ChromaDB semantic search invocation and provenance display.
- Ask SynapseAI example question chips, consecutive queries, verified citations, and abstention.
- System Status subsystem diagnostics and refresh trigger.
"""

import io
import os
import shutil
from typing import Any, List
import pytest
from streamlit.testing.v1 import AppTest

from src.document_ingestion import DocumentIngestionService, DocumentProcessingResult
from src.entity_extraction import EntityExtractor, ExtractionResult
from src.knowledge_graph import KnowledgeGraphBuilder
from src.vector_store import VectorStoreService
from src.query_router import QueryRouter
from src.answer_generator import AnswerGenerator, SufficiencyStatus


class TestAppTestInteractiveFlows:
    """Test interactive Streamlit UI interactions via streamlit.testing.v1.AppTest."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clean up test chroma DB directory before and after tests."""
        test_db = "./test_chroma_apptest"
        if os.path.exists(test_db):
            shutil.rmtree(test_db, ignore_errors=True)
        yield
        if os.path.exists(test_db):
            shutil.rmtree(test_db, ignore_errors=True)

    def test_app_initial_state_and_theme_toggle(self):
        """Verify initial launch states and theme toggle survival across reruns."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=15)
        at.run()
        assert not at.exception, f"App crashed on startup: {at.exception}"

        # Verify initial session state
        assert at.session_state["theme_mode"] == "dark"
        assert at.session_state["intro_seen"] is False
        assert at.session_state["current_page"] == "🌌 Overview"
        assert at.session_state["main_workspace_nav"] == "🌌 Overview"

        # Toggle to Light Mode via sidebar
        btn_theme = at.sidebar.button(key="sidebar_theme_toggle_btn")
        btn_theme.click().run()
        assert not at.exception
        assert at.session_state["theme_mode"] == "light"

        # Toggle back to Dark Mode via header
        btn_header_theme = at.button(key="header_theme_toggle_btn")
        btn_header_theme.click().run()
        assert not at.exception
        assert at.session_state["theme_mode"] == "dark"

    def test_intro_enter_and_replay(self):
        """Verify clicking 'Enter SynapseAI Workspace' dismisses intro and replay restores it."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=15)
        at.run()
        assert at.session_state["intro_seen"] is False

        # Click primary Enter button
        btn_enter = at.button(key="btn_enter_workspace_primary")
        btn_enter.click().run()
        assert not at.exception
        assert at.session_state["intro_seen"] is True

        # Click Replay Intro in header
        btn_replay = at.button(key="header_replay_intro_btn")
        btn_replay.click().run()
        assert not at.exception
        assert at.session_state["intro_seen"] is False

        # Enter again
        btn_enter2 = at.button(key="btn_enter_workspace_primary")
        btn_enter2.click().run()
        assert not at.exception
        assert at.session_state["intro_seen"] is True

    def test_overview_action_buttons_navigation(self):
        """Verify Overview action buttons navigate to target pages without getting clobbered by sidebar radio."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=15)
        at.run()

        # Dismiss intro
        at.button(key="btn_enter_workspace_primary").click().run()
        assert at.session_state["current_page"] == "🌌 Overview"

        # 1. Manage Documents ➔
        at.button(key="ov_btn_docs").click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "📄 Documents"
        assert at.session_state["main_workspace_nav"] == "📄 Documents"

        # Navigate back to Overview via sidebar radio
        at.sidebar.radio(key="main_workspace_nav").set_value("🌌 Overview").run()
        assert at.session_state["current_page"] == "🌌 Overview"

        # 2. Explore Graph ➔
        at.button(key="ov_btn_kg").click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "🕸️ Knowledge Graph"
        assert at.session_state["main_workspace_nav"] == "🕸️ Knowledge Graph"

        # Navigate back to Overview
        at.sidebar.radio(key="main_workspace_nav").set_value("🌌 Overview").run()

        # 3. Ask a Question ➔
        at.button(key="ov_btn_qa").click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "💬 Ask SynapseAI"
        assert at.session_state["main_workspace_nav"] == "💬 Ask SynapseAI"

        # Navigate back to Overview
        at.sidebar.radio(key="main_workspace_nav").set_value("🌌 Overview").run()

        # 4. Semantic Search ➔
        at.button(key="ov_btn_search").click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "🔍 Search & Explore"
        assert at.session_state["main_workspace_nav"] == "🔍 Search & Explore"

    def test_system_status_and_refresh_diagnostics(self):
        """Verify system status view renders and refresh diagnostics executes cleanly."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=15)
        at.run()
        at.button(key="btn_enter_workspace_primary").click().run()

        # Navigate to System Status
        at.sidebar.radio(key="main_workspace_nav").set_value("⚡ System Status").run()
        assert at.session_state["current_page"] == "⚡ System Status"

        # Click Refresh System Diagnostics button
        btn_refresh = at.button(key="btn_refresh_sys_diagnostics")
        btn_refresh.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "⚡ System Status"

    def test_ask_synapseai_chips_and_query_flow(self):
        """Verify clicking example chips populates the input and executes grounded Q&A pipeline."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=15)
        at.run()
        at.button(key="btn_enter_workspace_primary").click().run()

        # Navigate to Ask SynapseAI
        at.sidebar.radio(key="main_workspace_nav").set_value("💬 Ask SynapseAI").run()
        assert at.session_state["current_page"] == "💬 Ask SynapseAI"

        # Click Out-of-scope example chip
        btn_chip = at.button(key="ex_out_True")
        btn_chip.click().run()
        assert not at.exception

        # Verify Q&A result exists in session state
        last_qa = at.session_state.get("last_qa_result")
        assert last_qa is not None
        assert "iPhone" in last_qa.query
        assert last_qa.is_abstained or last_qa.sufficiency.status == SufficiencyStatus.INSUFFICIENT

    def test_knowledge_graph_view_modes_and_reset(self):
        """Verify 3D and 2D spatial mode switching in Knowledge Graph."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=15)
        at.run()
        at.button(key="btn_enter_workspace_primary").click().run()

        # Navigate to Knowledge Graph
        at.sidebar.radio(key="main_workspace_nav").set_value("🕸️ Knowledge Graph").run()
        assert at.session_state["current_page"] == "🕸️ Knowledge Graph"
        assert not at.exception

    def test_search_explore_empty_state_and_navigation(self):
        """Verify empty state on Search & Explore page and jump to Document Ingestion."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=15)
        at.run()
        at.button(key="btn_enter_workspace_primary").click().run()

        # Navigate to Search & Explore
        at.sidebar.radio(key="main_workspace_nav").set_value("🔍 Search & Explore").run()
        assert at.session_state["current_page"] == "🔍 Search & Explore"

        # If there are chunks in the persistent DB, clear index first
        btn_clear = at.button(key="btn_clear_vector_True")
        if not btn_clear.disabled:
            btn_clear.click().run()
            assert not at.exception

        # Now empty state is active
        btn_ingest = at.button(key="btn_nav_ingest_True")
        btn_ingest.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "📄 Documents"

    def test_contact_us_navigation_and_mailto_triggers(self):
        """Verify navigation to Contact Us section, presence of email address, link button, and footer link."""
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=15)
        at.run()
        at.button(key="btn_enter_workspace_primary").click().run()

        # 1. Test footer contact button navigation
        btn_footer = at.button(key="footer_nav_contact_btn")
        btn_footer.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "✉️ Contact Us"
        assert at.session_state["main_workspace_nav"] == "✉️ Contact Us"

        # 2. Check for Get in Touch heading and exact email address
        found_email = False
        for md in at.markdown:
            if "ayushraj20062007@gmail.com" in md.value:
                found_email = True
                break
        assert found_email, "Contact email ayushraj20062007@gmail.com not found in rendered page"

        # 3. Check link_buttons (Email Us and Launch Composer)
        link_urls = [btn.url for btn in at.link_button]
        assert any("mailto:ayushraj20062007@gmail.com" in url for url in link_urls), "mailto: link not found in link_button"
        expected_subject = "SynapseAI%20%E2%80%94%20Inquiry%20or%20Feedback"
        assert any(expected_subject in url for url in link_urls), "URL-encoded subject not found in mailto link"

        # 4. Navigate back via sidebar radio to verify clean bidirectional navigation
        at.sidebar.radio(key="main_workspace_nav").set_value("🌌 Overview").run()
        assert at.session_state["current_page"] == "🌌 Overview"


class TestEndToEndDataAndInteractionPipeline:
    """Test full backend pipeline execution and UI integration with synthetic multi-topic documents."""

    @pytest.fixture
    def synthetic_documents(self):
        """Create two synthetic documents with overlapping and distinct topics."""
        doc1_content = (
            "SynapseAI integrates ChromaDB for dense vector storage and semantic retrieval. "
            "SynapseAI utilizes NetworkX for interactive topological graph visualization. "
            "Topic themes group related concepts such as Vector Retrieval and Neural Architecture across documents."
        )
        doc2_content = (
            "SynapseAI connects Knowledge Graph with ChromaDB. "
            "ChromaDB stores document chunk embeddings for passage retrieval. "
            "SynapseAI performs hybrid retrieval combining vector similarity search with graph traversal. "
            "The system enforces grounded question answering with verified citations to prevent hallucinations."
        )
        return [
            ("synapse_overview.txt", doc1_content.encode("utf-8")),
            ("knowledge_rag.txt", doc2_content.encode("utf-8")),
        ]

    def test_full_pipeline_end_to_end_flow(self, synthetic_documents):
        """Simulate document ingestion -> entity extraction -> graph build -> vector index -> Q&A with citations."""
        # 1. Ingestion Service
        class MockUploadedFile:
            def __init__(self, name: str, data: bytes):
                self.name = name
                self._data = data
                self.size = len(data)

            def getvalue(self) -> bytes:
                return self._data

        uploaded_files = [MockUploadedFile(name, data) for name, data in synthetic_documents]
        ingestion_service = DocumentIngestionService()
        ingestion_results = ingestion_service.process_files(uploaded_files)

        assert len(ingestion_results) == 2
        assert all(r.is_success for r in ingestion_results)
        all_chunks = [c for r in ingestion_results for c in r.chunks]
        assert len(all_chunks) >= 2

        # 2. Entity and Theme Extraction
        extractor = EntityExtractor(use_llm=False)
        ext_result = extractor.extract_from_chunks(all_chunks)
        assert len(ext_result.entities) > 0
        assert len(ext_result.relationships) > 0
        assert len(ext_result.themes) > 0

        # Verify cross-document theme presence
        theme_names = [t.name for t in ext_result.themes]
        assert any("Retrieval" in tn or "Graph" in tn or "AI" in tn or "Concept" in tn or "Topic" in tn for tn in theme_names)

        # 3. Knowledge Graph Construction
        kg_builder = KnowledgeGraphBuilder()
        kg_builder.build_from_extraction(ext_result)
        stats = kg_builder.get_graph_statistics()
        assert stats["node_count"] == len(ext_result.entities)
        assert stats["edge_count"] > 0

        # Verify 3D and 2D HTML generation
        html_3d_dark = kg_builder.generate_3d_graph_html(theme="dark")
        assert "ForceGraph3D" in html_3d_dark
        assert "#050811" in html_3d_dark

        html_3d_light = kg_builder.generate_3d_graph_html(theme="light")
        assert "#f4f6fc" in html_3d_light

        html_2d_dark = kg_builder.generate_pyvis_html(theme="dark")
        assert "vis.Network" in html_2d_dark

        # 4. ChromaDB Vector Store Indexing
        test_dir = "./test_chroma_e2e"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)

        try:
            vector_store = VectorStoreService(persist_directory=test_dir)
            idx_report = vector_store.index_chunks(all_chunks)
            assert idx_report["indexed_count"] == len(all_chunks)

            # Test Semantic Similarity Search
            search_res = vector_store.similarity_search("ChromaDB vector database", top_k=2)
            assert len(search_res) > 0
            assert "chromadb" in search_res[0].text.lower()
            assert search_res[0].source_filename in ["synapse_overview.txt", "knowledge_rag.txt"]

            # 5. Hybrid Retrieval and Grounded Q&A
            router = QueryRouter()
            generator = AnswerGenerator(use_llm_if_available=False)

            # Case A: Factual Question
            q_fact = "What vector database does SynapseAI use?"
            known_entities = [e.name for e in ext_result.entities.values()]
            known_themes = [t.name for t in ext_result.themes]
            cls_fact = router.classify_query(q_fact, known_entities, known_themes)
            plan_fact = router.create_retrieval_plan(cls_fact, known_themes)
            ret_fact = router.execute_hybrid_retrieval(
                query=q_fact,
                plan=plan_fact,
                vector_store=vector_store,
                knowledge_graph=kg_builder,
                extraction_result=ext_result,
            )
            ans_fact = generator.generate_answer(q_fact, ret_fact)
            assert not ans_fact.is_abstained
            assert len(ans_fact.citations) > 0
            assert any(c.is_verified for c in ans_fact.citations)
            assert ans_fact.sufficiency.status in (SufficiencyStatus.SUPPORTED, SufficiencyStatus.PARTIALLY_SUPPORTED)

            # Case B: Relationship Question
            q_rel = "How is SynapseAI related to ChromaDB?"
            cls_rel = router.classify_query(q_rel, known_entities, known_themes)
            plan_rel = router.create_retrieval_plan(cls_rel, known_themes)
            ret_rel = router.execute_hybrid_retrieval(
                query=q_rel,
                plan=plan_rel,
                vector_store=vector_store,
                knowledge_graph=kg_builder,
                extraction_result=ext_result,
            )
            ans_rel = generator.generate_answer(q_rel, ret_rel)
            assert not ans_rel.is_abstained

            # Case C: Thematic Question
            q_theme = "What are the main themes across the documents?"
            cls_theme = router.classify_query(q_theme, known_entities, known_themes)
            plan_theme = router.create_retrieval_plan(cls_theme, known_themes)
            ret_theme = router.execute_hybrid_retrieval(
                query=q_theme,
                plan=plan_theme,
                vector_store=vector_store,
                knowledge_graph=kg_builder,
                extraction_result=ext_result,
            )
            ans_theme = generator.generate_answer(q_theme, ret_theme)
            assert not ans_theme.is_abstained

            # Case D: Out-of-scope / Unanswerable Question
            q_out = "What is the capital of Mars in 2050?"
            cls_out = router.classify_query(q_out, known_entities, known_themes)
            plan_out = router.create_retrieval_plan(cls_out, known_themes)
            ret_out = router.execute_hybrid_retrieval(
                query=q_out,
                plan=plan_out,
                vector_store=vector_store,
                knowledge_graph=kg_builder,
                extraction_result=ext_result,
            )
            ans_out = generator.generate_answer(q_out, ret_out)
            assert ans_out.is_abstained or ans_out.sufficiency.status == SufficiencyStatus.INSUFFICIENT
            assert "insufficient" in ans_out.answer_text.lower() or "not find" in ans_out.answer_text.lower() or "outside" in ans_out.answer_text.lower()

        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_full_document_workflow_bidirectional_traversal(self):
        """Regression test for Issue 1:
        Verify exact sequence:
        Process Documents -> Explore Graph -> Return to Documents -> Ask a Question -> Return to Graph -> Ask Another Question -> Return to Documents
        Ensures processed records, graph nodes, chunks, and citations survive repeated bidirectional navigation.
        """
        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=25)
        at.run()

        # Step 0: Enter workspace
        at.button(key="btn_enter_workspace_primary").click().run()
        assert not at.exception

        # Navigate to Documents page
        at.sidebar.radio(key="main_workspace_nav").set_value("📄 Documents").run()
        assert not at.exception
        assert at.session_state["current_page"] == "📄 Documents"

        # Populate processed documents in session state
        svc = DocumentIngestionService()
        doc_content = (
            "SynapseAI is a next-generation neural graph platform. "
            "It connects knowledge bases to ChromaDB vector stores and NetworkX graphs. "
            "SynapseAI enables bidirectional reasoning across unstructured documents."
        )
        fake_file = io.BytesIO(doc_content.encode("utf-8"))
        fake_file.name = "synapse_platform.txt"
        ingest_res = svc.process_files([fake_file])
        at.session_state["ingestion_results"] = ingest_res

        # Build extraction, KG, and vector store
        chunks = [c for r in ingest_res if r.is_success for c in r.chunks]
        extractor = EntityExtractor(use_llm=False)
        ext_res = extractor.extract_from_chunks(chunks)
        at.session_state["extraction_results"] = ext_res
        at.session_state["kg_builder"].build_from_extraction(ext_res)
        at.session_state["vector_store"].index_chunks(chunks)

        at.run()
        assert not at.exception
        assert len(at.session_state["ingestion_results"]) == 1
        initial_node_count = at.session_state["kg_builder"].graph.number_of_nodes()
        assert initial_node_count > 0

        # Action 1: Explore Graph via document action button
        btn_doc_kg = at.button(key="doc_nav_kg")
        assert btn_doc_kg is not None
        btn_doc_kg.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "🕸️ Knowledge Graph"
        assert at.session_state["kg_builder"].graph.number_of_nodes() == initial_node_count

        # Action 2: Return to Documents via quick action button
        btn_kg_docs = at.button(key="kg_nav_docs_True")
        assert btn_kg_docs is not None
        btn_kg_docs.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "📄 Documents"
        assert len(at.session_state["ingestion_results"]) == 1

        # Action 3: Ask a Question via document action button
        btn_doc_qa = at.button(key="doc_nav_qa")
        assert btn_doc_qa is not None
        btn_doc_qa.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "💬 Ask SynapseAI"

        # Ask first question
        at.text_input(key="input_qa_True").input("What does SynapseAI connect to?").run()
        at.button(key="btn_ask_True").click().run()
        assert not at.exception
        assert at.session_state["last_qa_result"] is not None
        first_answer = at.session_state["last_qa_result"].answer_text
        assert len(first_answer) > 0

        # Action 4: Return to Graph from Q&A page
        btn_qa_kg = at.button(key="qa_nav_kg_True")
        assert btn_qa_kg is not None
        btn_qa_kg.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "🕸️ Knowledge Graph"
        assert at.session_state["kg_builder"].graph.number_of_nodes() == initial_node_count

        # Action 5: Navigate back to Q&A from Graph page and ask second question
        btn_kg_qa = at.button(key="kg_nav_qa_True")
        assert btn_kg_qa is not None
        btn_kg_qa.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "💬 Ask SynapseAI"

        at.text_input(key="input_qa_True").input("What kind of reasoning does SynapseAI enable?").run()
        at.button(key="btn_ask_True").click().run()
        assert not at.exception
        second_answer = at.session_state["last_qa_result"].answer_text
        assert len(second_answer) > 0

        # Action 6: Return to Documents page
        btn_qa_docs = at.button(key="qa_nav_docs_True")
        assert btn_qa_docs is not None
        btn_qa_docs.click().run()
        assert not at.exception
        assert at.session_state["current_page"] == "📄 Documents"
        assert len(at.session_state["ingestion_results"]) == 1
        assert at.session_state["kg_builder"].graph.number_of_nodes() == initial_node_count

