"""Comprehensive test suite for Neo4j persistence, cross-document themes, and graph highlighting.

Covers:
- Neo4jGraphService connection checking, offline fallback, and parameterized Cypher queries.
- Schema creation with uniqueness constraints and indexes.
- Repeat-safe document upsert, provenance tracking, and document deletion.
- Cross-document theme community detection (TopicTheme.is_cross_document).
- HybridRetrievalResult query highlighting contract for factual, relational, thematic, and unanswerable queries.
- PyVis visual highlighting and auto-zoom script injection.
- End-to-end multi-document workflow simulation.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.document_ingestion import DocumentIngestionService, TextChunk
from src.entity_extraction import (
    Entity,
    EntityExtractor,
    EntityMention,
    ExtractionResult,
    Relationship,
    RelationshipEvidence,
    ThemeGenerator,
    TopicTheme,
)
from src.graph_database import GraphConnectionStatus, Neo4jGraphService
from src.knowledge_graph import KnowledgeGraphBuilder
from src.query_router import (
    HybridRetrievalResult,
    QueryClassification,
    QueryIntent,
    QueryRouter,
    RetrievalPlan,
)
from src.vector_store import VectorStoreService


class TestNeo4jGraphService:
    """Tests for Neo4j graph database integration, Cypher queries, and error handling."""

    def test_default_offline_behavior(self, monkeypatch):
        """When Neo4j is offline or credentials are dummy, service handles gracefully."""
        # Point to unreachable port
        monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:9999")
        svc = Neo4jGraphService(uri="bolt://127.0.0.1:9999")
        assert svc.is_connected is False
        assert svc.driver is None

        # check_connection reports status tuple without crashing
        status, msg = svc.check_connection()
        assert status != "connected"
        assert svc.is_connected is False
        assert len(msg) > 0

        # Operations fail gracefully with empty returns
        stats = svc.get_statistics()
        assert stats["node_count"] == 0
        assert stats["is_connected"] is False

        rels = svc.query_relationships(["SynapseAI"])
        assert rels == []

        themes = svc.query_themes()
        assert themes == []

    def test_schema_creation_cypher_calls(self):
        """ensure_schema executes parameterized Cypher constraint and index statements."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        svc = Neo4jGraphService(uri="bolt://localhost:7687", auto_connect=False)
        svc.driver = mock_driver
        svc._cached_status = (GraphConnectionStatus.CONNECTED, "Connected")

        result = svc.ensure_schema()
        assert result is True

        # Verify that run was called for constraints and indexes
        executed_queries = [call[0][0] for call in mock_session.run.call_args_list]
        assert any("CONSTRAINT" in q and "Entity" in q for q in executed_queries)
        assert any("CONSTRAINT" in q and "Document" in q for q in executed_queries)
        assert any("CONSTRAINT" in q and "Chunk" in q for q in executed_queries)
        assert any("CONSTRAINT" in q and "Theme" in q for q in executed_queries)

    def test_upsert_document_graph_cypher_structure(self):
        """upsert_document_graph runs deletion then batch node/relationship creations."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        svc = Neo4jGraphService(uri="bolt://localhost:7687", auto_connect=False)
        svc.driver = mock_driver
        svc._cached_status = (GraphConnectionStatus.CONNECTED, "Connected")

        # Create extraction sample
        ent1 = Entity(
            entity_id="ent_chromadb",
            name="ChromaDB",
            entity_type="DATABASE",
            description="Vector database",
            mention_count=2,
        )
        ent2 = Entity(
            entity_id="ent_synapseai",
            name="SynapseAI",
            entity_type="SYSTEM",
            description="Knowledge Graph system",
            mention_count=3,
        )
        rel = Relationship(
            relationship_id="rel_1",
            subject_id="ent_synapseai",
            predicate="USES",
            object_id="ent_chromadb",
            confidence=0.95,
            evidence=[
                RelationshipEvidence(
                    source_filename="synapse_doc.txt",
                    page_number=1,
                    source_chunk_id="chk_1",
                    evidence_text="SynapseAI uses ChromaDB for persistent vector search.",
                )
            ],
        )
        theme = TopicTheme(
            theme_id="theme_storage",
            name="Storage Architecture",
            description="Persistence layer",
            entity_ids=["ent_chromadb", "ent_synapseai"],
            document_filenames=["synapse_doc.txt"],
        )
        ext_result = ExtractionResult(
            entities={"ent_chromadb": ent1, "ent_synapseai": ent2},
            relationships={"rel_1": rel},
            themes=[theme],
        )
        chunks = [
            TextChunk(
                chunk_id="chk_1",
                text="SynapseAI uses ChromaDB for persistent vector search.",
                source_filename="synapse_doc.txt",
                page_number=1,
                chunk_index=0,
                char_count=52,
            )
        ]

        report = svc.upsert_document_graph("synapse_doc.txt", ext_result, chunks)
        assert report["document_filename"] == "synapse_doc.txt"
        assert report["entities_upserted"] == 2
        assert report["relationships_upserted"] == 1
        assert report["themes_upserted"] == 1

        # Check Cypher queries executed in session
        queries = [call[0][0] for call in mock_session.run.call_args_list]
        # Must delete prior records for repeatability
        assert any("DETACH DELETE" in q for q in queries)
        # Must merge document
        assert any("MERGE (d:Document" in q for q in queries)
        # Must create Chunk and CONTAINS
        assert any("MERGE (c:Chunk" in q and "CONTAINS" in q for q in queries)
        # Must create Entity and MENTIONS
        assert any("MERGE (e:Entity" in q and "MENTIONS" in q for q in queries)
        # Must create factual RELATED_TO with predicate
        assert any("MERGE (s)-[r:RELATED_TO" in q for q in queries)
        # Must create Theme and BELONGS_TO
        assert any("MERGE (t:Theme" in q and "BELONGS_TO" in q for q in queries)

    def test_delete_document_graph(self):
        """delete_document_graph cleanly removes document-specific chunks and relationships."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        svc = Neo4jGraphService(uri="bolt://localhost:7687", auto_connect=False)
        svc.driver = mock_driver
        svc._cached_status = (GraphConnectionStatus.CONNECTED, "Connected")

        success = svc.delete_document_graph("test_doc.pdf")
        assert success is True

        delete_query = mock_session.run.call_args_list[0][0][0]
        params = mock_session.run.call_args_list[0][0][1]
        assert "MATCH (d:Document {filename: $filename})" in delete_query
        assert params["filename"] == "test_doc.pdf"

    def test_export_to_networkx(self):
        """export_to_networkx rebuilds a NetworkX DiGraph from Cypher query results."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        svc = Neo4jGraphService(uri="bolt://localhost:7687", auto_connect=False)
        svc.driver = mock_driver
        svc._cached_status = (GraphConnectionStatus.CONNECTED, "Connected")

        # Mock node record
        mock_node_record = MagicMock()
        mock_node_record.__getitem__.side_effect = lambda k: {
            "id": "ent_1",
            "name": "Alpha",
            "type": "CONCEPT",
            "desc": "Alpha concept",
            "mentions": 2,
        }[k]

        # Mock edge record
        mock_edge_record = MagicMock()
        mock_edge_record.__getitem__.side_effect = lambda k: {
            "source_id": "ent_1",
            "target_id": "ent_2",
            "predicate": "LINKS",
            "confidence": 0.9,
            "evidence": "Evidence text",
            "source_file": "doc1.txt",
        }[k]

        # Two calls to run: nodes query then edges query
        mock_session.run.side_effect = [[mock_node_record], [mock_edge_record]]

        nx_graph = svc.export_to_networkx()
        assert nx_graph.has_node("ent_1")
        assert nx_graph.nodes["ent_1"]["name"] == "Alpha"
        assert nx_graph.has_edge("ent_1", "ent_2")
        assert nx_graph.get_edge_data("ent_1", "ent_2")["predicate"] == "LINKS"


class TestCrossDocumentThemes:
    """Tests for multi-document theme detection and provenance tracking."""

    def test_topic_theme_is_cross_document_property(self):
        """TopicTheme.is_cross_document is True if and only if len(document_filenames) > 1."""
        single_doc_theme = TopicTheme(
            theme_id="theme_1",
            name="Quantum Algorithms",
            description="Specific to physics paper",
            document_filenames=["physics_paper.pdf"],
        )
        assert single_doc_theme.is_cross_document is False

        multi_doc_theme = TopicTheme(
            theme_id="theme_2",
            name="Machine Learning",
            description="Shared across AI and medical papers",
            document_filenames=["ai_overview.pdf", "medical_imaging.pdf"],
        )
        assert multi_doc_theme.is_cross_document is True

        empty_theme = TopicTheme(
            theme_id="theme_3",
            name="Empty",
            description="",
            document_filenames=[],
        )
        assert empty_theme.is_cross_document is False

    def test_theme_generator_populates_cross_document_filenames(self):
        """ThemeGenerator accurately collects filenames from entity provenance and chunks."""
        generator = ThemeGenerator()

        ent_a = Entity(
            entity_id="ent_a",
            name="Transformer",
            entity_type="MODEL",
            source_references=[EntityMention(source_filename="nlp_guide.pdf", page_number=1, chunk_id="chk_1", text_snippet="Transformer architecture")],
        )
        ent_b = Entity(
            entity_id="ent_b",
            name="Self-Attention",
            entity_type="TECHNIQUE",
            source_references=[EntityMention(source_filename="vision_transformer.pdf", page_number=1, chunk_id="chk_2", text_snippet="Self-Attention mechanism")],
        )
        rel = Relationship(
            relationship_id="rel_ab",
            subject_id="ent_a",
            predicate="IMPLEMENTS",
            object_id="ent_b",
        )
        chunks = [
            TextChunk(chunk_id="chk_1", text="...", source_filename="nlp_guide.pdf", page_number=1, chunk_index=0, char_count=3),
            TextChunk(chunk_id="chk_2", text="...", source_filename="vision_transformer.pdf", page_number=1, chunk_index=1, char_count=3),
        ]

        themes = generator.generate_themes(
            entities={"ent_a": ent_a, "ent_b": ent_b},
            relationships={"rel_ab": rel},
            chunks=chunks,
        )

        assert len(themes) >= 1
        found_multi = any(t.is_cross_document for t in themes)
        assert found_multi is True
        for t in themes:
            if t.is_cross_document:
                assert "nlp_guide.pdf" in t.document_filenames
                assert "vision_transformer.pdf" in t.document_filenames


class TestGraphHighlightAndQueryRouterContract:
    """Verifies that HybridRetrievalResult returns exact highlight IDs for UI focusing."""

    def setup_method(self):
        self.router = QueryRouter()
        self.kg_builder = KnowledgeGraphBuilder()

        # Build a 3-node connected graph
        self.kg_builder.build_graph(
            entities=[
                {"entity_id": "ent_synapse", "name": "SynapseAI", "entity_type": "SYSTEM", "source_references": [{"source_filename": "synapse.pdf", "chunk_id": "c1", "page_number": 1}]},
                {"entity_id": "ent_chroma", "name": "ChromaDB", "entity_type": "DATABASE", "source_references": [{"source_filename": "synapse.pdf", "chunk_id": "c1", "page_number": 1}]},
                {"entity_id": "ent_neo4j", "name": "Neo4j", "entity_type": "DATABASE", "source_references": [{"source_filename": "database.txt", "chunk_id": "c2", "page_number": None}]},
            ],
            relations=[
                {"relationship_id": "r1", "subject_id": "ent_synapse", "object_id": "ent_chroma", "predicate": "STORES_VECTORS_IN", "confidence": 0.95, "evidence": [{"source_filename": "synapse.pdf", "page_number": 1, "source_chunk_id": "c1", "evidence_text": "SynapseAI stores vectors in ChromaDB."}]},
                {"relationship_id": "r2", "subject_id": "ent_synapse", "object_id": "ent_neo4j", "predicate": "PERSISTS_GRAPH_IN", "confidence": 0.90, "evidence": [{"source_filename": "database.txt", "page_number": None, "source_chunk_id": "c2", "evidence_text": "SynapseAI persists graph in Neo4j."}]},
            ],
            themes=[
                {"theme_id": "theme_infra", "name": "Infrastructure", "description": "Storage systems", "entity_ids": ["ent_chroma", "ent_neo4j"], "document_filenames": ["synapse.pdf", "database.txt"]}
            ],
        )

    def test_relationship_query_populates_path_and_edge_highlights(self):
        """Relationship retrieval returns nodes and edges along the relationship path."""
        plan = RetrievalPlan(
            intent=QueryIntent.RELATIONSHIP,
            use_vector_search=False,
            use_graph_search=True,
            use_thematic_search=False,
            vector_top_k=0,
            target_entities=["SynapseAI", "ChromaDB"],
        )

        res = self.router.execute_hybrid_retrieval(
            query="How is SynapseAI connected to ChromaDB?",
            plan=plan,
            knowledge_graph=self.kg_builder,
        )

        assert not res.is_empty
        assert "ent_synapse" in res.highlight_node_ids
        assert "ent_chroma" in res.highlight_node_ids
        assert ("ent_synapse", "ent_chroma") in res.highlight_edge_ids
        assert "synapse.pdf" in res.contributing_document_ids

    def test_thematic_query_populates_theme_and_cluster_highlights(self):
        """Thematic retrieval highlights all entities and edges in the theme community."""
        plan = RetrievalPlan(
            intent=QueryIntent.THEMATIC,
            use_vector_search=False,
            use_graph_search=False,
            use_thematic_search=True,
            vector_top_k=0,
            target_themes=["Infrastructure"],
        )

        res = self.router.execute_hybrid_retrieval(
            query="What is the overall infrastructure theme across documents?",
            plan=plan,
            knowledge_graph=self.kg_builder,
        )

        assert not res.is_empty
        assert "theme_infra" in res.highlight_theme_ids
        assert "ent_chroma" in res.highlight_node_ids
        assert "ent_neo4j" in res.highlight_node_ids
        assert "synapse.pdf" in res.contributing_document_ids
        assert "database.txt" in res.contributing_document_ids

    def test_unanswerable_query_guarantees_empty_highlights(self):
        """Unanswerable queries MUST NOT highlight unrelated graph elements."""
        plan = RetrievalPlan(
            intent=QueryIntent.UNANSWERABLE,
            use_vector_search=False,
            use_graph_search=False,
            use_thematic_search=False,
            vector_top_k=0,
        )

        res = self.router.execute_hybrid_retrieval(
            query="What is the average speed of a peregrine falcon in flight?",
            plan=plan,
            knowledge_graph=self.kg_builder,
        )

        assert res.is_empty is True
        assert res.highlight_node_ids == []
        assert res.highlight_edge_ids == []
        assert res.highlight_theme_ids == []
        assert res.contributing_document_ids == []


class TestPyVisHighlightingAndZoom:
    """Verifies that PyVis visualization injects dynamic styling and auto-zoom scripts."""

    def test_generate_pyvis_with_evidence_highlights(self):
        """Highlighted nodes and edges are emphasized while unhighlighted nodes are dimmed."""
        builder = KnowledgeGraphBuilder()
        builder.build_graph(
            entities=[
                {"entity_id": "ent_1", "name": "NodeOne", "entity_type": "CONCEPT"},
                {"entity_id": "ent_2", "name": "NodeTwo", "entity_type": "CONCEPT"},
                {"entity_id": "ent_3", "name": "NodeThree", "entity_type": "CONCEPT"},
            ],
            relations=[
                {"relationship_id": "r1", "subject_id": "ent_1", "object_id": "ent_2", "predicate": "CONNECTS"},
                {"relationship_id": "r2", "subject_id": "ent_2", "object_id": "ent_3", "predicate": "CONNECTS"},
            ],
        )

        html = builder.generate_pyvis_html(
            highlight_nodes=["ent_1", "ent_2"],
            highlight_edges=[("ent_1", "ent_2")],
        )

        # Highlighted node styling is present
        assert "#38bdf8" in html
        # RELEVANT EVIDENCE badge in tooltip
        assert "[RELEVANT EVIDENCE]" in html
        # Auto-fit zoom script is injected
        assert "network.fit" in html
        assert "ent_1" in html and "ent_2" in html


class Test3DSpatialGraphGeneration:
    """Verifies that 3D WebGL / ForceGraph3D and 3D landing empty-state constellation HTML are generated correctly."""

    def test_generate_3d_graph_html_structure(self):
        """3D graph HTML contains 3D Force Graph library, WebGL canvas, and offline canvas fallback."""
        builder = KnowledgeGraphBuilder()
        builder.build_graph(
            entities=[
                {"entity_id": "ent_1", "name": "NodeOne", "entity_type": "CONCEPT"},
                {"entity_id": "ent_2", "name": "NodeTwo", "entity_type": "TECHNOLOGY"},
            ],
            relations=[
                {"relationship_id": "r1", "subject_id": "ent_1", "object_id": "ent_2", "predicate": "USES"},
            ],
        )

        html_3d = builder.generate_3d_graph_html(
            highlight_nodes=["ent_1"],
            highlight_edges=[("ent_1", "ent_2")],
            height="550px",
        )

        assert "Synapse 3D Knowledge Space" in html_3d
        assert "3d-graph" in html_3d
        assert "ForceGraph3D" in html_3d
        assert "fallback-canvas" in html_3d
        assert "runFallbackCanvas" in html_3d
        assert "NodeOne" in html_3d
        assert "NodeTwo" in html_3d

    def test_generate_empty_state_3d_html_structure(self):
        """Empty state HTML renders interactive 3D constellation with hackathon headline."""
        builder = KnowledgeGraphBuilder()
        empty_html = builder.generate_empty_state_3d_html(height="460px")

        assert "Your documents. Connected knowledge. Clear answers." in empty_html
        assert '<canvas id="c"></canvas>' in empty_html
        assert "Synaptic Knowledge Graph Engine" in empty_html
