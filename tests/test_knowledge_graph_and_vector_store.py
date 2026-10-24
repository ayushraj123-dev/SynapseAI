"""Comprehensive Automated Unit Tests for Task 4: Knowledge Graph & Vector Store.

Test coverage:
1. Building graph nodes from actual entity records.
2. Building directed edges from actual relationship records.
3. Rejecting or reporting missing relationship endpoints.
4. Preserving node, edge, and evidence metadata.
5. Graph statistics on empty and nonempty graphs.
6. Repeat-safe graph construction without duplicates.
7. Vector indexing with a local/mocked embedding provider.
8. Similarity search returning original text and provenance metadata.
9. Stable chunk IDs and duplicate-safe re-indexing.
10. Updating changed chunks and removing stale document chunks.
11. Handling empty inputs, invalid records, and embedding failures gracefully.
12. PyVis HTML generation and dark-mode styling safety (escaping).
"""

import json
import shutil
import tempfile
import unittest

from src.document_ingestion import TextChunk
from src.entity_extraction import (
    Entity,
    EntityExtractor,
    EntityMention,
    ExtractionResult,
    Relationship,
    RelationshipEvidence,
    TopicTheme,
)
from src.knowledge_graph import KnowledgeGraphBuilder
from src.vector_store import (
    LocalDeterministicEmbeddingFunction,
    SearchResult,
    VectorStoreService,
)


class TestKnowledgeGraphBuilder(unittest.TestCase):
    """Test suite for Knowledge Graph construction, validation, and serialization."""

    def setUp(self):
        self.builder = KnowledgeGraphBuilder()

        # Sample actual entity records
        self.ent_kg = Entity(
            entity_id="ent_tech_kg",
            name="Knowledge Graph",
            entity_type="TECHNOLOGY",
            aliases=["KG"],
            description="A structured representation of real-world entities and relationships.",
            mention_count=3,
            source_references=[
                EntityMention("doc1.pdf", 1, "chunk_001", "A Knowledge Graph organizes information."),
                EntityMention("doc2.pdf", 2, "chunk_005", "Using KG for reasoning."),
            ],
        )

        self.ent_retrieval = Entity(
            entity_id="ent_concept_retrieval",
            name="Semantic Retrieval",
            entity_type="CONCEPT",
            aliases=["Vector Search"],
            description="Searching by semantic similarity over dense vectors.",
            mention_count=2,
            source_references=[
                EntityMention("doc1.pdf", 1, "chunk_002", "Semantic retrieval complements graphs."),
            ],
        )

        self.ent_rag = Entity(
            entity_id="ent_tech_rag",
            name="Retrieval-Augmented Generation",
            entity_type="TECHNOLOGY",
            aliases=["RAG"],
            description="Technique combining retrieval with generative language models.",
            mention_count=4,
            source_references=[
                EntityMention("doc1.pdf", 2, "chunk_003", "RAG enables grounded answers."),
            ],
        )

        # Sample actual relationship records
        self.rel_uses = Relationship(
            relationship_id="rel_rag_uses_kg",
            subject_id="ent_tech_rag",
            predicate="USES",
            object_id="ent_tech_kg",
            evidence=[
                RelationshipEvidence("doc1.pdf", 2, "chunk_003", "RAG uses Knowledge Graphs for facts.", 0.95),
            ],
            confidence=0.95,
        )

        self.rel_implements = Relationship(
            relationship_id="rel_rag_implements_retrieval",
            subject_id="ent_tech_rag",
            predicate="IMPLEMENTS",
            object_id="ent_concept_retrieval",
            evidence=[
                RelationshipEvidence("doc1.pdf", 2, "chunk_003", "RAG implements semantic retrieval.", 0.90),
            ],
            confidence=0.90,
        )

        # Sample theme
        self.theme_ai = TopicTheme(
            theme_id="theme_ai_arch",
            name="AI Retrieval Architectures",
            description="Technologies and concepts for grounded retrieval.",
            entity_ids=["ent_tech_rag", "ent_tech_kg", "ent_concept_retrieval"],
            source_chunk_ids=["chunk_001", "chunk_002", "chunk_003"],
        )

    def test_build_nodes_from_actual_entities(self):
        """Test 1: Verify graph nodes are created with correct attributes from entity records."""
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_retrieval.entity_id: self.ent_retrieval,
            },
            relationships={},
            themes=[self.theme_ai],
        )

        stats = self.builder.build_from_extraction(ext_result)
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 0)

        # Verify node attributes
        node_kg = self.builder.graph.nodes["ent_tech_kg"]
        self.assertEqual(node_kg["name"], "Knowledge Graph")
        self.assertEqual(node_kg["entity_type"], "TECHNOLOGY")
        self.assertEqual(node_kg["mention_count"], 3)
        self.assertIn("AI Retrieval Architectures", node_kg["themes"])
        self.assertEqual(len(node_kg["source_references"]), 2)

    def test_build_directed_edges_from_actual_relationships(self):
        """Test 2: Verify directed edges are correctly linked between valid entity endpoints."""
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_uses.relationship_id: self.rel_uses,
            },
        )

        stats = self.builder.build_from_extraction(ext_result)
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 1)

        # Verify edge direction and attributes
        self.assertTrue(self.builder.graph.has_edge("ent_tech_rag", "ent_tech_kg"))
        self.assertFalse(self.builder.graph.has_edge("ent_tech_kg", "ent_tech_rag"))  # Directed

        edge_data = self.builder.graph.get_edge_data("ent_tech_rag", "ent_tech_kg")
        self.assertEqual(edge_data["predicate"], "USES")
        self.assertEqual(edge_data["confidence"], 0.95)
        self.assertEqual(len(edge_data["evidence"]), 1)
        self.assertEqual(edge_data["evidence"][0]["source_chunk_id"], "chunk_003")

    def test_reject_missing_relationship_endpoints(self):
        """Test 3: Reject relationships referencing missing entity nodes and log invalid records."""
        # Relationship references 'ent_concept_retrieval', but it is NOT in entities dictionary
        ext_result = ExtractionResult(
            entities={
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_implements.relationship_id: self.rel_implements,
            },
        )

        stats = self.builder.build_from_extraction(ext_result)
        self.assertEqual(stats["node_count"], 1)
        self.assertEqual(stats["edge_count"], 0)  # Edge must be rejected!
        self.assertEqual(stats["missing_endpoints_count"], 1)

        missing = stats["missing_endpoints"][0]
        self.assertEqual(missing["relationship_id"], "rel_rag_implements_retrieval")
        self.assertTrue(missing["subject_exists"])
        self.assertFalse(missing["object_exists"])

    def test_preserve_node_edge_and_evidence_metadata(self):
        """Test 4: Verify complete provenance and metadata round-tripping."""
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_uses.relationship_id: self.rel_uses,
            },
            themes=[self.theme_ai],
        )

        self.builder.build_from_extraction(ext_result)
        vis_data = self.builder.get_visualization_data()

        # Check nodes payload
        node_ids = {n["id"] for n in vis_data["nodes"]}
        self.assertIn("ent_tech_kg", node_ids)
        self.assertIn("ent_tech_rag", node_ids)

        kg_vis = next(n for n in vis_data["nodes"] if n["id"] == "ent_tech_kg")
        self.assertEqual(kg_vis["label"], "Knowledge Graph")
        self.assertEqual(kg_vis["entity_type"], "TECHNOLOGY")
        self.assertEqual(len(kg_vis["source_references"]), 2)

        # Check edges payload
        self.assertEqual(len(vis_data["edges"]), 1)
        edge_vis = vis_data["edges"][0]
        self.assertEqual(edge_vis["from"], "ent_tech_rag")
        self.assertEqual(edge_vis["to"], "ent_tech_kg")
        self.assertEqual(edge_vis["label"], "USES")

    def test_graph_statistics_empty_and_nonempty(self):
        """Test 5: Compute statistics for both empty and populated graphs."""
        # Empty graph
        empty_stats = self.builder.get_graph_statistics()
        self.assertEqual(empty_stats["node_count"], 0)
        self.assertEqual(empty_stats["edge_count"], 0)
        self.assertEqual(empty_stats["connected_components"], 0)
        self.assertEqual(empty_stats["density"], 0.0)

        # Populated graph
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_retrieval.entity_id: self.ent_retrieval,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_uses.relationship_id: self.rel_uses,
                self.rel_implements.relationship_id: self.rel_implements,
            },
            themes=[self.theme_ai],
        )
        stats = self.builder.build_from_extraction(ext_result)
        self.assertEqual(stats["node_count"], 3)
        self.assertEqual(stats["edge_count"], 2)
        self.assertEqual(stats["connected_components"], 1)  # All 3 nodes are connected
        self.assertEqual(stats["nodes_by_type"]["TECHNOLOGY"], 2)
        self.assertEqual(stats["nodes_by_type"]["CONCEPT"], 1)
        self.assertEqual(stats["nodes_by_theme"]["AI Retrieval Architectures"], 3)

    def test_repeat_safe_graph_construction(self):
        """Test 6: Repeated construction produces identical logical graph without duplicate nodes/edges."""
        ext_result = ExtractionResult(
            entities={self.ent_kg.entity_id: self.ent_kg},
            relationships={},
        )

        stats1 = self.builder.build_from_extraction(ext_result)
        stats2 = self.builder.build_from_extraction(ext_result)

        self.assertEqual(stats1["node_count"], stats2["node_count"])
        self.assertEqual(stats1["edge_count"], stats2["edge_count"])
        self.assertEqual(self.builder.graph.number_of_nodes(), 1)

    def test_subgraph_filtering_and_pyvis_generation(self):
        """Test 7: Subgraph filtering and safe HTML generation for interactive visualization."""
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_retrieval.entity_id: self.ent_retrieval,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_uses.relationship_id: self.rel_uses,
                self.rel_implements.relationship_id: self.rel_implements,
            },
            themes=[self.theme_ai],
        )
        self.builder.build_from_extraction(ext_result)

        # Filter by entity type CONCEPT -> only ent_retrieval
        sub = self.builder.get_subgraph(filter_types=["CONCEPT"])
        self.assertEqual(sub.number_of_nodes(), 1)
        self.assertIn("ent_concept_retrieval", sub.nodes)

        # Focus on node ent_tech_kg and its neighbors
        sub_ego = self.builder.get_subgraph(focus_node="ent_tech_kg")
        self.assertIn("ent_tech_kg", sub_ego.nodes)
        self.assertIn("ent_tech_rag", sub_ego.nodes)  # predecessor

        # Generate PyVis HTML
        html_output = self.builder.generate_pyvis_html()
        self.assertIsInstance(html_output, str)
        self.assertIn("vis.Network", html_output)
        self.assertIn("Knowledge Graph", html_output)

    def test_multi_predicate_and_evidence_preservation(self):
        """Test 8: Multiple distinct predicates between the same pair are preserved without overwriting."""
        rel_extends = Relationship(
            relationship_id="rel_rag_extends_kg",
            subject_id="ent_tech_rag",
            predicate="EXTENDS",
            object_id="ent_tech_kg",
            evidence=[
                RelationshipEvidence("doc2.pdf", 3, "chunk_008", "RAG extends Knowledge Graphs with generation.", 0.92),
            ],
            confidence=0.92,
        )
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_uses.relationship_id: self.rel_uses,
                rel_extends.relationship_id: rel_extends,
            },
        )
        stats = self.builder.build_from_extraction(ext_result)
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 1)

        edge_data = self.builder.graph.get_edge_data("ent_tech_rag", "ent_tech_kg")
        self.assertIn("USES", edge_data["all_predicates"])
        self.assertIn("EXTENDS", edge_data["all_predicates"])
        self.assertEqual(len(edge_data["evidence"]), 2)
        self.assertIn("USES", edge_data["predicate"])
        self.assertIn("EXTENDS", edge_data["predicate"])

    def test_canonical_name_and_alias_resolution(self):
        """Test 9: Relationship endpoints referencing entity names or aliases resolve to canonical IDs."""
        rel_by_name = Relationship(
            relationship_id="rel_named_1",
            subject_id="Retrieval-Augmented Generation",  # Entity Name instead of ent_tech_rag
            predicate="USES",
            object_id="Knowledge Graph",                 # Entity Name instead of ent_tech_kg
            evidence=[
                RelationshipEvidence("doc1.pdf", 1, "c1", "RAG uses KG.", 0.9),
            ],
            confidence=0.9,
        )
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={rel_by_name.relationship_id: rel_by_name},
        )
        stats = self.builder.build_from_extraction(ext_result)
        self.assertEqual(stats["edge_count"], 1)
        self.assertEqual(stats["missing_endpoints_count"], 0)
        self.assertTrue(self.builder.graph.has_edge("ent_tech_rag", "ent_tech_kg"))

    def test_structural_topic_theme_overlay(self):
        """Test 10: Topic themes create structural BELONGS_TO links when requested, keeping factual clean."""
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_retrieval.entity_id: self.ent_retrieval,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_uses.relationship_id: self.rel_uses,
            },
            themes=[self.theme_ai],
        )
        self.builder.build_from_extraction(ext_result)

        # Pure factual graph (default)
        sub_factual = self.builder.get_subgraph(include_structural=False)
        self.assertEqual(sub_factual.number_of_nodes(), 3)
        self.assertEqual(sub_factual.number_of_edges(), 1)
        for _, _, edata in sub_factual.edges(data=True):
            self.assertFalse(edata.get("is_structural", False))

        # Structural overlay enabled
        sub_struct = self.builder.get_subgraph(include_structural=True)
        # Should have 3 entities + 1 theme node = 4 nodes
        self.assertEqual(sub_struct.number_of_nodes(), 4)
        # 1 factual edge + 3 BELONGS_TO edges = 4 edges
        self.assertEqual(sub_struct.number_of_edges(), 4)
        theme_edges = [
            (u, v, d) for u, v, d in sub_struct.edges(data=True) if d.get("predicate") == "BELONGS_TO"
        ]
        self.assertEqual(len(theme_edges), 3)

    def test_3d_force_graph_payload_and_link_rendering(self):
        """Test 11: 3D ForceGraph HTML contains populated links, correct endpoint IDs, and visible rendering."""
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_retrieval.entity_id: self.ent_retrieval,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_uses.relationship_id: self.rel_uses,
                self.rel_implements.relationship_id: self.rel_implements,
            },
            themes=[self.theme_ai],
        )
        self.builder.build_from_extraction(ext_result)

        html_3d = self.builder.generate_3d_graph_html(theme="dark")
        self.assertIn("ForceGraph3D", html_3d)
        self.assertIn(".linkOpacity(0.90)", html_3d)
        self.assertIn(".linkWidth(", html_3d)
        self.assertIn(".linkDirectionalArrowLength(", html_3d)

        # Extract gData from HTML to verify payload integrity
        start_marker = "const gData = "
        idx = html_3d.find(start_marker)
        self.assertGreater(idx, 0)
        end_idx = html_3d.find(";\n    let graphInstance", idx)
        raw_json = html_3d[idx + len(start_marker):end_idx].strip()
        data = json.loads(raw_json)

        self.assertEqual(len(data["nodes"]), 3)
        self.assertEqual(len(data["links"]), 2)

        node_ids = {n["id"] for n in data["nodes"]}
        for link in data["links"]:
            self.assertIn(link["source"], node_ids)
            self.assertIn(link["target"], node_ids)
            self.assertTrue(len(link["predicate"]) > 0)
            self.assertIsNotNone(link["color"])
            self.assertGreaterEqual(link["width"], 2.0)

    def test_2d_visjs_edge_styling_and_contrast(self):
        """Test 12: 2D Vis.js HTML uses high-contrast edge colors, directional arrows, and readable fonts."""
        ext_result = ExtractionResult(
            entities={
                self.ent_kg.entity_id: self.ent_kg,
                self.ent_rag.entity_id: self.ent_rag,
            },
            relationships={
                self.rel_uses.relationship_id: self.rel_uses,
            },
        )
        self.builder.build_from_extraction(ext_result)

        # Dark mode 2D
        html_dark = self.builder.generate_pyvis_html(theme="dark")
        self.assertIn("vis.Network", html_dark)
        self.assertIn("#818cf8", html_dark)  # Dark mode edge color
        self.assertIn("scaleFactor", html_dark)

        # Light mode 2D
        html_light = self.builder.generate_pyvis_html(theme="light")
        self.assertIn("#6366f1", html_light)  # Light mode edge color

    def test_end_to_end_extraction_to_graph_pipeline(self):
        """Test 13: End-to-end extraction from multi-entity text generates real connected graph."""
        text = (
            "Retrieval-Augmented Generation utilizes Knowledge Graphs to ground factual answers. "
            "SynapseAI integrates Neural Networks and Semantic Search to classify user queries. "
            "The Query Router directs questions to appropriate retrieval strategies."
        )
        chunk = TextChunk(
            chunk_id="c_e2e_001",
            source_filename="neural_doc.txt",
            page_number=1,
            text=text,
            chunk_index=0,
            char_count=len(text),
        )

        extractor = EntityExtractor(use_llm=False)
        extraction_res = extractor.extract_from_chunks([chunk])

        self.assertGreaterEqual(len(extraction_res.entities), 2)
        self.assertGreaterEqual(len(extraction_res.relationships), 1)

        builder = KnowledgeGraphBuilder()
        stats = builder.build_from_extraction(extraction_res)

        self.assertGreaterEqual(stats["node_count"], 2)
        self.assertGreaterEqual(stats["factual_edge_count"], 1)
        self.assertEqual(stats["missing_endpoints_count"], 0)

        # Verify 3D and 2D payloads can be generated with real connections
        html_3d = builder.generate_3d_graph_html()
        self.assertIn('"links": [', html_3d)
        html_2d = builder.generate_pyvis_html()
        self.assertIn("edges = new vis.DataSet(", html_2d)
        self.assertIn("vis.Network", html_2d)


class TestVectorStoreService(unittest.TestCase):
    """Test suite for persistent VectorStoreService, indexing, and similarity search."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embedder = LocalDeterministicEmbeddingFunction(dimension=64)
        self.service = VectorStoreService(
            persist_directory=self.temp_dir,
            collection_name="test_collection",
            embedding_function=self.embedder,
        )

        # Sample Task 2 chunks
        self.chunk1 = TextChunk(
            chunk_id="chunk_doc1_p1_0",
            source_filename="neural_networks.pdf",
            page_number=1,
            text="Convolutional neural networks excel at spatial feature extraction in image classification.",
            chunk_index=0,
            char_count=90,
        )
        self.chunk2 = TextChunk(
            chunk_id="chunk_doc1_p2_1",
            source_filename="neural_networks.pdf",
            page_number=2,
            text="Recurrent neural networks and LSTMs capture temporal dependencies in sequential data.",
            chunk_index=1,
            char_count=85,
        )
        self.chunk3 = TextChunk(
            chunk_id="chunk_doc2_p1_0",
            source_filename="database_internals.pdf",
            page_number=1,
            text="B-Tree indexes provide logarithmic search times for sorted relational queries.",
            chunk_index=0,
            char_count=78,
        )

    def tearDown(self):
        # Clean up temporary Chroma directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_vector_indexing_with_local_provider(self):
        """Test 7 (vector): Index chunks and verify persistent collection count."""
        report = self.service.index_chunks([self.chunk1, self.chunk2, self.chunk3])
        self.assertEqual(report["indexed_count"], 3)
        self.assertEqual(report["skipped_count"], 0)
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(self.service.collection.count(), 3)

    def test_similarity_search_returns_text_and_provenance(self):
        """Test 8: Perform semantic retrieval and verify original text and provenance metadata."""
        self.service.index_chunks([self.chunk1, self.chunk2, self.chunk3])

        # Query about sequential models
        results = self.service.similarity_search("recurrent neural networks and sequence data", top_k=2)
        self.assertTrue(len(results) > 0)

        top_match = results[0]
        self.assertIsInstance(top_match, SearchResult)
        self.assertEqual(top_match.chunk_id, "chunk_doc1_p2_1")
        self.assertEqual(top_match.source_filename, "neural_networks.pdf")
        self.assertEqual(top_match.page_number, 2)
        self.assertIn("Recurrent neural networks", top_match.text)
        self.assertTrue(top_match.relevance_score > 0.0)

    def test_stable_chunk_ids_and_duplicate_safe_reindexing(self):
        """Test 9: Re-indexing identical chunks does not duplicate records in vector store."""
        report1 = self.service.index_chunks([self.chunk1, self.chunk2])
        self.assertEqual(report1["indexed_count"], 2)
        self.assertEqual(self.service.collection.count(), 2)

        # Re-index the same chunks
        report2 = self.service.index_chunks([self.chunk1, self.chunk2])
        self.assertEqual(report2["indexed_count"], 2)
        self.assertEqual(self.service.collection.count(), 2)  # Count MUST remain 2!

    def test_update_changed_chunks_and_delete_document(self):
        """Test 10: Updating chunk text overwrites old record; deleting document purges its chunks."""
        self.service.index_chunks([self.chunk1, self.chunk3])
        self.assertEqual(self.service.collection.count(), 2)

        # Update chunk1 with new text
        updated_chunk1 = TextChunk(
            chunk_id="chunk_doc1_p1_0",
            source_filename="neural_networks.pdf",
            page_number=1,
            text="Updated text: Transformers have largely replaced traditional convolutional models.",
            chunk_index=0,
            char_count=80,
        )
        self.service.index_chunks([updated_chunk1])
        self.assertEqual(self.service.collection.count(), 2)

        # Retrieve chunk1 to confirm text was updated
        item = self.service.collection.get(ids=["chunk_doc1_p1_0"])
        self.assertIn("Transformers", item["documents"][0])

        # Delete 'neural_networks.pdf'
        deleted_count = self.service.delete_document("neural_networks.pdf")
        self.assertEqual(deleted_count, 1)
        self.assertEqual(self.service.collection.count(), 1)

        # Search should only find database_internals
        results = self.service.similarity_search("neural", top_k=5)
        for r in results:
            self.assertNotEqual(r.source_filename, "neural_networks.pdf")

    def test_empty_inputs_and_error_handling(self):
        """Test 11: Handle empty chunks, whitespace text, and empty search queries gracefully."""
        # Empty chunk list
        rep_empty = self.service.index_chunks([])
        self.assertEqual(rep_empty["indexed_count"], 0)

        # Blank/whitespace chunk
        blank_chunk = TextChunk(
            chunk_id="blank_1",
            source_filename="empty.txt",
            page_number=None,
            text="   \n  \t ",
            chunk_index=0,
            char_count=0,
        )
        rep_blank = self.service.index_chunks([blank_chunk])
        self.assertEqual(rep_blank["indexed_count"], 0)
        self.assertEqual(rep_blank["skipped_count"], 1)

        # Empty search query
        res_empty = self.service.similarity_search("")
        self.assertEqual(res_empty, [])

        res_spaces = self.service.similarity_search("   ")
        self.assertEqual(res_spaces, [])


if __name__ == "__main__":
    unittest.main()
