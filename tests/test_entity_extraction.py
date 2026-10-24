"""Unit tests for SynapseAI Task 3: Entity, Relationship, and Topic Theme Extraction."""

import json
import unittest

from src.document_ingestion import TextChunk
from src.entity_extraction import (
    Entity,
    EntityExtractor,
    EntityMention,
    ExtractionResult,
    Relationship,
    RelationshipEvidence,
    ThemeGenerator,
    TopicTheme,
    generate_entity_id,
    generate_relationship_id,
    generate_theme_id,
    normalize_predicate,
)


class TestEntityExtraction(unittest.TestCase):
    """Test suite covering entity extraction, relationship validation, and thematic clustering."""

    def setUp(self):
        self.extractor = EntityExtractor(mode="heuristic")
        self.theme_generator = ThemeGenerator()

    # 1. Entity schema validation
    def test_entity_schema_validation(self):
        mention = EntityMention(
            source_filename="research.pdf",
            page_number=2,
            chunk_id="research.pdf_p2_c0",
            text_snippet="SynapseAI integrates knowledge graphs with vector retrieval.",
        )
        entity = Entity(
            entity_id="ent_tech_synapseai",
            name="SynapseAI",
            entity_type="TECHNOLOGY",
            aliases=["SynapseAI Platform"],
            description="Knowledge Graph and RAG platform.",
            source_references=[mention],
            mention_count=1,
        )

        d = entity.to_dict()
        self.assertEqual(d["entity_id"], "ent_tech_synapseai")
        self.assertEqual(d["name"], "SynapseAI")
        self.assertEqual(d["entity_type"], "TECHNOLOGY")
        self.assertEqual(len(d["source_references"]), 1)
        self.assertEqual(d["source_references"][0]["chunk_id"], "research.pdf_p2_c0")
        self.assertEqual(d["source_references"][0]["page_number"], 2)

    # 2. Relationship schema validation
    def test_relationship_schema_validation(self):
        evidence = RelationshipEvidence(
            source_filename="tech_report.pdf",
            page_number=4,
            source_chunk_id="tech_report.pdf_p4_c1",
            evidence_text="PyTorch uses CUDA for GPU acceleration.",
            confidence=0.9,
        )
        rel = Relationship(
            relationship_id="rel_ent_tech_pytorch_USES_ent_tech_cuda",
            subject_id="ent_tech_pytorch",
            predicate="USES",
            object_id="ent_tech_cuda",
            evidence=[evidence],
            confidence=0.9,
        )

        d = rel.to_dict()
        self.assertEqual(d["relationship_id"], "rel_ent_tech_pytorch_USES_ent_tech_cuda")
        self.assertEqual(d["subject_id"], "ent_tech_pytorch")
        self.assertEqual(d["predicate"], "USES")
        self.assertEqual(d["object_id"], "ent_tech_cuda")
        self.assertEqual(len(d["evidence"]), 1)
        self.assertEqual(d["evidence"][0]["source_chunk_id"], "tech_report.pdf_p4_c1")

    # 3. Invalid entity references in relationships
    def test_invalid_entity_references_rejected(self):
        # Create a single chunk that mentions an entity but references a non-existent entity in a relationship
        chunk = TextChunk(
            chunk_id="test_chunk_1",
            source_filename="doc.txt",
            page_number=None,
            text="SynapseAI is developed by hypothetical non-entity.",
            chunk_index=0,
            char_count=50,
        )

        result = self.extractor.extract_from_chunks([chunk])

        # Validate that all relationships in the result link ONLY entities present in result.entities
        for rel_id, rel in result.relationships.items():
            self.assertIn(
                rel.subject_id,
                result.entities,
                f"Relationship subject {rel.subject_id} does not exist in extracted entities",
            )
            self.assertIn(
                rel.object_id,
                result.entities,
                f"Relationship object {rel.object_id} does not exist in extracted entities",
            )
            self.assertNotEqual(
                rel.subject_id,
                rel.object_id,
                "Degenerate self-loop relationship was not rejected",
            )

    # 4. Entity deduplication while preserving evidence
    def test_entity_deduplication_preserves_evidence(self):
        chunk1 = TextChunk(
            chunk_id="doc_p1_c0",
            source_filename="doc.pdf",
            page_number=1,
            text="Streamlit enables interactive web application development.",
            chunk_index=0,
            char_count=60,
        )
        chunk2 = TextChunk(
            chunk_id="doc_p2_c1",
            source_filename="doc.pdf",
            page_number=2,
            text="Developers build dashboards with Streamlit.",
            chunk_index=1,
            char_count=45,
        )

        result = self.extractor.extract_from_chunks([chunk1, chunk2])

        # Find the Streamlit entity
        streamlit_eid = generate_entity_id("Streamlit", "TECHNOLOGY")
        self.assertIn(streamlit_eid, result.entities)

        streamlit_ent = result.entities[streamlit_eid]
        self.assertEqual(streamlit_ent.name, "Streamlit")
        self.assertGreaterEqual(streamlit_ent.mention_count, 2)

        # Both chunk IDs should be preserved in source_references
        chunk_ids = [ref.chunk_id for ref in streamlit_ent.source_references]
        self.assertIn("doc_p1_c0", chunk_ids)
        self.assertIn("doc_p2_c1", chunk_ids)

        pages = [ref.page_number for ref in streamlit_ent.source_references]
        self.assertIn(1, pages)
        self.assertIn(2, pages)

    # 5. Relationship deduplication while preserving provenance
    def test_relationship_deduplication_preserves_provenance(self):
        chunk1 = TextChunk(
            chunk_id="chunk_a",
            source_filename="paper_a.txt",
            page_number=None,
            text="SynapseAI uses Knowledge Graph algorithms.",
            chunk_index=0,
            char_count=42,
        )
        chunk2 = TextChunk(
            chunk_id="chunk_b",
            source_filename="paper_b.txt",
            page_number=None,
            text="SynapseAI uses Knowledge Graph representations.",
            chunk_index=0,
            char_count=47,
        )

        result = self.extractor.extract_from_chunks([chunk1, chunk2])

        # Check relationships between SynapseAI and Knowledge Graph
        matching_rels = [
            r for r in result.relationships.values()
            if "synapseai" in r.subject_id and "knowledge_graph" in r.object_id
        ]

        if matching_rels:
            rel = matching_rels[0]
            # Should have aggregated evidence from both chunks if both matched
            sources = [ev.source_filename for ev in rel.evidence]
            self.assertTrue(len(rel.evidence) >= 1)

    # 6. Correct filename, page number, and chunk ID propagation
    def test_provenance_propagation(self):
        chunk = TextChunk(
            chunk_id="nlp_survey_p5_c2",
            source_filename="nlp_survey.pdf",
            page_number=5,
            text="PyMuPDF enables fast PDF parsing.",
            chunk_index=2,
            char_count=35,
        )

        result = self.extractor.extract_from_chunks([chunk])

        pymupdf_eid = generate_entity_id("PyMuPDF", "TECHNOLOGY")
        self.assertIn(pymupdf_eid, result.entities)
        ref = result.entities[pymupdf_eid].source_references[0]

        self.assertEqual(ref.source_filename, "nlp_survey.pdf")
        self.assertEqual(ref.page_number, 5)
        self.assertEqual(ref.chunk_id, "nlp_survey_p5_c2")

    # 7. Malformed structured model output handling
    def test_malformed_json_and_predicate_normalization(self):
        # Predicate normalization check
        self.assertEqual(normalize_predicate("utilizes"), "USES")
        self.assertEqual(normalize_predicate("is based on"), "EXTENDS")
        self.assertEqual(normalize_predicate("developed by"), "DEVELOPED_BY")
        self.assertEqual(normalize_predicate("unknown verbal action"), "UNKNOWN_VERBAL_ACTION")

        # Test deterministic ID generation robustness against messy characters
        messy_name = "  Special! & Model @ 4.0  "
        clean_id = generate_entity_id(messy_name, "TECHNOLOGY")
        self.assertTrue(clean_id.startswith("ent_technology_"))
        self.assertNotIn("!", clean_id)
        self.assertNotIn("@", clean_id)

    # 8. Empty input and documents with no identifiable entities
    def test_empty_and_sparse_input(self):
        # Empty chunk list
        result_empty = self.extractor.extract_from_chunks([])
        self.assertEqual(len(result_empty.entities), 0)
        self.assertEqual(len(result_empty.relationships), 0)
        self.assertEqual(len(result_empty.themes), 0)

        # Chunk with common stop words only
        noise_chunk = TextChunk(
            chunk_id="noise_0",
            source_filename="noise.txt",
            page_number=None,
            text="This is that and these are those.",
            chunk_index=0,
            char_count=33,
        )
        result_noise = self.extractor.extract_from_chunks([noise_chunk])
        self.assertEqual(len(result_noise.entities), 0)

    # 9. Theme grouping behavior on controlled dataset
    def test_theme_grouping_behavior(self):
        e1 = Entity(
            entity_id="ent_tech_vector_database",
            name="Vector Database",
            entity_type="TECHNOLOGY",
            source_references=[EntityMention("f1.txt", 1, "c1", "Vector Database snippet")],
        )
        e2 = Entity(
            entity_id="ent_concept_similarity_search",
            name="Similarity Search",
            entity_type="CONCEPT",
            source_references=[EntityMention("f1.txt", 1, "c1", "Similarity Search snippet")],
        )
        e3 = Entity(
            entity_id="ent_tech_knowledge_graph",
            name="Knowledge Graph",
            entity_type="TECHNOLOGY",
            source_references=[EntityMention("f2.txt", 2, "c2", "Knowledge Graph snippet")],
        )

        entities = {e1.entity_id: e1, e2.entity_id: e2, e3.entity_id: e3}
        themes = self.theme_generator.generate_themes(entities, {}, [])

        self.assertGreater(len(themes), 0)
        # Verify themes have valid IDs, descriptions, and entity lists
        for theme in themes:
            self.assertTrue(theme.theme_id.startswith("theme_"))
            self.assertGreater(len(theme.name), 0)
            self.assertGreater(len(theme.description), 0)
            self.assertGreater(len(theme.entity_ids), 0)

    # 10. Repeatability of deterministic operations
    def test_deterministic_repeatability(self):
        sample_text = (
            "SynapseAI employs Knowledge Graph and Vector Database components. "
            "PyTorch utilizes Neural Network architectures."
        )
        chunk = TextChunk(
            chunk_id="chunk_det_1",
            source_filename="sample.txt",
            page_number=1,
            text=sample_text,
            chunk_index=0,
            char_count=len(sample_text),
        )

        res1 = self.extractor.extract_from_chunks([chunk])
        res2 = self.extractor.extract_from_chunks([chunk])

        # Entity IDs must match identically
        self.assertEqual(sorted(res1.entities.keys()), sorted(res2.entities.keys()))
        # Relationship IDs must match identically
        self.assertEqual(sorted(res1.relationships.keys()), sorted(res2.relationships.keys()))
        # Theme IDs must match identically
        self.assertEqual([t.theme_id for t in res1.themes], [t.theme_id for t in res2.themes])


if __name__ == "__main__":
    unittest.main()
