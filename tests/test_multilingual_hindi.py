"""Automated Tests for Multilingual Hindi-English Cross-Lingual Q&A (Feature 5).

Test Coverage:
1. Devanagari Script and Hinglish Marker Detection.
2. Cross-Lingual Entity Resolution (Hindi names -> English canonical graph nodes).
3. Cross-Lingual Hybrid Semantic Retrieval over English document chunks.
4. Preserved Knowledge Graph Path Traversal (Hindi question -> English graph path -> Hindi explanation).
5. Grounded Hindi Answer Generation with verifiable English source citations.
6. Conservative Hindi Abstention when evidence is lacking in the English corpus.
"""

import shutil
import tempfile
import unittest

from src.answer_generator import Citation, GroundedAnswer, SufficiencyStatus
from src.document_ingestion import TextChunk
from src.entity_extraction import Entity, ExtractionResult, Relationship, RelationshipEvidence
from src.knowledge_graph import KnowledgeGraphBuilder
from src.multilingual import MultilingualQueryHandler, MultilingualQueryInfo
from src.vector_store import LocalDeterministicEmbeddingFunction, VectorStoreService


class TestMultilingualHindiRetrieval(unittest.TestCase):
    """Test suite validating cross-lingual Hindi-to-English retrieval, graph traversal, and grounded QA."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embedder = LocalDeterministicEmbeddingFunction(dimension=256)
        self.vector_store = VectorStoreService(
            persist_directory=self.temp_dir,
            collection_name="test_hindi_col",
            embedding_function=self.embedder,
        )
        self.kg = KnowledgeGraphBuilder()

        # English Corpus
        self.chunks = [
            TextChunk(
                chunk_id="chunk_en_1",
                source_filename="synapse_tech_overview.pdf",
                page_number=1,
                text="SynapseAI integrates NetworkX as its primary directed graph engine for topological reasoning.",
                chunk_index=0,
                char_count=98,
            ),
            TextChunk(
                chunk_id="chunk_en_2",
                source_filename="synapse_tech_overview.pdf",
                page_number=2,
                text="ChromaDB provides persistent vector embedding indexing and similarity search for SynapseAI.",
                chunk_index=1,
                char_count=94,
            ),
        ]
        self.vector_store.index_chunks(self.chunks)

        # Knowledge Graph in English
        ent_syn = Entity(entity_id="SynapseAI", name="SynapseAI", entity_type="TECHNOLOGY")
        ent_nx = Entity(entity_id="NetworkX", name="NetworkX", entity_type="TECHNOLOGY")
        ent_chroma = Entity(entity_id="ChromaDB", name="ChromaDB", entity_type="TECHNOLOGY")

        rel1 = Relationship(
            relationship_id="rel_1",
            subject_id="SynapseAI",
            predicate="IMPLEMENTS",
            object_id="NetworkX",
            evidence=[RelationshipEvidence(
                source_filename="synapse_tech_overview.pdf",
                page_number=1,
                source_chunk_id="chunk_en_1",
                evidence_text="SynapseAI integrates NetworkX as its primary directed graph engine",
            )],
        )
        rel2 = Relationship(
            relationship_id="rel_2",
            subject_id="SynapseAI",
            predicate="USES",
            object_id="ChromaDB",
            evidence=[RelationshipEvidence(
                source_filename="synapse_tech_overview.pdf",
                page_number=2,
                source_chunk_id="chunk_en_2",
                evidence_text="ChromaDB provides persistent vector embedding indexing",
            )],
        )

        ext_res = ExtractionResult(
            entities={"SynapseAI": ent_syn, "NetworkX": ent_nx, "ChromaDB": ent_chroma},
            relationships={"rel_1": rel1, "rel_2": rel2},
            themes=[],
        )
        self.kg.build_from_extraction(ext_res)

        self.handler = MultilingualQueryHandler(
            graph_builder=self.kg,
            vector_store=self.vector_store,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_script_and_language_detection(self):
        """Verify proper classification of Devanagari Hindi, Hinglish, and English queries."""
        # Devanagari Hindi
        q_dev = "सिनेप्सएआई और नेटवर्कएक्स में क्या संबंध है?"
        info_dev = self.handler.analyze_query(q_dev)
        self.assertEqual(info_dev.detected_language, "hindi")
        self.assertTrue(info_dev.is_devanagari)

        # Hinglish
        q_hinglish = "SynapseAI aur ChromaDB mein kya sambandh hai?"
        info_hinglish = self.handler.analyze_query(q_hinglish)
        self.assertEqual(info_hinglish.detected_language, "hinglish")
        self.assertFalse(info_hinglish.is_devanagari)

        # English
        q_en = "What is the relation between SynapseAI and NetworkX?"
        info_en = self.handler.analyze_query(q_en)
        self.assertEqual(info_en.detected_language, "english")

    def test_cross_lingual_entity_resolution(self):
        """Verify Hindi transliterations and domain words resolve to canonical English entity IDs."""
        q = "सिनेप्सएआई नॉलेज ग्राफ कैसे बनाता है?"
        info = self.handler.analyze_query(q)

        detected_eids = [eid for _, eid in info.detected_entity_aliases]
        self.assertIn("SynapseAI", detected_eids)
        self.assertIn("Knowledge Graph", detected_eids)

        # Normalized English query retains technical terms
        self.assertIn("SynapseAI", info.normalized_english_query)

    def test_cross_lingual_retrieval_over_english_documents(self):
        """Verify Hindi query retrieves relevant English document chunks."""
        q = "सिनेप्सएआई किस ग्राफ इंजन का उपयोग करता है?"
        info = self.handler.analyze_query(q)
        results = self.handler.retrieve_cross_lingual(info, top_k=2)

        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].source_filename, "synapse_tech_overview.pdf")
        self.assertIn("NetworkX", results[0].text)

    def test_preserve_graph_path_and_explain_in_hindi(self):
        """Verify that a relationship query in Hindi discovers the English graph path and explains it in Hindi."""
        q = "सिनेप्सएआई और नेटवर्कएक्स में क्या संबंध है?"
        info = self.handler.analyze_query(q)

        path_info = self.handler.find_graph_path_for_hindi_query(info)
        self.assertIsNotNone(path_info)
        self.assertEqual(path_info["node_path"], ["SynapseAI", "NetworkX"])
        self.assertEqual(path_info["edges"][0]["predicate"], "IMPLEMENTS")

        # Generate answer with path explanation
        results = self.handler.retrieve_cross_lingual(info, top_k=2)
        answer = self.handler.generate_hindi_answer(info, results, graph_path=path_info)

        self.assertFalse(answer.is_abstained)
        self.assertIn("ज्ञान ग्राफ (Knowledge Graph)", answer.answer_text)
        self.assertIn("SynapseAI ➔ NetworkX", answer.answer_text)
        self.assertIn("IMPLEMENTS", answer.answer_text)

        # Verified English citations
        self.assertTrue(len(answer.citations) > 0)
        self.assertEqual(answer.citations[0].source_filename, "synapse_tech_overview.pdf")

    def test_hindi_abstention_when_evidence_is_lacking(self):
        """Verify transparent refusal in Hindi when English corpus has no relevant information."""
        q = "सौर ऊर्जा से चलने वाले हवाई जहाज की गति कितनी है?"  # Solar airplane speed (not in corpus)
        info = self.handler.analyze_query(q)
        results = self.handler.retrieve_cross_lingual(info, top_k=2)
        path_info = self.handler.find_graph_path_for_hindi_query(info)

        # Filter out false matches
        valid_results = [r for r in results if r.relevance_score > 0.85]
        answer = self.handler.generate_hindi_answer(info, valid_results, graph_path=path_info)

        self.assertTrue(answer.is_abstained)
        self.assertEqual(answer.sufficiency.status, SufficiencyStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn("पर्याप्त साक्ष्य नहीं मिले", answer.answer_text)
        self.assertEqual(len(answer.citations), 0)


if __name__ == "__main__":
    unittest.main()
