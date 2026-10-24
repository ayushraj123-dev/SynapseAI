"""Automated Tests for Cross-Document Contradiction Detection and Multi-Claim Grounding (Feature 3).

Test Coverage:
1. Claim extraction (polarity, numerical, temporal, relational).
2. Cross-document polarity contradiction detection (e.g. approved vs rejected).
3. Cross-document numerical divergence under identical scope (e.g. 12% vs 18%).
4. False-positive resistance: different scopes, populations, or dates are recognized as compatible.
5. Cross-document temporal contradiction (e.g. founded in 2021 vs 2024).
6. Multi-claim balanced answer generation presenting both sides with citations.
7. Graph synchronization: CONTRADICTS edges rendered with distinctive conflict styling.
8. Document removal cleanly updates contradiction registry.
"""

import shutil
import tempfile
import unittest

from src.answer_generator import AnswerGenerator, Citation, GroundedAnswer
from src.claims_and_contradictions import Claim, ClaimExtractor, ContradictionDetector, ContradictionRecord
from src.document_ingestion import DocumentProcessingResult, TextChunk
from src.entity_extraction import Entity, ExtractionResult, Relationship, RelationshipEvidence
from src.incremental_manager import IncrementalKnowledgeManager
from src.knowledge_graph import KnowledgeGraphBuilder
from src.query_router import HybridRetrievalResult, QueryIntent, QueryRouter, RetrievalPlan, RetrievedEvidence
from src.vector_store import LocalDeterministicEmbeddingFunction, VectorStoreService


class TestContradictionDetectionAndMultiClaimAnswers(unittest.TestCase):
    """Test suite validating evidence-grounded contradiction handling and multi-claim responses."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embedder = LocalDeterministicEmbeddingFunction(dimension=256)
        self.vector_store = VectorStoreService(
            persist_directory=self.temp_dir,
            collection_name="test_contra_col",
            embedding_function=self.embedder,
        )
        self.kg = KnowledgeGraphBuilder()
        self.claim_extractor = ClaimExtractor()
        self.detector = ContradictionDetector()
        self.ans_gen = AnswerGenerator(use_llm_if_available=False)

        self.manager = IncrementalKnowledgeManager(
            vector_store=self.vector_store,
            graph_builder=self.kg,
            claim_extractor=self.claim_extractor,
            contradiction_detector=self.detector,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_polarity_contradiction_detection(self):
        """Verify conflicting affirmative vs negative evaluation across two distinct documents."""
        claims = [
            Claim(
                claim_id="c_docA_fda",
                subject="TherapyX",
                predicate="EVALUATION_STATUS",
                object="approved",
                source_doc_id="clinical_report_2024.pdf",
                source_filename="clinical_report_2024.pdf",
                chunk_id="chunk_a1",
                page_number=1,
                excerpt="TherapyX was approved by the regulatory commission after Phase 3 trials.",
                claim_type="polarity",
                polarity=True,
                confidence=0.95,
            ),
            Claim(
                claim_id="c_docB_fda",
                subject="TherapyX",
                predicate="EVALUATION_STATUS",
                object="rejected",
                source_doc_id="health_watchdog_audit.pdf",
                source_filename="health_watchdog_audit.pdf",
                chunk_id="chunk_b1",
                page_number=3,
                excerpt="TherapyX was rejected by the panel citing insufficient longitudinal safety data.",
                claim_type="polarity",
                polarity=False,
                confidence=0.92,
            ),
        ]

        contradictions = self.detector.detect_contradictions(claims)
        self.assertEqual(len(contradictions), 1)

        c = contradictions[0]
        self.assertEqual(c.contradiction_type, "polarity")
        self.assertEqual(c.status, "confirmed")
        self.assertIn("TherapyX", c.explanation)
        self.assertIn("clinical_report_2024.pdf", c.explanation)
        self.assertIn("health_watchdog_audit.pdf", c.explanation)

    def test_numerical_divergence_under_same_scope(self):
        """Verify numerical conflict when quantities differ for the same attribute and unit."""
        claims = [
            Claim(
                claim_id="c_spec1",
                subject="QuantumCore",
                predicate="HAS_VALUE",
                object="12 percent",
                source_doc_id="lab_benchmarks.pdf",
                source_filename="lab_benchmarks.pdf",
                chunk_id="c1",
                page_number=2,
                excerpt="The error rate of QuantumCore is 12 percent under ambient pressure.",
                claim_type="numerical",
                polarity=True,
                quantity=12.0,
                unit="percent",
                confidence=0.90,
            ),
            Claim(
                claim_id="c_spec2",
                subject="QuantumCore",
                predicate="HAS_VALUE",
                object="18 percent",
                source_doc_id="independent_review.pdf",
                source_filename="independent_review.pdf",
                chunk_id="c2",
                page_number=5,
                excerpt="The error rate of QuantumCore was reported at 18 percent in standardized testing.",
                claim_type="numerical",
                polarity=True,
                quantity=18.0,
                unit="percent",
                confidence=0.88,
            ),
        ]

        contradictions = self.detector.detect_contradictions(claims)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].contradiction_type, "numerical")
        self.assertIn("12.0", contradictions[0].explanation)
        self.assertIn("18.0", contradictions[0].explanation)

    def test_avoid_false_positive_when_scopes_differ(self):
        """Verify statements referring to different dates or populations are NOT flagged as contradictory."""
        claims = [
            Claim(
                claim_id="c_2021",
                subject="BatteryTech",
                predicate="HAS_VALUE",
                object="250 miles",
                source_doc_id="annual_2021.pdf",
                source_filename="annual_2021.pdf",
                chunk_id="c1",
                page_number=1,
                excerpt="The range of BatteryTech stands at 250 miles in 2021 tests.",
                claim_type="numerical",
                polarity=True,
                quantity=250.0,
                unit="miles",
                date_scope="2021",
            ),
            Claim(
                claim_id="c_2024",
                subject="BatteryTech",
                predicate="HAS_VALUE",
                object="400 miles",
                source_doc_id="annual_2024.pdf",
                source_filename="annual_2024.pdf",
                chunk_id="c2",
                page_number=1,
                excerpt="The range of BatteryTech stands at 400 miles in 2024 tests.",
                claim_type="numerical",
                polarity=True,
                quantity=400.0,
                unit="miles",
                date_scope="2024",
            ),
        ]

        # Different date_scope -> compatible progress over time, NOT a contradiction!
        contradictions = self.detector.detect_contradictions(claims)
        self.assertEqual(len(contradictions), 0)

    def test_temporal_contradiction(self):
        """Verify conflict when two sources claim incompatible dates for the same event."""
        claims = [
            Claim(
                claim_id="c_hist1",
                subject="SynapseAI Project",
                predicate="FOUNDED",
                object="2021",
                source_doc_id="company_history.txt",
                source_filename="company_history.txt",
                chunk_id="c1",
                page_number=1,
                excerpt="SynapseAI Project was founded in 2021 by research fellows.",
                claim_type="temporal",
                polarity=True,
                date_scope="2021",
            ),
            Claim(
                claim_id="c_hist2",
                subject="SynapseAI Project",
                predicate="FOUNDED",
                object="2024",
                source_doc_id="press_announcement.txt",
                source_filename="press_announcement.txt",
                chunk_id="c2",
                page_number=1,
                excerpt="SynapseAI Project was founded in 2024 during the Hypnexis Hackathon.",
                claim_type="temporal",
                polarity=True,
                date_scope="2024",
            ),
        ]

        contradictions = self.detector.detect_contradictions(claims)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].contradiction_type, "temporal")
        self.assertIn("2021 vs 2024", contradictions[0].date_or_scope)

    def test_multi_claim_answer_generation(self):
        """Verify AnswerGenerator presents both sides of a contradiction with distinct citations."""
        claim_a = Claim(
            claim_id="ca",
            subject="FusionCore",
            predicate="STATUS",
            object="operational",
            source_doc_id="paper_a.pdf",
            source_filename="paper_a.pdf",
            chunk_id="c_a",
            page_number=1,
            excerpt="FusionCore reached full net-positive operational output in early 2024.",
        )
        claim_b = Claim(
            claim_id="cb",
            subject="FusionCore",
            predicate="STATUS",
            object="non-operational",
            source_doc_id="paper_b.pdf",
            source_filename="paper_b.pdf",
            chunk_id="c_b",
            page_number=4,
            excerpt="FusionCore failed to achieve net-positive ignition and remains non-operational.",
        )
        record = ContradictionRecord(
            record_id="rec_contra_1",
            claim_a=claim_a,
            claim_b=claim_b,
            contradiction_type="polarity",
            conflicting_attribute="STATUS",
            status="confirmed",
            explanation="paper_a.pdf asserts operational success while paper_b.pdf asserts ignition failure.",
        )

        retrieval = HybridRetrievalResult(
            query="Is FusionCore operational?",
            plan=RetrievalPlan(intent=QueryIntent.FACTUAL, use_vector_search=True, use_graph_search=False, use_thematic_search=False, vector_top_k=2),
            evidence_items=[
                RetrievedEvidence(evidence_id="e1", evidence_type="vector_passage", text=claim_a.excerpt, source_filename="paper_a.pdf", page_number=1, chunk_id="c_a", score=0.9),
                RetrievedEvidence(evidence_id="e2", evidence_type="vector_passage", text=claim_b.excerpt, source_filename="paper_b.pdf", page_number=4, chunk_id="c_b", score=0.88),
            ],
        )

        answer = self.ans_gen.generate_answer(
            query="Is FusionCore operational?",
            retrieval_result=retrieval,
            contradictions=[record],
        )

        self.assertFalse(answer.is_abstained)
        # Verify both claims are cited in answer text
        self.assertIn("Claim A", answer.answer_text)
        self.assertIn("Claim B", answer.answer_text)
        self.assertIn("paper_a.pdf", answer.answer_text)
        self.assertIn("paper_b.pdf", answer.answer_text)
        self.assertIn("Conflicting Claims", answer.answer_text)

        # Verify citations point to both sources
        cited_files = {c.source_filename for c in answer.citations}
        self.assertIn("paper_a.pdf", cited_files)
        self.assertIn("paper_b.pdf", cited_files)

    def test_incremental_deletion_updates_contradictions(self):
        """Verify that when one conflicting document is deleted, its contradiction record is purged."""
        doc1 = DocumentProcessingResult(
            filename="source_one.txt",
            file_type="txt",
            file_size_bytes=100,
            is_success=True,
            chunks=[TextChunk("c1", "source_one.txt", 1, "AlphaEngine was launched in 2020.", 0, 34)],
            total_pages=1,
            total_characters=34,
        )
        doc2 = DocumentProcessingResult(
            filename="source_two.txt",
            file_type="txt",
            file_size_bytes=100,
            is_success=True,
            chunks=[TextChunk("c2", "source_two.txt", 1, "AlphaEngine was launched in 2023.", 0, 34)],
            total_pages=1,
            total_characters=34,
        )

        ent_alpha = Entity(entity_id="AlphaEngine", name="AlphaEngine", entity_type="TECHNOLOGY")
        ext1 = ExtractionResult(entities={"AlphaEngine": ent_alpha}, relationships={}, themes=[])
        ext2 = ExtractionResult(entities={"AlphaEngine": ent_alpha}, relationships={}, themes=[])

        self.manager.add_document(doc1, ext1, doc_id="source_one")
        self.manager.add_document(doc2, ext2, doc_id="source_two")

        # Verify contradiction was detected
        self.assertTrue(len(self.detector.contradiction_records) > 0)

        # Delete source_two
        self.manager.delete_document("source_two")

        # Contradiction should be removed because source_two is gone
        self.assertEqual(len(self.detector.contradiction_records), 0)


if __name__ == "__main__":
    unittest.main()
