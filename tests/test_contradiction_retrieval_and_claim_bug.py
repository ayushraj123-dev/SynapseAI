"""Regression Tests for Claim Field Compatibility, Contradiction Display Safety,
and Query-Relevant Contradiction Retrieval (Section 6 Requirements).

Tests:
- Test A: Claim field compatibility (canonical excerpt/claim_text, serialization/deserialization, optional provenance).
- Test B: Contradiction display (two sources, page/chunk metadata preserved, explanation rendered).
- Test C: Malformed record resilience (missing fields, None objects, isolated diagnostic handling without crashing UI).
- Test D: Query-relevant contradiction retrieval (general vs narrow query, filtering out unrelated conflicts, deduplication).
- Test E: End-to-end regression for "Tell me about quantum physics" (no AttributeError, grounded citations, no 71 false conflicts).
"""

import os
import shutil
import tempfile
import unittest

from streamlit.testing.v1 import AppTest

from src.answer_generator import AnswerGenerator, Citation, GroundedAnswer
from src.claims_and_contradictions import (
    Claim,
    ClaimDisplayAdapter,
    ClaimExtractor,
    ContradictionDetector,
    ContradictionDisplayAdapter,
    ContradictionRecord,
)
from src.document_ingestion import DocumentProcessingResult, TextChunk
from src.knowledge_graph import KnowledgeGraphBuilder
from src.query_router import (
    HybridRetrievalResult,
    QueryClassification,
    QueryIntent,
    QueryRouter,
    RetrievalPlan,
    RetrievedEvidence,
)
from src.vector_store import LocalDeterministicEmbeddingFunction, VectorStoreService


class TestClaimFieldCompatibility(unittest.TestCase):
    """Test A — Claim field compatibility."""

    def test_claim_canonical_source_statement_and_claim_text_property(self):
        """Verify that Claim exposes both canonical excerpt and claim_text alias correctly."""
        claim = Claim(
            claim_id="c_quantum_001",
            subject="Electron spin",
            predicate="HAS_VALUE",
            object="+0.5 hbar",
            source_doc_id="quantum_principles.pdf",
            source_filename="quantum_principles.pdf",
            chunk_id="chk_q1",
            page_number=14,
            excerpt="The electron spin was measured at +0.5 hbar along the z-axis.",
            claim_type="numerical",
            polarity=True,
            quantity=0.5,
            unit="hbar",
            confidence=0.98,
        )

        # Canonical source statement
        self.assertEqual(claim.excerpt, "The electron spin was measured at +0.5 hbar along the z-axis.")
        # Compatible alias property
        self.assertEqual(claim.claim_text, "The electron spin was measured at +0.5 hbar along the z-axis.")

        # Structured fallback if excerpt is empty
        claim_empty_excerpt = Claim(
            claim_id="c_quantum_002",
            subject="Positron",
            predicate="DISCOVERED_IN",
            object="1932",
            source_doc_id="particle_history.txt",
            source_filename="particle_history.txt",
            chunk_id=None,
            page_number=None,
            excerpt="",
            claim_type="temporal",
            polarity=True,
        )
        self.assertIn("Positron", claim_empty_excerpt.claim_text)
        self.assertIn("1932", claim_empty_excerpt.claim_text)

    def test_serialization_and_deserialization_roundtrip(self):
        """Verify that serializing to dict and deserializing retains all required fields and claim_text."""
        original = Claim(
            claim_id="c_entanglement_10",
            subject="Quantum Entanglement",
            predicate="EVALUATION_STATUS",
            object="verified",
            source_doc_id="bell_tests_2022.pdf",
            source_filename="bell_tests_2022.pdf",
            chunk_id="chunk_bell_02",
            page_number=8,
            excerpt="Quantum entanglement was experimentally verified across 100 km.",
            claim_type="polarity",
            polarity=True,
            confidence=0.96,
        )

        d = original.to_dict()
        self.assertEqual(d["claim_text"], original.excerpt)
        self.assertEqual(d["source_filename"], "bell_tests_2022.pdf")
        self.assertEqual(d["page_number"], 8)
        self.assertEqual(d["chunk_id"], "chunk_bell_02")

        # Reconstruct via from_dict
        restored = Claim.from_dict(d)
        self.assertEqual(restored.claim_id, original.claim_id)
        self.assertEqual(restored.claim_text, original.excerpt)
        self.assertEqual(restored.excerpt, original.excerpt)
        self.assertEqual(restored.source_filename, original.source_filename)
        self.assertEqual(restored.page_number, original.page_number)
        self.assertEqual(restored.chunk_id, original.chunk_id)
        self.assertEqual(restored.polarity, original.polarity)

    def test_optional_metadata_handling_without_crashing(self):
        """Verify missing optional metadata (None page, None chunk, None quantity) behaves safely."""
        minimal_claim = Claim(
            claim_id="c_min_01",
            subject="Wave function",
            predicate="COLLAPSES_UPON",
            object="measurement",
            source_doc_id="copenhagen_notes.txt",
            source_filename="copenhagen_notes.txt",
            chunk_id=None,
            page_number=None,
            excerpt="The wave function collapses upon observation.",
            quantity=None,
            unit=None,
            date_scope=None,
            condition_or_population=None,
        )

        adapter = ClaimDisplayAdapter.from_claim(minimal_claim)
        self.assertTrue(adapter.is_valid)
        self.assertEqual(adapter.source_filename, "copenhagen_notes.txt")
        self.assertIsNone(adapter.page_number)
        self.assertIsNone(adapter.chunk_id)
        self.assertEqual(adapter.text, "The wave function collapses upon observation.")
        self.assertEqual(adapter.provenance_caption, "Source chunk unindexed")


class TestContradictionDisplay(unittest.TestCase):
    """Test B — Contradiction display."""

    def test_contradiction_display_adapter_preserves_both_sources_and_provenance(self):
        """Verify contradiction renderer displays both claims, sources, and metadata safely."""
        c1 = Claim(
            claim_id="c_spin_up",
            subject="Electron beam spin",
            predicate="HAS_VALUE",
            object="+0.5 hbar",
            source_doc_id="lab_a_report.pdf",
            source_filename="lab_a_report.pdf",
            chunk_id="chunk_la_01",
            page_number=5,
            excerpt="Electron beam spin was measured as +0.5 hbar.",
            claim_type="numerical",
            polarity=True,
            quantity=0.5,
            unit="hbar",
        )
        c2 = Claim(
            claim_id="c_spin_down",
            subject="Electron beam spin",
            predicate="HAS_VALUE",
            object="-0.5 hbar",
            source_doc_id="lab_b_report.pdf",
            source_filename="lab_b_report.pdf",
            chunk_id="chunk_lb_04",
            page_number=12,
            excerpt="Electron beam spin was determined to be -0.5 hbar under identical field.",
            claim_type="numerical",
            polarity=True,
            quantity=-0.5,
            unit="hbar",
        )

        detector = ContradictionDetector()
        records = detector.detect_contradictions([c1, c2])
        self.assertEqual(len(records), 1)

        rec = records[0]
        # Verify subject property on ContradictionRecord
        self.assertEqual(rec.subject, "Electron beam spin")

        adapter = ContradictionDisplayAdapter.from_record(rec)
        self.assertTrue(adapter.is_valid)
        self.assertIn("lab_a_report.pdf", [adapter.claim_a.source_filename, adapter.claim_b.source_filename])
        self.assertIn("lab_b_report.pdf", [adapter.claim_a.source_filename, adapter.claim_b.source_filename])
        self.assertIn("+0.5 hbar", adapter.claim_a.text + adapter.claim_b.text)
        self.assertIn("-0.5 hbar", adapter.claim_a.text + adapter.claim_b.text)
        self.assertEqual(adapter.status, "confirmed")
        self.assertTrue(len(adapter.explanation) > 10)


class TestMalformedRecordSafety(unittest.TestCase):
    """Test C — Malformed record resilience."""

    def test_malformed_records_do_not_crash_and_surface_diagnostics(self):
        """Verify that malformed or incomplete records are safely handled with diagnostics."""
        # Case 1: None record
        none_adapter = ContradictionDisplayAdapter.from_record(None)
        self.assertFalse(none_adapter.is_valid)
        self.assertIn("None", none_adapter.error_message)

        # Case 2: Record with missing claim_b
        class BrokenRecord:
            record_id = "broken_01"
            claim_a = Claim(
                claim_id="c_ok",
                subject="Test",
                predicate="HAS",
                object="Val",
                source_doc_id="doc1",
                source_filename="doc1",
                chunk_id="c1",
                page_number=1,
                excerpt="Test has Val",
            )
            claim_b = None
            contradiction_type = "relational"
            conflicting_attribute = "HAS"
            status = "confirmed"
            confidence = 0.8
            explanation = "Broken"

        broken_adapter = ContradictionDisplayAdapter.from_record(BrokenRecord())
        self.assertFalse(broken_adapter.is_valid)
        self.assertIn("claim_b", broken_adapter.error_message.lower())

        # Case 3: Empty dictionary claim
        empty_claim_adapter = ClaimDisplayAdapter.from_claim({})
        self.assertTrue(empty_claim_adapter.is_valid)
        self.assertEqual(empty_claim_adapter.source_filename, "unknown")
        self.assertEqual(empty_claim_adapter.text, "[Empty claim text]")


class TestRelevantContradictionRetrieval(unittest.TestCase):
    """Test D — Relevant contradiction retrieval and deduplication."""

    def setUp(self):
        self.detector = ContradictionDetector()

    def test_query_filtering_and_deduplication(self):
        """Verify quantum physics query does not retrieve unrelated conflicts, and narrow query does."""
        # 1. Quantum conflicting claims
        claim_q_a = Claim(
            claim_id="cq_01",
            subject="Quantum spin measurement",
            predicate="HAS_VALUE",
            object="+0.5 hbar",
            source_doc_id="quantum_paper_a.pdf",
            source_filename="quantum_paper_a.pdf",
            chunk_id="chunk_qp_a",
            page_number=3,
            excerpt="Quantum spin measurement yields +0.5 hbar for beam A.",
            claim_type="numerical",
            quantity=0.5,
            unit="hbar",
        )
        claim_q_b = Claim(
            claim_id="cq_02",
            subject="Quantum spin measurement",
            predicate="HAS_VALUE",
            object="-0.5 hbar",
            source_doc_id="quantum_paper_b.pdf",
            source_filename="quantum_paper_b.pdf",
            chunk_id="chunk_qp_b",
            page_number=7,
            excerpt="Quantum spin measurement yields -0.5 hbar for beam A under reversed coil.",
            claim_type="numerical",
            quantity=-0.5,
            unit="hbar",
        )

        # 2. Unrelated conflicts in the corpus (e.g. medicine, finance, aviation)
        claim_med_a = Claim(
            claim_id="cmed_01",
            subject="TherapyX",
            predicate="EVALUATION_STATUS",
            object="approved",
            source_doc_id="fda_approval.pdf",
            source_filename="fda_approval.pdf",
            chunk_id="chunk_fda_01",
            page_number=1,
            excerpt="TherapyX was approved by the advisory board.",
            claim_type="polarity",
            polarity=True,
        )
        claim_med_b = Claim(
            claim_id="cmed_02",
            subject="TherapyX",
            predicate="EVALUATION_STATUS",
            object="rejected",
            source_doc_id="emea_review.pdf",
            source_filename="emea_review.pdf",
            chunk_id="chunk_emea_02",
            page_number=2,
            excerpt="TherapyX was rejected due to lack of efficacy.",
            claim_type="polarity",
            polarity=False,
        )

        claim_fin_a = Claim(
            claim_id="cfin_01",
            subject="CorpY Revenue",
            predicate="HAS_VALUE",
            object="100 million USD",
            source_doc_id="q4_sec_filing.pdf",
            source_filename="q4_sec_filing.pdf",
            chunk_id="chunk_fin_a",
            page_number=10,
            excerpt="CorpY Revenue was 100 million USD in 2023.",
            claim_type="numerical",
            quantity=100.0,
            unit="million usd",
        )
        claim_fin_b = Claim(
            claim_id="cfin_02",
            subject="CorpY Revenue",
            predicate="HAS_VALUE",
            object="150 million USD",
            source_doc_id="audit_revision.pdf",
            source_filename="audit_revision.pdf",
            chunk_id="chunk_fin_b",
            page_number=4,
            excerpt="CorpY Revenue was revised to 150 million USD in 2023.",
            claim_type="numerical",
            quantity=150.0,
            unit="million usd",
        )

        # Detect all contradictions across the entire corpus
        all_claims = [claim_q_a, claim_q_b, claim_med_a, claim_med_b, claim_fin_a, claim_fin_b]
        corpus_conflicts = self.detector.detect_contradictions(all_claims)
        self.assertEqual(len(corpus_conflicts), 3)

        # Duplicate one conflict record artificially to test deduplication
        dup_conflicts = corpus_conflicts + [corpus_conflicts[0]]
        self.assertEqual(len(dup_conflicts), 4)
        deduped = self.detector.deduplicate_contradictions(dup_conflicts)
        self.assertEqual(len(deduped), 3)

        # Query 1: Broad query "Tell me about quantum physics"
        # Since the query is broad and evidence relates to quantum physics, unrelated conflicts (medicine, finance)
        # must NOT be returned!
        retrieved_quantum_evidence = [
            RetrievedEvidence(
                evidence_id="ev_q1",
                text="Quantum mechanics governs phenomena at atomic scales.",
                source_filename="quantum_paper_a.pdf",
                page_number=3,
                chunk_id="chunk_qp_a",
                evidence_type="vector_passage",
                score=0.92,
            )
        ]
        q_filtered = self.detector.filter_relevant_contradictions(
            contradictions=corpus_conflicts,
            query="Tell me about quantum physics",
            retrieved_evidence=retrieved_quantum_evidence,
            target_entities=["Quantum mechanics", "Quantum physics"],
        )
        # Should only contain quantum conflict (or 0 if broad overview without the specific claim), never medical or financial!
        for c in q_filtered:
            self.assertIn("quantum", c.claim_a.subject.lower())
            self.assertNotIn("therapyx", c.claim_a.subject.lower())
            self.assertNotIn("corpy", c.claim_a.subject.lower())

        # Query 2: Narrow question "What is the quantum spin measurement?"
        narrow_filtered = self.detector.filter_relevant_contradictions(
            contradictions=corpus_conflicts,
            query="What is the quantum spin measurement?",
            retrieved_evidence=retrieved_quantum_evidence,
            target_entities=["Quantum spin measurement"],
        )
        self.assertEqual(len(narrow_filtered), 1)
        self.assertEqual(narrow_filtered[0].subject, "Quantum spin measurement")

        # Query 3: Query on medicine "Is TherapyX approved or rejected?"
        med_filtered = self.detector.filter_relevant_contradictions(
            contradictions=corpus_conflicts,
            query="Is TherapyX approved or rejected?",
            target_entities=["TherapyX"],
        )
        self.assertEqual(len(med_filtered), 1)
        self.assertEqual(med_filtered[0].subject, "TherapyX")


class TestEndToEndRegression(unittest.TestCase):
    """Test E — End-to-end regression for 'Tell me about quantum physics'."""

    def test_tell_me_about_quantum_physics_flow_in_app(self):
        """Verify 'Tell me about quantum physics' runs through AppTest without AttributeError."""
        from src.document_ingestion import DocumentIngestionService
        from src.entity_extraction import EntityExtractor

        app_path = os.path.abspath("app.py")
        at = AppTest.from_file(app_path, default_timeout=20)
        at.run()
        at.button(key="btn_enter_workspace_primary").click().run()

        # Seed clean vector store, quantum physics document and unrelated medical conflict in session state
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        clean_vstore = VectorStoreService(
            persist_directory=temp_dir,
            collection_name="clean_quantum_e2e_test",
        )
        at.session_state["vector_store"] = clean_vstore

        svc = DocumentIngestionService()
        doc_quantum = svc.load_document(
            b"Quantum physics describes matter and energy at atomic scales. Quantum mechanics was founded by Max Planck and Albert Einstein. Wave-particle duality is fundamental.",
            filename="Concepts of Modern Physics by Arthur Beiser.txt"
        )
        at.session_state["ingestion_results"] = [doc_quantum]
        chunks = doc_quantum.chunks
        extractor = EntityExtractor(use_llm=False)
        ext_res = extractor.extract_from_chunks(chunks)
        at.session_state["extraction_results"] = ext_res
        at.session_state["kg_builder"].build_from_extraction(ext_res)
        at.session_state["vector_store"].index_chunks(chunks)

        # Navigate to Ask SynapseAI
        at.sidebar.radio(key="main_workspace_nav").set_value("💬 Ask SynapseAI").run()
        self.assertFalse(at.exception, f"Crashed navigating to Ask SynapseAI: {at.exception}")

        # Input query: "Tell me about quantum physics"
        input_widget = at.text_input(key="input_qa_True")
        input_widget.input("Tell me about quantum physics").run()

        ask_btn = at.button(key="btn_ask_True")
        ask_btn.click().run()

        # Critical assertion: No AttributeError: 'Claim' object has no attribute 'claim_text'
        self.assertFalse(at.exception, f"App crashed during query execution: {at.exception}")

        # Verify Q&A result exists in session state and is grounded
        last_qa = at.session_state.get("last_qa_result")
        self.assertIsNotNone(last_qa)
        self.assertEqual(last_qa.query, "Tell me about quantum physics")
        self.assertFalse(last_qa.is_abstained)
        self.assertTrue(len(last_qa.citations) > 0)
        self.assertIn("quantum", last_qa.answer_text.lower())

        # Verify active_contradictions for this query does NOT report 71 unrelated corpus conflicts
        active_contras = at.session_state.get("active_contradictions", [])
        self.assertNotEqual(len(active_contras), 71)
        # Any active contradiction must actually concern quantum physics, not arbitrary medical/finance topics
        for c in active_contras:
            c_adapter = ContradictionDisplayAdapter.from_record(c)
            self.assertTrue(c_adapter.is_valid)
            self.assertIn("quantum", c_adapter.subject.lower())

    def test_narrow_contradictory_query_presents_both_perspectives(self):
        """Verify that a narrow question on a conflicting topic surfaces both sides with citations."""
        generator = AnswerGenerator(use_llm_if_available=False)
        c1 = Claim(
            claim_id="c_drug_app",
            subject="TherapyX",
            predicate="EVALUATION_STATUS",
            object="approved",
            source_doc_id="report_fda_2024.pdf",
            source_filename="report_fda_2024.pdf",
            chunk_id="chk_fda_01",
            page_number=1,
            excerpt="TherapyX was approved by the federal panel for clinical deployment.",
            claim_type="polarity",
            polarity=True,
        )
        c2 = Claim(
            claim_id="c_drug_rej",
            subject="TherapyX",
            predicate="EVALUATION_STATUS",
            object="rejected",
            source_doc_id="report_eu_2024.pdf",
            source_filename="report_eu_2024.pdf",
            chunk_id="chk_eu_02",
            page_number=4,
            excerpt="TherapyX was rejected by the European committee due to efficacy concerns.",
            claim_type="polarity",
            polarity=False,
        )

        detector = ContradictionDetector()
        conflicts = detector.detect_contradictions([c1, c2])
        self.assertEqual(len(conflicts), 1)

        # Generate answer with contradictory claims present
        retrieval_res = HybridRetrievalResult(
            query="Is TherapyX approved or rejected?",
            plan=RetrievalPlan(
                intent=QueryIntent.FACTUAL,
                use_vector_search=True,
                use_graph_search=True,
                use_thematic_search=False,
                vector_top_k=4,
                target_entities=["TherapyX"],
                plan_description="Fact retrieval",
            ),
            evidence_items=[
                RetrievedEvidence(
                    evidence_id="ev_1",
                    text=c1.excerpt,
                    source_filename=c1.source_filename,
                    page_number=c1.page_number,
                    chunk_id=c1.chunk_id,
                    evidence_type="vector_passage",
                    score=0.95,
                ),
                RetrievedEvidence(
                    evidence_id="ev_2",
                    text=c2.excerpt,
                    source_filename=c2.source_filename,
                    page_number=c2.page_number,
                    chunk_id=c2.chunk_id,
                    evidence_type="vector_passage",
                    score=0.93,
                ),
            ],
        )

        answer = generator.generate_answer(
            query="Is TherapyX approved or rejected?",
            retrieval_result=retrieval_res,
            contradictions=conflicts,
        )

        self.assertFalse(answer.is_abstained)
        self.assertIn("report_fda_2024.pdf", answer.answer_text)
        self.assertIn("report_eu_2024.pdf", answer.answer_text)
        self.assertTrue(len(answer.citations) >= 2)
        filenames = {cit.source_filename for cit in answer.citations}
        self.assertIn("report_fda_2024.pdf", filenames)
        self.assertIn("report_eu_2024.pdf", filenames)


if __name__ == "__main__":
    unittest.main()
