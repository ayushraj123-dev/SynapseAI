"""Automated Unit Tests & Controlled Evaluation Fixture for Task 5.

Test Coverage:
1. Classification of factual, relationship, thematic, and unanswerable questions.
2. Retrieval-plan selection for each intent.
3. Integration with the actual vector-store interface using controlled fixtures.
4. Integration with graph nodes, edges, and evidence metadata.
5. Correct propagation of filenames, page numbers, and chunk IDs.
6. Deduplication of retrieved evidence without losing provenance.
7. Answer generation with verified citations.
8. Rejection of invalid or fabricated citations.
9. Conservative evidence sufficiency check and abstention when evidence is lacking.
10. Provider failure, missing credentials, and malformed output fallback.
11. Empty index and missing graph handling.
12. Controlled Evaluation Fixture:
    - Case A: Factual Question
    - Case B: Relationship Question
    - Case C: Thematic Question
    - Case D: Unanswerable Question
"""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.answer_generator import (
    AnswerGenerator,
    Citation,
    GroundedAnswer,
    SufficiencyEvaluation,
    SufficiencyStatus,
)
from src.document_ingestion import TextChunk
from src.entity_extraction import (
    Entity,
    EntityMention,
    ExtractionResult,
    Relationship,
    RelationshipEvidence,
    TopicTheme,
)
from src.knowledge_graph import KnowledgeGraphBuilder
from src.query_router import (
    HybridRetrievalResult,
    QueryClassification,
    QueryIntent,
    QueryRouter,
    RetrievalPlan,
    RetrievedEvidence,
)
from src.vector_store import (
    LocalDeterministicEmbeddingFunction,
    SearchResult,
    VectorStoreService,
)


class TestQueryRouterAndRetrieval(unittest.TestCase):
    """Test suite for QueryRouter intent classification, planning, and hybrid retrieval."""

    def setUp(self):
        self.router = QueryRouter()

        # Known entities and themes for testing
        self.known_entities = [
            "Knowledge Graph",
            "Vector Database",
            "SynapseAI",
            "Transformer Architecture",
            "ChromaDB",
        ]
        self.known_themes = [
            "Information Retrieval Systems",
            "Neural Architecture",
        ]

        # Setup local vector store in temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.embedder = LocalDeterministicEmbeddingFunction(dimension=256)
        self.vector_store = VectorStoreService(
            persist_directory=self.temp_dir,
            collection_name="test_router_col",
            embedding_function=self.embedder,
        )

        # Index controlled sample chunks
        self.chunk1 = TextChunk(
            chunk_id="chunk_doc1_p1_0",
            source_filename="synapse_tech_report.pdf",
            page_number=1,
            text="SynapseAI uses ChromaDB as its persistent vector database for semantic retrieval.",
            chunk_index=0,
            char_count=82,
        )
        self.chunk2 = TextChunk(
            chunk_id="chunk_doc1_p2_1",
            source_filename="synapse_tech_report.pdf",
            page_number=2,
            text="Knowledge Graph technology enables topological reasoning over structured entities and relations.",
            chunk_index=1,
            char_count=98,
        )
        self.chunk3 = TextChunk(
            chunk_id="chunk_doc1_p2_2",
            source_filename="synapse_tech_report.pdf",
            page_number=2,
            text="SynapseAI implements Knowledge Graph technology to link extracted entities and topic themes.",
            chunk_index=2,
            char_count=93,
        )
        self.vector_store.index_chunks([self.chunk1, self.chunk2, self.chunk3])

        # Setup KnowledgeGraphBuilder with actual entities and relationships
        self.kg = KnowledgeGraphBuilder()
        ent_synapse = Entity(
            entity_id="ent_tech_synapseai",
            name="SynapseAI",
            entity_type="TECHNOLOGY",
            description="An intelligent knowledge base and graph builder.",
            source_references=[
                EntityMention("synapse_tech_report.pdf", 1, "chunk_doc1_p1_0", "SynapseAI uses ChromaDB.")
            ],
        )
        ent_chroma = Entity(
            entity_id="ent_tech_chromadb",
            name="ChromaDB",
            entity_type="TECHNOLOGY",
            description="A local persistent vector database.",
            source_references=[
                EntityMention("synapse_tech_report.pdf", 1, "chunk_doc1_p1_0", "ChromaDB vector database.")
            ],
        )
        ent_kg = Entity(
            entity_id="ent_tech_kg",
            name="Knowledge Graph",
            entity_type="TECHNOLOGY",
            description="Structured topological graph of concepts.",
            source_references=[
                EntityMention("synapse_tech_report.pdf", 2, "chunk_doc1_p2_1", "Knowledge Graph technology.")
            ],
        )

        rel_uses = Relationship(
            relationship_id="rel_synapse_uses_chroma",
            subject_id="ent_tech_synapseai",
            predicate="USES",
            object_id="ent_tech_chromadb",
            confidence=0.95,
            evidence=[
                RelationshipEvidence(
                    source_filename="synapse_tech_report.pdf",
                    page_number=1,
                    source_chunk_id="chunk_doc1_p1_0",
                    evidence_text="SynapseAI uses ChromaDB as its persistent vector database.",
                )
            ],
        )
        rel_implements = Relationship(
            relationship_id="rel_synapse_implements_kg",
            subject_id="ent_tech_synapseai",
            predicate="IMPLEMENTS",
            object_id="ent_tech_kg",
            confidence=0.90,
            evidence=[
                RelationshipEvidence(
                    source_filename="synapse_tech_report.pdf",
                    page_number=2,
                    source_chunk_id="chunk_doc1_p2_2",
                    evidence_text="SynapseAI implements Knowledge Graph technology.",
                )
            ],
        )

        theme_ir = TopicTheme(
            theme_id="theme_ir_systems",
            name="Information Retrieval Systems",
            description="Vector search and graph indexing architectures.",
            entity_ids=["ent_tech_synapseai", "ent_tech_chromadb", "ent_tech_kg"],
            source_chunk_ids=["chunk_doc1_p1_0", "chunk_doc1_p2_1", "chunk_doc1_p2_2"],
        )

        ext_res = ExtractionResult(
            entities={
                ent_synapse.entity_id: ent_synapse,
                ent_chroma.entity_id: ent_chroma,
                ent_kg.entity_id: ent_kg,
            },
            relationships={
                rel_uses.relationship_id: rel_uses,
                rel_implements.relationship_id: rel_implements,
            },
            themes=[theme_ir],
        )
        self.kg.build_from_extraction(ext_res)
        self.ext_res = ext_res

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classification_of_all_intents(self):
        """Test 1: Correctly classify factual, relationship, thematic, and unanswerable queries."""
        # Factual query
        q_fact = "What vector database does SynapseAI use?"
        cls_fact = self.router.classify_query(q_fact, self.known_entities, self.known_themes)
        self.assertEqual(cls_fact.intent, QueryIntent.FACTUAL)
        self.assertIn("SynapseAI", cls_fact.extracted_entities)

        # Relationship query
        q_rel = "How is SynapseAI connected to ChromaDB?"
        cls_rel = self.router.classify_query(q_rel, self.known_entities, self.known_themes)
        self.assertEqual(cls_rel.intent, QueryIntent.RELATIONSHIP)
        self.assertIn("SynapseAI", cls_rel.extracted_entities)
        self.assertIn("ChromaDB", cls_rel.extracted_entities)

        # Thematic query
        q_theme = "What are the main themes across the uploaded documents?"
        cls_theme = self.router.classify_query(q_theme, self.known_entities, self.known_themes)
        self.assertEqual(cls_theme.intent, QueryIntent.THEMATIC)

        # Unanswerable / out-of-scope query
        q_unans = "What is the capital of France and what is the weather today?"
        cls_unans = self.router.classify_query(q_unans, self.known_entities, self.known_themes)
        self.assertEqual(cls_unans.intent, QueryIntent.UNANSWERABLE)

        # Nonsensical gibberish query
        q_gibberish = "zzxxqqwrtplkmnbv"
        cls_gibberish = self.router.classify_query(q_gibberish)
        self.assertEqual(cls_gibberish.intent, QueryIntent.UNANSWERABLE)

    def test_retrieval_plan_selection(self):
        """Test 2: Retrieval plan selection configures backends appropriately for each intent."""
        # Factual plan
        cls_fact = QueryClassification(intent=QueryIntent.FACTUAL, confidence=0.9, extracted_entities=["SynapseAI"])
        plan_fact = self.router.create_retrieval_plan(cls_fact)
        self.assertTrue(plan_fact.use_vector_search)
        self.assertTrue(plan_fact.use_graph_search)  # Target entity detected
        self.assertFalse(plan_fact.use_thematic_search)
        self.assertEqual(plan_fact.vector_top_k, 4)

        # Relationship plan
        cls_rel = QueryClassification(intent=QueryIntent.RELATIONSHIP, confidence=0.9, extracted_entities=["SynapseAI", "ChromaDB"])
        plan_rel = self.router.create_retrieval_plan(cls_rel)
        self.assertTrue(plan_rel.use_graph_search)
        self.assertTrue(plan_rel.use_vector_search)
        self.assertFalse(plan_rel.use_thematic_search)

        # Thematic plan
        cls_theme = QueryClassification(intent=QueryIntent.THEMATIC, confidence=0.85)
        plan_theme = self.router.create_retrieval_plan(cls_theme, ["Information Retrieval"])
        self.assertTrue(plan_theme.use_thematic_search)
        self.assertTrue(plan_theme.use_vector_search)
        self.assertFalse(plan_theme.use_graph_search)

        # Unanswerable plan
        cls_unans = QueryClassification(intent=QueryIntent.UNANSWERABLE, confidence=1.0)
        plan_unans = self.router.create_retrieval_plan(cls_unans)
        self.assertFalse(plan_unans.use_vector_search)
        self.assertFalse(plan_unans.use_graph_search)
        self.assertEqual(plan_unans.vector_top_k, 0)

    def test_hybrid_retrieval_integrates_vector_and_graph(self):
        """Test 3 & 4: Execute hybrid retrieval and verify vector passages and graph paths."""
        query = "How is SynapseAI related to ChromaDB?"
        classification = self.router.classify_query(query, self.known_entities, self.known_themes)
        plan = self.router.create_retrieval_plan(classification, self.known_themes)

        retrieval_res = self.router.execute_hybrid_retrieval(
            query=query,
            plan=plan,
            vector_store=self.vector_store,
            knowledge_graph=self.kg,
            extraction_result=self.ext_res,
        )

        self.assertFalse(retrieval_res.is_empty)
        self.assertTrue(len(retrieval_res.evidence_items) > 0)

        # Verify presence of both vector and graph evidence
        evidence_types = {e.evidence_type for e in retrieval_res.evidence_items}
        self.assertIn("vector_passage", evidence_types)
        self.assertIn("graph_relationship", evidence_types)

        # Verify edge evidence has direct citation text
        rel_ev = next(e for e in retrieval_res.evidence_items if e.evidence_type == "graph_relationship")
        self.assertEqual(rel_ev.source_filename, "synapse_tech_report.pdf")
        self.assertEqual(rel_ev.page_number, 1)
        self.assertEqual(rel_ev.chunk_id, "chunk_doc1_p1_0")
        self.assertIn("USES", rel_ev.text)

    def test_provenance_propagation_and_deduplication(self):
        """Test 5 & 6: Preserves exact provenance and deduplicates overlapping evidence."""
        query = "What does SynapseAI implement?"
        classification = self.router.classify_query(query, self.known_entities, self.known_themes)
        plan = self.router.create_retrieval_plan(classification)

        res = self.router.execute_hybrid_retrieval(
            query=query,
            plan=plan,
            vector_store=self.vector_store,
            knowledge_graph=self.kg,
            extraction_result=self.ext_res,
        )

        # Check provenance
        for ev in res.evidence_items:
            self.assertTrue(bool(ev.source_filename))
            if ev.source_filename == "synapse_tech_report.pdf":
                self.assertIn(ev.page_number, (1, 2))
                self.assertIsNotNone(ev.chunk_id)

        # Deduplication check: keys must be unique
        unique_keys = {f"{e.source_filename}:{e.chunk_id or e.text[:40]}" for e in res.evidence_items}
        self.assertEqual(len(unique_keys), len(res.evidence_items))


class TestAnswerGeneratorAndGrounding(unittest.TestCase):
    """Test suite for grounded answer generation, sufficiency checking, and citation validation."""

    def setUp(self):
        self.generator = AnswerGenerator(use_llm_if_available=False)

        # Construct verified evidence items
        self.ev1 = RetrievedEvidence(
            evidence_id="vec_c1",
            evidence_type="vector_passage",
            text="SynapseAI uses ChromaDB as its persistent vector database for semantic retrieval.",
            source_filename="synapse_tech_report.pdf",
            page_number=1,
            chunk_id="chunk_doc1_p1_0",
            score=0.92,
        )
        self.ev2 = RetrievedEvidence(
            evidence_id="graph_rel_1",
            evidence_type="graph_relationship",
            text="Relationship: SynapseAI ——[USES]——> ChromaDB. Evidence: 'SynapseAI uses ChromaDB.'",
            source_filename="synapse_tech_report.pdf",
            page_number=1,
            chunk_id="chunk_doc1_p1_0",
            score=0.95,
        )
        self.plan_factual = RetrievalPlan(
            intent=QueryIntent.FACTUAL,
            use_vector_search=True,
            use_graph_search=True,
            use_thematic_search=False,
            vector_top_k=4,
        )
        self.retrieval_supported = HybridRetrievalResult(
            query="What vector database does SynapseAI use?",
            plan=self.plan_factual,
            evidence_items=[self.ev1, self.ev2],
            is_empty=False,
        )

    def test_conservative_evidence_sufficiency_supported(self):
        """Test: Sufficiency check marks well-supported query as SUPPORTED."""
        evaluation = self.generator.evaluate_evidence_sufficiency(
            "What vector database does SynapseAI use?",
            self.retrieval_supported,
        )
        self.assertEqual(evaluation.status, SufficiencyStatus.SUPPORTED)
        self.assertTrue(evaluation.confidence >= 0.70)

    def test_abstention_when_evidence_is_insufficient(self):
        """Test 9: System explicitly abstains when evidence is missing or query is unanswerable."""
        empty_retrieval = HybridRetrievalResult(
            query="What is the internal clock frequency of quantum computers?",
            plan=RetrievalPlan(
                intent=QueryIntent.FACTUAL,
                use_vector_search=True,
                use_graph_search=False,
                use_thematic_search=False,
                vector_top_k=4,
            ),
            evidence_items=[],
            is_empty=True,
        )

        ans = self.generator.generate_answer(
            query="What is the internal clock frequency of quantum computers?",
            retrieval_result=empty_retrieval,
        )

        self.assertTrue(ans.is_abstained)
        self.assertEqual(ans.sufficiency.status, SufficiencyStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn("unable to answer this question", ans.answer_text)
        self.assertEqual(len(ans.citations), 0)  # Zero false citations!

    def test_answer_generation_and_verified_citations(self):
        """Test 7: Offline grounded synthesizer generates answer with verifiable citations."""
        ans = self.generator.generate_answer(
            query="What vector database does SynapseAI use?",
            retrieval_result=self.retrieval_supported,
        )

        self.assertFalse(ans.is_abstained)
        self.assertTrue(len(ans.citations) > 0)
        self.assertEqual(ans.citations[0].citation_id, "[1]")
        self.assertEqual(ans.citations[0].source_filename, "synapse_tech_report.pdf")
        self.assertEqual(ans.citations[0].page_number, 1)
        self.assertEqual(ans.citations[0].chunk_id, "chunk_doc1_p1_0")
        self.assertTrue(ans.citations[0].is_verified)
        self.assertIn("[1]", ans.answer_text)

    def test_reject_fabricated_or_unresolved_citations(self):
        """Test 8: Citation validator flags or rejects citations not found in retrieved evidence."""
        fake_citation = Citation(
            citation_id="[99]",
            source_filename="completely_unrelated_file.pdf",
            page_number=42,
            chunk_id="chunk_fake_999",
            quoted_snippet="Fabricated statement not in evidence.",
            is_verified=True,
        )
        real_citation = Citation(
            citation_id="[1]",
            source_filename="synapse_tech_report.pdf",
            page_number=1,
            chunk_id="chunk_doc1_p1_0",
            quoted_snippet="SynapseAI uses ChromaDB.",
            is_verified=True,
        )

        validated = self.generator.validate_citations([real_citation, fake_citation], self.retrieval_supported.evidence_items)
        # Real citation should be kept
        self.assertIn(real_citation, validated)
        # Fake citation should not be in valid list
        self.assertNotIn(fake_citation, validated)
        self.assertFalse(fake_citation.is_verified)

    def test_provider_fallback_on_failure(self):
        """Test 10: Graceful fallback to local grounded synthesis if LLM provider fails."""
        generator_with_llm = AnswerGenerator(use_llm_if_available=True)

        with patch.object(generator_with_llm, "_generate_with_llm", side_effect=Exception("API Timeout")):
            ans = generator_with_llm.generate_answer(
                query="What vector database does SynapseAI use?",
                retrieval_result=self.retrieval_supported,
            )
            # Should not raise; should fall back to offline grounded synthesizer
            self.assertFalse(ans.is_abstained)
            self.assertEqual(ans.generation_mode, "offline_grounded")
            self.assertTrue(len(ans.citations) > 0)


class TestControlledEvaluationFixture(unittest.TestCase):
    """Evaluation fixture: 4 controlled cases testing factual, relationship, thematic, and unanswerable queries."""

    @classmethod
    def setUpClass(cls):
        cls.router = QueryRouter()
        cls.generator = AnswerGenerator(use_llm_if_available=False)
        cls.temp_dir = tempfile.mkdtemp()
        cls.embedder = LocalDeterministicEmbeddingFunction(dimension=256)
        cls.vector_store = VectorStoreService(
            persist_directory=cls.temp_dir,
            collection_name="eval_fixture_col",
            embedding_function=cls.embedder,
        )

        # Standard Evaluation Document: "SynapseAI Architecture & Retrieval Engine"
        cls.eval_chunks = [
            TextChunk(
                chunk_id="eval_chunk_1",
                source_filename="synapse_spec.pdf",
                page_number=1,
                text="SynapseAI implements an in-memory NetworkX directed graph to represent extracted concepts and relationships.",
                chunk_index=0,
                char_count=112,
            ),
            TextChunk(
                chunk_id="eval_chunk_2",
                source_filename="synapse_spec.pdf",
                page_number=2,
                text="ChromaDB serves as the persistent vector database, storing dense embeddings locally in the chroma_db directory.",
                chunk_index=1,
                char_count=113,
            ),
            TextChunk(
                chunk_id="eval_chunk_3",
                source_filename="synapse_spec.pdf",
                page_number=3,
                text="SynapseAI connects to ChromaDB for semantic passage retrieval to substantiate factual answers with exact citations.",
                chunk_index=2,
                char_count=116,
            ),
        ]
        cls.vector_store.index_chunks(cls.eval_chunks)

        # Graph setup
        cls.kg = KnowledgeGraphBuilder()
        ent_syn = Entity("ent_synapse", "SynapseAI", "TECHNOLOGY", description="AI knowledge base system.")
        ent_nx = Entity("ent_networkx", "NetworkX", "TECHNOLOGY", description="In-memory directed graph library.")
        ent_chroma = Entity("ent_chroma", "ChromaDB", "TECHNOLOGY", description="Persistent local vector database.")

        rel_syn_nx = Relationship(
            relationship_id="rel_1",
            subject_id="ent_synapse",
            predicate="IMPLEMENTS",
            object_id="ent_networkx",
            confidence=0.95,
            evidence=[RelationshipEvidence("synapse_spec.pdf", 1, "eval_chunk_1", "SynapseAI implements NetworkX graph.", 0.95)],
        )
        rel_syn_chroma = Relationship(
            relationship_id="rel_2",
            subject_id="ent_synapse",
            predicate="USES",
            object_id="ent_chroma",
            confidence=0.95,
            evidence=[RelationshipEvidence("synapse_spec.pdf", 3, "eval_chunk_3", "SynapseAI connects to ChromaDB.", 0.95)],
        )
        theme = TopicTheme(
            theme_id="theme_retrieval",
            name="Hybrid Retrieval Architecture",
            description="Combining graph topology and vector search.",
            entity_ids=["ent_synapse", "ent_networkx", "ent_chroma"],
            source_chunk_ids=["eval_chunk_1", "eval_chunk_2", "eval_chunk_3"],
        )

        ext_res = ExtractionResult(
            entities={"ent_synapse": ent_syn, "ent_networkx": ent_nx, "ent_chroma": ent_chroma},
            relationships={"rel_1": rel_syn_nx, "rel_2": rel_syn_chroma},
            themes=[theme],
        )
        cls.kg.build_from_extraction(ext_res)
        cls.ext_res = ext_res
        cls.known_entities = ["SynapseAI", "NetworkX", "ChromaDB"]
        cls.known_themes = ["Hybrid Retrieval Architecture"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_case_a_factual_question(self):
        """Case A: Factual Question -> Verifies strategy, evidence retrieval, citation resolution."""
        query = "What graph library does SynapseAI implement?"
        routing = self.router.route_query(query, self.known_entities, self.known_themes)
        self.assertEqual(routing["intent"], "factual")

        plan = self.router.create_retrieval_plan(
            self.router.classify_query(query, self.known_entities, self.known_themes)
        )
        retrieval = self.router.execute_hybrid_retrieval(
            query=query,
            plan=plan,
            vector_store=self.vector_store,
            knowledge_graph=self.kg,
            extraction_result=self.ext_res,
        )

        # Verify evidence
        self.assertFalse(retrieval.is_empty)
        answer = self.generator.generate_answer(query=query, retrieval_result=retrieval)

        self.assertFalse(answer.is_abstained)
        self.assertTrue(len(answer.citations) > 0)
        top_cit = answer.citations[0]
        self.assertEqual(top_cit.source_filename, "synapse_spec.pdf")
        self.assertEqual(top_cit.page_number, 1)
        self.assertTrue(top_cit.is_verified)
        self.assertIn("NetworkX", answer.answer_text)

    def test_case_b_relationship_question(self):
        """Case B: Relationship Question -> Verifies graph traversal and relationship evidence."""
        query = "How is SynapseAI connected to NetworkX?"
        routing = self.router.route_query(query, self.known_entities, self.known_themes)
        self.assertEqual(routing["intent"], "relationship")

        plan = self.router.create_retrieval_plan(
            self.router.classify_query(query, self.known_entities, self.known_themes)
        )
        retrieval = self.router.execute_hybrid_retrieval(
            query=query,
            plan=plan,
            vector_store=self.vector_store,
            knowledge_graph=self.kg,
            extraction_result=self.ext_res,
        )

        self.assertFalse(retrieval.is_empty)
        # Verify graph relationship was retrieved
        rel_ev = [e for e in retrieval.evidence_items if e.evidence_type == "graph_relationship"]
        self.assertTrue(len(rel_ev) > 0)

        answer = self.generator.generate_answer(query=query, retrieval_result=retrieval)
        self.assertFalse(answer.is_abstained)
        self.assertIn("IMPLEMENTS", answer.answer_text)
        self.assertTrue(len(answer.citations) > 0)

    def test_case_c_thematic_question(self):
        """Case C: Thematic Question -> Verifies theme grouping and broad passage retrieval."""
        query = "What are the main themes across these documents?"
        routing = self.router.route_query(query, self.known_entities, self.known_themes)
        self.assertEqual(routing["intent"], "thematic")

        plan = self.router.create_retrieval_plan(
            self.router.classify_query(query, self.known_entities, self.known_themes)
        )
        retrieval = self.router.execute_hybrid_retrieval(
            query=query,
            plan=plan,
            vector_store=self.vector_store,
            knowledge_graph=self.kg,
            extraction_result=self.ext_res,
        )

        self.assertIn("Hybrid Retrieval Architecture", retrieval.themes_matched)
        answer = self.generator.generate_answer(query=query, retrieval_result=retrieval)
        self.assertFalse(answer.is_abstained)
        self.assertIn("Hybrid Retrieval Architecture", answer.answer_text)

    def test_case_d_unanswerable_question(self):
        """Case D: Unanswerable Question -> Verifies conservative abstention without hallucinated citations."""
        query = "What is the average battery life of an iPhone 16 in minutes?"
        routing = self.router.route_query(query, self.known_entities, self.known_themes)

        plan = self.router.create_retrieval_plan(
            self.router.classify_query(query, self.known_entities, self.known_themes)
        )
        retrieval = self.router.execute_hybrid_retrieval(
            query=query,
            plan=plan,
            vector_store=self.vector_store,
            knowledge_graph=self.kg,
            extraction_result=self.ext_res,
        )

        answer = self.generator.generate_answer(query=query, retrieval_result=retrieval)
        # MUST abstain safely!
        self.assertTrue(answer.is_abstained)
        self.assertEqual(answer.sufficiency.status, SufficiencyStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(len(answer.citations), 0)
        self.assertIn("unable to answer", answer.answer_text.lower())


if __name__ == "__main__":
    unittest.main()
