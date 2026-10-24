"""Phase 1 Quality Evaluation and Regression Benchmark Module for SynapseAI.

Phase 2 Requirement:
- Formal 100-point evaluation rubric spanning:
  1. Factual Accuracy & Citation Fidelity (25 pts)
  2. Relationship Reasoning & Graph Path Correctness (25 pts)
  3. Cross-Document Thematic Synthesis (25 pts)
  4. Conservative Abstention on Unanswerable Queries (25 pts)
- Explicit constraint: The post-Phase-2 score must remain within 3 absolute points
  of the pre-extension baseline (Decline <= 3.0 pts).
- Grounded, deterministic evaluation dataset and reproducible runner.
"""

from dataclasses import dataclass, field
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from src.answer_generator import AnswerGenerator, SufficiencyStatus
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
from src.query_router import QueryIntent, QueryRouter
from src.vector_store import LocalDeterministicEmbeddingFunction, VectorStoreService


@dataclass
class QualityBenchmarkReport:
    """Structured report of Phase 1 answer quality and regression metrics."""
    benchmark_id: str
    total_score: float  # out of 100
    baseline_score: float  # out of 100
    absolute_point_difference: float
    is_within_3_points: bool
    category_scores: Dict[str, float] = field(default_factory=dict)
    test_case_results: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "total_score": round(self.total_score, 2),
            "baseline_score": round(self.baseline_score, 2),
            "absolute_point_difference": round(self.absolute_point_difference, 2),
            "is_within_3_points": self.is_within_3_points,
            "category_scores": {k: round(v, 2) for k, v in self.category_scores.items()},
            "test_case_results": self.test_case_results,
            "evaluation_conditions": self.evaluation_conditions,
        }


class Phase1QualityEvaluator:
    """Executes the standard 100-point Phase 1 evaluation suite and checks for regressions."""

    def __init__(self, baseline_score: float = 98.0) -> None:
        self.baseline_score = baseline_score

    def run_evaluation(self, generator: Optional[AnswerGenerator] = None) -> QualityBenchmarkReport:
        """Run the standard 100-point evaluation test suite."""
        benchmark_id = f"qual_bench_{int(time.time())}"
        ans_gen = generator or AnswerGenerator(use_llm_if_available=False)
        router = QueryRouter()

        temp_dir = tempfile.mkdtemp()
        try:
            embedder = LocalDeterministicEmbeddingFunction(dimension=256)
            vector_store = VectorStoreService(
                persist_directory=temp_dir,
                collection_name="eval_suite_col",
                embedding_function=embedder,
            )

            # 1. Corpus Fixtures
            c1 = TextChunk(
                chunk_id="chunk_eval_1",
                source_filename="synapse_spec.pdf",
                page_number=1,
                text="SynapseAI implements NetworkX as its primary directed graph engine for knowledge graph modeling.",
                chunk_index=0,
                char_count=107,
            )
            c2 = TextChunk(
                chunk_id="chunk_eval_2",
                source_filename="synapse_spec.pdf",
                page_number=2,
                text="ChromaDB provides persistent vector storage and cosine similarity retrieval for extracted text passages.",
                chunk_index=1,
                char_count=111,
            )
            c3 = TextChunk(
                chunk_id="chunk_eval_3",
                source_filename="synapse_arch.pdf",
                page_number=1,
                text="The Hybrid Retrieval Architecture synthesizes dense vector search with structured graph traversals across all modules.",
                chunk_index=0,
                char_count=123,
            )
            vector_store.index_chunks([c1, c2, c3])

            # 2. Graph Fixtures
            kg = KnowledgeGraphBuilder()
            ent_syn = Entity(entity_id="ent_synapse", name="SynapseAI", entity_type="TECHNOLOGY", description="AI system")
            ent_nx = Entity(entity_id="ent_networkx", name="NetworkX", entity_type="TECHNOLOGY", description="Graph library")
            ent_chroma = Entity(entity_id="ent_chroma", name="ChromaDB", entity_type="TECHNOLOGY", description="Vector database")

            rel_syn_nx = Relationship(
                relationship_id="rel_1",
                subject_id="ent_synapse",
                predicate="IMPLEMENTS",
                object_id="ent_networkx",
                confidence=0.96,
                evidence=[RelationshipEvidence(
                    source_filename="synapse_spec.pdf",
                    page_number=1,
                    source_chunk_id="chunk_eval_1",
                    evidence_text="SynapseAI implements NetworkX as its primary directed graph engine",
                )],
            )
            rel_syn_chroma = Relationship(
                relationship_id="rel_2",
                subject_id="ent_synapse",
                predicate="USES",
                object_id="ent_chroma",
                confidence=0.94,
                evidence=[RelationshipEvidence(
                    source_filename="synapse_spec.pdf",
                    page_number=2,
                    source_chunk_id="chunk_eval_2",
                    evidence_text="ChromaDB provides persistent vector storage",
                )],
            )
            theme = TopicTheme(
                theme_id="theme_arch",
                name="Hybrid Retrieval Architecture",
                description="Cross-document architecture uniting vector and graph reasoning.",
                entity_ids=["ent_synapse", "ent_networkx", "ent_chroma"],
                source_chunk_ids=["chunk_eval_1", "chunk_eval_2", "chunk_eval_3"],
                confidence=0.98,
                document_filenames=["synapse_spec.pdf", "synapse_arch.pdf"],
            )

            ext_res = ExtractionResult(
                entities={"ent_synapse": ent_syn, "ent_networkx": ent_nx, "ent_chroma": ent_chroma},
                relationships={"rel_1": rel_syn_nx, "rel_2": rel_syn_chroma},
                themes=[theme],
            )
            kg.build_from_extraction(ext_res)

            known_entities = ["SynapseAI", "NetworkX", "ChromaDB"]
            known_themes = ["Hybrid Retrieval Architecture"]

            test_results: List[Dict[str, Any]] = []

            # --- Category 1: Factual Accuracy & Citation Fidelity (25 pts max) ---
            cat1_pts = 0.0
            q1 = "What graph library does SynapseAI implement?"
            plan1 = router.create_retrieval_plan(router.classify_query(q1, known_entities, known_themes))
            ret1 = router.execute_hybrid_retrieval(q1, plan1, vector_store, kg, ext_res)
            ans1 = ans_gen.generate_answer(q1, ret1)

            if "NetworkX" in ans1.answer_text:
                cat1_pts += 10.0
            if len(ans1.citations) > 0 and ans1.citations[0].source_filename == "synapse_spec.pdf":
                cat1_pts += 10.0
            if not ans1.is_abstained and all(c.is_verified for c in ans1.citations):
                cat1_pts += 5.0

            test_results.append({
                "category": "Factual",
                "query": q1,
                "score": cat1_pts,
                "max_score": 25.0,
                "passed": cat1_pts >= 23.0,
            })

            # --- Category 2: Relationship Reasoning & Graph Traversal (25 pts max) ---
            cat2_pts = 0.0
            q2 = "How is SynapseAI connected to NetworkX?"
            plan2 = router.create_retrieval_plan(router.classify_query(q2, known_entities, known_themes))
            ret2 = router.execute_hybrid_retrieval(q2, plan2, vector_store, kg, ext_res)
            ans2 = ans_gen.generate_answer(q2, ret2)

            rel_items = [e for e in ret2.evidence_items if e.evidence_type == "graph_relationship"]
            if len(rel_items) > 0:
                cat2_pts += 10.0
            if "IMPLEMENTS" in ans2.answer_text or "implements" in ans2.answer_text.lower():
                cat2_pts += 10.0
            if len(ans2.citations) > 0 and not ans2.is_abstained:
                cat2_pts += 5.0

            test_results.append({
                "category": "Relationship",
                "query": q2,
                "score": cat2_pts,
                "max_score": 25.0,
                "passed": cat2_pts >= 23.0,
            })

            # --- Category 3: Cross-Document Thematic Synthesis (25 pts max) ---
            cat3_pts = 0.0
            q3 = "What are the main themes across these documents?"
            plan3 = router.create_retrieval_plan(router.classify_query(q3, known_entities, known_themes))
            ret3 = router.execute_hybrid_retrieval(q3, plan3, vector_store, kg, ext_res)
            ans3 = ans_gen.generate_answer(q3, ret3)

            if "Hybrid Retrieval Architecture" in ret3.themes_matched:
                cat3_pts += 10.0
            if "Hybrid Retrieval Architecture" in ans3.answer_text:
                cat3_pts += 10.0
            if not ans3.is_abstained:
                cat3_pts += 5.0

            test_results.append({
                "category": "Thematic",
                "query": q3,
                "score": cat3_pts,
                "max_score": 25.0,
                "passed": cat3_pts >= 23.0,
            })

            # --- Category 4: Conservative Abstention on Unanswerable Queries (25 pts max) ---
            cat4_pts = 0.0
            q4 = "What is the average battery life of an iPhone 16 in minutes?"
            plan4 = router.create_retrieval_plan(router.classify_query(q4, known_entities, known_themes))
            ret4 = router.execute_hybrid_retrieval(q4, plan4, vector_store, kg, ext_res)
            ans4 = ans_gen.generate_answer(q4, ret4)

            if ans4.is_abstained:
                cat4_pts += 15.0
            if len(ans4.citations) == 0:
                cat4_pts += 5.0
            if ans4.sufficiency.status == SufficiencyStatus.INSUFFICIENT_EVIDENCE:
                cat4_pts += 5.0

            test_results.append({
                "category": "Unanswerable_Abstention",
                "query": q4,
                "score": cat4_pts,
                "max_score": 25.0,
                "passed": cat4_pts >= 25.0,
            })

            total_score = cat1_pts + cat2_pts + cat3_pts + cat4_pts
            diff = total_score - self.baseline_score
            is_within_3 = (self.baseline_score - total_score) <= 3.0

            category_scores = {
                "Factual": cat1_pts,
                "Relationship": cat2_pts,
                "Thematic": cat3_pts,
                "Unanswerable_Abstention": cat4_pts,
            }

            return QualityBenchmarkReport(
                benchmark_id=benchmark_id,
                total_score=total_score,
                baseline_score=self.baseline_score,
                absolute_point_difference=diff,
                is_within_3_points=is_within_3,
                category_scores=category_scores,
                test_case_results=test_results,
                evaluation_conditions={
                    "total_categories": 4,
                    "max_possible_points": 100.0,
                    "baseline_target": self.baseline_score,
                    "allowed_decline_points": 3.0,
                },
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
