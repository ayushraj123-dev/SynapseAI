"""Comprehensive Automated Tests for Incremental Document Mutations (Feature 2).

Validates:
1. Incremental Document Addition (<60s SLA, no full-corpus rebuild).
2. Incremental Document Replacement / Editing (<60s SLA, stale evidence purged).
3. Incremental Document Deletion (<60s SLA, shared entities/relationships retained).
4. Provenance and dependency tracking integrity.
5. Retrieval verification: newly added content is queryable, deleted content is absent.
"""

import shutil
import tempfile
import time
import unittest

from src.claims_and_contradictions import ClaimExtractor, ContradictionDetector
from src.document_ingestion import DocumentProcessingResult, TextChunk
from src.entity_extraction import (
    Entity,
    EntityMention,
    ExtractionResult,
    Relationship,
    RelationshipEvidence,
    TopicTheme,
)
from src.incremental_manager import IncrementalKnowledgeManager
from src.knowledge_graph import KnowledgeGraphBuilder
from src.vector_store import LocalDeterministicEmbeddingFunction, VectorStoreService


class TestIncrementalDocumentMutations(unittest.TestCase):
    """Test suite validating sub-60-second incremental add, edit, and delete operations."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.embedder = LocalDeterministicEmbeddingFunction(dimension=256)
        self.vector_store = VectorStoreService(
            persist_directory=self.temp_dir,
            collection_name="test_inc_col",
            embedding_function=self.embedder,
        )
        self.kg = KnowledgeGraphBuilder()
        self.claim_extractor = ClaimExtractor()
        self.contra_detector = ContradictionDetector()

        self.manager = IncrementalKnowledgeManager(
            vector_store=self.vector_store,
            graph_builder=self.kg,
            claim_extractor=self.claim_extractor,
            contradiction_detector=self.contra_detector,
        )

        # Baseline document 1
        self.doc1_chunks = [
            TextChunk(
                chunk_id="c_doc1_1",
                source_filename="doc_alpha.pdf",
                page_number=1,
                text="SynapseAI uses ChromaDB for persistent vector retrieval.",
                chunk_index=0,
                char_count=57,
            ),
            TextChunk(
                chunk_id="c_doc1_2",
                source_filename="doc_alpha.pdf",
                page_number=2,
                text="SynapseAI implements NetworkX for graph topological reasoning.",
                chunk_index=1,
                char_count=63,
            ),
        ]
        self.doc1_result = DocumentProcessingResult(
            filename="doc_alpha.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            is_success=True,
            chunks=self.doc1_chunks,
            total_pages=2,
            total_characters=120,
        )

        self.ent_syn = Entity(entity_id="SynapseAI", name="SynapseAI", entity_type="TECHNOLOGY")
        self.ent_chroma = Entity(entity_id="ChromaDB", name="ChromaDB", entity_type="TECHNOLOGY")
        self.ent_nx = Entity(entity_id="NetworkX", name="NetworkX", entity_type="TECHNOLOGY")

        self.rel1 = Relationship(
            relationship_id="rel_alpha_1",
            subject_id="SynapseAI",
            predicate="USES",
            object_id="ChromaDB",
            confidence=0.95,
            evidence=[RelationshipEvidence(source_filename="doc_alpha.pdf", page_number=1, source_chunk_id="c_doc1_1", evidence_text="SynapseAI uses ChromaDB")],
        )
        self.rel2 = Relationship(
            relationship_id="rel_alpha_2",
            subject_id="SynapseAI",
            predicate="IMPLEMENTS",
            object_id="NetworkX",
            confidence=0.92,
            evidence=[RelationshipEvidence(source_filename="doc_alpha.pdf", page_number=2, source_chunk_id="c_doc1_2", evidence_text="SynapseAI implements NetworkX")],
        )
        self.doc1_ext = ExtractionResult(
            entities={"SynapseAI": self.ent_syn, "ChromaDB": self.ent_chroma, "NetworkX": self.ent_nx},
            relationships={"rel_alpha_1": self.rel1, "rel_alpha_2": self.rel2},
            themes=[],
        )

        # Ingest baseline doc 1
        self.manager.add_document(self.doc1_result, self.doc1_ext, doc_id="doc_alpha")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_document_incrementally(self):
        """Verify adding Document 2 updates graph and vector index within 60s without wiping Doc 1."""
        doc2_chunks = [
            TextChunk(
                chunk_id="c_doc2_1",
                source_filename="doc_beta.pdf",
                page_number=1,
                text="ChromaDB integrates with LangChain for hybrid document chunking.",
                chunk_index=0,
                char_count=65,
            )
        ]
        doc2_result = DocumentProcessingResult(
            filename="doc_beta.pdf",
            file_type="pdf",
            file_size_bytes=512,
            is_success=True,
            chunks=doc2_chunks,
            total_pages=1,
            total_characters=65,
        )

        ent_langchain = Entity(entity_id="LangChain", name="LangChain", entity_type="TECHNOLOGY")
        rel_beta = Relationship(
            relationship_id="rel_beta_1",
            subject_id="ChromaDB",
            predicate="INTEGRATES_WITH",
            object_id="LangChain",
            confidence=0.90,
            evidence=[RelationshipEvidence(source_filename="doc_beta.pdf", page_number=1, source_chunk_id="c_doc2_1", evidence_text="ChromaDB integrates with LangChain")],
        )
        doc2_ext = ExtractionResult(
            entities={"ChromaDB": self.ent_chroma, "LangChain": ent_langchain},
            relationships={"rel_beta_1": rel_beta},
            themes=[],
        )

        mutation = self.manager.add_document(doc2_result, doc2_ext, doc_id="doc_beta")

        # 1. Verify SLA
        self.assertTrue(mutation.is_success)
        self.assertTrue(mutation.within_60s_sla)
        self.assertLess(mutation.elapsed_seconds, 60.0)

        # 2. Verify graph nodes: ChromaDB shared, LangChain added, Doc 1 nodes retained
        self.assertTrue(self.kg.graph.has_node("SynapseAI"))
        self.assertTrue(self.kg.graph.has_node("ChromaDB"))
        self.assertTrue(self.kg.graph.has_node("NetworkX"))
        self.assertTrue(self.kg.graph.has_node("LangChain"))

        # 3. Verify edges
        self.assertTrue(self.kg.graph.has_edge("SynapseAI", "ChromaDB"))
        self.assertTrue(self.kg.graph.has_edge("ChromaDB", "LangChain"))

        # 4. Verify vector store has both Doc 1 and Doc 2 chunks
        sr = self.vector_store.similarity_search("LangChain", top_k=2)
        self.assertTrue(any("LangChain" in r.text for r in sr))
        self.assertEqual(self.vector_store.collection.count(), 3)

    def test_delete_document_preserves_shared_entities(self):
        """Verify deleting Document 2 leaves shared ChromaDB intact because Document 1 still references it."""
        # Add doc 2 first
        self.test_add_document_incrementally()

        # Delete doc 2
        del_mutation = self.manager.delete_document("doc_beta")

        self.assertTrue(del_mutation.is_success)
        self.assertTrue(del_mutation.within_60s_sla)
        self.assertLess(del_mutation.elapsed_seconds, 60.0)

        # 1. LangChain (orphaned) was removed
        self.assertFalse(self.kg.graph.has_node("LangChain"))
        # 2. ChromaDB (shared with Doc 1) is still present!
        self.assertTrue(self.kg.graph.has_node("ChromaDB"))
        self.assertTrue(self.kg.graph.has_node("SynapseAI"))
        self.assertTrue(self.kg.graph.has_edge("SynapseAI", "ChromaDB"))

        # 3. Vector chunks of doc_beta are gone
        self.assertEqual(self.vector_store.collection.count(), 2)
        sr = self.vector_store.similarity_search("LangChain", top_k=5)
        self.assertFalse(any(r.source_filename == "doc_beta.pdf" for r in sr))

    def test_edit_or_replace_document(self):
        """Verify replacing doc_alpha with an updated version replaces its stale evidence."""
        updated_chunks = [
            TextChunk(
                chunk_id="c_doc1_v2_1",
                source_filename="doc_alpha.pdf",
                page_number=1,
                text="SynapseAI has transitioned to Neo4j for enterprise knowledge graph modeling.",
                chunk_index=0,
                char_count=82,
            )
        ]
        updated_result = DocumentProcessingResult(
            filename="doc_alpha.pdf",
            file_type="pdf",
            file_size_bytes=512,
            is_success=True,
            chunks=updated_chunks,
            total_pages=1,
            total_characters=82,
        )

        ent_neo4j = Entity(entity_id="Neo4j", name="Neo4j", entity_type="TECHNOLOGY")
        rel_updated = Relationship(
            relationship_id="rel_alpha_v2",
            subject_id="SynapseAI",
            predicate="IMPLEMENTS",
            object_id="Neo4j",
            confidence=0.98,
            evidence=[RelationshipEvidence(source_filename="doc_alpha.pdf", page_number=1, source_chunk_id="c_doc1_v2_1", evidence_text="SynapseAI has transitioned to Neo4j")],
        )
        updated_ext = ExtractionResult(
            entities={"SynapseAI": self.ent_syn, "Neo4j": ent_neo4j},
            relationships={"rel_alpha_v2": rel_updated},
            themes=[],
        )

        edit_mutation = self.manager.edit_or_replace_document(
            doc_id_or_filename="doc_alpha",
            new_doc_result=updated_result,
            new_extraction_result=updated_ext,
        )

        self.assertTrue(edit_mutation.is_success)
        self.assertTrue(edit_mutation.within_60s_sla)
        self.assertLess(edit_mutation.elapsed_seconds, 60.0)

        # 1. New node Neo4j exists, stale node NetworkX removed
        self.assertTrue(self.kg.graph.has_node("Neo4j"))
        self.assertFalse(self.kg.graph.has_node("NetworkX"))

        # 2. New edge exists
        self.assertTrue(self.kg.graph.has_edge("SynapseAI", "Neo4j"))

        # 3. Vector search returns updated text and no old chunks
        sr = self.vector_store.similarity_search("Neo4j", top_k=2)
        self.assertTrue(any("Neo4j" in r.text for r in sr))
        self.assertEqual(self.vector_store.collection.count(), 1)


if __name__ == "__main__":
    unittest.main()
