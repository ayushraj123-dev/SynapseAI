"""Incremental Knowledge Graph and Vector Index Manager for SynapseAI.

Phase 2 Implementation:
- Precise document-level dependency tracking:
  - doc_id -> chunk IDs, entity IDs, relationship keys, claim IDs, theme IDs.
  - Reference counting for shared entities and relationship evidence across documents.
- True incremental operations (< 60 seconds hard requirement):
  - Add Document: Embeds and indexes ONLY new chunks, adds only new nodes/edges or
    attaches evidence to shared nodes/edges, updates claims and contradictions.
  - Edit/Replace Document: Atomically removes stale single-document evidence, updates
    shared references, indexes new chunks, updates claims, and recalculates conflicts.
  - Delete Document: Purges document chunks from vector store, removes document/chunk nodes,
    removes relationships supported exclusively by the deleted document, retains shared
    entities/relationships supported by other documents, cleans up contradictions.
- Provenance integrity: avoids orphan records, broken endpoints, or cross-store state divergence.
- Mutation timing: measures wall-clock execution time from invocation to queryable state.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from src.claims_and_contradictions import Claim, ClaimExtractor, ContradictionDetector, ContradictionRecord
from src.document_ingestion import DocumentProcessingResult, TextChunk
from src.entity_extraction import (
    Entity,
    EntityMention,
    ExtractionResult,
    Relationship,
    RelationshipEvidence,
    TopicTheme,
)
from src.knowledge_graph import KnowledgeGraphBuilder
from src.vector_store import VectorStoreService


@dataclass
class MutationRecord:
    """Telemetry and outcome of an incremental document mutation."""
    mutation_id: str
    mutation_type: str  # "add", "edit", "delete"
    doc_id: str
    filename: str
    elapsed_seconds: float
    is_success: bool
    within_60s_sla: bool
    chunks_affected: int
    entities_affected: int
    relationships_affected: int
    claims_affected: int
    contradictions_affected: int
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "mutation_type": self.mutation_type,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "is_success": self.is_success,
            "within_60s_sla": self.within_60s_sla,
            "chunks_affected": self.chunks_affected,
            "entities_affected": self.entities_affected,
            "relationships_affected": self.relationships_affected,
            "claims_affected": self.claims_affected,
            "contradictions_affected": self.contradictions_affected,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


class DocumentDependencyTracker:
    """Maintains bidirectional dependency mappings and reference counts across documents."""

    def __init__(self) -> None:
        # Document ID -> set of chunk IDs
        self.doc_to_chunks: Dict[str, Set[str]] = {}
        # Document ID -> set of entity IDs mentioned
        self.doc_to_entities: Dict[str, Set[str]] = {}
        # Document ID -> set of relationship keys (subj, obj, pred)
        self.doc_to_relationships: Dict[str, Set[Tuple[str, str, str]]] = {}
        # Document ID -> set of claim IDs
        self.doc_to_claims: Dict[str, Set[str]] = {}
        # Document ID -> filename mapping
        self.doc_to_filename: Dict[str, str] = {}
        self.filename_to_doc: Dict[str, str] = {}

        # Reference counting across documents
        # Entity ID -> set of doc_ids that reference it
        self.entity_ref_counts: Dict[str, Set[str]] = {}
        # Relationship key (subj, obj, pred) -> set of doc_ids that reference it
        self.relationship_ref_counts: Dict[Tuple[str, str, str], Set[str]] = {}
        # Chunk registry: chunk_id -> chunk object
        self.chunk_registry: Dict[str, Any] = {}

    @property
    def active_documents(self) -> Dict[str, str]:
        """Dictionary of active doc_id -> filename."""
        return dict(self.doc_to_filename)

    def get_document_chunks(self, doc_id: str) -> List[Any]:
        """Retrieve chunk objects belonging to a document."""
        cids = self.doc_to_chunks.get(doc_id, set())
        return [self.chunk_registry[cid] for cid in cids if cid in self.chunk_registry]

    def register_document(
        self,
        doc_id: str,
        filename: str,
        chunk_ids: List[str],
        entity_ids: List[str],
        relationships: List[Tuple[str, str, str]],
        claim_ids: List[str],
        chunks: Optional[List[Any]] = None,
    ) -> None:
        """Register dependency links and increment reference counts for a document."""
        self.doc_to_filename[doc_id] = filename
        self.filename_to_doc[filename] = doc_id

        self.doc_to_chunks[doc_id] = set(chunk_ids)
        self.doc_to_entities[doc_id] = set(entity_ids)
        self.doc_to_relationships[doc_id] = set(relationships)
        self.doc_to_claims[doc_id] = set(claim_ids)

        if chunks:
            for c in chunks:
                cid = c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id", "")
                if cid:
                    self.chunk_registry[cid] = c

        for eid in entity_ids:
            self.entity_ref_counts.setdefault(eid, set()).add(doc_id)

        for rel_key in relationships:
            self.relationship_ref_counts.setdefault(rel_key, set()).add(doc_id)

    def unregister_document(self, doc_id: str) -> Dict[str, Any]:
        """Unregister a document and compute orphaned entities and relationships.

        Returns:
            Dictionary with orphaned_entities and orphaned_relationships.
        """
        orphaned_entities: Set[str] = set()
        orphaned_relationships: Set[Tuple[str, str, str]] = set()

        # Decrement entity references
        for eid in self.doc_to_entities.get(doc_id, set()):
            if eid in self.entity_ref_counts:
                self.entity_ref_counts[eid].discard(doc_id)
                if not self.entity_ref_counts[eid]:
                    orphaned_entities.add(eid)
                    del self.entity_ref_counts[eid]

        # Decrement relationship references
        for rel_key in self.doc_to_relationships.get(doc_id, set()):
            if rel_key in self.relationship_ref_counts:
                self.relationship_ref_counts[rel_key].discard(doc_id)
                if not self.relationship_ref_counts[rel_key]:
                    orphaned_relationships.add(rel_key)
                    del self.relationship_ref_counts[rel_key]

        chunks = self.doc_to_chunks.pop(doc_id, set())
        entities = self.doc_to_entities.pop(doc_id, set())
        relationships = self.doc_to_relationships.pop(doc_id, set())
        claims = self.doc_to_claims.pop(doc_id, set())
        fname = self.doc_to_filename.pop(doc_id, None)
        if fname and fname in self.filename_to_doc:
            del self.filename_to_doc[fname]

        return {
            "doc_id": doc_id,
            "filename": fname,
            "chunk_ids": chunks,
            "claim_ids": claims,
            "orphaned_entities": orphaned_entities,
            "orphaned_relationships": orphaned_relationships,
            "shared_entities_retained": entities - orphaned_entities,
            "shared_relationships_retained": relationships - orphaned_relationships,
        }


class IncrementalKnowledgeManager:
    """Orchestrates safe, sub-60-second document mutations across Vector Store, Graph, and Claims."""

    def __init__(
        self,
        vector_store: VectorStoreService,
        graph_builder: KnowledgeGraphBuilder,
        claim_extractor: Optional[ClaimExtractor] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        entity_extractor: Optional[Any] = None,
    ) -> None:
        self.vector_store = vector_store
        self.graph_builder = graph_builder
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self.entity_extractor = entity_extractor
        self.dependency_tracker = DocumentDependencyTracker()
        self.mutation_history: List[MutationRecord] = []
        self.all_claims: Dict[str, Claim] = {}

    @property
    def tracker(self) -> DocumentDependencyTracker:
        """Alias for dependency_tracker."""
        return self.dependency_tracker

    def add_document(
        self,
        doc_result: Union[DocumentProcessingResult, Dict[str, Any]],
        extraction_result: Optional[Union[ExtractionResult, Dict[str, Any]]] = None,
        doc_id: Optional[str] = None,
    ) -> MutationRecord:
        """Incrementally add a new document to the knowledge base without full rebuild."""
        start_time = time.time()
        mutation_id = f"mut_add_{int(start_time * 1000)}"

        # 1. Parse document metadata
        if isinstance(doc_result, DocumentProcessingResult):
            filename = doc_result.filename
            chunks = doc_result.chunks
            resolved_doc_id = doc_id or getattr(doc_result, "document_id", None) or filename
        else:
            filename = doc_result.get("filename", "unknown.txt")
            chunks = doc_result.get("chunks", [])
            resolved_doc_id = doc_id or doc_result.get("document_id", filename)

        if extraction_result is None:
            if self.entity_extractor is not None:
                extraction_result = self.entity_extractor.extract_from_chunks(chunks)
            else:
                from src.entity_extraction import EntityExtractor
                extraction_result = EntityExtractor(use_llm=False).extract_from_chunks(chunks)

        if isinstance(doc_result, DocumentProcessingResult):
            filename = doc_result.filename
            chunks = doc_result.chunks
            resolved_doc_id = doc_id or getattr(doc_result, "document_id", None) or filename
        else:
            filename = doc_result.get("filename", "unknown.txt")
            chunks = doc_result.get("chunks", [])
            resolved_doc_id = doc_id or doc_result.get("document_id", filename)

        try:
            # 2. Vector indexing: index ONLY this document's chunks
            self.vector_store.index_chunks(chunks)
            chunk_ids = [c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id", "") for c in chunks]

            # 3. Entity & Relationship parsing
            if isinstance(extraction_result, ExtractionResult):
                entities_dict = extraction_result.entities
                relationships_dict = extraction_result.relationships
                themes_list = extraction_result.themes
            else:
                entities_dict = extraction_result.get("entities", {})
                relationships_dict = extraction_result.get("relationships", {})
                themes_list = extraction_result.get("themes", [])

            # 4. Incremental Knowledge Graph Update
            # Add entity nodes if not present, or merge references
            entity_ids: List[str] = []
            for eid, ent in entities_dict.items():
                if hasattr(ent, "entity_id"):
                    name = ent.name
                    etype = ent.entity_type
                    aliases = ent.aliases
                    desc = ent.description
                    mentions = ent.mention_count
                    refs = [r.to_dict() if hasattr(r, "to_dict") else r for r in ent.source_references]
                else:
                    name = ent.get("name", eid)
                    etype = ent.get("entity_type", "CONCEPT")
                    aliases = ent.get("aliases", [])
                    desc = ent.get("description", "")
                    mentions = ent.get("mention_count", 1)
                    refs = ent.get("source_references", [])

                entity_ids.append(eid)
                if self.graph_builder.graph.has_node(eid):
                    # Existing shared node: update mentions and append unique source refs
                    node_data = self.graph_builder.graph.nodes[eid]
                    node_data["mention_count"] = node_data.get("mention_count", 0) + mentions
                    existing_refs = node_data.get("source_references", [])
                    for r in refs:
                        if r not in existing_refs:
                            existing_refs.append(r)
                    node_data["source_references"] = existing_refs
                else:
                    self.graph_builder.graph.add_node(
                        eid,
                        entity_id=eid,
                        name=name,
                        entity_type=etype,
                        aliases=aliases,
                        description=desc,
                        mention_count=mentions,
                        source_references=refs,
                        themes=[],
                    )

            # Add relationship edges
            rel_keys: List[Tuple[str, str, str]] = []
            for rid, rel in relationships_dict.items():
                if hasattr(rel, "subject_id"):
                    sid = rel.subject_id
                    oid = rel.object_id
                    pred = rel.predicate
                    conf = rel.confidence
                    evidence = [e.to_dict() if hasattr(e, "to_dict") else e for e in rel.evidence]
                else:
                    sid = rel.get("subject_id", "")
                    oid = rel.get("object_id", "")
                    pred = rel.get("predicate", "RELATES_TO")
                    conf = float(rel.get("confidence", 1.0))
                    evidence = rel.get("evidence", [])

                if not (self.graph_builder.graph.has_node(sid) and self.graph_builder.graph.has_node(oid)):
                    continue
                if sid == oid:
                    continue

                rel_key = (sid, oid, pred)
                rel_keys.append(rel_key)

                if self.graph_builder.graph.has_edge(sid, oid):
                    edge_data = self.graph_builder.graph[sid][oid]
                    preds = edge_data.get("all_predicates", [edge_data.get("predicate", "")])
                    if pred not in preds:
                        preds.append(pred)
                    edge_data["predicate"] = " / ".join(preds)
                    edge_data["all_predicates"] = preds
                    ev_list = edge_data.get("evidence", [])
                    for ev in evidence:
                        if ev not in ev_list:
                            ev_list.append(ev)
                    edge_data["evidence"] = ev_list
                else:
                    self.graph_builder.graph.add_edge(
                        sid,
                        oid,
                        relationship_id=rid,
                        predicate=pred,
                        all_predicates=[pred],
                        confidence=conf,
                        evidence=evidence,
                        edge_type="FACTUAL",
                    )

            # 5. Extract and Register Claims
            new_claims: List[Claim] = []
            for c in chunks:
                extracted = self.claim_extractor.extract_claims_from_chunk(c, doc_id=resolved_doc_id)
                new_claims.extend(extracted)
            rel_claims = self.claim_extractor.extract_claims_from_relationships(
                list(relationships_dict.values()), source_doc_id=resolved_doc_id
            )
            new_claims.extend(rel_claims)

            for claim in new_claims:
                self.all_claims[claim.claim_id] = claim

            # Run contradiction detection with updated claims
            contradictions = self.contradiction_detector.detect_contradictions(list(self.all_claims.values()))

            # Reflect contradictions in graph with distinct CONTRADICTS edges
            self._sync_contradictions_to_graph(contradictions)

            # 6. Register dependency links
            claim_ids = [c.claim_id for c in new_claims]
            self.dependency_tracker.register_document(
                doc_id=resolved_doc_id,
                filename=filename,
                chunk_ids=chunk_ids,
                entity_ids=entity_ids,
                relationships=rel_keys,
                claim_ids=claim_ids,
                chunks=chunks,
            )

            elapsed = time.time() - start_time
            rec = MutationRecord(
                mutation_id=mutation_id,
                mutation_type="add",
                doc_id=resolved_doc_id,
                filename=filename,
                elapsed_seconds=elapsed,
                is_success=True,
                within_60s_sla=elapsed <= 60.0,
                chunks_affected=len(chunk_ids),
                entities_affected=len(entity_ids),
                relationships_affected=len(rel_keys),
                claims_affected=len(claim_ids),
                contradictions_affected=len(contradictions),
            )
            self.mutation_history.append(rec)
            return rec

        except Exception as e:
            elapsed = time.time() - start_time
            rec = MutationRecord(
                mutation_id=mutation_id,
                mutation_type="add",
                doc_id=resolved_doc_id,
                filename=filename,
                elapsed_seconds=elapsed,
                is_success=False,
                within_60s_sla=elapsed <= 60.0,
                chunks_affected=0,
                entities_affected=0,
                relationships_affected=0,
                claims_affected=0,
                contradictions_affected=0,
                error_message=str(e),
            )
            self.mutation_history.append(rec)
            return rec

    def delete_document(self, doc_id_or_filename: str) -> MutationRecord:
        """Incrementally delete a document, retaining shared entities/relationships."""
        start_time = time.time()
        mutation_id = f"mut_del_{int(start_time * 1000)}"

        # Resolve doc_id and filename
        if doc_id_or_filename in self.dependency_tracker.doc_to_filename:
            doc_id = doc_id_or_filename
            filename = self.dependency_tracker.doc_to_filename[doc_id]
        elif doc_id_or_filename in self.dependency_tracker.filename_to_doc:
            filename = doc_id_or_filename
            doc_id = self.dependency_tracker.filename_to_doc[filename]
        else:
            filename = doc_id_or_filename
            doc_id = doc_id_or_filename

        try:
            # 1. Purge chunks from vector store
            removed_chunks = self.vector_store.delete_document(filename)

            # 2. Unregister dependencies and determine orphans
            dep_res = self.dependency_tracker.unregister_document(doc_id)
            orphaned_entities = dep_res["orphaned_entities"]
            orphaned_relationships = dep_res["orphaned_relationships"]
            claim_ids = dep_res["claim_ids"]

            # 3. Clean up orphaned edges from Knowledge Graph
            for sid, oid, pred in orphaned_relationships:
                if self.graph_builder.graph.has_edge(sid, oid):
                    edge_data = self.graph_builder.graph[sid][oid]
                    # Filter out evidence from this deleted document
                    filtered_ev = [
                        ev for ev in edge_data.get("evidence", [])
                        if (ev.get("source_filename") != filename if isinstance(ev, dict) else getattr(ev, "source_filename", "") != filename)
                    ]
                    if not filtered_ev:
                        self.graph_builder.graph.remove_edge(sid, oid)
                    else:
                        edge_data["evidence"] = filtered_ev

            # 4. Clean up orphaned nodes from Knowledge Graph
            for eid in orphaned_entities:
                if self.graph_builder.graph.has_node(eid):
                    self.graph_builder.graph.remove_node(eid)

            # 5. Remove claims and recalculate contradictions
            for cid in claim_ids:
                self.all_claims.pop(cid, None)
            self.contradiction_detector.remove_document_contradictions(filename)
            self.contradiction_detector.remove_document_contradictions(doc_id)

            # Remove obsolete contradiction edges from graph
            self._purge_contradiction_edges_for_doc(filename)

            elapsed = time.time() - start_time
            rec = MutationRecord(
                mutation_id=mutation_id,
                mutation_type="delete",
                doc_id=doc_id,
                filename=filename,
                elapsed_seconds=elapsed,
                is_success=True,
                within_60s_sla=elapsed <= 60.0,
                chunks_affected=removed_chunks,
                entities_affected=len(orphaned_entities),
                relationships_affected=len(orphaned_relationships),
                claims_affected=len(claim_ids),
                contradictions_affected=0,
            )
            self.mutation_history.append(rec)
            return rec

        except Exception as e:
            elapsed = time.time() - start_time
            rec = MutationRecord(
                mutation_id=mutation_id,
                mutation_type="delete",
                doc_id=doc_id,
                filename=filename,
                elapsed_seconds=elapsed,
                is_success=False,
                within_60s_sla=elapsed <= 60.0,
                chunks_affected=0,
                entities_affected=0,
                relationships_affected=0,
                claims_affected=0,
                contradictions_affected=0,
                error_message=str(e),
            )
            self.mutation_history.append(rec)
            return rec

    def edit_or_replace_document(
        self,
        doc_id_or_filename: str,
        new_doc_result: Union[DocumentProcessingResult, Dict[str, Any]],
        new_extraction_result: Optional[Union[ExtractionResult, Dict[str, Any]]] = None,
    ) -> MutationRecord:
        """Edit or replace a document's content while keeping document identity stable."""
        start_time = time.time()
        mutation_id = f"mut_edit_{int(start_time * 1000)}"

        # 1. First remove old document traces
        del_rec = self.delete_document(doc_id_or_filename)
        if not del_rec.is_success:
            elapsed = time.time() - start_time
            return MutationRecord(
                mutation_id=mutation_id,
                mutation_type="edit",
                doc_id=del_rec.doc_id,
                filename=del_rec.filename,
                elapsed_seconds=elapsed,
                is_success=False,
                within_60s_sla=elapsed <= 60.0,
                chunks_affected=0,
                entities_affected=0,
                relationships_affected=0,
                claims_affected=0,
                contradictions_affected=0,
                error_message=f"Edit pre-cleanup failed: {del_rec.error_message}",
            )

        # 2. Add the replacement content with same/stable identity
        add_rec = self.add_document(new_doc_result, new_extraction_result, doc_id=del_rec.doc_id)
        elapsed = time.time() - start_time

        rec = MutationRecord(
            mutation_id=mutation_id,
            mutation_type="edit",
            doc_id=del_rec.doc_id,
            filename=add_rec.filename,
            elapsed_seconds=elapsed,
            is_success=add_rec.is_success,
            within_60s_sla=elapsed <= 60.0,
            chunks_affected=add_rec.chunks_affected,
            entities_affected=add_rec.entities_affected,
            relationships_affected=add_rec.relationships_affected,
            claims_affected=add_rec.claims_affected,
            contradictions_affected=add_rec.contradictions_affected,
            error_message=add_rec.error_message,
        )
        self.mutation_history.append(rec)
        return rec

    def edit_document(
        self,
        doc_id_or_filename: str,
        new_doc_result: Union[DocumentProcessingResult, Dict[str, Any]],
        new_extraction_result: Optional[Union[ExtractionResult, Dict[str, Any]]] = None,
    ) -> MutationRecord:
        """Convenience alias for edit_or_replace_document."""
        return self.edit_or_replace_document(doc_id_or_filename, new_doc_result, new_extraction_result)

    def _sync_contradictions_to_graph(self, contradictions: List[ContradictionRecord]) -> None:
        """Add distinct CONTRADICTS edges between entities involved in contradictions."""
        for c in contradictions:
            sid = c.claim_a.subject
            oid = c.claim_b.subject

            # Find matching node IDs in graph (case-insensitive)
            node_map = {n.lower(): n for n in self.graph_builder.graph.nodes()}
            nid_a = node_map.get(sid.lower())
            nid_b = node_map.get(oid.lower())

            if nid_a and nid_b and nid_a != nid_b:
                self.graph_builder.graph.add_edge(
                    nid_a,
                    nid_b,
                    relationship_id=c.record_id,
                    predicate="CONTRADICTS",
                    all_predicates=["CONTRADICTS"],
                    confidence=c.confidence,
                    contradiction_type=c.contradiction_type,
                    edge_type="CONTRADICTION",
                    color="#f43f5e",  # Vibrant Rose/Amber for conflicts
                    dashes=True,
                    explanation=c.explanation,
                )

    def _purge_contradiction_edges_for_doc(self, filename: str) -> None:
        """Remove contradiction edges that were supported by the deleted document."""
        to_remove = []
        for u, v, d in self.graph_builder.graph.edges(data=True):
            if d.get("edge_type") == "CONTRADICTION":
                rec_id = d.get("relationship_id", "")
                if rec_id not in self.contradiction_detector.contradiction_records:
                    to_remove.append((u, v))
        for u, v in to_remove:
            if self.graph_builder.graph.has_edge(u, v):
                self.graph_builder.graph.remove_edge(u, v)

    def get_latest_mutation_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics of recent mutations and SLA compliance."""
        if not self.mutation_history:
            return {
                "total_mutations": 0,
                "latest_mutation": None,
                "max_latency_seconds": 0.0,
                "avg_latency_seconds": 0.0,
                "all_within_60s": True,
            }

        latencies = [m.elapsed_seconds for m in self.mutation_history]
        return {
            "total_mutations": len(self.mutation_history),
            "latest_mutation": self.mutation_history[-1].to_dict(),
            "max_latency_seconds": round(max(latencies), 3),
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 3),
            "all_within_60s": all(m.within_60s_sla for m in self.mutation_history),
            "history": [m.to_dict() for m in self.mutation_history[-10:]],
        }
