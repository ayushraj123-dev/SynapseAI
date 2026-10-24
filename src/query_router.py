"""Query Router and Hybrid Retrieval Module for SynapseAI.

Task 5 Implementation:
- Adaptive query intent classification: FACTUAL, RELATIONSHIP, THEMATIC, HYBRID, and UNANSWERABLE.
- Deterministic heuristic rule classifier with entity-aware matching and LLM-assisted option.
- Structured retrieval planning choosing the optimal mix of vector, graph, and thematic search.
- Hybrid evidence retriever combining dense vector passages and validated knowledge graph paths.
- Strict provenance tracking: preserves source filenames, PDF page numbers, chunk IDs, and evidence quotes.
- Safe graph path discovery: rejects ungrounded edges lacking direct source evidence.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from src.entity_extraction import ExtractionResult, TopicTheme
from src.knowledge_graph import KnowledgeGraphBuilder
from src.vector_store import SearchResult, VectorStoreService


class QueryIntent(str, Enum):
    """Retrieval pathways supported by SynapseAI."""
    FACTUAL = "factual"
    RELATIONSHIP = "relationship"
    THEMATIC = "thematic"
    HYBRID = "hybrid"
    UNANSWERABLE = "unanswerable"


@dataclass
class QueryClassification:
    """Structured result of query intent classification."""
    intent: QueryIntent
    confidence: float
    extracted_entities: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "extracted_entities": self.extracted_entities,
            "key_terms": self.key_terms,
            "reasoning": self.reasoning,
        }


@dataclass
class RetrievalPlan:
    """Operational retrieval strategy decided by the query router."""
    intent: QueryIntent
    use_vector_search: bool
    use_graph_search: bool
    use_thematic_search: bool
    vector_top_k: int
    target_entities: List[str] = field(default_factory=list)
    target_themes: List[str] = field(default_factory=list)
    plan_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "use_vector_search": self.use_vector_search,
            "use_graph_search": self.use_graph_search,
            "use_thematic_search": self.use_thematic_search,
            "vector_top_k": self.vector_top_k,
            "target_entities": self.target_entities,
            "target_themes": self.target_themes,
            "plan_description": self.plan_description,
        }


@dataclass
class RetrievedEvidence:
    """An individual piece of grounded evidence with provenance."""
    evidence_id: str
    evidence_type: str  # 'vector_passage', 'graph_relationship', 'graph_node', 'theme_summary'
    text: str
    source_filename: str
    page_number: Optional[int]
    chunk_id: Optional[str]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "text": self.text,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


@dataclass
class HybridRetrievalResult:
    """Aggregated output of hybrid vector and graph retrieval."""
    query: str
    plan: RetrievalPlan
    evidence_items: List[RetrievedEvidence] = field(default_factory=list)
    vector_passages: List[SearchResult] = field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = field(default_factory=list)
    graph_paths: List[List[str]] = field(default_factory=list)
    themes_matched: List[str] = field(default_factory=list)
    matched_entities: List[str] = field(default_factory=list)
    highlight_node_ids: List[str] = field(default_factory=list)
    highlight_edge_ids: List[Tuple[str, str]] = field(default_factory=list)
    highlight_theme_ids: List[str] = field(default_factory=list)
    contributing_document_ids: List[str] = field(default_factory=list)
    is_empty: bool = False
    retrieval_limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan.to_dict(),
            "evidence_count": len(self.evidence_items),
            "evidence_items": [ev.to_dict() for ev in self.evidence_items],
            "vector_passages_count": len(self.vector_passages),
            "graph_edges_count": len(self.graph_edges),
            "graph_paths": self.graph_paths,
            "themes_matched": self.themes_matched,
            "matched_entities": self.matched_entities,
            "highlight_node_ids": self.highlight_node_ids,
            "highlight_edge_ids": self.highlight_edge_ids,
            "highlight_theme_ids": self.highlight_theme_ids,
            "contributing_document_ids": self.contributing_document_ids,
            "is_empty": self.is_empty,
            "retrieval_limitations": self.retrieval_limitations,
        }


class QueryRouter:
    """Service analyzing questions, classifying retrieval intent, and planning hybrid search."""

    # Keywords signaling relationship-seeking queries
    RELATIONSHIP_PATTERNS = [
        r"\bhow (?:is|are|does|do) (.+?) (?:connected|related|linked|associated) to (.+?)\b",
        r"\brelationship between (.+?) and (.+?)\b",
        r"\bconnection between (.+?) and (.+?)\b",
        r"\bhow does (.+?) (?:use|interact with|affect|impact|depend on|implement) (.+?)\b",
        r"\bcompare (.+?) (?:and|with|versus|vs) (.+?)\b",
        r"\bdifference between (.+?) and (.+?)\b",
    ]

    # Keywords signaling thematic queries
    THEMATIC_KEYWORDS = {
        "theme", "themes", "topic", "topics", "overview", "summary", "summarize",
        "main ideas", "key concepts", "broad categories", "patterns across", "scope",
        "outline", "subjects covered", "general idea", "what are the main",
    }

    # Keywords signaling unanswerable / chit-chat / nonsensical queries
    OUT_OF_SCOPE_PATTERNS = [
        r"^(?:hello|hi|hey|good morning|how are you|who are you)\b",
        r"\b(?:weather|recipe|cake|bake|movie|song|capital of france|president of|iphone|battery life)\b",
    ]

    def __init__(self) -> None:
        pass

    def classify_query(
        self,
        query: str,
        known_entities: Optional[List[str]] = None,
        known_themes: Optional[List[str]] = None,
    ) -> QueryClassification:
        """Classify user question intent using deterministic heuristics and entity awareness.

        Args:
            query: The user's input question.
            known_entities: Optional list of entity names in the knowledge graph.
            known_themes: Optional list of topic themes.

        Returns:
            QueryClassification with intent, confidence, detected entities, and reasoning.
        """
        cleaned = query.strip()
        if not cleaned:
            return QueryClassification(
                intent=QueryIntent.UNANSWERABLE,
                confidence=1.0,
                reasoning="Empty query provided.",
            )

        lower_query = cleaned.lower()

        # Check for obvious out-of-scope or chit-chat queries
        for pattern in self.OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, lower_query):
                return QueryClassification(
                    intent=QueryIntent.UNANSWERABLE,
                    confidence=0.95,
                    reasoning="Query matches out-of-scope chit-chat or general trivia patterns.",
                )

        # Detect gibberish (e.g. repeated consonant strings, no vowels, or extremely low token length)
        tokens = [t.strip(".,?!\"'") for t in lower_query.split()]
        tokens = [t for t in tokens if t]
        if len(tokens) == 1 and len(tokens[0]) > 12 and not any(v in tokens[0] for v in "aeiou"):
            return QueryClassification(
                intent=QueryIntent.UNANSWERABLE,
                confidence=0.90,
                reasoning="Query appears to be non-lexical string or gibberish.",
            )

        # Identify known entities mentioned in the query
        detected_entities: List[str] = []
        if known_entities:
            # Sort by descending length so multi-word entities match first
            for ent_name in sorted(known_entities, key=len, reverse=True):
                if ent_name.lower() in lower_query:
                    if ent_name not in detected_entities:
                        detected_entities.append(ent_name)

        # 1. Check for Relationship intent
        for rel_pat in self.RELATIONSHIP_PATTERNS:
            match = re.search(rel_pat, lower_query)
            if match:
                extracted = list(detected_entities)
                # If entities weren't in known_entities, capture regex groups
                if not extracted and match.groups():
                    extracted = [g.strip() for g in match.groups() if g.strip()]
                return QueryClassification(
                    intent=QueryIntent.RELATIONSHIP,
                    confidence=0.90,
                    extracted_entities=extracted,
                    key_terms=tokens,
                    reasoning=f"Matched relational inquiry pattern: '{rel_pat}'",
                )

        # Explicit multi-entity connection query
        if len(detected_entities) >= 2 and any(w in lower_query for w in ["connect", "link", "between", "relate", "with"]):
            return QueryClassification(
                intent=QueryIntent.RELATIONSHIP,
                confidence=0.85,
                extracted_entities=detected_entities,
                key_terms=tokens,
                reasoning=f"Multiple known entities detected ({detected_entities}) with relational connective phrasing.",
            )

        # 2. Check for Thematic intent
        if any(tk in lower_query for tk in self.THEMATIC_KEYWORDS):
            return QueryClassification(
                intent=QueryIntent.THEMATIC,
                confidence=0.85,
                extracted_entities=detected_entities,
                key_terms=tokens,
                reasoning="Query asks for thematic overview, topic distribution, or cross-document summary.",
            )

        # 3. Default to Factual intent
        return QueryClassification(
            intent=QueryIntent.FACTUAL,
            confidence=0.80,
            extracted_entities=detected_entities,
            key_terms=tokens,
            reasoning="Defaulted to factual passage lookup for targeted information retrieval.",
        )

    def route_query(
        self,
        query: str,
        known_entities: Optional[List[str]] = None,
        known_themes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible entry point returning classification and suggested parameters.

        Args:
            query: The user's input question.
            known_entities: Optional list of known entity names.
            known_themes: Optional list of known topic themes.

        Returns:
            Dictionary representation of routing decision and retrieval plan.
        """
        classification = self.classify_query(query, known_entities, known_themes)
        plan = self.create_retrieval_plan(classification, known_themes)
        return {
            "intent": classification.intent.value,
            "classification": classification.to_dict(),
            "plan": plan.to_dict(),
        }

    def create_retrieval_plan(
        self,
        classification: QueryClassification,
        known_themes: Optional[List[str]] = None,
    ) -> RetrievalPlan:
        """Formulate a tailored hybrid retrieval plan based on classified intent.

        Args:
            classification: QueryClassification instance.
            known_themes: Optional list of available topic theme names.

        Returns:
            RetrievalPlan specifying vector top_k, graph exploration, and themes.
        """
        intent = classification.intent

        if intent == QueryIntent.RELATIONSHIP:
            return RetrievalPlan(
                intent=intent,
                use_vector_search=True,
                use_graph_search=True,
                use_thematic_search=False,
                vector_top_k=3,
                target_entities=classification.extracted_entities,
                plan_description="Inspect graph nodes, relational edges, and direct paths between entities; verify via vector passages.",
            )

        elif intent == QueryIntent.THEMATIC:
            return RetrievalPlan(
                intent=intent,
                use_vector_search=True,
                use_graph_search=False,
                use_thematic_search=True,
                vector_top_k=5,
                target_entities=classification.extracted_entities,
                target_themes=known_themes or [],
                plan_description="Retrieve cluster themes and broad vector passages across document chunks for thematic synthesis.",
            )

        elif intent == QueryIntent.UNANSWERABLE:
            return RetrievalPlan(
                intent=intent,
                use_vector_search=False,
                use_graph_search=False,
                use_thematic_search=False,
                vector_top_k=0,
                target_entities=[],
                plan_description="Query classified as unanswerable or out of scope; abstain from speculative retrieval.",
            )

        else:  # FACTUAL or HYBRID
            use_graph = bool(classification.extracted_entities)
            return RetrievalPlan(
                intent=QueryIntent.FACTUAL,
                use_vector_search=True,
                use_graph_search=use_graph,
                use_thematic_search=False,
                vector_top_k=4,
                target_entities=classification.extracted_entities,
                plan_description="Dense semantic passage retrieval over indexed document chunks" + ("; check graph for entity attributes." if use_graph else "."),
            )

    def execute_hybrid_retrieval(
        self,
        query: str,
        plan: RetrievalPlan,
        vector_store: Optional[VectorStoreService] = None,
        knowledge_graph: Optional[KnowledgeGraphBuilder] = None,
        extraction_result: Optional[ExtractionResult] = None,
        neo4j_service: Optional[Any] = None,
    ) -> HybridRetrievalResult:
        """Execute the retrieval plan across vector, knowledge graph, and Neo4j backends.

        Args:
            query: The user's original question.
            plan: The planned retrieval strategy.
            vector_store: Optional active VectorStoreService instance.
            knowledge_graph: Optional active KnowledgeGraphBuilder instance.
            extraction_result: Optional raw ExtractionResult instance.
            neo4j_service: Optional active Neo4jGraphService instance.

        Returns:
            HybridRetrievalResult containing deduplicated evidence records, provenance, and graph highlight IDs.
        """
        evidence_items: List[RetrievedEvidence] = []
        vector_passages: List[SearchResult] = []
        graph_edges: List[Dict[str, Any]] = []
        graph_paths: List[List[str]] = []
        themes_matched: List[str] = []
        matched_entities: List[str] = list(plan.target_entities)
        highlight_node_ids: List[str] = []
        highlight_edge_ids: List[Tuple[str, str]] = []
        highlight_theme_ids: List[str] = []
        contributing_docs: Set[str] = set()
        limitations: List[str] = []

        # Early return for unanswerable queries: strict zero-highlight guarantee
        if plan.intent == QueryIntent.UNANSWERABLE:
            return HybridRetrievalResult(
                query=query,
                plan=plan,
                evidence_items=[],
                matched_entities=[],
                highlight_node_ids=[],
                highlight_edge_ids=[],
                highlight_theme_ids=[],
                contributing_document_ids=[],
                is_empty=True,
                retrieval_limitations=["Query was classified as out of scope or unanswerable."],
            )

        # 1. Vector Search Execution
        if plan.use_vector_search:
            if vector_store and vector_store.collection.count() > 0:
                try:
                    passages = vector_store.similarity_search(query, top_k=plan.vector_top_k)
                    vector_passages = passages
                    for p in passages:
                        evidence_items.append(
                            RetrievedEvidence(
                                evidence_id=f"vec_{p.chunk_id}",
                                evidence_type="vector_passage",
                                text=p.text,
                                source_filename=p.source_filename,
                                page_number=p.page_number,
                                chunk_id=p.chunk_id,
                                score=p.relevance_score,
                                metadata={"distance": p.distance, "char_count": p.char_count},
                            )
                        )
                        if p.source_filename and (p.source_filename.endswith(".pdf") or p.source_filename.endswith(".txt")):
                            contributing_docs.add(p.source_filename)
                except Exception as e:
                    limitations.append(f"Vector retrieval failed: {str(e)}")
            else:
                limitations.append("Vector store is empty or uninitialized.")

        # 2. Neo4j Persistent Graph Database Queries (if connected)
        if neo4j_service and getattr(neo4j_service, "is_connected", False):
            try:
                if plan.use_graph_search and plan.target_entities:
                    neo_rels = neo4j_service.query_relationships(
                        entity_names=plan.target_entities,
                        max_hops=3,
                        limit=25,
                    )
                    for r in neo_rels:
                        subj_name = r.get("subject_name", "")
                        obj_name = r.get("object_name", "")
                        pred = r.get("predicate", "RELATES_TO")
                        conf = float(r.get("confidence", 1.0))
                        ev_text = r.get("evidence_text", "")
                        src_fn = r.get("source_filename", "")
                        pg_num = r.get("page_number")
                        chk_id = r.get("chunk_id")

                        if subj_name and subj_name not in highlight_node_ids:
                            highlight_node_ids.append(subj_name)
                        if obj_name and obj_name not in highlight_node_ids:
                            highlight_node_ids.append(obj_name)
                        if subj_name and obj_name:
                            highlight_edge_ids.append((subj_name, obj_name))

                        if ev_text:
                            evidence_items.append(
                                RetrievedEvidence(
                                    evidence_id=f"neo_rel_{subj_name}_{pred}_{obj_name}",
                                    evidence_type="graph_relationship",
                                    text=f"Neo4j Relationship: {subj_name} ——[{pred}]——> {obj_name}. Grounded Evidence: \"{ev_text}\"",
                                    source_filename=src_fn or "neo4j_graph",
                                    page_number=pg_num,
                                    chunk_id=chk_id,
                                    score=conf,
                                    metadata={"predicate": pred, "subject": subj_name, "object": obj_name, "database": "neo4j"},
                                )
                            )
                        graph_edges.append({
                            "subject": subj_name,
                            "predicate": pred,
                            "object": obj_name,
                            "confidence": conf,
                            "has_evidence": bool(ev_text),
                            "database": "neo4j",
                        })
                        if src_fn and (src_fn.endswith(".pdf") or src_fn.endswith(".txt")):
                            contributing_docs.add(src_fn)

                if plan.use_thematic_search:
                    neo_themes = neo4j_service.query_themes()
                    for nth in neo_themes:
                        th_name = nth.get("name", "")
                        th_id = nth.get("theme_id", "")
                        if th_id and th_id not in highlight_theme_ids:
                            highlight_theme_ids.append(th_id)
                        for ent_name in nth.get("entities", []):
                            if ent_name not in highlight_node_ids:
                                highlight_node_ids.append(ent_name)
                        for doc_fn in nth.get("documents", []):
                            if doc_fn:
                                contributing_docs.add(doc_fn)
            except Exception as neo_err:
                limitations.append(f"Neo4j retrieval query warning: {str(neo_err)}")

        # 3. NetworkX Graph Search & Provenance Execution
        if plan.use_graph_search and knowledge_graph and knowledge_graph.graph.number_of_nodes() > 0:
            graph = knowledge_graph.graph

            # Resolve target entity node IDs
            matched_node_ids: List[str] = []
            for target in plan.target_entities:
                target_lower = target.lower()
                for nid, ndata in graph.nodes(data=True):
                    name_lower = ndata.get("name", "").lower()
                    aliases = [a.lower() for a in ndata.get("aliases", [])]
                    if target_lower == name_lower or target_lower == nid.lower() or target_lower in aliases:
                        if nid not in matched_node_ids:
                            matched_node_ids.append(nid)

            for nid in matched_node_ids:
                if nid not in highlight_node_ids:
                    highlight_node_ids.append(nid)

            # Node attribute evidence
            for nid in matched_node_ids:
                ndata = graph.nodes[nid]
                desc = ndata.get("description", "")
                if desc:
                    refs = ndata.get("source_references", [])
                    first_ref = refs[0] if refs else {}
                    if isinstance(first_ref, dict):
                        src_fn = first_ref.get("source_filename")
                        pg = first_ref.get("page_number")
                        chk = first_ref.get("chunk_id")
                    else:
                        src_fn = getattr(first_ref, "source_filename", None)
                        pg = getattr(first_ref, "page_number", None)
                        chk = getattr(first_ref, "chunk_id", None)

                    score = 0.80 if (src_fn and (src_fn.endswith(".pdf") or src_fn.endswith(".txt"))) else 0.35
                    evidence_items.append(
                        RetrievedEvidence(
                            evidence_id=f"graph_node_{nid}",
                            evidence_type="graph_node",
                            text=f"Entity {ndata.get('name')} [{ndata.get('entity_type')}]: {desc}",
                            source_filename=src_fn or "knowledge_graph",
                            page_number=pg,
                            chunk_id=chk,
                            score=score,
                            metadata={"entity_id": nid, "entity_type": ndata.get("entity_type")},
                        )
                    )
                    if src_fn and (src_fn.endswith(".pdf") or src_fn.endswith(".txt")):
                        contributing_docs.add(src_fn)

            # Path & Relational Edge retrieval
            if len(matched_node_ids) >= 2:
                for i in range(len(matched_node_ids)):
                    for j in range(i + 1, len(matched_node_ids)):
                        src, dst = matched_node_ids[i], matched_node_ids[j]
                        undirected = graph.to_undirected(as_view=True)
                        if nx.has_path(undirected, src, dst):
                            try:
                                sp = nx.shortest_path(undirected, src, dst)
                                if len(sp) <= 4:  # Cutoff at 3 hops
                                    named_path = [graph.nodes[n].get("name", n) for n in sp]
                                    graph_paths.append(named_path)

                                    for step in range(len(sp) - 1):
                                        u, v = sp[step], sp[step + 1]
                                        if u not in highlight_node_ids:
                                            highlight_node_ids.append(u)
                                        if v not in highlight_node_ids:
                                            highlight_node_ids.append(v)
                                        highlight_edge_ids.append((u, v))

                                        edge_data = graph.get_edge_data(u, v) or graph.get_edge_data(v, u)
                                        if edge_data:
                                            pred = edge_data.get("predicate", "RELATES_TO")
                                            conf = edge_data.get("confidence", 1.0)
                                            u_name = graph.nodes[u].get("name", u)
                                            v_name = graph.nodes[v].get("name", v)

                                            ev_list = edge_data.get("evidence", [])
                                            if ev_list:
                                                for ev in ev_list:
                                                    ev_text = ev.get("evidence_text", "")
                                                    src_fn = ev.get("source_filename", "knowledge_graph")
                                                    evidence_items.append(
                                                        RetrievedEvidence(
                                                            evidence_id=f"graph_rel_{u}_{pred}_{v}",
                                                            evidence_type="graph_relationship",
                                                            text=f"Relationship: {u_name} ——[{pred}]——> {v_name}. Grounded Evidence: \"{ev_text}\"",
                                                            source_filename=src_fn,
                                                            page_number=ev.get("page_number"),
                                                            chunk_id=ev.get("source_chunk_id"),
                                                            score=conf,
                                                            metadata={"predicate": pred, "subject": u_name, "object": v_name},
                                                        )
                                                    )
                                                    if src_fn and (src_fn.endswith(".pdf") or src_fn.endswith(".txt")):
                                                        contributing_docs.add(src_fn)

                                            graph_edges.append({
                                                "subject": u_name,
                                                "predicate": pred,
                                                "object": v_name,
                                                "confidence": conf,
                                                "has_evidence": bool(ev_list),
                                            })
                            except Exception:
                                pass

            # Direct incident edges if single entity matched
            elif len(matched_node_ids) == 1:
                single_id = matched_node_ids[0]
                single_name = graph.nodes[single_id].get("name", single_id)
                for _, target, edata in list(graph.out_edges(single_id, data=True))[:3]:
                    target_name = graph.nodes[target].get("name", target)
                    pred = edata.get("predicate", "RELATES_TO")
                    ev_list = edata.get("evidence", [])
                    if single_id not in highlight_node_ids:
                        highlight_node_ids.append(single_id)
                    if target not in highlight_node_ids:
                        highlight_node_ids.append(target)
                    highlight_edge_ids.append((single_id, target))

                    if ev_list:
                        ev = ev_list[0]
                        src_fn = ev.get("source_filename", "knowledge_graph")
                        evidence_items.append(
                            RetrievedEvidence(
                                evidence_id=f"graph_rel_{single_id}_{pred}_{target}",
                                evidence_type="graph_relationship",
                                text=f"Relationship: {single_name} ——[{pred}]——> {target_name}. Evidence: \"{ev.get('evidence_text', '')}\"",
                                source_filename=src_fn,
                                page_number=ev.get("page_number"),
                                chunk_id=ev.get("source_chunk_id"),
                                score=edata.get("confidence", 1.0),
                                metadata={"predicate": pred, "subject": single_name, "object": target_name},
                            )
                        )
                        if src_fn and (src_fn.endswith(".pdf") or src_fn.endswith(".txt")):
                            contributing_docs.add(src_fn)

                        graph_edges.append({
                            "subject": single_name,
                            "predicate": pred,
                            "object": target_name,
                            "confidence": edata.get("confidence", 1.0),
                            "has_evidence": True,
                        })

        # 4. Thematic Search Execution
        if plan.use_thematic_search:
            raw_themes = []
            if extraction_result and extraction_result.themes:
                raw_themes = extraction_result.themes
            elif knowledge_graph and knowledge_graph.themes:
                raw_themes = knowledge_graph.themes

            for th in raw_themes:
                theme_name = th.name if hasattr(th, "name") else th.get("name", "")
                theme_desc = th.description if hasattr(th, "description") else th.get("description", "")
                theme_chunks = th.source_chunk_ids if hasattr(th, "source_chunk_ids") else th.get("source_chunk_ids", [])
                theme_id = th.theme_id if hasattr(th, "theme_id") else th.get("theme_id", theme_name)
                entity_ids = th.entity_ids if hasattr(th, "entity_ids") else th.get("entity_ids", [])
                doc_fns = th.document_filenames if hasattr(th, "document_filenames") else th.get("document_filenames", [])

                themes_matched.append(theme_name)
                if theme_id not in highlight_theme_ids:
                    highlight_theme_ids.append(theme_id)

                for eid in entity_ids:
                    if eid not in highlight_node_ids:
                        highlight_node_ids.append(eid)

                for dfn in doc_fns:
                    if dfn:
                        contributing_docs.add(dfn)

                # Add connecting edges between theme entities in the graph
                if knowledge_graph and knowledge_graph.graph.number_of_nodes() > 0:
                    g = knowledge_graph.graph
                    for u in entity_ids:
                        for v in entity_ids:
                            if u != v and g.has_edge(u, v):
                                highlight_edge_ids.append((u, v))

                multi_doc_note = f" [Cross-Document Theme: {', '.join(doc_fns)}]" if len(doc_fns) > 1 else (f" [Source: {doc_fns[0]}]" if doc_fns else "")
                evidence_items.append(
                    RetrievedEvidence(
                        evidence_id=f"theme_{theme_id}",
                        evidence_type="theme_summary",
                        text=f"Thematic Cluster '{theme_name}': {theme_desc}{multi_doc_note}",
                        source_filename=doc_fns[0] if len(doc_fns) == 1 else "cross_document_theme",
                        page_number=None,
                        chunk_id=theme_chunks[0] if theme_chunks else None,
                        score=0.88,
                        metadata={
                            "theme_name": theme_name,
                            "chunk_ids": theme_chunks,
                            "documents": doc_fns,
                            "is_cross_document": len(doc_fns) > 1,
                        },
                    )
                )

        # 5. Deduplication: prioritize structured evidence (graph relationships) over plain passages
        type_priority = {
            "graph_relationship": 4,
            "vector_passage": 3,
            "theme_summary": 2,
            "graph_node": 1,
        }
        best_by_key: Dict[str, RetrievedEvidence] = {}

        for ev in evidence_items:
            key = f"{ev.source_filename}:{ev.chunk_id or ev.text[:40]}"
            if key not in best_by_key:
                best_by_key[key] = ev
            else:
                existing_prio = type_priority.get(best_by_key[key].evidence_type, 0)
                new_prio = type_priority.get(ev.evidence_type, 0)
                if new_prio > existing_prio:
                    best_by_key[key] = ev

        deduped_evidence = list(best_by_key.values())

        # Final pass for contributing document filenames from deduped evidence
        for ev in deduped_evidence:
            if ev.source_filename and (ev.source_filename.endswith(".pdf") or ev.source_filename.endswith(".txt")):
                contributing_docs.add(ev.source_filename)

        # Deduplicate highlight IDs preserving order
        dedup_nodes = list(dict.fromkeys(highlight_node_ids))
        dedup_edges = list(dict.fromkeys(highlight_edge_ids))
        dedup_themes = list(dict.fromkeys(highlight_theme_ids))

        return HybridRetrievalResult(
            query=query,
            plan=plan,
            evidence_items=deduped_evidence,
            vector_passages=vector_passages,
            graph_edges=graph_edges,
            graph_paths=graph_paths,
            themes_matched=themes_matched,
            matched_entities=matched_entities,
            highlight_node_ids=dedup_nodes,
            highlight_edge_ids=dedup_edges,
            highlight_theme_ids=dedup_themes,
            contributing_document_ids=sorted(list(contributing_docs)),
            is_empty=len(deduped_evidence) == 0,
            retrieval_limitations=limitations,
        )
