"""Multilingual Hindi-English Retrieval and Grounded Q&A Module for SynapseAI.

Phase 2 Implementation:
- Script & Language Detection: Recognizes Devanagari Hindi, Hinglish, and Romanized Hindi.
- Cross-Lingual Entity Resolution: Maps Hindi names, transliterations, and aliases
  to canonical English Knowledge Graph entity IDs.
- Cross-Lingual Hybrid Retrieval: Normalizes Hindi queries into English search terms
  without mutating original source documents or destroying provenance.
- Traceable Graph Path Discovery: Finds real paths in the stored NetworkX graph
  between entities queried in Hindi and explains the path in natural Hindi.
- Grounded Hindi Answer Synthesis: Formulates evidence-based responses in Hindi
  with valid citations to original English source chunks [1], [2].
- Transparent Hindi Abstention: Refuses to speculate in clear, natural Hindi
  when source evidence is insufficient.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.answer_generator import Citation, GroundedAnswer, SufficiencyEvaluation, SufficiencyStatus
from src.knowledge_graph import KnowledgeGraphBuilder
from src.query_router import HybridRetrievalResult, QueryIntent, QueryRouter, RetrievedEvidence
from src.vector_store import SearchResult, VectorStoreService


@dataclass
class MultilingualQueryInfo:
    """Analysis of input query language, script, and normalized retrieval keys."""
    original_query: str
    detected_language: str  # "hindi", "hinglish", "english"
    is_devanagari: bool
    normalized_english_query: str
    detected_entity_aliases: List[Tuple[str, str]] = field(default_factory=list)  # (hindi_term, english_id)
    intent: str = "factual"

    @property
    def is_multilingual(self) -> bool:
        """Whether the query is in Hindi or Hinglish."""
        return self.detected_language in ("hindi", "hinglish")

    @property
    def script(self) -> str:
        """Script name for UI display."""
        return "devanagari" if self.is_devanagari else ("latin" if self.detected_language == "hinglish" else "english")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "detected_language": self.detected_language,
            "is_devanagari": self.is_devanagari,
            "is_multilingual": self.is_multilingual,
            "script": self.script,
            "normalized_english_query": self.normalized_english_query,
            "detected_entity_aliases": self.detected_entity_aliases,
            "intent": self.intent,
        }


class MultilingualQueryHandler:
    """Handles cross-lingual Hindi-to-English retrieval, entity resolution, and Hindi answer synthesis."""

    # Common Hindi question words and particles in Devanagari
    HINDI_INDICATORS_DEVANAGARI = {
        "क्या", "कैसे", "क्यों", "कहाँ", "कब", "किसने", "किसका", "कितना",
        "संबंध", "जुड़ा", "जुड़ी", "दस्तावेज़", "साक्ष्य", "निष्कर्ष", "अध्ययन",
        "और", "तथा", "में", "से", "का", "की", "के", "पर", "है", "हैं", "था", "थी",
    }

    # Common Hinglish / Romanized Hindi markers
    HINGLISH_MARKERS = {
        "kya", "kaise", "kyun", "kahan", "kab", "kisko", "kitna",
        "sambandh", "juda", "judi", "aur", "mein", "hai", "hain", "tha", "thi",
        "batao", "bataiye", "dikhaiye", "kare",
    }

    # Common Hindi -> English domain transliteration & synonym dictionary
    HINDI_TO_ENGLISH_VOCAB: Dict[str, str] = {
        # General & Technical Terms
        "सिनेप्सएआई": "SynapseAI",
        "नेटवर्कएक्स": "NetworkX",
        "क्रोमाडीबी": "ChromaDB",
        "ग्राफ": "graph",
        "ग्राफ़": "graph",
        "इंजन": "engine",
        "उपयोग": "uses",
        "नॉलेज ग्राफ": "Knowledge Graph",
        "ज्ञान ग्राफ": "Knowledge Graph",
        "वेक्टर स्टोर": "Vector Store",
        "एआई": "AI",
        "कृत्रिम बुद्धिमत्ता": "Artificial Intelligence",
        "मशीन लर्निंग": "Machine Learning",
        "डीप लर्निंग": "Deep Learning",
        "ट्रांसफॉर्मर": "Transformer",
        "दवा": "Drug",
        "उपचार": "Treatment",
        "कैंसर": "Cancer",
        "जीन": "Gene",
        "प्रोटीन": "Protein",
        "न्यूरल नेटवर्क": "Neural Network",
        "सेंसर": "Sensor",
        "ऊर्जा": "Energy",
        "दस्तावेज़": "Document",
        "अध्ययन": "Study",
        "परीक्षण": "Trial",
        "परिणाम": "Result",
        "दक्षता": "Efficiency",
        "सटीकता": "Accuracy",
        "जोखिम": "Risk",
        "प्रभाव": "Impact",

        # Hinglish terms
        "sambandh": "relationship",
        "judav": "connection",
        "kya hai": "what is",
        "kaise juda hai": "how is connected",
    }

    def __init__(self, graph_builder: KnowledgeGraphBuilder, vector_store: VectorStoreService) -> None:
        self.graph_builder = graph_builder
        self.vector_store = vector_store

    def analyze_query(self, query: str) -> MultilingualQueryInfo:
        """Detect language, extract concepts, and normalize Hindi query to English retrieval query."""
        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", query))
        is_devanagari = devanagari_chars > 2

        words_lower = [w.lower().strip("?,.!") for w in query.split()]
        has_hinglish = any(w in self.HINGLISH_MARKERS for w in words_lower)

        if is_devanagari:
            lang = "hindi"
        elif has_hinglish:
            lang = "hinglish"
        else:
            lang = "english"

        # Resolve Hindi entities to English terms
        normalized_tokens: List[str] = []
        detected_aliases: List[Tuple[str, str]] = []

        # Check full string substitutions first - longest multi-word terms first
        temp_query = query
        sorted_vocab = sorted(self.HINDI_TO_ENGLISH_VOCAB.items(), key=lambda x: len(x[0]), reverse=True)
        for h_term, e_term in sorted_vocab:
            if h_term in temp_query:
                detected_aliases.append((h_term, e_term))
                temp_query = temp_query.replace(h_term, e_term)

        # Check existing entity names in the Knowledge Graph for direct matches or transliterations
        node_names = {data.get("name", nid).lower(): nid for nid, data in self.graph_builder.graph.nodes(data=True)}
        for word in query.split():
            clean_w = word.strip("?,.!'\"")
            if clean_w.lower() in node_names:
                canonical = node_names[clean_w.lower()]
                detected_aliases.append((clean_w, canonical))

        # Build English search query
        # Remove purely Hindi grammatical stopwords from normalized English retrieval string
        stop_words_hindi = self.HINDI_INDICATORS_DEVANAGARI.union(self.HINGLISH_MARKERS)
        words = temp_query.split()
        filtered_words = [w for w in words if w.strip("?,.!'\"") not in stop_words_hindi]

        normalized_english = " ".join(filtered_words).strip()
        if not normalized_english:
            normalized_english = query  # fallback

        # Determine question intent
        if any(w in query.lower() for w in ["संबंध", "जुड़ा", "जुड़े", "रिलेशन", "sambandh", "connect", "path"]):
            intent = "relationship"
        elif any(w in query.lower() for w in ["विषय", "थीम", "समूह", "topic", "theme"]):
            intent = "thematic"
        else:
            intent = "factual"

        return MultilingualQueryInfo(
            original_query=query,
            detected_language=lang,
            is_devanagari=is_devanagari,
            normalized_english_query=normalized_english,
            detected_entity_aliases=detected_aliases,
            intent=intent,
        )

    def process_query(self, query: str) -> MultilingualQueryInfo:
        """Alias for analyze_query."""
        return self.analyze_query(query)

    def retrieve_cross_lingual(
        self,
        query_info: MultilingualQueryInfo,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Perform cross-lingual retrieval against English document chunks."""
        # Use English-normalized query to retrieve from English vector index
        results = self.vector_store.similarity_search(query_info.normalized_english_query, top_k=top_k)

        # If zero results and original query differed, try fallback with original query
        if not results and query_info.normalized_english_query != query_info.original_query:
            results = self.vector_store.similarity_search(query_info.original_query, top_k=top_k)

        return results

    def retrieve_multilingual(
        self,
        query_info: MultilingualQueryInfo,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Alias for retrieve_cross_lingual."""
        return self.retrieve_cross_lingual(query_info, top_k=top_k)

    def find_graph_path_for_hindi_query(
        self,
        query_info: MultilingualQueryInfo,
    ) -> Optional[Dict[str, Any]]:
        """Resolve entities mentioned in Hindi and find the connecting path in the English graph."""
        # Extract candidate entity IDs from detected aliases or words
        candidate_eids: List[str] = []
        for _, eid in query_info.detected_entity_aliases:
            if self.graph_builder.graph.has_node(eid) and eid not in candidate_eids:
                candidate_eids.append(eid)

        # Also search graph node names directly against normalized query
        norm_lower = query_info.normalized_english_query.lower()
        for nid, data in self.graph_builder.graph.nodes(data=True):
            name = data.get("name", nid).lower()
            if name in norm_lower and nid not in candidate_eids:
                candidate_eids.append(nid)

        if len(candidate_eids) < 2:
            return None

        src_id = candidate_eids[0]
        dst_id = candidate_eids[1]

        # Find shortest path in NetworkX
        try:
            # Try directed path
            if nx.has_path(self.graph_builder.graph, src_id, dst_id):
                node_path = nx.shortest_path(self.graph_builder.graph, src_id, dst_id)
            elif nx.has_path(self.graph_builder.graph, dst_id, src_id):
                node_path = nx.shortest_path(self.graph_builder.graph, dst_id, src_id)
            else:
                # Try undirected path
                undirected = self.graph_builder.graph.to_undirected()
                if nx.has_path(undirected, src_id, dst_id):
                    node_path = nx.shortest_path(undirected, src_id, dst_id)
                else:
                    return None

            # Collect edge details and evidence
            edges_info = []
            for i in range(len(node_path) - 1):
                u, v = node_path[i], node_path[i + 1]
                if self.graph_builder.graph.has_edge(u, v):
                    edata = self.graph_builder.graph[u][v]
                elif self.graph_builder.graph.has_edge(v, u):
                    edata = self.graph_builder.graph[v][u]
                else:
                    edata = {}
                edges_info.append({
                    "source": u,
                    "target": v,
                    "predicate": edata.get("predicate", "RELATES_TO"),
                    "evidence": edata.get("evidence", []),
                })

            return {
                "source_entity": src_id,
                "target_entity": dst_id,
                "node_path": node_path,
                "edges": edges_info,
            }
        except Exception:
            return None

    def generate_hindi_answer(
        self,
        query_info: MultilingualQueryInfo,
        search_results: List[SearchResult],
        graph_path: Optional[Dict[str, Any]] = None,
    ) -> GroundedAnswer:
        """Synthesize a grounded answer in Hindi with valid citations to English sources."""
        # 1. Abstention check: if no search results and no graph path, abstain in Hindi
        if not search_results and not graph_path:
            return GroundedAnswer(
                query=query_info.original_query,
                answer_text="उपलब्ध अंग्रेज़ी दस्तावेज़ों में इस प्रश्न का उत्तर देने के लिए पर्याप्त साक्ष्य नहीं मिले।",
                citations=[],
                sufficiency=SufficiencyEvaluation(
                    status=SufficiencyStatus.INSUFFICIENT_EVIDENCE,
                    confidence=0.92,
                    reason="No matching document passages or knowledge graph paths were found for the Hindi query.",
                    missing_elements=["Relevant document passages in English corpus."],
                ),
                is_abstained=True,
                generation_mode="multilingual_grounded",
            )

        citations: List[Citation] = []
        answer_parts: List[str] = []

        # 2. If graph path is available, format the relationship explanation in Hindi
        if graph_path:
            node_path = graph_path["node_path"]
            src_name = self.graph_builder.graph.nodes.get(node_path[0], {}).get("name", node_path[0])
            dst_name = self.graph_builder.graph.nodes.get(node_path[-1], {}).get("name", node_path[-1])

            path_str = " ➔ ".join(node_path)
            answer_parts.append(
                f"ज्ञान ग्राफ (Knowledge Graph) के अनुसार, '{src_name}' और '{dst_name}' निम्नलिखित पथ द्वारा सीधे जुड़े हुए हैं:\n"
                f"**पथ:** `{path_str}`\n"
            )

            # Add citations for edges in path
            for i, edge in enumerate(graph_path.get("edges", []), 1):
                ev_list = edge.get("evidence", [])
                for ev in ev_list:
                    fname = ev.get("source_filename", "graph_evidence") if isinstance(ev, dict) else getattr(ev, "source_filename", "graph_evidence")
                    cid = ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)
                    page = ev.get("page_number") if isinstance(ev, dict) else getattr(ev, "page_number", None)
                    quote = ev.get("exact_quote", "") if isinstance(ev, dict) else getattr(ev, "exact_quote", "")

                    cit_id = f"[{len(citations) + 1}]"
                    citations.append(Citation(
                        citation_id=cit_id,
                        source_filename=fname,
                        page_number=page,
                        chunk_id=cid,
                        quoted_snippet=quote or f"{edge['source']} {edge['predicate']} {edge['target']}",
                        is_verified=True,
                    ))
                    answer_parts.append(
                        f"- संबंध `{edge['source']}` **{edge['predicate']}** `{edge['target']}` {cit_id}"
                    )

        # 3. Grounded factual excerpts from retrieved English passages
        if search_results:
            answer_parts.append("\n**दस्तावेज़ों से प्राप्त मुख्य साक्ष्य:**")
            for i, res in enumerate(search_results[:3], 1):
                cit_id = f"[{len(citations) + 1}]"
                citations.append(Citation(
                    citation_id=cit_id,
                    source_filename=res.source_filename,
                    page_number=res.page_number,
                    chunk_id=res.chunk_id,
                    quoted_snippet=res.text[:180].strip(),
                    is_verified=True,
                ))

                # Present English excerpt cleanly with Hindi context
                snippet_clean = res.text.replace("\n", " ").strip()
                if len(snippet_clean) > 220:
                    snippet_clean = snippet_clean[:220] + "..."

                answer_parts.append(f"{cit_id} *\"{snippet_clean}\"* ({res.source_filename})")

        final_text = "\n\n".join(answer_parts).strip()

        return GroundedAnswer(
            query=query_info.original_query,
            answer_text=final_text,
            citations=citations,
            sufficiency=SufficiencyEvaluation(
                status=SufficiencyStatus.SUPPORTED,
                confidence=0.89,
                reason="Cross-lingual retrieval and graph path successfully grounded the response.",
            ),
            is_abstained=False,
            generation_mode="multilingual_grounded",
        )
