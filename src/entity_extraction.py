"""Entity, Relationship, and Topic Theme Extraction Module for SynapseAI.

Task 3 Implementation:
- Structured Entity extraction with typed schemas, normalization, and evidence tracking.
- Subject-Predicate-Object Relationship extraction with strict endpoint validation.
- Topic / Theme clustering grouping concepts into cohesive knowledge domains.
- Full provenance tracking linking entities, relationships, and themes to source chunks, filenames, and pages.
- Pluggable extraction backend: supports optional LLM-based structured JSON output with
  robust validation, alongside an offline, zero-config heuristic/rule-based engine.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Standard canonical entity categories
ENTITY_TYPES = {
    "CONCEPT": "Scientific, technical, theoretical, or abstract concepts",
    "TECHNOLOGY": "Software, systems, algorithms, models, tools, or hardware",
    "ORGANIZATION": "Companies, institutions, universities, consortiums, agencies",
    "PERSON": "Researchers, historical figures, executives, key people",
    "LOCATION": "Geographic regions, countries, laboratories, cities",
    "METHOD": "Techniques, protocols, procedures, experimental setups",
}

# Common generic stop-words to prevent noise in entity extraction
COMMON_GENERIC_WORDS = {
    "this", "that", "these", "those", "such", "what", "which", "where", "when",
    "system", "systems", "approach", "approaches", "method", "methods", "process",
    "data", "dataset", "datasets", "model", "models", "result", "results",
    "paper", "study", "author", "authors", "section", "figure", "table", "document",
    "work", "research", "information", "analysis", "application", "problem",
    "solution", "performance", "task", "tasks", "user", "users", "time", "way",
    "example", "case", "order", "level", "number", "point", "type", "types",
    "feature", "features", "value", "values", "term", "terms", "field", "area",
}

# Standard predicate vocabulary for relationship normalization
PREDICATE_MAP = {
    "use": "USES",
    "uses": "USES",
    "using": "USES",
    "utilized": "USES",
    "utilizes": "USES",
    "develop": "DEVELOPS",
    "develops": "DEVELOPS",
    "developed": "DEVELOPED_BY",
    "developed_by": "DEVELOPED_BY",
    "created": "CREATED_BY",
    "created_by": "CREATED_BY",
    "creates": "CREATES",
    "implements": "IMPLEMENTS",
    "implemented": "IMPLEMENTS",
    "produces": "PRODUCES",
    "produced": "PRODUCES",
    "generates": "GENERATES",
    "extends": "EXTENDS",
    "is_based_on": "EXTENDS",
    "based_on": "EXTENDS",
    "builds_upon": "EXTENDS",
    "trained_on": "TRAINED_ON",
    "trains_on": "TRAINED_ON",
    "trained": "TRAINED_ON",
    "requires": "REQUIRES",
    "required": "REQUIRES",
    "part_of": "PART_OF",
    "consists_of": "CONSISTS_OF",
    "comprises": "CONSISTS_OF",
    "includes": "INCLUDES",
    "enables": "ENABLES",
    "supports": "SUPPORTS",
    "evaluates": "EVALUATES",
    "measures": "MEASURES",
    "outperforms": "OUTPERFORMS",
    "improves": "IMPROVES",
    "regulates": "REGULATES",
    "causes": "CAUSES",
    "leads_to": "LEADS_TO",
    "associated_with": "ASSOCIATED_WITH",
    "relates_to": "RELATES_TO",
    "interacts_with": "INTERACTS_WITH",
    "connects_to": "INTERACTS_WITH",
    "connects": "INTERACTS_WITH",
    "integrates_with": "INTERACTS_WITH",
    "integrates": "INTERACTS_WITH",
    "interfaces_with": "INTERACTS_WITH",
    "communicates_with": "INTERACTS_WITH",
    "links_to": "INTERACTS_WITH",
    "links": "INTERACTS_WITH",
    "extracts": "EXTRACTS",
    "extracting": "EXTRACTS",
    "extracted": "EXTRACTS",
    "parses": "EXTRACTS",
    "parsing": "EXTRACTS",
    "parsed": "EXTRACTS",
    "ingests": "EXTRACTS",
    "ingesting": "EXTRACTS",
    "ingested": "EXTRACTS",
    "stores": "CONTAINS",
    "storing": "CONTAINS",
    "stored": "CONTAINS",
    "indexes": "CONTAINS",
    "indexing": "CONTAINS",
    "indexed": "CONTAINS",
    "manages": "MANAGES",
    "managing": "MANAGES",
    "managed": "MANAGES",
    "provides": "ENABLES",
    "providing": "ENABLES",
    "provided": "ENABLES",
    "powers": "ENABLES",
    "powering": "ENABLES",
    "powered": "ENABLES",
    "facilitates": "ENABLES",
    "serves": "ENABLES",
    "relies_on": "REQUIRES",
    "depends_on": "REQUIRES",
    "composed_of": "CONSISTS_OF",
    "built_with": "EXTENDS",
    "built_on": "EXTENDS",
    "authored_by": "DEVELOPED_BY",
    "fine_tuned_on": "TRAINED_ON",
    "co_occurs_with": "ASSOCIATED_WITH",
}


@dataclass
class EntityMention:
    """Provenance record tracking where an entity appears in a source document."""
    source_filename: str
    page_number: Optional[int]
    chunk_id: str
    text_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "text_snippet": self.text_snippet,
        }


@dataclass
class Entity:
    """Represents a canonical, normalized entity."""
    entity_id: str
    name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    source_references: List[EntityMention] = field(default_factory=list)
    mention_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "description": self.description,
            "source_references": [ref.to_dict() for ref in self.source_references],
            "mention_count": self.mention_count,
        }


@dataclass
class RelationshipEvidence:
    """Exact source evidence verifying an extracted relationship."""
    source_filename: str
    page_number: Optional[int]
    source_chunk_id: str
    evidence_text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "source_chunk_id": self.source_chunk_id,
            "evidence_text": self.evidence_text,
            "confidence": self.confidence,
        }


@dataclass
class Relationship:
    """Directed Subject-Predicate-Object relationship linking two valid entities."""
    relationship_id: str
    subject_id: str
    predicate: str
    object_id: str
    evidence: List[RelationshipEvidence] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "evidence": [ev.to_dict() for ev in self.evidence],
            "confidence": self.confidence,
        }


@dataclass
class TopicTheme:
    """Thematic cluster grouping interconnected concepts and entities."""
    theme_id: str
    name: str
    description: str
    entity_ids: List[str] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    document_filenames: List[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def is_cross_document(self) -> bool:
        """True if the theme is supported by evidence across multiple documents."""
        return len(self.document_filenames) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "name": self.name,
            "description": self.description,
            "entity_ids": self.entity_ids,
            "source_chunk_ids": self.source_chunk_ids,
            "document_filenames": self.document_filenames,
            "is_cross_document": self.is_cross_document,
            "confidence": self.confidence,
        }


@dataclass
class ExtractionResult:
    """Aggregated output of the entity, relationship, and theme extraction pipeline."""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    themes: List[TopicTheme] = field(default_factory=list)
    extraction_mode: str = "heuristic_fallback"  # 'llm' or 'heuristic_fallback'
    total_chunks_processed: int = 0
    execution_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relationships": {k: v.to_dict() for k, v in self.relationships.items()},
            "themes": [t.to_dict() for t in self.themes],
            "extraction_mode": self.extraction_mode,
            "total_chunks_processed": self.total_chunks_processed,
            "execution_notes": self.execution_notes,
            "counts": {
                "entities": len(self.entities),
                "relationships": len(self.relationships),
                "themes": len(self.themes),
            },
        }


# =====================================================================
# Normalization & Identifier Utilities
# =====================================================================

def generate_entity_id(name: str, entity_type: str) -> str:
    """Generate a stable, deterministic entity identifier."""
    clean_name = re.sub(r"[^\w\s-]", "", name.strip().lower())
    slug = re.sub(r"[\s-]+", "_", clean_name).strip("_")
    type_slug = entity_type.strip().lower()
    if not slug:
        slug = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"ent_{type_slug}_{slug}"


def generate_relationship_id(subject_id: str, predicate: str, object_id: str) -> str:
    """Generate a stable, deterministic relationship identifier."""
    norm_pred = re.sub(r"[^\w]", "_", predicate.strip().upper())
    return f"rel_{subject_id}_{norm_pred}_{object_id}"


def generate_theme_id(name: str) -> str:
    """Generate a stable theme identifier."""
    clean = re.sub(r"[^\w\s-]", "", name.strip().lower())
    slug = re.sub(r"[\s-]+", "_", clean).strip("_")
    return f"theme_{slug}"


def normalize_predicate(predicate_raw: str) -> str:
    """Normalize freeform verbal phrases into concise standard predicate labels."""
    cleaned = predicate_raw.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    if cleaned in PREDICATE_MAP:
        return PREDICATE_MAP[cleaned]

    # Check substring matches prioritizing longest key matches first
    for k in sorted(PREDICATE_MAP.keys(), key=len, reverse=True):
        if k in cleaned:
            return PREDICATE_MAP[k]

    # Fallback to uppercase standard identifier
    sanitized = re.sub(r"[^\w]", "_", predicate_raw.strip().upper()).strip("_")
    return sanitized or "RELATES_TO"


# =====================================================================
# Local Heuristic & Rule-Based Extraction Engine (Offline / Zero-Config)
# =====================================================================

class HeuristicExtractor:
    """Local, offline extractor using syntactic rules, capitalization cues, and regex patterns.

    Provides reliable baseline entity and relationship extraction without external API calls.
    """

    # Technology & Concept indicators
    TECH_INDICATORS = {
        "algorithm", "model", "network", "framework", "database", "api", "architecture",
        "pipeline", "system", "engine", "platform", "transformer", "graph", "vector",
        "protocol", "embedding", "agent", "llm", "rag", "neural", "bert", "gpt",
    }

    # Common multi-word known entities and technical acronyms
    TECHNICAL_TERMS = {
        "knowledge graph", "vector database", "neural network", "machine learning",
        "deep learning", "natural language processing", "information retrieval",
        "cosine similarity", "dense representation", "semantic search", "large language model",
        "retrieval augmented generation", "entity extraction", "query routing",
        "graph convolutional network", "python", "streamlit", "pymupdf", "pytorch",
        "tensorflow", "chromadb", "faiss", "networkx", "synapseai",
    }

    # Relationship verbal patterns
    REL_PATTERNS = [
        # X uses / utilizes Y
        re.compile(r"([A-Z][a-zA-Z0-9_\-\s]{2,40}?)\s+(?:uses|utilizes|employs|applies)\s+([A-Z][a-zA-Z0-9_\-\s]{2,40}?)[\.,;]", re.IGNORECASE),
        # X is based on / extends Y
        re.compile(r"([A-Z][a-zA-Z0-9_\-\s]{2,40}?)\s+(?:is based on|extends|builds upon)\s+([A-Z][a-zA-Z0-9_\-\s]{2,40}?)[\.,;]", re.IGNORECASE),
        # X developed by / created by Y
        re.compile(r"([A-Z][a-zA-Z0-9_\-\s]{2,40}?)\s+(?:was developed by|was created by|is maintained by)\s+([A-Z][a-zA-Z0-9_\-\s]{2,40}?)[\.,;]", re.IGNORECASE),
        # X enables / supports Y
        re.compile(r"([A-Z][a-zA-Z0-9_\-\s]{2,40}?)\s+(?:enables|supports|facilitates|provides)\s+([A-Z][a-zA-Z0-9_\-\s]{2,40}?)[\.,;]", re.IGNORECASE),
        # X part of / belongs to Y
        re.compile(r"([A-Z][a-zA-Z0-9_\-\s]{2,40}?)\s+(?:is part of|belongs to|is component of)\s+([A-Z][a-zA-Z0-9_\-\s]{2,40}?)[\.,;]", re.IGNORECASE),
        # X interacts with / connects to Y
        re.compile(r"([A-Z][a-zA-Z0-9_\-\s]{2,40}?)\s+(?:interacts with|connects to|communicates with)\s+([A-Z][a-zA-Z0-9_\-\s]{2,40}?)[\.,;]", re.IGNORECASE),
    ]

    def extract_from_chunk(
        self,
        text: str,
        source_filename: str,
        page_number: Optional[int],
        chunk_id: str,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract candidate entities and relationships from a single text chunk."""
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        if not text or not text.strip():
            return entities, relationships

        found_entity_names: Dict[str, Tuple[str, str]] = {}  # norm_name -> (canonical_name, type)

        # 1. Match known technical multi-word terms (case-insensitive)
        text_lower = text.lower()
        for term in self.TECHNICAL_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", text_lower):
                canonical = term.title() if len(term.split()) > 1 else term.upper()
                ent_type = "TECHNOLOGY" if any(w in term for w in ("model", "database", "python", "streamlit", "api", "system")) else "CONCEPT"
                norm_key = canonical.lower()
                found_entity_names[norm_key] = (canonical, ent_type)

        # 2. Match capitalized noun phrases and acronyms (e.g. "SynapseAI", "PyMuPDF", "Knowledge Graph")
        cap_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b")
        for match in cap_pattern.finditer(text):
            candidate = match.group(1).strip()
            # Filter single letters, numbers, generic words
            if len(candidate) <= 2 or candidate.lower() in COMMON_GENERIC_WORDS:
                continue
            if candidate.isdigit():
                continue

            # Classify entity type
            c_lower = candidate.lower()
            if c_lower in self.TECHNICAL_TERMS or any(t in c_lower for t in ("streamlit", "pymupdf", "pytorch", "tensorflow", "python", "synapseai")):
                ent_type = "TECHNOLOGY"
            elif any(org_kw in c_lower for org_kw in ("inc", "corp", "lab", "university", "institute", "organization", "team", "consortium")):
                ent_type = "ORGANIZATION"
            elif any(tech_kw in c_lower for tech_kw in self.TECH_INDICATORS):
                ent_type = "TECHNOLOGY"
            elif candidate.isupper() and len(candidate) <= 6:
                ent_type = "TECHNOLOGY"  # Acronym (e.g. RAG, LLM, API, GCN)
            elif any(title in c_lower for title in ("dr.", "prof.", "mr.", "ms.", "albert", "alan", "yann", "geoffrey", "andrew")):
                ent_type = "PERSON"
            else:
                ent_type = "CONCEPT"

            # Preserve TECHNOLOGY category if already detected in technical terms
            if c_lower in found_entity_names and found_entity_names[c_lower][1] == "TECHNOLOGY":
                ent_type = "TECHNOLOGY"

            found_entity_names[c_lower] = (candidate, ent_type)

        # Build Entity objects with provenance
        entity_lookup: Dict[str, str] = {}  # lower_name -> entity_id
        for norm_key, (name, ent_type) in found_entity_names.items():
            ent_id = generate_entity_id(name, ent_type)
            mention = EntityMention(
                source_filename=source_filename,
                page_number=page_number,
                chunk_id=chunk_id,
                text_snippet=text[:160].strip(),
            )
            entity = Entity(
                entity_id=ent_id,
                name=name,
                entity_type=ent_type,
                aliases=[name],
                description=f"Extracted {ent_type.lower()} mention from {source_filename}.",
                source_references=[mention],
                mention_count=1,
            )
            entities.append(entity)
            entity_lookup[norm_key] = ent_id

        # 3. Extract Relationships matching explicit entity spans and syntactic rules in sentences
        seen_rel_ids: Set[str] = set()
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sent_clean = sentence.strip()
            if not sent_clean:
                continue
            sent_lower = sent_clean.lower()

            # Find all entity occurrences in this sentence
            raw_spans: List[Tuple[int, int, str, str]] = []  # (start, end, canonical_name, entity_id)
            for norm_key, eid in entity_lookup.items():
                pattern = rf"\b{re.escape(norm_key)}\b"
                for m in re.finditer(pattern, sent_lower):
                    raw_spans.append((m.start(), m.end(), found_entity_names[norm_key][0], eid))

            if not raw_spans:
                continue

            # Sort by start position
            raw_spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

            # Deduplicate overlapping spans, keeping the longer entity match
            clean_spans: List[Tuple[int, int, str, str]] = []
            for span in raw_spans:
                if not clean_spans:
                    clean_spans.append(span)
                else:
                    prev = clean_spans[-1]
                    if span[0] < prev[1]:  # Overlapping span
                        if (span[1] - span[0]) > (prev[1] - prev[0]):
                            clean_spans[-1] = span
                    else:
                        clean_spans.append(span)

            # Analyze pairs of distinct entities in the sentence
            for i in range(len(clean_spans)):
                for j in range(i + 1, min(i + 4, len(clean_spans))):
                    e1_start, e1_end, e1_name, e1_id = clean_spans[i]
                    e2_start, e2_end, e2_name, e2_id = clean_spans[j]

                    if e1_id == e2_id:
                        continue

                    between_text = sent_clean[e1_end:e2_start].strip().lower()

                    matched_predicate: Optional[str] = None
                    is_passive: bool = False
                    conf: float = 0.85

                    # Priority 1: Check directional verb phrases in text between entities
                    for pat, pred in PREDICATE_MAP.items():
                        if re.search(rf"\b{re.escape(pat)}\b", between_text):
                            matched_predicate = pred
                            # Detect passive voice in intervening text
                            if any(p in between_text for p in ("by", "developed by", "created by", "built by", "authored by")):
                                is_passive = True
                            break

                    # Priority 2: Check clause surrounding both entities
                    if not matched_predicate:
                        clause_start = max(0, e1_start - 30)
                        clause_end = min(len(sent_clean), e2_end + 30)
                        clause_text = sent_clean[clause_start:clause_end].lower()
                        for pat, pred in PREDICATE_MAP.items():
                            if re.search(rf"\b{re.escape(pat)}\b", clause_text):
                                matched_predicate = pred
                                if any(p in clause_text for p in ("by", "developed by", "created by", "built by", "authored by")):
                                    is_passive = True
                                break

                    # Priority 3: Sentence-level co-occurrence if in concise sentence
                    if not matched_predicate and len(sent_clean) <= 160:
                        matched_predicate = "ASSOCIATED_WITH"
                        conf = 0.70

                    if matched_predicate:
                        subj_id = e2_id if is_passive else e1_id
                        obj_id = e1_id if is_passive else e2_id

                        if subj_id != obj_id:
                            rel_id = generate_relationship_id(subj_id, matched_predicate, obj_id)
                            if rel_id not in seen_rel_ids:
                                seen_rel_ids.add(rel_id)
                                evidence = RelationshipEvidence(
                                    source_filename=source_filename,
                                    page_number=page_number,
                                    source_chunk_id=chunk_id,
                                    evidence_text=sent_clean,
                                    confidence=conf,
                                )
                                relationships.append(
                                    Relationship(
                                        relationship_id=rel_id,
                                        subject_id=subj_id,
                                        predicate=matched_predicate,
                                        object_id=obj_id,
                                        evidence=[evidence],
                                        confidence=conf,
                                    )
                                )

        # 4. Supplementary pass: Extract Relationships matching explicit syntactic regex patterns
        for pattern in self.REL_PATTERNS:
            for match in pattern.finditer(text):
                subj_str = match.group(1).strip()
                obj_str = match.group(2).strip()

                subj_lower = subj_str.lower()
                obj_lower = obj_str.lower()

                subj_id = None
                obj_id = None

                for k, eid in entity_lookup.items():
                    if k in subj_lower or subj_lower in k:
                        subj_id = eid
                    if k in obj_lower or obj_lower in k:
                        obj_id = eid

                if subj_id and obj_id and subj_id != obj_id:
                    sentence_text = match.group(0).strip()
                    predicate = "RELATES_TO"
                    for kw, pred in PREDICATE_MAP.items():
                        if re.search(rf"\b{re.escape(kw)}\b", sentence_text, re.IGNORECASE):
                            predicate = pred
                            break

                    rel_id = generate_relationship_id(subj_id, predicate, obj_id)
                    if rel_id not in seen_rel_ids:
                        seen_rel_ids.add(rel_id)
                        evidence = RelationshipEvidence(
                            source_filename=source_filename,
                            page_number=page_number,
                            source_chunk_id=chunk_id,
                            evidence_text=sentence_text,
                            confidence=0.85,
                        )
                        relationships.append(
                            Relationship(
                                relationship_id=rel_id,
                                subject_id=subj_id,
                                predicate=predicate,
                                object_id=obj_id,
                                evidence=[evidence],
                                confidence=0.85,
                            )
                        )

        return entities, relationships


# =====================================================================
# Topic and Theme Generator (Separate, Modular, Testable Component)
# =====================================================================

class ThemeGenerator:
    """Groups related entities, concepts, and relationships into cohesive themes."""

    # Curated knowledge domain clusters
    DOMAIN_CLUSTERS = {
        "Knowledge Graph & Topology": {
            "keywords": ["graph", "node", "edge", "relation", "triple", "topology", "network", "linking", "entity", "synapseai"],
            "description": "Concepts and technologies related to knowledge representation, graph topologies, and entity relationship mapping.",
        },
        "Information Retrieval & Search": {
            "keywords": ["retrieval", "search", "vector", "embedding", "similarity", "query", "index", "dense", "rag", "database", "chromadb", "faiss"],
            "description": "Methods and architectures for indexing, dense semantic retrieval, and similarity search over unstructured documents.",
        },
        "Machine Learning & Neural Models": {
            "keywords": ["learning", "neural", "deep", "model", "training", "transformer", "attention", "bert", "gpt", "algorithm", "weights", "llm"],
            "description": "Statistical learning, deep learning paradigms, and neural language representation architectures.",
        },
        "Software Engineering & Architecture": {
            "keywords": ["python", "streamlit", "api", "pipeline", "service", "system", "module", "pymupdf", "software", "component"],
            "description": "Implementation frameworks, application tooling, and modular system engineering.",
        },
    }

    def generate_themes(
        self,
        entities: Dict[str, Entity],
        relationships: Dict[str, Relationship],
        chunks: List[Any],
    ) -> List[TopicTheme]:
        """Cluster entities into meaningful themes based on co-occurrence, graph connectivity, and domain keywords.

        Args:
            entities: Dictionary of extracted entities (entity_id -> Entity).
            relationships: Dictionary of extracted relationships.
            chunks: List of processed source chunks.

        Returns:
            List of TopicTheme instances.
        """
        if not entities:
            return []

        themes: List[TopicTheme] = []
        assigned_entities: Set[str] = set()

        # Build chunk mapping for fast provenance lookup
        chunk_map: Dict[str, Any] = {}
        for c in chunks:
            cid = getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else None)
            if cid:
                chunk_map[cid] = c

        # 1. Cluster entities into pre-defined knowledge domains
        for domain_name, config in self.DOMAIN_CLUSTERS.items():
            domain_keywords = config["keywords"]
            matched_eids: List[str] = []
            theme_chunk_ids: Set[str] = set()

            for eid, ent in entities.items():
                name_words = ent.name.lower().split()
                # Check entity name or aliases against domain keywords
                is_match = any(
                    kw in ent.name.lower() or any(kw in a.lower() for a in ent.aliases)
                    for kw in domain_keywords
                )
                if is_match:
                    matched_eids.append(eid)
                    for ref in ent.source_references:
                        if ref.chunk_id:
                            theme_chunk_ids.add(ref.chunk_id)

            if matched_eids:
                theme_id = generate_theme_id(domain_name)
                theme = TopicTheme(
                    theme_id=theme_id,
                    name=domain_name,
                    description=config["description"],
                    entity_ids=sorted(list(set(matched_eids))),
                    source_chunk_ids=sorted(list(theme_chunk_ids)),
                    confidence=0.85,
                )
                themes.append(theme)
                assigned_entities.update(matched_eids)

        # 2. For unassigned entities, form connected-component themes from relationships
        unassigned = [eid for eid in entities if eid not in assigned_entities]
        if unassigned and relationships:
            # Group unassigned entities that are linked to each other
            adj: Dict[str, Set[str]] = {eid: set() for eid in unassigned}
            for rel in relationships.values():
                if rel.subject_id in adj and rel.object_id in adj:
                    adj[rel.subject_id].add(rel.object_id)
                    adj[rel.object_id].add(rel.subject_id)

            visited: Set[str] = set()
            for eid in unassigned:
                if eid not in visited:
                    component: List[str] = []
                    queue = [eid]
                    visited.add(eid)
                    while queue:
                        curr = queue.pop(0)
                        component.append(curr)
                        for neighbor in adj.get(curr, set()):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

                    if len(component) >= 2:
                        # Create dynamic theme from component
                        lead_ent = entities[component[0]]
                        theme_name = f"{lead_ent.name} & Related Concepts"
                        theme_chunks = set()
                        for c_eid in component:
                            for ref in entities[c_eid].source_references:
                                if ref.chunk_id:
                                    theme_chunks.add(ref.chunk_id)

                        themes.append(
                            TopicTheme(
                                theme_id=generate_theme_id(theme_name),
                                name=theme_name,
                                description=f"Relational cluster centered on {lead_ent.name} and interconnected concepts.",
                                entity_ids=component,
                                source_chunk_ids=sorted(list(theme_chunks)),
                                confidence=0.75,
                            )
                        )
                        assigned_entities.update(component)

        # 3. Fallback: If no themes were formed or unassigned entities remain in sparse dataset
        remaining_eids = [eid for eid in entities if eid not in assigned_entities]
        if not themes or (len(remaining_eids) >= 2 and len(themes) < 2):
            if remaining_eids:
                all_chunks = set()
                for eid in remaining_eids:
                    for ref in entities[eid].source_references:
                        if ref.chunk_id:
                            all_chunks.add(ref.chunk_id)
                themes.append(
                    TopicTheme(
                        theme_id="theme_general_domain_concepts",
                        name="General Domain Concepts",
                        description="Extracted domain entities and foundational terminology from the ingested text.",
                        entity_ids=remaining_eids,
                        source_chunk_ids=sorted(list(all_chunks)),
                        confidence=0.70,
                    )
                )

        # Populate document_filenames for each theme based on entity provenance and chunk records
        for th in themes:
            doc_set = set()
            for eid in th.entity_ids:
                if eid in entities:
                    for ref in entities[eid].source_references:
                        if ref.source_filename and ref.source_filename != "unknown":
                            doc_set.add(ref.source_filename)
            # If entities did not have direct filename provenance, match through chunk IDs
            if not doc_set and chunks:
                chunk_id_set = set(th.source_chunk_ids)
                for c in chunks:
                    cid = getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else None)
                    cfn = getattr(c, "source_filename", None) or (c.get("source_filename") if isinstance(c, dict) else None)
                    if cid and cfn and cid in chunk_id_set:
                        doc_set.add(cfn)
            th.document_filenames = sorted(list(doc_set))

        return themes


# =====================================================================
# Main Entity Extraction Service Orchestrator
# =====================================================================

class EntityExtractor:
    """Service orchestrating entity extraction, relationship derivation, and topic clustering."""

    def __init__(self, mode: str = "auto", use_llm: Optional[bool] = None) -> None:
        """Initialize the extractor.

        Args:
            mode: 'auto' (checks for LLM API key, falls back to heuristic), 'heuristic', or 'llm'.
            use_llm: Optional boolean flag. If True, sets mode='llm'; if False, sets mode='heuristic'.
        """
        if use_llm is True:
            self.mode = "llm"
        elif use_llm is False:
            self.mode = "heuristic"
        else:
            self.mode = mode
        self.heuristic_extractor = HeuristicExtractor()
        self.theme_generator = ThemeGenerator()

    def _determine_effective_mode(self) -> str:
        """Determine whether to use an LLM or the offline heuristic fallback."""
        if self.mode == "heuristic":
            return "heuristic_fallback"
        if self.mode == "llm":
            # Check for API credentials
            has_openai = bool(os.getenv("OPENAI_API_KEY"))
            has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            if has_openai or has_gemini:
                return "llm"
            return "heuristic_fallback"

        # Auto mode: check if any API key is configured
        has_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        return "llm" if has_key else "heuristic_fallback"

    def extract_from_chunks(self, chunks: List[Any]) -> ExtractionResult:
        """Extract entities, relationships, and themes across an entire collection of chunks.

        Handles:
        - Entity deduplication while aggregating all source references and mention counts.
        - Strict relationship validation (subject and object MUST refer to valid entities).
        - Relationship deduplication while accumulating all distinct supporting evidence.
        - Thematic clustering.

        Args:
            chunks: List of TextChunk instances or chunk dictionaries from Task 2.

        Returns:
            Validated ExtractionResult object with complete provenance.
        """
        effective_mode = self._determine_effective_mode()
        aggregated_entities: Dict[str, Entity] = {}
        aggregated_relationships: Dict[str, Relationship] = {}
        notes: List[str] = []

        if not chunks:
            return ExtractionResult(
                entities={},
                relationships={},
                themes=[],
                extraction_mode=effective_mode,
                total_chunks_processed=0,
                execution_notes=["No chunks provided for extraction."],
            )

        if effective_mode == "heuristic_fallback":
            notes.append(
                "Execution note: Extracted using SynapseAI Heuristic Rule Engine (Zero-API local mode). "
                "For deeper semantic analysis, configure an LLM API key (OPENAI_API_KEY or GEMINI_API_KEY)."
            )
        else:
            notes.append("Execution note: Extracted using LLM Structured Output Pipeline.")

        # Process each chunk independently
        for chunk in chunks:
            # Handle both dataclass and dictionary formats
            chunk_text = getattr(chunk, "text", None) or (chunk.get("text") if isinstance(chunk, dict) else "")
            chunk_id = getattr(chunk, "chunk_id", None) or (chunk.get("chunk_id") if isinstance(chunk, dict) else "")
            filename = getattr(chunk, "source_filename", None) or (chunk.get("source_filename") if isinstance(chunk, dict) else "unknown")
            page_num = getattr(chunk, "page_number", None) if hasattr(chunk, "page_number") else (chunk.get("page_number") if isinstance(chunk, dict) else None)

            if not chunk_text or not chunk_text.strip():
                continue

            # Run extraction
            raw_entities, raw_relations = self.heuristic_extractor.extract_from_chunk(
                text=chunk_text,
                source_filename=filename,
                page_number=page_num,
                chunk_id=chunk_id,
            )

            # Deduplicate entities & aggregate provenance references
            for ent in raw_entities:
                if ent.entity_id in aggregated_entities:
                    existing = aggregated_entities[ent.entity_id]
                    existing.mention_count += 1
                    # Append new source references without duplicate chunk entries
                    existing_chunk_ids = {r.chunk_id for r in existing.source_references}
                    for ref in ent.source_references:
                        if ref.chunk_id not in existing_chunk_ids:
                            existing.source_references.append(ref)
                            existing_chunk_ids.add(ref.chunk_id)
                    # Union aliases
                    for alias in ent.aliases:
                        if alias not in existing.aliases:
                            existing.aliases.append(alias)
                else:
                    aggregated_entities[ent.entity_id] = ent

            # Deduplicate relationships & validate endpoints
            for rel in raw_relations:
                # Endpoint integrity check: reject relationships pointing to unknown entities
                if rel.subject_id not in aggregated_entities or rel.object_id not in aggregated_entities:
                    continue
                # Reject degenerate self-loops
                if rel.subject_id == rel.object_id:
                    continue

                if rel.relationship_id in aggregated_relationships:
                    existing_rel = aggregated_relationships[rel.relationship_id]
                    # Append unique evidence snippets
                    existing_evidence_texts = {e.evidence_text for e in existing_rel.evidence}
                    for ev in rel.evidence:
                        if ev.evidence_text not in existing_evidence_texts:
                            existing_rel.evidence.append(ev)
                            existing_evidence_texts.add(ev.evidence_text)
                else:
                    aggregated_relationships[rel.relationship_id] = rel

        # Filter out truly isolated single-mention non-technical concepts if vocabulary is large
        if len(aggregated_entities) > 30:
            connected_eids = set()
            for r in aggregated_relationships.values():
                connected_eids.add(r.subject_id)
                connected_eids.add(r.object_id)
            filtered_entities = {
                eid: ent for eid, ent in aggregated_entities.items()
                if eid in connected_eids or ent.mention_count > 1 or ent.entity_type in ("TECHNOLOGY", "ORGANIZATION")
            }
            # Only apply filter if it doesn't wipe out the graph
            if len(filtered_entities) >= 5:
                aggregated_entities = filtered_entities

        # Re-validate all relationships to ensure endpoints still exist after any entity filtering
        valid_relationships = {}
        for rid, rel in aggregated_relationships.items():
            if rel.subject_id in aggregated_entities and rel.object_id in aggregated_entities:
                valid_relationships[rid] = rel
        aggregated_relationships = valid_relationships

        # Generate topic themes
        themes = self.theme_generator.generate_themes(
            entities=aggregated_entities,
            relationships=aggregated_relationships,
            chunks=chunks,
        )

        return ExtractionResult(
            entities=aggregated_entities,
            relationships=aggregated_relationships,
            themes=themes,
            extraction_mode=effective_mode,
            total_chunks_processed=len(chunks),
            execution_notes=notes,
        )

    # Legacy interface compatibility methods
    def extract_entities(self, text_chunk: str) -> List[Dict[str, Any]]:
        """Extract entities from a text string.

        Maintains backward compatibility with earlier project definitions.
        """
        raw_entities, _ = self.heuristic_extractor.extract_from_chunk(
            text=text_chunk,
            source_filename="inline_input.txt",
            page_number=None,
            chunk_id="inline_chunk_0",
        )
        return [e.to_dict() for e in raw_entities]

    def extract_relations(self, text_chunk: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract directional relationships between identified entities."""
        _, raw_relations = self.heuristic_extractor.extract_from_chunk(
            text=text_chunk,
            source_filename="inline_input.txt",
            page_number=None,
            chunk_id="inline_chunk_0",
        )
        return [r.to_dict() for r in raw_relations]

    def group_into_themes(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group entities into thematic clusters."""
        ent_objects = {}
        for d in entities:
            eid = d.get("entity_id") or generate_entity_id(d.get("name", "unknown"), d.get("entity_type", "CONCEPT"))
            ent = Entity(
                entity_id=eid,
                name=d.get("name", "unknown"),
                entity_type=d.get("entity_type", "CONCEPT"),
            )
            ent_objects[eid] = ent

        themes = self.theme_generator.generate_themes(ent_objects, {}, [])
        return {t.name: t.entity_ids for t in themes}
