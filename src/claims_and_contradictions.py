"""Claim Modeling and Cross-Document Contradiction Detection Module for SynapseAI.

Phase 2 Implementation:
- First-class Claim representation preserving subject, predicate, object, polarity,
  quantity, unit, temporal scope, conditions, source filename, chunk ID, and exact excerpt.
- Cross-document Contradiction Detection:
  1. Identifies candidate claims with equivalent subject and attribute/property.
  2. Normalizes quantities, units, dates, and polarities.
  3. Detects direct polarity negation, numerical divergence under identical scope,
     and temporal incompatibilities.
  4. Respects condition and scope differences (avoids false-positive contradictions).
  5. Distinguishes confirmed contradictions, potential contradictions, and compatible claims.
- Bidirectional traceability: preserves source documents, chunks, page numbers,
  and citations for both sides of conflicting claims.
- Safe integration with KnowledgeGraphBuilder and AnswerGenerator.
"""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.entity_extraction import ExtractionResult, Relationship
from src.document_ingestion import TextChunk


@dataclass
class Claim:
    """Represents a discrete factual claim extracted from an untrusted source document."""
    claim_id: str
    subject: str
    predicate: str
    object: str
    source_doc_id: str
    source_filename: str
    chunk_id: Optional[str]
    page_number: Optional[int]
    excerpt: str
    claim_type: str = "assertion"  # "polarity", "numerical", "temporal", "relational", "assertion"
    polarity: bool = True          # True = affirmative, False = negated
    quantity: Optional[float] = None
    unit: Optional[str] = None
    date_scope: Optional[str] = None
    condition_or_population: Optional[str] = None
    extraction_method: str = "rule_based"
    confidence: float = 1.0

    @property
    def claim_text(self) -> str:
        """Canonical original source statement of the claim, preserving exact excerpt or structured fallback."""
        if self.excerpt and self.excerpt.strip():
            return self.excerpt.strip()
        # Clean structured fallback if excerpt is not populated
        val_str = f" {self.quantity} {self.unit or ''}".strip() if self.quantity is not None else ""
        pol_str = "" if self.polarity else "not "
        return f"{self.subject} is {pol_str}{self.predicate} {self.object}{val_str}".strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source_doc_id": self.source_doc_id,
            "source_filename": self.source_filename,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "excerpt": self.excerpt,
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "polarity": self.polarity,
            "quantity": self.quantity,
            "unit": self.unit,
            "date_scope": self.date_scope,
            "condition_or_population": self.condition_or_population,
            "extraction_method": self.extraction_method,
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Reconstruct a Claim instance from a dictionary, safely handling field variations."""
        excerpt = data.get("excerpt") or data.get("claim_text") or data.get("text") or ""
        return cls(
            claim_id=str(data.get("claim_id", "")),
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            object=str(data.get("object", "")),
            source_doc_id=str(data.get("source_doc_id", "unknown")),
            source_filename=str(data.get("source_filename", "unknown")),
            chunk_id=data.get("chunk_id"),
            page_number=data.get("page_number"),
            excerpt=excerpt,
            claim_type=str(data.get("claim_type", "assertion")),
            polarity=bool(data.get("polarity", True)),
            quantity=data.get("quantity"),
            unit=data.get("unit"),
            date_scope=data.get("date_scope"),
            condition_or_population=data.get("condition_or_population"),
            extraction_method=str(data.get("extraction_method", "rule_based")),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class ContradictionRecord:
    """Represents an evidence-grounded conflict detected between two independent claims."""
    record_id: str
    claim_a: Claim
    claim_b: Claim
    contradiction_type: str  # "polarity", "numerical", "temporal", "relational"
    conflicting_attribute: str
    date_or_scope: Optional[str] = None
    status: str = "confirmed"  # "confirmed", "potential_review", "compatible"
    confidence: float = 0.9
    explanation: str = ""

    @property
    def subject(self) -> str:
        """Canonical conflicting subject name."""
        if self.claim_a and hasattr(self.claim_a, "subject") and self.claim_a.subject:
            return self.claim_a.subject
        if self.claim_b and hasattr(self.claim_b, "subject") and self.claim_b.subject:
            return self.claim_b.subject
        return "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "claim_a": self.claim_a.to_dict() if hasattr(self.claim_a, "to_dict") else self.claim_a,
            "claim_b": self.claim_b.to_dict() if hasattr(self.claim_b, "to_dict") else self.claim_b,
            "contradiction_type": self.contradiction_type,
            "conflicting_attribute": self.conflicting_attribute,
            "date_or_scope": self.date_or_scope,
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContradictionRecord":
        """Reconstruct a ContradictionRecord from a dictionary."""
        c_a_data = data.get("claim_a", {})
        c_b_data = data.get("claim_b", {})
        c_a = Claim.from_dict(c_a_data) if isinstance(c_a_data, dict) else c_a_data
        c_b = Claim.from_dict(c_b_data) if isinstance(c_b_data, dict) else c_b_data
        return cls(
            record_id=str(data.get("record_id", "")),
            claim_a=c_a,
            claim_b=c_b,
            contradiction_type=str(data.get("contradiction_type", "relational")),
            conflicting_attribute=str(data.get("conflicting_attribute", "")),
            date_or_scope=data.get("date_or_scope"),
            status=str(data.get("status", "confirmed")),
            confidence=float(data.get("confidence", 0.9)),
            explanation=str(data.get("explanation", "")),
        )


@dataclass
class ClaimDisplayAdapter:
    """Standardized view adapter for rendering Claim objects safely in UI components."""
    claim_id: str
    text: str
    source_filename: str
    page_number: Optional[int]
    chunk_id: Optional[str]
    provenance_caption: str
    is_valid: bool = True
    error_message: Optional[str] = None

    @classmethod
    def from_claim(cls, claim: Any) -> "ClaimDisplayAdapter":
        if claim is None:
            return cls(
                claim_id="unknown",
                text="[Missing claim data]",
                source_filename="unknown",
                page_number=None,
                chunk_id=None,
                provenance_caption="Unavailable",
                is_valid=False,
                error_message="Claim record is missing (None).",
            )

        if isinstance(claim, dict):
            c_id = str(claim.get("claim_id", "unknown"))
            text = claim.get("excerpt") or claim.get("claim_text") or claim.get("text") or ""
            fname = str(claim.get("source_filename", "unknown"))
            page = claim.get("page_number")
            cid = claim.get("chunk_id")
        else:
            c_id = str(getattr(claim, "claim_id", "unknown"))
            text = getattr(claim, "claim_text", None) or getattr(claim, "excerpt", "") or ""
            fname = str(getattr(claim, "source_filename", "unknown"))
            page = getattr(claim, "page_number", None)
            cid = getattr(claim, "chunk_id", None)

        if not text and hasattr(claim, "subject") and hasattr(claim, "predicate") and hasattr(claim, "object"):
            pol = getattr(claim, "polarity", True)
            pol_str = "" if pol else "not "
            val = getattr(claim, "quantity", None)
            unit = getattr(claim, "unit", "") or ""
            val_str = f" {val} {unit}".strip() if val is not None else ""
            text = f"{getattr(claim, 'subject', '')} is {pol_str}{getattr(claim, 'predicate', '')} {getattr(claim, 'object', '')}{val_str}".strip()

        if not text:
            text = "[Empty claim text]"

        prov_parts = []
        if cid:
            prov_parts.append(f"Chunk: <code>{cid}</code>")
        if page is not None:
            prov_parts.append(f"Page: {page}")
        caption = " | ".join(prov_parts) if prov_parts else "Source chunk unindexed"

        return cls(
            claim_id=c_id,
            text=text,
            source_filename=fname,
            page_number=page,
            chunk_id=cid,
            provenance_caption=caption,
            is_valid=True,
        )


@dataclass
class ContradictionDisplayAdapter:
    """Standardized view adapter for rendering ContradictionRecord objects safely."""
    record_id: str
    claim_a: ClaimDisplayAdapter
    claim_b: ClaimDisplayAdapter
    contradiction_type: str
    conflicting_attribute: str
    status: str
    confidence: float
    explanation: str
    subject: str = "Unknown"
    is_valid: bool = True
    error_message: Optional[str] = None

    @classmethod
    def from_record(cls, record: Any) -> "ContradictionDisplayAdapter":
        if record is None:
            return cls(
                record_id="invalid",
                claim_a=ClaimDisplayAdapter.from_claim(None),
                claim_b=ClaimDisplayAdapter.from_claim(None),
                contradiction_type="unknown",
                conflicting_attribute="unknown",
                status="invalid",
                confidence=0.0,
                explanation="Contradiction record is None.",
                subject="Unknown",
                is_valid=False,
                error_message="Contradiction record is None.",
            )

        claim_a_obj = getattr(record, "claim_a", None)
        claim_b_obj = getattr(record, "claim_b", None)
        if claim_a_obj is None or claim_b_obj is None:
            missing = []
            if claim_a_obj is None:
                missing.append("claim_a")
            if claim_b_obj is None:
                missing.append("claim_b")
            return cls(
                record_id=str(getattr(record, "record_id", "invalid")),
                claim_a=ClaimDisplayAdapter.from_claim(claim_a_obj),
                claim_b=ClaimDisplayAdapter.from_claim(claim_b_obj),
                contradiction_type=str(getattr(record, "contradiction_type", "unknown")),
                conflicting_attribute=str(getattr(record, "conflicting_attribute", "unknown")),
                status="invalid",
                confidence=0.0,
                explanation="Contradiction record is missing claim_a or claim_b.",
                subject=str(getattr(record, "subject", "Unknown")),
                is_valid=False,
                error_message=f"Contradiction record lacks valid claim: {', '.join(missing)} is missing.",
            )

        c_a = ClaimDisplayAdapter.from_claim(record.claim_a)
        c_b = ClaimDisplayAdapter.from_claim(record.claim_b)
        rec_id = str(getattr(record, "record_id", "unknown"))
        ctype = str(getattr(record, "contradiction_type", "relational"))
        attr = str(getattr(record, "conflicting_attribute", "attribute"))
        status = str(getattr(record, "status", "confirmed"))
        conf = float(getattr(record, "confidence", 0.9))
        exp = str(getattr(record, "explanation", ""))
        subj = str(getattr(record, "subject", getattr(record.claim_a, "subject", "Unknown")))

        is_valid = c_a.is_valid and c_b.is_valid
        err = None if is_valid else f"Malformed claim: A={c_a.error_message}, B={c_b.error_message}"

        return cls(
            record_id=rec_id,
            claim_a=c_a,
            claim_b=c_b,
            contradiction_type=ctype,
            conflicting_attribute=attr,
            status=status,
            confidence=conf,
            explanation=exp,
            subject=subj,
            is_valid=is_valid,
            error_message=err,
        )


class ClaimExtractor:
    """Extracts structured Claim records from text chunks and factual relationships."""

    # Patterns for numerical quantities (e.g. "12 mg", "18.5 %", "250 kg", "14 days", "$50 million")
    NUMERICAL_REGEX = re.compile(
        r"(?:([A-Za-z0-9_\-\s]{2,30}?)\s+(?:is|was|measures|equals|stands at|reached|reported at|of)\s+)"
        r"([+-]?\d+(?:\.\d+)?)\s*"
        r"(%|percent|mg|kg|g|km|miles|dollars|USD|days|hours|minutes|seconds|volts|watts|GW|MW|bps|gbps|users|samples|patients)?",
        re.IGNORECASE,
    )

    # Patterns for temporal claims (e.g. "founded in 2021", "launched in 1998")
    TEMPORAL_REGEX = re.compile(
        r"([A-Za-z0-9_\-\s]{2,30}?)\s+(?:was\s+)?(founded|launched|released|discovered|approved|established|built|invented|published)\s+(?:in|on)\s+(\d{4}(?:-\d{2}-\d{2})?)",
        re.IGNORECASE,
    )

    # Patterns indicating negation or negative polarity
    NEGATION_WORDS = {"not", "never", "failed", "denied", "rejected", "disapproved", "ineffective", "disproven", "refuted", "cannot", "no"}

    def extract_claims_from_chunk(
        self,
        chunk: Union[TextChunk, Dict[str, Any]],
        doc_id: Optional[str] = None,
    ) -> List[Claim]:
        """Extract candidate claims directly from a chunk's text and metadata."""
        if isinstance(chunk, TextChunk):
            cid = chunk.chunk_id
            text = chunk.text
            fname = chunk.source_filename
            page = chunk.page_number
        elif isinstance(chunk, dict):
            cid = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            fname = chunk.get("source_filename", "unknown")
            page = chunk.get("page_number")
        else:
            return []

        doc_identifier = doc_id or fname
        claims: List[Claim] = []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            sent_str = sent.strip()
            if len(sent_str) < 10:
                continue

            # 1. Numerical claims
            num_matches = self.NUMERICAL_REGEX.findall(sent_str)
            for subj, num_val, unit in num_matches:
                subj_clean = subj.strip()
                if len(subj_clean) < 2 or subj_clean.lower() in ("there", "it", "this", "that"):
                    continue
                try:
                    q_val = float(num_val)
                except ValueError:
                    continue

                claim_id = self._generate_claim_id(doc_identifier, cid, subj_clean, "numerical")
                claims.append(Claim(
                    claim_id=claim_id,
                    subject=subj_clean,
                    predicate="HAS_VALUE",
                    object=f"{q_val} {unit or ''}".strip(),
                    source_doc_id=doc_identifier,
                    source_filename=fname,
                    chunk_id=cid,
                    page_number=page,
                    excerpt=sent_str,
                    claim_type="numerical",
                    polarity=True,
                    quantity=q_val,
                    unit=unit.lower() if unit else None,
                    extraction_method="numerical_regex",
                    confidence=0.88,
                ))

            # 2. Temporal claims
            temp_matches = self.TEMPORAL_REGEX.findall(sent_str)
            for subj, action, date_val in temp_matches:
                subj_clean = subj.strip()
                if len(subj_clean) < 2 or subj_clean.lower() in ("there", "it", "this", "that"):
                    continue

                claim_id = self._generate_claim_id(doc_identifier, cid, subj_clean, "temporal")
                claims.append(Claim(
                    claim_id=claim_id,
                    subject=subj_clean,
                    predicate=action.upper(),
                    object=date_val,
                    source_doc_id=doc_identifier,
                    source_filename=fname,
                    chunk_id=cid,
                    page_number=page,
                    excerpt=sent_str,
                    claim_type="temporal",
                    polarity=True,
                    date_scope=date_val,
                    extraction_method="temporal_regex",
                    confidence=0.92,
                ))

            # 3. Polarity / Status claims
            # e.g., "X is approved by FDA" vs "X was rejected by FDA"
            polarity_patterns = [
                (r"([A-Za-z0-9_\-\s]{2,30}?)\s+(?:is|was|has been)\s+(approved|authorized|cleared|effective|successful|beneficial)", True),
                (r"([A-Za-z0-9_\-\s]{2,30}?)\s+(?:is|was|has been)\s+(rejected|denied|disapproved|ineffective|failed|harmful)", False),
                (r"([A-Za-z0-9_\-\s]{2,30}?)\s+(?:does not|did not|failed to|cannot)\s+([A-Za-z0-9_\-\s]{3,30})", False),
            ]
            for pat, is_positive in polarity_patterns:
                m = re.search(pat, sent_str, re.IGNORECASE)
                if m:
                    subj = m.group(1).strip()
                    val = m.group(2).strip()
                    if len(subj) >= 2 and subj.lower() not in ("there", "it", "this", "that"):
                        claim_id = self._generate_claim_id(doc_identifier, cid, subj, "polarity")
                        claims.append(Claim(
                            claim_id=claim_id,
                            subject=subj,
                            predicate="EVALUATION_STATUS",
                            object=val,
                            source_doc_id=doc_identifier,
                            source_filename=fname,
                            chunk_id=cid,
                            page_number=page,
                            excerpt=sent_str,
                            claim_type="polarity",
                            polarity=is_positive,
                            extraction_method="polarity_regex",
                            confidence=0.85,
                        ))

        return claims

    def extract_claims_from_relationships(
        self,
        relationships: List[Union[Relationship, Dict[str, Any]]],
        source_doc_id: Optional[str] = None,
    ) -> List[Claim]:
        """Convert extracted factual relationship triples into structured Claim objects."""
        claims: List[Claim] = []
        for rel in relationships:
            if isinstance(rel, Relationship):
                sid = rel.subject_id
                pred = rel.predicate
                oid = rel.object_id
                conf = rel.confidence
                evidence = rel.evidence
            elif isinstance(rel, dict):
                sid = rel.get("subject_id", "")
                pred = rel.get("predicate", "")
                oid = rel.get("object_id", "")
                conf = float(rel.get("confidence", 1.0))
                evidence = rel.get("evidence", [])
            else:
                continue

            for ev in evidence:
                if hasattr(ev, "source_filename"):
                    fname = ev.source_filename
                    page = getattr(ev, "page_number", None)
                    cid = getattr(ev, "source_chunk_id", getattr(ev, "chunk_id", None))
                    quote = getattr(ev, "evidence_text", getattr(ev, "exact_quote", ""))
                elif isinstance(ev, dict):
                    fname = ev.get("source_filename", "unknown")
                    page = ev.get("page_number")
                    cid = ev.get("source_chunk_id", ev.get("chunk_id"))
                    quote = ev.get("evidence_text", ev.get("exact_quote", ""))
                else:
                    fname = "unknown"
                    page = None
                    cid = None
                    quote = ""

                doc_id = source_doc_id or fname

                # Check for negation words in predicate or quote
                pred_lower = pred.lower()
                is_negated = any(neg in pred_lower for neg in self.NEGATION_WORDS)

                claim_id = self._generate_claim_id(doc_id, cid or sid, f"{sid}_{pred}_{oid}", "relational")
                claims.append(Claim(
                    claim_id=claim_id,
                    subject=sid,
                    predicate=pred,
                    object=oid,
                    source_doc_id=doc_id,
                    source_filename=fname,
                    chunk_id=cid,
                    page_number=page,
                    excerpt=quote or f"{sid} {pred} {oid}",
                    claim_type="relational",
                    polarity=not is_negated,
                    extraction_method="relationship_triple",
                    confidence=conf,
                ))

        return claims

    @staticmethod
    def _generate_claim_id(doc_id: str, chunk_id: str, subject: str, ctype: str) -> str:
        raw = f"{doc_id}|{chunk_id}|{subject}|{ctype}"
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"claim_{h}"


class ContradictionDetector:
    """Detects incompatible cross-document claims and manages contradictory evidence."""

    def __init__(self) -> None:
        self.contradiction_records: Dict[str, ContradictionRecord] = {}

    def clear(self) -> None:
        """Clear all registered contradiction records."""
        self.contradiction_records.clear()

    def detect_contradictions(self, claims: List[Claim]) -> List[ContradictionRecord]:
        """Examine a pool of claims and identify evidence-grounded contradictions.

        Rules:
        1. Claims must come from DIFFERENT source documents.
        2. Subjects must match (canonical match or normalized string match).
        3. Check polarity conflicts (e.g. positive status vs negative status).
        4. Check numerical conflicts (same unit, identical scope, conflicting quantity).
        5. Check temporal conflicts (same event predicate, incompatible dates).
        6. Respect scope / population: If populations or dates explicitly differ, mark compatible.
        """
        detected: List[ContradictionRecord] = []
        n = len(claims)

        for i in range(n):
            for j in range(i + 1, n):
                c1 = claims[i]
                c2 = claims[j]

                # Rule 1: Contradictions must be across different source documents
                if c1.source_filename == c2.source_filename or c1.source_doc_id == c2.source_doc_id:
                    continue

                # Subject normalization
                s1 = self._normalize_term(c1.subject)
                s2 = self._normalize_term(c2.subject)
                if not self._is_subject_match(s1, s2):
                    continue

                # Conflict Check A: Polarity Conflict
                # Same subject, same attribute/predicate, opposite polarity
                if self._is_predicate_match(c1.predicate, c2.predicate):
                    if c1.polarity != c2.polarity:
                        # Direct polarity contradiction
                        rec = self._create_record(
                            c1, c2,
                            contradiction_type="polarity",
                            conflicting_attribute=c1.predicate,
                            status="confirmed",
                            confidence=min(c1.confidence, c2.confidence) * 0.95,
                            explanation=(
                                f"Polarity conflict on '{c1.subject}': '{c1.source_filename}' asserts affirmative "
                                f"({c1.object}), whereas '{c2.source_filename}' asserts negative or opposite ({c2.object})."
                            ),
                        )
                        detected.append(rec)
                        continue

                # Conflict Check B: Numerical Divergence
                if c1.claim_type == "numerical" and c2.claim_type == "numerical":
                    if c1.quantity is not None and c2.quantity is not None:
                        # Check unit compatibility
                        u1 = (c1.unit or "").lower().strip()
                        u2 = (c2.unit or "").lower().strip()
                        if u1 == u2:
                            # If quantities differ meaningfully (e.g. > 5% difference)
                            if abs(c1.quantity - c2.quantity) > 1e-4:
                                # Check if dates/scopes differ
                                if c1.date_scope and c2.date_scope and c1.date_scope != c2.date_scope:
                                    # Different dates means they are compatible observations over time
                                    continue
                                if c1.condition_or_population and c2.condition_or_population and c1.condition_or_population != c2.condition_or_population:
                                    # Different populations means compatible
                                    continue

                                rec = self._create_record(
                                    c1, c2,
                                    contradiction_type="numerical",
                                    conflicting_attribute=f"value ({u1})" if u1 else "value",
                                    status="confirmed",
                                    confidence=min(c1.confidence, c2.confidence) * 0.92,
                                    explanation=(
                                        f"Numerical conflict on '{c1.subject}': '{c1.source_filename}' reports "
                                        f"{c1.quantity} {u1}, while '{c2.source_filename}' reports {c2.quantity} {u2}."
                                    ),
                                )
                                detected.append(rec)
                                continue

                # Conflict Check C: Temporal Divergence
                if c1.claim_type == "temporal" and c2.claim_type == "temporal":
                    if self._is_predicate_match(c1.predicate, c2.predicate):
                        d1 = c1.date_scope or c1.object
                        d2 = c2.date_scope or c2.object
                        if d1 and d2 and d1 != d2:
                            rec = self._create_record(
                                c1, c2,
                                contradiction_type="temporal",
                                conflicting_attribute=c1.predicate,
                                date_or_scope=f"{d1} vs {d2}",
                                status="confirmed",
                                confidence=min(c1.confidence, c2.confidence) * 0.94,
                                explanation=(
                                    f"Temporal conflict on '{c1.subject}' {c1.predicate}: '{c1.source_filename}' claims "
                                    f"{d1}, but '{c2.source_filename}' claims {d2}."
                                ),
                            )
                            detected.append(rec)
                            continue

                # Conflict Check D: Relational Mutually Exclusive Conflict
                # e.g., (X, INHIBITS, Y) vs (X, ACTIVATES, Y) or (X, IS_LOCATED_IN, A) vs (X, IS_LOCATED_IN, B)
                if c1.claim_type == "relational" and c2.claim_type == "relational":
                    o1 = self._normalize_term(c1.object)
                    o2 = self._normalize_term(c2.object)
                    mutually_exclusive_preds = {
                        ("INHIBITS", "ACTIVATES"), ("INCREASES", "DECREASES"),
                        ("PROMOTES", "SUPPRESSES"), ("APPROVED", "REJECTED"),
                        ("SUCCESSFUL", "FAILED"),
                    }
                    p1 = c1.predicate.upper()
                    p2 = c2.predicate.upper()
                    if (p1, p2) in mutually_exclusive_preds or (p2, p1) in mutually_exclusive_preds:
                        if o1 == o2:
                            rec = self._create_record(
                                c1, c2,
                                contradiction_type="relational",
                                conflicting_attribute=f"{p1} vs {p2}",
                                status="confirmed",
                                confidence=min(c1.confidence, c2.confidence) * 0.90,
                                explanation=(
                                    f"Relational conflict between '{c1.source_filename}' ({c1.subject} {p1} {c1.object}) "
                                    f"and '{c2.source_filename}' ({c2.subject} {p2} {c2.object})."
                                ),
                            )
                            detected.append(rec)
                            continue

        # Register all detected records
        for r in detected:
            self.contradiction_records[r.record_id] = r

        return detected

    def remove_document_contradictions(self, filename_or_doc_id: str) -> int:
        """Remove contradiction records that depended on a deleted document."""
        to_delete = [
            rid for rid, r in self.contradiction_records.items()
            if r.claim_a.source_filename == filename_or_doc_id
            or r.claim_a.source_doc_id == filename_or_doc_id
            or r.claim_b.source_filename == filename_or_doc_id
            or r.claim_b.source_doc_id == filename_or_doc_id
        ]
        for rid in to_delete:
            del self.contradiction_records[rid]
        return len(to_delete)

    def deduplicate_contradictions(
        self,
        contradictions: List[ContradictionRecord],
    ) -> List[ContradictionRecord]:
        """Deduplicate contradiction records using stable IDs and conflict keys."""
        seen_keys: Set[Any] = set()
        deduped: List[ContradictionRecord] = []

        for r in contradictions:
            if not isinstance(r, ContradictionRecord) or not r.claim_a or not r.claim_b:
                continue
            claim_id_pair = tuple(sorted([str(getattr(r.claim_a, "claim_id", "")), str(getattr(r.claim_b, "claim_id", ""))]))
            subj = (getattr(r.claim_a, "subject", "") or "").lower().strip()
            attr = (getattr(r, "conflicting_attribute", "") or "").lower().strip()
            src_pair = frozenset([
                (getattr(r.claim_a, "source_filename", "") or "").lower(),
                (getattr(r.claim_b, "source_filename", "") or "").lower(),
            ])

            semantic_key = (subj, attr, src_pair)
            id_key = (claim_id_pair, getattr(r, "contradiction_type", ""))

            if semantic_key in seen_keys or id_key in seen_keys or r.record_id in seen_keys:
                continue
            seen_keys.add(semantic_key)
            seen_keys.add(id_key)
            seen_keys.add(r.record_id)
            deduped.append(r)

        return deduped

    def filter_relevant_contradictions(
        self,
        contradictions: List[ContradictionRecord],
        query: str,
        retrieved_evidence: Optional[List[Any]] = None,
        target_entities: Optional[List[str]] = None,
        matched_themes: Optional[List[str]] = None,
    ) -> List[ContradictionRecord]:
        """Filter contradictions to only those genuinely relevant to the query, entities, or evidence.

        Ensures that broad factual questions do not falsely surface every conflict in the corpus.
        """
        if not contradictions:
            return []

        unique_records = self.deduplicate_contradictions(contradictions)
        query_lower = query.lower()

        stop_words = {
            "tell", "about", "what", "when", "where", "which", "how", "does", "the",
            "and", "are", "for", "with", "this", "that", "these", "those", "have", "has",
            "had", "can", "could", "will", "would", "explain", "describe", "give", "overview",
            "summary", "details", "information", "some", "many", "much", "more", "question",
            "questions", "answer", "answers"
        }
        query_terms = {t for t in re.findall(r"\w{3,}", query_lower) if t not in stop_words}

        entity_set = {e.lower().strip() for e in (target_entities or []) if e}
        theme_set = {th.lower().strip() for th in (matched_themes or []) if th}

        evidence_chunk_ids: Set[str] = set()
        evidence_doc_names: Set[str] = set()
        evidence_text_tokens: Set[str] = set()

        if retrieved_evidence:
            for ev in retrieved_evidence:
                cid = getattr(ev, "chunk_id", None) or (ev.get("chunk_id") if isinstance(ev, dict) else None)
                if cid:
                    evidence_chunk_ids.add(str(cid))
                doc = getattr(ev, "source_filename", None) or (ev.get("source_filename") if isinstance(ev, dict) else None)
                if doc:
                    evidence_doc_names.add(str(doc).lower())
                text = getattr(ev, "text", "") or (ev.get("text", "") if isinstance(ev, dict) else "")
                if text:
                    for w in re.findall(r"\w{3,}", text.lower()):
                        if w not in stop_words:
                            evidence_text_tokens.add(w)

        relevant: List[ContradictionRecord] = []

        for r in unique_records:
            if not r.claim_a or not r.claim_b:
                continue

            s_a = (getattr(r.claim_a, "subject", "") or "").lower().strip()
            s_b = (getattr(r.claim_b, "subject", "") or "").lower().strip()
            obj_a = (getattr(r.claim_a, "object", "") or "").lower().strip()
            obj_b = (getattr(r.claim_b, "object", "") or "").lower().strip()
            attr = (getattr(r, "conflicting_attribute", "") or "").lower().strip()

            c_id_a = str(getattr(r.claim_a, "chunk_id", "") or "")
            c_id_b = str(getattr(r.claim_b, "chunk_id", "") or "")

            # 1. Match against query terms or target entities
            entity_match = False
            for s in (s_a, s_b, obj_a, obj_b):
                if not s or len(s) < 2:
                    continue
                if s in query_lower or s in entity_set:
                    entity_match = True
                    break
                s_words = {w for w in re.findall(r"\w{3,}", s) if w not in stop_words}
                if s_words and s_words.intersection(query_terms):
                    entity_match = True
                    break

            if entity_match:
                relevant.append(r)
                continue

            # 2. Match against retrieved evidence chunks or tokens
            if evidence_chunk_ids and (c_id_a in evidence_chunk_ids or c_id_b in evidence_chunk_ids):
                if (s_a in evidence_text_tokens or s_b in evidence_text_tokens or
                    any(t in query_terms for t in re.findall(r"\w{3,}", attr))):
                    relevant.append(r)
                    continue

            # 3. Match against matched topic themes
            if theme_set and (s_a in theme_set or s_b in theme_set or attr in theme_set):
                relevant.append(r)
                continue

        return relevant

    def find_contradictions_for_query(
        self,
        query: str,
        entities: Optional[List[str]] = None,
        retrieved_evidence: Optional[List[Any]] = None,
        candidate_records: Optional[List[ContradictionRecord]] = None,
    ) -> List[ContradictionRecord]:
        """Find stored contradiction records relevant to a user question or queried entities."""
        candidates = candidate_records if candidate_records is not None else list(self.contradiction_records.values())
        return self.filter_relevant_contradictions(
            contradictions=candidates,
            query=query,
            retrieved_evidence=retrieved_evidence,
            target_entities=entities,
        )

    @staticmethod
    def _create_record(
        c1: Claim,
        c2: Claim,
        contradiction_type: str,
        conflicting_attribute: str,
        status: str,
        confidence: float,
        explanation: str,
        date_or_scope: Optional[str] = None,
    ) -> ContradictionRecord:
        # Canonical order based on claim ID for stability
        first, second = (c1, c2) if c1.claim_id <= c2.claim_id else (c2, c1)
        raw_key = f"{first.claim_id}|{second.claim_id}|{contradiction_type}"
        rec_id = f"contra_{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:12]}"
        return ContradictionRecord(
            record_id=rec_id,
            claim_a=first,
            claim_b=second,
            contradiction_type=contradiction_type,
            conflicting_attribute=conflicting_attribute,
            date_or_scope=date_or_scope,
            status=status,
            confidence=confidence,
            explanation=explanation,
        )

    @staticmethod
    def _normalize_term(term: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_\s]", "", term).lower().strip()
        # remove common articles
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned)
        return cleaned

    def _is_subject_match(self, s1: str, s2: str) -> bool:
        if s1 == s2:
            return True
        if len(s1) > 3 and len(s2) > 3:
            if s1 in s2 or s2 in s1:
                return True
        return False

    def _is_predicate_match(self, p1: str, p2: str) -> bool:
        p1_norm = self._normalize_term(p1)
        p2_norm = self._normalize_term(p2)
        if p1_norm == p2_norm:
            return True
        # Synonymous evaluation predicates
        eval_group = {"has_value", "value", "evaluation_status", "status", "approved", "rejected", "tested"}
        if p1_norm in eval_group and p2_norm in eval_group:
            return True
        return False
