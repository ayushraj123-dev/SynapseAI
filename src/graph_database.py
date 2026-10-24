"""Persistent Graph Database Integration for SynapseAI.

Implements genuine persistent graph storage using Neo4j alongside ChromaDB vector storage.
Provides:
- Neo4j driver connection management and connection health verification.
- Explicit persistent graph schema with uniqueness constraints and indexes.
- Repeat-safe document upserts (Document -> Chunk -> Entity -> Theme).
- Relationship storage with exact source quotation provenance and confidence.
- Clean document deletion and update without corrupting other documents.
- Parameterized Cypher queries for relationship path traversal and theme queries.
- In-memory NetworkX export for PyVis interactive canvas visualization.
- Transparent fallback reporting when Neo4j is not configured or unavailable.
"""

from enum import Enum
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# Check if neo4j driver is available
try:
    import neo4j
    from neo4j import GraphDatabase, Driver, Session
    from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError
    NEO4J_DRIVER_AVAILABLE = True
except ImportError:
    NEO4J_DRIVER_AVAILABLE = False
    GraphDatabase = None
    Driver = None
    Session = None


class GraphConnectionStatus(str, Enum):
    """Enumeration of graph database connection statuses."""
    CONNECTED = "connected"
    CONFIGURED_UNAVAILABLE = "configured_unavailable"
    NOT_CONFIGURED = "not_configured"
    LOCAL_FALLBACK = "local_fallback"


class Neo4jGraphService:
    """Service managing persistent graph operations in Neo4j with safe parameterized Cypher."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        auto_connect: bool = True,
    ) -> None:
        """Initialize Neo4j connection parameters.

        Reads from environment variables if not provided.

        Args:
            uri: Neo4j Bolt/Neo4j URI (e.g. 'bolt://localhost:7687' or 'neo4j+s://...').
            username: Username for database authentication (e.g. 'neo4j').
            password: Password for database authentication.
            database: Target database name (default: 'neo4j').
            auto_connect: If True, optionally verifies connectivity lazily or on demand.
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.auto_connect = auto_connect

        self._driver: Optional[Any] = None
        self._cached_status: Optional[Tuple[GraphConnectionStatus, str]] = None
        self.connection_error: str = ""

    @property
    def driver(self) -> Optional[Any]:
        return self._driver

    @driver.setter
    def driver(self, d: Optional[Any]) -> None:
        self._driver = d
        self._cached_status = None

    def close(self) -> None:
        """Close driver connection pools gracefully."""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception as e:
                logger.warning(f"Error closing Neo4j driver: {e}")
            finally:
                self._driver = None
                self._cached_status = None

    def get_driver(self) -> Optional[Any]:
        """Obtain or initialize the active Neo4j driver."""
        if not NEO4J_DRIVER_AVAILABLE:
            return None

        if self._driver is not None:
            return self._driver

        if not self.uri or not self.password:
            return None

        try:
            auth = (self.username, self.password)
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=auth,
                max_connection_lifetime=3600,
                connection_timeout=1.0,
            )
        except Exception as e:
            logger.warning(f"Failed to create Neo4j driver: {e}")
            self._driver = None

        return self._driver

    def check_connection(self) -> Tuple[GraphConnectionStatus, str]:
        """Verify database connectivity and return status and descriptive message.

        Returns:
            Tuple of (GraphConnectionStatus, detail_message).
        """
        if not NEO4J_DRIVER_AVAILABLE:
            msg = "Neo4j Python driver is not installed. Running in local NetworkX mode."
            self.connection_error = msg
            self._cached_status = (GraphConnectionStatus.LOCAL_FALLBACK, msg)
            return self._cached_status

        if not self.uri or not self.password:
            msg = (
                "Neo4j environment variables (NEO4J_URI, NEO4J_PASSWORD) are not set. "
                "Operating in Local Fallback mode using NetworkX."
            )
            self.connection_error = msg
            self._cached_status = (GraphConnectionStatus.NOT_CONFIGURED, msg)
            return self._cached_status

        try:
            driver = self.get_driver()
            if driver is None:
                msg = f"Unable to instantiate Neo4j driver for URI: {self.uri}"
                self.connection_error = msg
                self._cached_status = (GraphConnectionStatus.CONFIGURED_UNAVAILABLE, msg)
                return self._cached_status

            # Perform a lightweight connectivity check
            if hasattr(driver, "verify_connectivity"):
                driver.verify_connectivity()
            with driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS ping")
                record = result.single() if hasattr(result, "single") else None
                if record and record["ping"] == 1:
                    status = (
                        GraphConnectionStatus.CONNECTED,
                        f"Connected to Neo4j instance at {self.uri} (Database: {self.database})",
                    )
                    self._cached_status = status
                    self.connection_error = ""
                    return status

            msg = "Neo4j connection test did not return expected response."
            self.connection_error = msg
            self._cached_status = (GraphConnectionStatus.CONFIGURED_UNAVAILABLE, msg)
            return self._cached_status
        except Exception as e:
            msg = f"Could not connect to Neo4j at {self.uri}: {str(e)}"
            self.connection_error = msg
            self._cached_status = (GraphConnectionStatus.CONFIGURED_UNAVAILABLE, msg)
            return self._cached_status

    @property
    def is_connected(self) -> bool:
        """Check whether Neo4j is actively reachable."""
        if self._driver is None and not self.auto_connect:
            return False
        if self._cached_status is not None:
            return self._cached_status[0] == GraphConnectionStatus.CONNECTED
        status, _ = self.check_connection()
        return status == GraphConnectionStatus.CONNECTED

    def ensure_schema(self) -> bool:
        """Create schema uniqueness constraints and indexes using parameterized Cypher.

        Returns:
            True if schema was established, False otherwise.
        """
        if not self.is_connected:
            return False

        constraints = [
            "CREATE CONSTRAINT doc_filename_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.filename IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT theme_id_unique IF NOT EXISTS FOR (t:Theme) REQUIRE t.theme_id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX chunk_source_idx IF NOT EXISTS FOR (c:Chunk) ON (c.source_filename)",
            "CREATE INDEX theme_name_idx IF NOT EXISTS FOR (t:Theme) ON (t.name)",
        ]

        driver = self.get_driver()
        try:
            with driver.session(database=self.database) as session:
                for query in constraints + indexes:
                    try:
                        session.run(query)
                    except Exception as qe:
                        logger.debug(f"Schema statement '{query}' note: {qe}")
            return True
        except Exception as e:
            logger.error(f"Error creating Neo4j schema: {e}")
            return False

    def upsert_document_graph(
        self,
        filename: str,
        total_pages_or_extraction: Any = 1,
        chunks: Optional[List[Any]] = None,
        entities: Optional[Dict[str, Any]] = None,
        relationships: Optional[Dict[str, Any]] = None,
        themes: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Repeat-safe upsert of a document, its chunks, entities, relationships, and themes into Neo4j.

        Args:
            filename: Source document filename.
            total_pages_or_extraction: Total pages or ExtractionResult instance.
            chunks: List of TextChunk objects or dicts.
            entities: Dictionary of Entity objects or dicts.
            relationships: Dictionary of Relationship objects or dicts.
            themes: List of TopicTheme objects or dicts.

        Returns:
            Dictionary with counts of nodes and relationships created/updated.
        """
        if not self.is_connected:
            return {"status": 0, "message": "Neo4j not connected"}

        # Unpack ExtractionResult if passed as 2nd argument
        if hasattr(total_pages_or_extraction, "entities"):
            ext = total_pages_or_extraction
            entities = getattr(ext, "entities", {})
            relationships = getattr(ext, "relationships", {})
            themes = getattr(ext, "themes", [])
            total_pages = kwargs.get("total_pages", 1)
        else:
            total_pages = int(total_pages_or_extraction) if isinstance(total_pages_or_extraction, (int, float)) else 1
            chunks = chunks or []
            entities = entities or {}
            relationships = relationships or {}
            themes = themes or []

        chunks = chunks or []
        entities = entities or {}
        relationships = relationships or {}
        themes = themes or []

        self.ensure_schema()
        driver = self.get_driver()
        now_ts = int(time.time())

        counts = {
            "document_filename": filename,
            "documents_upserted": 0,
            "chunks_upserted": 0,
            "entities_upserted": 0,
            "relationships_upserted": 0,
            "themes_upserted": 0,
        }

        def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        with driver.session(database=self.database) as session:
            # 0. Clean prior document records for repeatability
            cleanup_cypher = """
            MATCH (d:Document {filename: $filename})
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
            DETACH DELETE c, d
            """
            session.run(cleanup_cypher, filename=filename)

            # 1. Upsert Document Node
            doc_cypher = """
            MERGE (d:Document {filename: $filename})
            SET d.total_pages = $total_pages,
                d.total_chunks = $total_chunks,
                d.updated_at = $timestamp
            RETURN d
            """
            session.run(
                doc_cypher,
                filename=filename,
                total_pages=total_pages,
                total_chunks=len(chunks),
                timestamp=now_ts,
            )
            counts["documents_upserted"] += 1

            # 2. Upsert Chunks and Link to Document
            chunk_cypher = """
            UNWIND $batch AS item
            MERGE (c:Chunk {chunk_id: item.chunk_id})
            SET c.source_filename = item.source_filename,
                c.page_number = item.page_number,
                c.chunk_index = item.chunk_index,
                c.char_count = item.char_count,
                c.text = item.text
            WITH c, item
            MATCH (d:Document {filename: item.source_filename})
            MERGE (d)-[:CONTAINS]->(c)
            """
            chunk_batch = []
            for c in chunks:
                chunk_batch.append({
                    "chunk_id": _get_attr(c, "chunk_id", None),
                    "source_filename": _get_attr(c, "source_filename", filename) or filename,
                    "page_number": _get_attr(c, "page_number", None),
                    "chunk_index": _get_attr(c, "chunk_index", 0),
                    "char_count": _get_attr(c, "char_count", 0),
                    "text": _get_attr(c, "text", ""),
                })
            if chunk_batch:
                session.run(chunk_cypher, batch=chunk_batch)
                counts["chunks_upserted"] = len(chunk_batch)

            # 3. Upsert Entities & Mention Links from Chunks
            ent_cypher = """
            UNWIND $batch AS item
            MERGE (e:Entity {entity_id: item.entity_id})
            SET e.name = item.name,
                e.entity_type = item.entity_type,
                e.description = item.description,
                e.aliases = item.aliases,
                e.mention_count = item.mention_count
            """
            ent_batch = []
            mention_batch = []
            for eid, ent in entities.items():
                name = _get_attr(ent, "name", eid)
                etype = _get_attr(ent, "entity_type", "CONCEPT")
                desc = _get_attr(ent, "description", "")
                aliases = _get_attr(ent, "aliases", [])
                mentions = _get_attr(ent, "mention_count", 1)

                ent_batch.append({
                    "entity_id": eid,
                    "name": name,
                    "entity_type": etype,
                    "description": desc,
                    "aliases": aliases,
                    "mention_count": mentions,
                })

                refs = _get_attr(ent, "source_references", []) or []
                if not refs and chunks:
                    for c in chunks:
                        c_text = _get_attr(c, "text", "")
                        c_id = _get_attr(c, "chunk_id", None)
                        if name.lower() in c_text.lower() and c_id:
                            mention_batch.append({
                                "chunk_id": c_id,
                                "entity_id": eid,
                                "text_snippet": f"Mention of {name}",
                            })
                else:
                    for ref in refs:
                        chk = _get_attr(ref, "chunk_id", None)
                        snip = _get_attr(ref, "text_snippet", "")
                        if chk:
                            mention_batch.append({
                                "chunk_id": chk,
                                "entity_id": eid,
                                "text_snippet": snip,
                            })

            if ent_batch:
                session.run(ent_cypher, batch=ent_batch)
                counts["entities_upserted"] = len(ent_batch)

            if mention_batch:
                mention_cypher = """
                UNWIND $batch AS item
                MATCH (c:Chunk {chunk_id: item.chunk_id})
                MERGE (e:Entity {entity_id: item.entity_id})
                MERGE (c)-[m:MENTIONS]->(e)
                SET m.text_snippet = item.text_snippet
                """
                session.run(mention_cypher, batch=mention_batch)

            # 4. Upsert Relationships (Factual Triples with Provenance)
            rel_cypher = """
            UNWIND $batch AS item
            MATCH (s:Entity {entity_id: item.subject_id})
            MATCH (o:Entity {entity_id: item.object_id})
            MERGE (s)-[r:RELATED_TO {relationship_id: item.relationship_id}]->(o)
            SET r.predicate = item.predicate,
                r.confidence = item.confidence,
                r.evidence_text = item.evidence_text,
                r.source_filename = item.source_filename,
                r.page_number = item.page_number,
                r.source_chunk_id = item.source_chunk_id
            """
            rel_batch = []
            for rid, rel in relationships.items():
                subj = _get_attr(rel, "subject_id", None)
                obj = _get_attr(rel, "object_id", None)
                pred = _get_attr(rel, "predicate", "RELATES_TO")
                conf = _get_attr(rel, "confidence", 1.0)
                ev_list = _get_attr(rel, "evidence", []) or []

                first_ev = ev_list[0] if ev_list else {}
                ev_text = _get_attr(first_ev, "evidence_text", "")
                src_fn = _get_attr(first_ev, "source_filename", filename) or filename
                pg = _get_attr(first_ev, "page_number", None)
                chk_id = _get_attr(first_ev, "source_chunk_id", None)

                rel_batch.append({
                    "relationship_id": rid,
                    "subject_id": subj,
                    "object_id": obj,
                    "predicate": pred,
                    "confidence": float(conf),
                    "evidence_text": ev_text,
                    "source_filename": src_fn,
                    "page_number": pg,
                    "source_chunk_id": chk_id,
                })

            if rel_batch:
                session.run(rel_cypher, batch=rel_batch)
                counts["relationships_upserted"] = len(rel_batch)

            # 5. Upsert Themes (Topic Communities)
            theme_cypher = """
            UNWIND $batch AS item
            MERGE (t:Theme {theme_id: item.theme_id})
            SET t.name = item.name,
                t.description = item.description,
                t.confidence = item.confidence,
                t.document_filenames = item.document_filenames
            """
            theme_batch = []
            theme_entity_batch = []
            theme_doc_batch = []

            for th in themes:
                tid = _get_attr(th, "theme_id", None)
                tname = _get_attr(th, "name", None)
                tdesc = _get_attr(th, "description", "")
                tconf = _get_attr(th, "confidence", 1.0)
                doc_fns = _get_attr(th, "document_filenames", [filename]) or [filename]
                if filename not in doc_fns:
                    doc_fns = list(doc_fns) + [filename]

                theme_batch.append({
                    "theme_id": tid,
                    "name": tname,
                    "description": tdesc,
                    "confidence": float(tconf),
                    "document_filenames": doc_fns,
                })

                eids = _get_attr(th, "entity_ids", []) or []
                for e in eids:
                    theme_entity_batch.append({"theme_id": tid, "entity_id": e})

                for df in doc_fns:
                    theme_doc_batch.append({"theme_id": tid, "filename": df})

            if theme_batch:
                session.run(theme_cypher, batch=theme_batch)
                counts["themes_upserted"] = len(theme_batch)

            if theme_entity_batch:
                te_cypher = """
                UNWIND $batch AS item
                MATCH (e:Entity {entity_id: item.entity_id})
                MERGE (t:Theme {theme_id: item.theme_id})
                MERGE (e)-[:BELONGS_TO]->(t)
                """
                session.run(te_cypher, batch=theme_entity_batch)

            if theme_doc_batch:
                td_cypher = """
                UNWIND $batch AS item
                MATCH (d:Document {filename: item.filename})
                MATCH (t:Theme {theme_id: item.theme_id})
                MERGE (t)-[:REPRESENTS_DOCUMENT]->(d)
                """
                session.run(td_cypher, batch=theme_doc_batch)

        return counts

    def delete_document_graph(self, filename: str) -> bool:
        """Delete all graph records specific to a document without corrupting other documents.

        Removes:
        - Document node and its Chunks.
        - Relational edges whose evidence came solely from this document.
        - Theme-document links.

        Args:
            filename: Target document filename.

        Returns:
            True if successfully executed, False otherwise.
        """
        if not self.is_connected:
            return False

        driver = self.get_driver()
        try:
            with driver.session(database=self.database) as session:
                # Delete Document node and its Chunks
                cypher = """
                MATCH (d:Document {filename: $filename})
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                DETACH DELETE c, d
                """
                session.run(cypher, {"filename": filename})

                # Clean up relationships whose evidence belonged to this file
                rel_cypher = """
                MATCH ()-[r:RELATED_TO {source_filename: $filename}]->()
                DELETE r
                """
                session.run(rel_cypher, {"filename": filename})

                return True
        except Exception as e:
            logger.warning(f"Error deleting document graph for {filename}: {e}")
            return False

    def query_relationships(
        self,
        entity_names: List[str],
        max_depth: int = 3,
    ) -> List[Dict[str, Any]]:
        """Query direct and multi-hop paths between entities using Neo4j Cypher.

        Args:
            entity_names: List of entity names to explore connections between.
            max_depth: Maximum path hop length (bounded, max 3).

        Returns:
            List of path edge dictionaries with exact evidence quotations.
        """
        if not self.is_connected or not entity_names:
            return []

        driver = self.get_driver()
        bounded_depth = min(max(1, max_depth), 3)

        cypher = f"""
        MATCH (s:Entity)
        WHERE s.name IN $names OR s.entity_id IN $names
        MATCH (o:Entity)
        WHERE (o.name IN $names OR o.entity_id IN $names) AND s <> o
        MATCH path = shortestPath((s)-[:RELATED_TO*1..{bounded_depth}]-(o))
        RETURN [node IN nodes(path) | {{id: node.entity_id, name: node.name, type: node.entity_type}}] AS path_nodes,
               [rel IN relationships(path) | {{
                   id: rel.relationship_id,
                   predicate: rel.predicate,
                   confidence: rel.confidence,
                   evidence_text: rel.evidence_text,
                   source_filename: rel.source_filename,
                   page_number: rel.page_number,
                   chunk_id: rel.source_chunk_id
               }}] AS path_edges
        LIMIT 10
        """

        results = []
        try:
            with driver.session(database=self.database) as session:
                records = session.run(cypher, names=entity_names)
                for rec in records:
                    results.append({
                        "path_nodes": rec["path_nodes"],
                        "path_edges": rec["path_edges"],
                    })
        except Exception as e:
            logger.warning(f"Neo4j relationship query failed: {e}")

        return results

    def query_themes(self) -> List[Dict[str, Any]]:
        """Query all topic communities, their member entities, and supporting documents.

        Returns:
            List of theme summary dictionaries.
        """
        if not self.is_connected:
            return []

        cypher = """
        MATCH (t:Theme)
        OPTIONAL MATCH (e:Entity)-[:BELONGS_TO]->(t)
        OPTIONAL MATCH (t)-[:REPRESENTS_DOCUMENT]->(d:Document)
        RETURN t.theme_id AS theme_id,
               t.name AS name,
               t.description AS description,
               t.confidence AS confidence,
               t.document_filenames AS stored_docs,
               collect(DISTINCT e.name) AS member_entities,
               collect(DISTINCT d.filename) AS linked_documents
        ORDER BY size(member_entities) DESC
        """

        themes = []
        try:
            driver = self.get_driver()
            with driver.session(database=self.database) as session:
                records = session.run(cypher)
                for rec in records:
                    docs = rec["linked_documents"] or rec["stored_docs"] or []
                    themes.append({
                        "theme_id": rec["theme_id"],
                        "name": rec["name"],
                        "description": rec["description"],
                        "confidence": rec["confidence"],
                        "member_entities": rec["member_entities"],
                        "supporting_documents": docs,
                        "is_cross_document": len(docs) > 1,
                    })
        except Exception as e:
            logger.warning(f"Neo4j theme query failed: {e}")

        return themes

    def get_statistics(self) -> Dict[str, Any]:
        """Retrieve node and relationship statistics from Neo4j.

        Returns:
            Dictionary of graph metrics and label distributions.
        """
        if not self.is_connected:
            return {"status": "disconnected", "is_connected": False, "node_count": 0, "edge_count": 0}

        driver = self.get_driver()
        try:
            with driver.session(database=self.database) as session:
                n_res = session.run("MATCH (n) RETURN count(n) AS total_nodes").single()
                r_res = session.run("MATCH ()-[r]->() RETURN count(r) AS total_edges").single()

                # Breakdown by label
                labels_res = session.run("CALL db.labels() YIELD label RETURN label")
                label_counts = {}
                for lrec in labels_res:
                    lbl = lrec["label"]
                    cnt_res = session.run(f"MATCH (n:{lbl}) RETURN count(n) AS cnt").single()
                    label_counts[lbl] = cnt_res["cnt"] if cnt_res else 0

                return {
                    "status": "connected",
                    "is_connected": True,
                    "total_nodes": n_res["total_nodes"] if n_res else 0,
                    "total_edges": r_res["total_edges"] if r_res else 0,
                    "node_count": label_counts.get("Entity", 0),
                    "document_count": label_counts.get("Document", 0),
                    "chunk_count": label_counts.get("Chunk", 0),
                    "theme_count": label_counts.get("Theme", 0),
                    "labels_distribution": label_counts,
                }
        except Exception as e:
            logger.warning(f"Error fetching Neo4j statistics: {e}")
            return {"status": "error", "error": str(e), "node_count": 0, "edge_count": 0}

    def export_to_networkx(self) -> nx.DiGraph:
        """Read all entities, factual relationships, and theme memberships from Neo4j into NetworkX.

        Returns:
            NetworkX DiGraph populated from persistent Neo4j records.
        """
        g = nx.DiGraph()
        if not self.is_connected:
            return g

        def _rec_get(record: Any, *keys: str, default: Any = None) -> Any:
            for k in keys:
                try:
                    val = record[k]
                    if val is not None:
                        return val
                except Exception:
                    pass
            return default

        driver = self.get_driver()
        try:
            with driver.session(database=self.database) as session:
                # 1. Fetch Entity Nodes
                ent_res = session.run("""
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[:BELONGS_TO]->(t:Theme)
                RETURN e.entity_id AS id,
                       e.name AS name,
                       e.entity_type AS entity_type,
                       e.description AS description,
                       e.aliases AS aliases,
                       e.mention_count AS mention_count,
                       collect(DISTINCT t.name) AS themes
                """)
                for rec in ent_res:
                    eid = _rec_get(rec, "id", "entity_id")
                    if not eid:
                        continue
                    g.add_node(
                        eid,
                        entity_id=eid,
                        name=_rec_get(rec, "name", default=eid),
                        entity_type=_rec_get(rec, "type", "entity_type", default="CONCEPT"),
                        description=_rec_get(rec, "desc", "description", default=""),
                        aliases=_rec_get(rec, "aliases", default=[]),
                        mention_count=_rec_get(rec, "mentions", "mention_count", default=1),
                        themes=_rec_get(rec, "themes", default=[]),
                        source_references=[],
                    )

                # 2. Fetch Factual Relationships
                rel_res = session.run("""
                MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
                RETURN s.entity_id AS subj,
                       o.entity_id AS obj,
                       r.relationship_id AS rel_id,
                       r.predicate AS predicate,
                       r.confidence AS confidence,
                       r.evidence_text AS evidence_text,
                       r.source_filename AS source_filename,
                       r.page_number AS page_number,
                       r.source_chunk_id AS source_chunk_id
                """)
                for rec in rel_res:
                    subj = _rec_get(rec, "source_id", "subj", "subject_id")
                    obj = _rec_get(rec, "target_id", "obj", "object_id")
                    if not subj or not obj:
                        continue
                    if not g.has_node(subj):
                        g.add_node(subj, name=subj, entity_type="CONCEPT")
                    if not g.has_node(obj):
                        g.add_node(obj, name=obj, entity_type="CONCEPT")

                    ev_text = _rec_get(rec, "evidence", "evidence_text", default="")
                    src_file = _rec_get(rec, "source_file", "source_filename", default="unknown")
                    evidence = []
                    if ev_text:
                        evidence.append({
                            "source_filename": src_file,
                            "page_number": _rec_get(rec, "page_number"),
                            "source_chunk_id": _rec_get(rec, "source_chunk_id", "chunk_id", default=""),
                            "evidence_text": ev_text,
                            "confidence": _rec_get(rec, "confidence", default=1.0),
                        })

                    g.add_edge(
                        subj,
                        obj,
                        relationship_id=_rec_get(rec, "rel_id", "relationship_id", default=f"rel_{subj}_{obj}"),
                        predicate=_rec_get(rec, "predicate", default="RELATES_TO"),
                        confidence=_rec_get(rec, "confidence", default=1.0),
                        evidence=evidence,
                    )
        except Exception as e:
            logger.warning(f"Error exporting Neo4j graph to NetworkX: {e}")

        return g
