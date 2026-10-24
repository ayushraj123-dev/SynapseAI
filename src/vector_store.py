"""Vector Store and Semantic Retrieval Module for SynapseAI.

Task 4 Implementation:
- Persistent vector collection using ChromaDB (default local directory: ./chroma_db).
- Repeat-safe indexing with stable chunk IDs: identical chunks update safely without duplication.
- Document replacement and cleanup: purge stale chunks by source filename.
- Complete provenance metadata: chunk ID, source filename, page number, chunk index, char count.
- Dense similarity retrieval returning original passage text, similarity scores, and source citations.
- Pluggable embedding providers: default local ONNX MiniLM, with an offline deterministic fallback
  and support for custom embedding functions during unit testing.
"""

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from src.document_ingestion import TextChunk


@dataclass
class SearchResult:
    """Represents a scored, provenance-backed passage retrieved from the vector store."""
    chunk_id: str
    text: str
    source_filename: str
    page_number: Optional[int]
    chunk_index: int
    char_count: int
    distance: float
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "distance": round(self.distance, 4),
            "relevance_score": round(self.relevance_score, 4),
        }


class LocalDeterministicEmbeddingFunction(EmbeddingFunction):
    """Offline, deterministic embedding provider requiring zero downloads or API keys.

    Generates 256-dimensional unit-normalized pseudo-semantic vectors using term hashing.
    Used as a safe fallback when ONNX models cannot be loaded, and for ultra-fast,
    reproducible unit testing without internet access.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    @classmethod
    def name(cls) -> str:
        return "local_deterministic_embedding"

    def get_config(self) -> Dict[str, Any]:
        return {"dimension": self.dimension}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "LocalDeterministicEmbeddingFunction":
        return cls(dimension=config.get("dimension", 256))

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: List[List[float]] = []
        for doc in input:
            vec = [0.0] * self.dimension
            # Clean and tokenize words
            tokens = [t.strip(".,;:!?()[]\"'") for t in doc.lower().split()]
            tokens = [t for t in tokens if len(t) > 1]
            if not tokens:
                embeddings.append(vec)
                continue

            for token in tokens:
                # Stable positive term hash
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                vec[idx] += 1.0

            # Normalize to unit length for cosine space
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 1e-9:
                vec = [x / norm for x in vec]
            embeddings.append(vec)

        return embeddings


class VectorStoreService:
    """Service for managing persistent chunk embeddings and semantic similarity queries."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "synapse_knowledge_base",
        embedding_function: Optional[EmbeddingFunction] = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_provider_name = "default"

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Configure embedding function
        if embedding_function is not None:
            self.embedding_function = embedding_function
            self.embedding_provider_name = getattr(
                embedding_function, "__class__", type(embedding_function)
            ).__name__
        else:
            try:
                from chromadb.utils import embedding_functions
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
                self.embedding_provider_name = "ChromaDefault_all-MiniLM-L6-v2"
            except Exception:
                # Fallback to local deterministic embedder if ONNX runtime fails
                self.embedding_function = LocalDeterministicEmbeddingFunction()
                self.embedding_provider_name = "LocalDeterministicFallback"

        # Get or create the persistent collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(
        self,
        chunks: List[Union[TextChunk, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Index a batch of text chunks into the persistent vector database.

        Uses upsert semantics to ensure repeated indexing of identical chunks
        does not create duplicates, while updated chunks overwrite older records.

        Args:
            chunks: List of TextChunk objects or chunk dictionaries.

        Returns:
            Indexing summary report with counts of indexed, skipped, and total chunks.
        """
        if not chunks:
            return {
                "indexed_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "collection_total": self.collection.count(),
                "chunk_ids": [],
            }

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        skipped_count = 0
        failed_count = 0

        for chunk in chunks:
            if isinstance(chunk, TextChunk):
                cid = chunk.chunk_id
                text = chunk.text
                fname = chunk.source_filename
                page = chunk.page_number
                cidx = chunk.chunk_index
                chars = chunk.char_count
            elif isinstance(chunk, dict):
                cid = chunk.get("chunk_id", "")
                text = chunk.get("text", "")
                fname = chunk.get("source_filename", "unknown")
                page = chunk.get("page_number")
                cidx = chunk.get("chunk_index", 0)
                chars = chunk.get("char_count", len(text))
            else:
                skipped_count += 1
                continue

            # Validate text content
            if not text or not text.strip():
                skipped_count += 1
                continue

            if not cid:
                skipped_count += 1
                continue

            # ChromaDB metadata values must be int, float, str, or bool
            metadata_record: Dict[str, Any] = {
                "source_filename": str(fname),
                "page_number": int(page) if page is not None else -1,
                "chunk_index": int(cidx),
                "char_count": int(chars),
            }

            ids.append(cid)
            documents.append(text)
            metadatas.append(metadata_record)

        if not ids:
            return {
                "indexed_count": 0,
                "skipped_count": skipped_count,
                "failed_count": failed_count,
                "collection_total": self.collection.count(),
                "chunk_ids": [],
            }

        try:
            # Batch upsert into ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            indexed_count = len(ids)
        except Exception as e:
            failed_count = len(ids)
            indexed_count = 0

        return {
            "indexed_count": indexed_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "collection_total": self.collection.count(),
            "chunk_ids": ids if indexed_count > 0 else [],
        }

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_filename: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform semantic vector retrieval over indexed chunks.

        Args:
            query: User's question or search phrase.
            top_k: Number of most similar passages to return.
            filter_filename: Optional source filename to restrict search.

        Returns:
            List of SearchResult objects sorted by descending relevance.
        """
        if not query or not query.strip():
            return []

        if self.collection.count() == 0:
            return []

        where_clause = None
        if filter_filename and filter_filename != "ALL":
            where_clause = {"source_filename": filter_filename}

        # Cap top_k to total records in collection
        actual_k = min(top_k, self.collection.count())
        if actual_k <= 0:
            return []

        query_kwargs: Dict[str, Any] = {
            "query_texts": [query.strip()],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        try:
            query_res = self.collection.query(**query_kwargs)
        except Exception:
            return []

        results: List[SearchResult] = []
        ids = query_res.get("ids", [[]])[0]
        docs = query_res.get("documents", [[]])[0]
        metas = query_res.get("metadatas", [[]])[0]
        dists = query_res.get("distances", [[]])[0]

        for i, cid in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            dist = float(dists[i]) if i < len(dists) else 1.0

            # Cosine distance to similarity score in [0.0, 1.0]
            # ChromaDB cosine distance ranges from 0 (identical) to 2 (opposite)
            relevance = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            raw_page = meta.get("page_number", -1)
            page_num = int(raw_page) if raw_page != -1 else None

            results.append(SearchResult(
                chunk_id=cid,
                text=text,
                source_filename=meta.get("source_filename", "unknown"),
                page_number=page_num,
                chunk_index=int(meta.get("chunk_index", 0)),
                char_count=int(meta.get("char_count", len(text))),
                distance=dist,
                relevance_score=relevance,
            ))

        # Sort by relevance descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def delete_document(self, filename: str) -> int:
        """Remove all chunks associated with a specific source filename.

        Ensures deleted or replaced documents do not leave stale chunks in the index.

        Args:
            filename: Target document filename to purge.

        Returns:
            Number of removed records (estimate based on collection difference).
        """
        before_count = self.collection.count()
        try:
            self.collection.delete(where={"source_filename": filename})
        except Exception:
            return 0
        after_count = self.collection.count()
        return max(0, before_count - after_count)

    def clear_collection(self) -> None:
        """Delete all indexed chunks in the active collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics for the vector collection."""
        total = self.collection.count()
        filenames = self.get_all_indexed_filenames()
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_provider": self.embedding_provider_name,
            "total_chunks": total,
            "indexed_files_count": len(filenames),
            "indexed_files": filenames,
        }

    def get_all_indexed_filenames(self) -> List[str]:
        """Fetch list of distinct document filenames currently indexed."""
        if self.collection.count() == 0:
            return []
        try:
            # Query collection metadatas to find distinct filenames
            data = self.collection.get(include=["metadatas"])
            metas = data.get("metadatas", [])
            names = {m.get("source_filename") for m in metas if m and "source_filename" in m}
            return sorted(list(names))
        except Exception:
            return []
