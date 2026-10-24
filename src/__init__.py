"""SynapseAI Core Package.

SynapseAI: Interactive Knowledge Base & Topic-Augmented Knowledge Graph Builder.

This package contains modular components designed to handle:
- Document Ingestion (PDF & TXT parsing, chunking)
- Entity & Relationship Extraction
- Topic-Augmented Knowledge Graph construction
- Vector Embeddings & Similarity Retrieval
- Adaptive Query Routing
- Grounded Answer Generation with Citations

Note: For Task 1 (Foundation), these modules define clear interfaces and
placeholders to guide upcoming implementation tasks.
"""

__version__ = "0.1.0"
__author__ = "SynapseAI Team"

__all__ = [
    "document_ingestion",
    "entity_extraction",
    "knowledge_graph",
    "vector_store",
    "query_router",
    "answer_generator",
    "graph_database",
]
