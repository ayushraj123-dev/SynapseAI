"""Scalable Ingestion Engine and 500+ Document Benchmark for SynapseAI.

Phase 2 Implementation:
- High-efficiency batch ingestion processing 500+ documents without memory leaks or unresponsiveness.
- Isolated document error handling: faulty documents are logged without aborting the batch.
- Comprehensive ingestion telemetry:
  - Input, success, and failure counts.
  - Total pages, chunks, entities, factual triples, themes, vector records.
  - Strict graph size classification: entity nodes, theme nodes, factual edges, structural edges.
  - Wall-clock runtime with breakdown: text extraction, entity extraction, graph update, vector indexing.
  - Document throughput (docs/sec) and peak process memory via psutil.
- Reproducible 500-document benchmark generator producing realistic, diverse multi-domain documents.
"""

from dataclasses import dataclass, field
import gc
import os
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil

from src.document_ingestion import DocumentIngestionService, DocumentProcessingResult, TextChunk
from src.entity_extraction import (
    Entity,
    EntityExtractor,
    ExtractionResult,
    Relationship,
    RelationshipEvidence,
    TopicTheme,
)
from src.knowledge_graph import KnowledgeGraphBuilder
from src.vector_store import LocalDeterministicEmbeddingFunction, VectorStoreService


@dataclass
class IngestionTelemetry:
    """Detailed telemetry record of a corpus ingestion run."""
    benchmark_id: str
    total_input_documents: int
    successful_documents: int
    failed_documents: int
    total_pages_extracted: int
    total_chunks_produced: int
    total_entities_extracted: int
    total_factual_triples: int
    total_themes_produced: int
    graph_entity_nodes: int
    graph_theme_nodes: int
    graph_total_nodes: int
    graph_factual_edges: int
    graph_structural_edges: int
    graph_total_edges: int
    vector_records_indexed: int
    text_extraction_time_seconds: float
    entity_extraction_time_seconds: float
    graph_build_time_seconds: float
    vector_indexing_time_seconds: float
    total_elapsed_time_seconds: float
    throughput_docs_per_second: float
    avg_time_per_doc_seconds: float
    peak_process_memory_mb: float
    hardware_environment: Dict[str, Any] = field(default_factory=dict)
    failure_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "total_input_documents": self.total_input_documents,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "total_pages_extracted": self.total_pages_extracted,
            "total_chunks_produced": self.total_chunks_produced,
            "total_entities_extracted": self.total_entities_extracted,
            "total_factual_triples": self.total_factual_triples,
            "total_themes_produced": self.total_themes_produced,
            "graph_entity_nodes": self.graph_entity_nodes,
            "graph_theme_nodes": self.graph_theme_nodes,
            "graph_total_nodes": self.graph_total_nodes,
            "graph_factual_edges": self.graph_factual_edges,
            "graph_structural_edges": self.graph_structural_edges,
            "graph_total_edges": self.graph_total_edges,
            "vector_records_indexed": self.vector_records_indexed,
            "text_extraction_time_seconds": round(self.text_extraction_time_seconds, 3),
            "entity_extraction_time_seconds": round(self.entity_extraction_time_seconds, 3),
            "graph_build_time_seconds": round(self.graph_build_time_seconds, 3),
            "vector_indexing_time_seconds": round(self.vector_indexing_time_seconds, 3),
            "total_elapsed_time_seconds": round(self.total_elapsed_time_seconds, 3),
            "throughput_docs_per_second": round(self.throughput_docs_per_second, 2),
            "avg_time_per_doc_seconds": round(self.avg_time_per_doc_seconds, 4),
            "peak_process_memory_mb": round(self.peak_process_memory_mb, 2),
            "hardware_environment": self.hardware_environment,
            "failure_details": self.failure_details,
        }


class ScaleBenchmarkRunner:
    """Manages the creation and execution of the reproducible 500-document benchmark."""

    DOMAINS = [
        ("AI_Research", "Artificial Intelligence", ["Neural Network", "Transformer", "Attention Mechanism", "Deep Learning", "SynapseAI"]),
        ("Genomics", "Biomedical Genetics", ["CRISPR", "Gene Therapy", "DNA Sequencing", "Protein Folding", "AlphaFold"]),
        ("Quantum_Computing", "Quantum Physics", ["Qubit", "Quantum Entanglement", "Superconducting Circuit", "Quantum Supremacy", "Cryostat"]),
        ("Renewable_Energy", "Clean Technology", ["Photovoltaic Cell", "Solid State Battery", "Wind Turbine", "Grid Storage", "Electrolyzer"]),
        ("Space_Exploration", "Aerospace Engineering", ["Propulsion Engine", "Orbital Habitat", "Heat Shield", "Satellite Constellation", "Telemetry"]),
    ]

    def __init__(self, output_dir: str = "./data/benchmark_500_corpus") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_synthetic_corpus(self, document_count: int = 500) -> List[str]:
        """Generate a diverse, multi-topic synthetic corpus of text documents."""
        file_paths: List[str] = []
        num_domains = len(self.DOMAINS)

        for i in range(document_count):
            dom_id, dom_name, entities = self.DOMAINS[i % num_domains]
            doc_filename = f"doc_{i+1:04d}_{dom_id}.txt"
            doc_path = os.path.join(self.output_dir, doc_filename)

            # Generate structured technical text with overlapping entities & varied lengths
            e1, e2, e3 = entities[0], entities[1], entities[2]
            partner_dom = self.DOMAINS[(i + 1) % num_domains]
            cross_entity = partner_dom[2][0]

            paragraphs = [
                f"# Technical Report {i+1:04d}: Advancements in {dom_name}\n",
                f"This document investigates the integration of {e1} with modern architectures. "
                f"Recent benchmark trials demonstrate that {e1} accelerates {e2} by a factor of {(i % 5) + 2}x. "
                f"The standard operational baseline for {e1} was established in {2018 + (i % 6)}.",

                f"Further experimentation reveals that {e2} directly interacts with {e3}. "
                f"When evaluated under controlled conditions, {e3} reduces overall latency to {(10 + (i % 20))} ms. "
                f"Researchers at the Hypnexis Laboratory observed that {e1} enhances {cross_entity} in hybrid deployments.",

                f"The measured power efficiency stands at {(80.0 + (i % 15)):.1f} percent across all test clusters. "
                f"In conclusion, the coupling of {e2} and {e3} establishes a new standard for {dom_name} applications."
            ]

            content = "\n\n".join(paragraphs)
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(content)

            file_paths.append(doc_path)

        return file_paths

    def run_benchmark(
        self,
        document_paths: List[str],
        batch_size: int = 50,
        use_fast_embeddings: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> IngestionTelemetry:
        """Execute full end-to-end ingestion, extraction, graph building, and vector indexing."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        peak_memory = initial_memory

        benchmark_id = f"bench_{int(time.time())}"
        start_time = time.time()

        # Step timers
        t_text_extract = 0.0
        t_entity_extract = 0.0
        t_graph_build = 0.0
        t_vector_index = 0.0

        # Services
        embedder = LocalDeterministicEmbeddingFunction() if use_fast_embeddings else None
        vector_store = VectorStoreService(
            persist_directory=f"./chroma_db_bench_{benchmark_id}",
            collection_name=f"col_bench_{benchmark_id}",
            embedding_function=embedder,
        )
        graph_builder = KnowledgeGraphBuilder()
        extractor = EntityExtractor()
        ingestion_service = DocumentIngestionService()

        successful_docs = 0
        failed_docs = 0
        failures: List[Dict[str, Any]] = []

        all_chunks: List[TextChunk] = []
        all_entities: Dict[str, Entity] = {}
        all_relationships: Dict[str, Relationship] = {}
        all_themes: List[TopicTheme] = []
        total_pages = 0

        total_docs = len(document_paths)

        # Batch processing loop
        for batch_start in range(0, total_docs, batch_size):
            batch_files = document_paths[batch_start : batch_start + batch_size]

            # 1. Text Ingestion & Chunking
            t0 = time.time()
            for filepath in batch_files:
                try:
                    res = ingestion_service.load_document(filepath)
                    if res.is_success or res.chunks:
                        successful_docs += 1
                        all_chunks.extend(res.chunks)
                        total_pages += res.page_count
                    else:
                        failed_docs += 1
                        failures.append({"file": filepath, "error": res.error_message})
                except Exception as e:
                    failed_docs += 1
                    failures.append({"file": filepath, "error": str(e)})
            t_text_extract += (time.time() - t0)

            # Monitor memory
            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_memory:
                peak_memory = current_mem

            if progress_callback:
                progress_callback(min(batch_start + batch_size, total_docs), total_docs, "Chunking & Text Extraction")

        # 2. Entity & Relationship Extraction
        t0 = time.time()
        # Extract per batch of chunks to control memory
        chunk_batch_size = 200
        for i in range(0, len(all_chunks), chunk_batch_size):
            batch_c = all_chunks[i : i + chunk_batch_size]
            ext_res = extractor.extract_from_chunks(batch_c)
            # Merge entities
            for eid, ent in ext_res.entities.items():
                if eid in all_entities:
                    all_entities[eid].mention_count += ent.mention_count
                    all_entities[eid].source_references.extend(ent.source_references)
                else:
                    all_entities[eid] = ent
            # Merge relationships
            for rid, rel in ext_res.relationships.items():
                if rid not in all_relationships:
                    all_relationships[rid] = rel
                else:
                    all_relationships[rid].evidence.extend(rel.evidence)
            all_themes.extend(ext_res.themes)

            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_memory:
                peak_memory = current_mem

        t_entity_extract += (time.time() - t0)

        # 3. Knowledge Graph Construction
        t0 = time.time()
        combined_extraction = ExtractionResult(
            entities=all_entities,
            relationships=all_relationships,
            themes=all_themes,
        )
        graph_stats = graph_builder.build_from_extraction(combined_extraction)
        t_graph_build += (time.time() - t0)

        # 4. Vector Store Indexing
        t0 = time.time()
        # Index chunks in batches
        for i in range(0, len(all_chunks), batch_size * 2):
            vec_batch = all_chunks[i : i + (batch_size * 2)]
            vector_store.index_chunks(vec_batch)
        t_vector_index += (time.time() - t0)

        total_elapsed = time.time() - start_time
        throughput = successful_docs / total_elapsed if total_elapsed > 0 else 0.0
        avg_per_doc = total_elapsed / successful_docs if successful_docs > 0 else 0.0

        # Detailed graph size categorization
        factual_edges = graph_stats.get("factual_edge_count", 0)
        structural_edges = graph_stats.get("structural_edge_count", 0)
        total_nodes = graph_stats.get("node_count", 0)
        total_edges = graph_stats.get("edge_count", 0)

        theme_nodes = sum(1 for _, d in graph_builder.graph.nodes(data=True) if d.get("entity_type") == "THEME")
        entity_nodes = total_nodes - theme_nodes

        hardware_env = {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "embedding_model": vector_store.embedding_provider_name,
            "vector_db": "ChromaDB (Local persistent)",
            "graph_engine": "NetworkX (In-memory directed multigraph)",
        }

        # Clean up temporary test collection files if needed
        gc.collect()

        return IngestionTelemetry(
            benchmark_id=benchmark_id,
            total_input_documents=total_docs,
            successful_documents=successful_docs,
            failed_documents=failed_docs,
            total_pages_extracted=total_pages,
            total_chunks_produced=len(all_chunks),
            total_entities_extracted=len(all_entities),
            total_factual_triples=len(all_relationships),
            total_themes_produced=len(all_themes),
            graph_entity_nodes=entity_nodes,
            graph_theme_nodes=theme_nodes,
            graph_total_nodes=total_nodes,
            graph_factual_edges=factual_edges,
            graph_structural_edges=structural_edges,
            graph_total_edges=total_edges,
            vector_records_indexed=vector_store.collection.count(),
            text_extraction_time_seconds=t_text_extract,
            entity_extraction_time_seconds=t_entity_extract,
            graph_build_time_seconds=t_graph_build,
            vector_indexing_time_seconds=t_vector_index,
            total_elapsed_time_seconds=total_elapsed,
            throughput_docs_per_second=throughput,
            avg_time_per_doc_seconds=avg_per_doc,
            peak_process_memory_mb=peak_memory,
            hardware_environment=hardware_env,
            failure_details=failures,
        )
