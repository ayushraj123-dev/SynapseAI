"""Execution script for running full 500-document scale benchmark and saving real telemetry."""
import os
import sys
import json
import time

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scale_benchmark import ScaleBenchmarkRunner

def main():
    print("=================================================================")
    print("SYNAPSEAI PHASE 2: 500-DOCUMENT SCALE BENCHMARK EXECUTION")
    print("=================================================================")
    runner = ScaleBenchmarkRunner()
    
    print("1. Generating 500 synthetic diverse technical documents...")
    doc_paths = runner.generate_synthetic_corpus(document_count=500)
    print(f"Generated {len(doc_paths)} documents across diverse domains.")
    
    def on_progress(done, total, stage):
        pct = (done / total) * 100
        print(f"[{stage}] Progress: {done}/{total} ({pct:.1f}%)")
        
    print("2. Ingesting, extracting entities/relations, building graph, and indexing ChromaDB vectors in batches...")
    telemetry = runner.run_benchmark(
        document_paths=doc_paths,
        batch_size=50,
        use_fast_embeddings=True,
        progress_callback=on_progress
    )
    
    print("\n================ BENCHMARK COMPLETED ================")
    print(f"Benchmark ID: {telemetry.benchmark_id}")
    print(f"Input Documents: {telemetry.total_input_documents}")
    print(f"Successful Docs: {telemetry.successful_documents}")
    print(f"Failed Docs: {telemetry.failed_documents}")
    print(f"Total Pages Extracted: {telemetry.total_pages_extracted}")
    print(f"Total Chunks: {telemetry.total_chunks_produced}")
    print(f"Entities Extracted: {telemetry.total_entities_extracted}")
    print(f"Factual Triples: {telemetry.total_factual_triples}")
    print(f"Themes Produced: {telemetry.total_themes_produced}")
    print(f"Graph Nodes (Entity): {telemetry.graph_entity_nodes}")
    print(f"Graph Nodes (Theme): {telemetry.graph_theme_nodes}")
    print(f"Graph Nodes (Total): {telemetry.graph_total_nodes}")
    print(f"Graph Edges (Factual): {telemetry.graph_factual_edges}")
    print(f"Graph Edges (Structural): {telemetry.graph_structural_edges}")
    print(f"Graph Edges (Total): {telemetry.graph_total_edges}")
    print(f"Vector Records: {telemetry.vector_records_indexed}")
    print(f"Total Elapsed Time: {telemetry.total_elapsed_time_seconds:.2f}s")
    print(f"Throughput: {telemetry.throughput_docs_per_second:.2f} docs/sec")
    print(f"Avg Time Per Doc: {telemetry.avg_time_per_doc_seconds * 1000:.1f} ms")
    print(f"Peak Process Memory: {telemetry.peak_process_memory_mb:.2f} MB")
    print(f"Step Timings:")
    print(f"  - Text extraction: {telemetry.text_extraction_time_seconds:.2f}s")
    print(f"  - Entity & relationship extraction: {telemetry.entity_extraction_time_seconds:.2f}s")
    print(f"  - Graph construction: {telemetry.graph_build_time_seconds:.2f}s")
    print(f"  - Vector indexing: {telemetry.vector_indexing_time_seconds:.2f}s")
    
    # Save results to data/latest_500_doc_benchmark.json
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest_500_doc_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(telemetry.to_dict(), f, indent=2)
    print(f"\nSaved telemetry report to: {out_path}")

if __name__ == "__main__":
    main()
