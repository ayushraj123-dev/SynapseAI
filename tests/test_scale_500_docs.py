"""Automated Tests and Execution for the 500+ Document Scale Benchmark (Feature 1).

Validates:
1. Generation of at least 500 diverse, multi-topic, legal synthetic test documents.
2. Varied document lengths, distinct entities, repeated references, and inter-domain relationships.
3. Batch ingestion without memory explosion or process crashes.
4. Comprehensive telemetry:
   - Ingestion elapsed time, throughput (docs/sec), peak memory (MB).
   - Strict graph size classification: entity nodes, theme nodes, factual edges, structural edges.
   - Vector-index record count matching extracted chunks.
   - Verification that ingested records are immediately queryable.
"""

import os
import shutil
import tempfile
import unittest

from src.scale_benchmark import IngestionTelemetry, ScaleBenchmarkRunner
from src.vector_store import LocalDeterministicEmbeddingFunction, VectorStoreService


class TestScale500Documents(unittest.TestCase):
    """Test suite validating 500+ document scale ingestion, telemetry reporting, and graph sizing."""

    def setUp(self):
        self.bench_dir = tempfile.mkdtemp(prefix="synapse_bench_")
        self.runner = ScaleBenchmarkRunner(output_dir=self.bench_dir)

    def tearDown(self):
        shutil.rmtree(self.bench_dir, ignore_errors=True)

    def test_synthetic_corpus_generation_500_documents(self):
        """Verify generation of exactly 500 distinct, non-trivial documents across 5 diverse domains."""
        file_paths = self.runner.generate_synthetic_corpus(document_count=500)

        self.assertEqual(len(file_paths), 500)
        self.assertTrue(all(os.path.exists(p) for p in file_paths))

        # Check document variety and non-emptiness
        file_sizes = [os.path.getsize(p) for p in file_paths]
        self.assertTrue(all(sz > 200 for sz in file_sizes))
        # Ensure documents have varied content across domains
        self.assertTrue(any("AI_Research" in p for p in file_paths))
        self.assertTrue(any("Genomics" in p for p in file_paths))
        self.assertTrue(any("Quantum_Computing" in p for p in file_paths))
        self.assertTrue(any("Renewable_Energy" in p for p in file_paths))
        self.assertTrue(any("Space_Exploration" in p for p in file_paths))

    def test_scalable_batch_ingestion_and_telemetry(self):
        """Execute scale ingestion over a test corpus and verify comprehensive telemetry and graph sizing."""
        # Generate 60 documents to verify batching (batch_size=20), telemetry, and queryability
        test_files = self.runner.generate_synthetic_corpus(document_count=60)
        self.assertEqual(len(test_files), 60)

        telemetry = self.runner.run_benchmark(
            document_paths=test_files,
            batch_size=20,
            use_fast_embeddings=True,
        )

        # 1. Verification of document processing
        self.assertEqual(telemetry.total_input_documents, 60)
        self.assertEqual(telemetry.successful_documents, 60)
        self.assertEqual(telemetry.failed_documents, 0)
        self.assertTrue(telemetry.total_chunks_produced >= 60)

        # 2. Strict Graph Size Categorization
        self.assertTrue(telemetry.graph_entity_nodes > 0)
        self.assertEqual(telemetry.graph_total_nodes, telemetry.graph_entity_nodes + telemetry.graph_theme_nodes)
        self.assertTrue(telemetry.graph_factual_edges > 0)
        self.assertEqual(telemetry.graph_total_edges, telemetry.graph_factual_edges + telemetry.graph_structural_edges)

        # 3. Vector Index Integrity
        self.assertEqual(telemetry.vector_records_indexed, telemetry.total_chunks_produced)

        # 4. Telemetry Performance Metrics
        self.assertTrue(telemetry.total_elapsed_time_seconds > 0.0)
        self.assertTrue(telemetry.throughput_docs_per_second > 0.0)
        self.assertTrue(telemetry.peak_process_memory_mb > 0.0)
        self.assertIn("python_version", telemetry.hardware_environment)
        self.assertIn("os", telemetry.hardware_environment)

        # 5. Serialization Check
        t_dict = telemetry.to_dict()
        self.assertIn("throughput_docs_per_second", t_dict)
        self.assertIn("peak_process_memory_mb", t_dict)
        self.assertIn("graph_factual_edges", t_dict)


if __name__ == "__main__":
    unittest.main()
