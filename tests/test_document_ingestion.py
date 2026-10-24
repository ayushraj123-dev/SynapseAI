"""Unit tests for SynapseAI Task 2: Document Ingestion and Text Chunking."""

import io
from pathlib import Path
import unittest

import pymupdf

from src.document_ingestion import (
    DocumentIngestionService,
    DocumentProcessingResult,
    TextChunk,
)


def create_in_memory_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate an in-memory PDF using PyMuPDF."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:  # insert text only if provided
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestDocumentIngestion(unittest.TestCase):
    """Test suite covering PDF & TXT extraction, validation, and chunking."""

    def setUp(self):
        self.service = DocumentIngestionService(
            max_file_size_bytes=20 * 1024 * 1024,
            target_chunk_size=400,
            chunk_overlap=80,
        )

    # 1. TXT extraction
    def test_txt_extraction_utf8(self):
        sample_text = "SynapseAI is an interactive knowledge base with topic-augmented graphs."
        raw_bytes = sample_text.encode("utf-8")
        result = self.service.load_document(raw_bytes, filename="test_intro.txt")

        self.assertTrue(result.is_success)
        self.assertEqual(result.file_type, "txt")
        self.assertEqual(result.filename, "test_intro.txt")
        self.assertIn("SynapseAI", result.pages[0].text)
        self.assertEqual(len(result.chunks), 1)
        self.assertIsNone(result.chunks[0].page_number)
        self.assertEqual(result.chunks[0].source_filename, "test_intro.txt")

    # 2. PDF extraction with page metadata
    def test_pdf_extraction_with_page_metadata(self):
        page_texts = [
            "Page 1: Introduction to knowledge graph representations and entity links.",
            "Page 2: Deep dive into dense vector embeddings and retrieval augmented generation.",
        ]
        pdf_bytes = create_in_memory_pdf(page_texts)
        result = self.service.load_document(pdf_bytes, filename="paper.pdf")

        self.assertTrue(result.is_success)
        self.assertEqual(result.file_type, "pdf")
        self.assertEqual(result.total_pages, 2)
        self.assertEqual(len(result.pages), 2)
        self.assertEqual(result.pages[0].page_number, 1)
        self.assertEqual(result.pages[1].page_number, 2)
        self.assertIn("Page 1", result.pages[0].text)
        self.assertIn("Page 2", result.pages[1].text)
        self.assertGreater(len(result.chunks), 0)

        # Check provenance in each chunk
        for chunk in result.chunks:
            self.assertEqual(chunk.source_filename, "paper.pdf")
            self.assertIn(chunk.page_number, [1, 2])

    # 3. Chunking and overlap behavior
    def test_chunking_and_overlap_behavior(self):
        # Create a document with 1200 characters to verify chunking and overlap
        sentence = "Neural networks discover dense representations for natural language concepts. "
        long_text = sentence * 20  # ~1560 characters
        raw_bytes = long_text.encode("utf-8")

        result = self.service.load_document(raw_bytes, filename="long_article.txt")
        self.assertTrue(result.is_success)
        self.assertGreater(len(result.chunks), 1)

        # Verify that chunk sizes do not excessively exceed target_chunk_size
        for chunk in result.chunks:
            self.assertGreater(chunk.char_count, 0)
            self.assertLessEqual(chunk.char_count, self.service.target_chunk_size + 150)

        # Verify overlap: consecutive chunks should share substring
        if len(result.chunks) >= 2:
            first_chunk_end = result.chunks[0].text[-40:]
            second_chunk = result.chunks[1].text
            # Overlap words should appear in adjacent chunk
            overlap_words = [w for w in first_chunk_end.split() if len(w) > 4]
            if overlap_words:
                self.assertTrue(any(w in second_chunk for w in overlap_words))

    # 4. Preservation of source metadata in chunks
    def test_preservation_of_source_metadata(self):
        chunks = self.service.chunk_text(
            text="First sentence for testing metadata. Second sentence with relevant concepts.",
            source_filename="research_report.pdf",
            page_number=3,
        )
        self.assertGreater(len(chunks), 0)
        chunk = chunks[0]
        self.assertEqual(chunk.source_filename, "research_report.pdf")
        self.assertEqual(chunk.page_number, 3)
        self.assertEqual(chunk.chunk_index, 0)
        self.assertTrue(chunk.chunk_id.startswith("research_report.pdf_p3_c"))
        self.assertEqual(chunk.char_count, len(chunk.text))

    # 5. Empty files
    def test_empty_file_handling(self):
        empty_bytes = b""
        result = self.service.load_document(empty_bytes, filename="empty_file.txt")
        self.assertFalse(result.is_success)
        self.assertIn("empty", result.error_message.lower())

    # 6. Unsupported file types
    def test_unsupported_file_types(self):
        dummy_bytes = b"header1,header2\nval1,val2"
        result = self.service.load_document(dummy_bytes, filename="dataset.csv")
        self.assertFalse(result.is_success)
        self.assertIn("unsupported", result.error_message.lower())

        result2 = self.service.load_document(dummy_bytes, filename="archive.zip")
        self.assertFalse(result2.is_success)

    # 7. Invalid or corrupt PDF handling
    def test_corrupt_pdf_handling(self):
        corrupt_bytes = b"%PDF-1.4\nCorrupted binary content that will fail parsing."
        result = self.service.load_document(corrupt_bytes, filename="damaged.pdf")
        self.assertFalse(result.is_success)
        self.assertIn("corrupt", result.error_message.lower())

    # 8. Multiple files where one fails but another succeeds
    def test_batch_processing_isolation(self):
        valid_txt = "Valid text file for SynapseAI ingestion."
        corrupt_pdf = b"NOT_A_PDF"
        valid_pdf = create_in_memory_pdf(["Valid PDF Page"])

        files = [
            (valid_txt.encode("utf-8"), "good_text.txt"),
            (corrupt_pdf, "broken.pdf"),
            (valid_pdf, "good_doc.pdf"),
        ]

        # Use mock objects with read and name attributes like Streamlit UploadedFile
        class MockUploadedFile:
            def __init__(self, data: bytes, name: str):
                self._data = data
                self.name = name

            def getvalue(self):
                return self._data

        mock_files = [MockUploadedFile(d, n) for d, n in files]
        results = self.service.process_files(mock_files)

        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].is_success)
        self.assertFalse(results[1].is_success)
        self.assertTrue(results[2].is_success)

        summary = self.service.get_batch_summary(results)
        self.assertEqual(summary["total_files"], 3)
        self.assertEqual(summary["successful_files"], 2)
        self.assertEqual(summary["failed_files"], 1)

    # 9. PDF with no extractable text (e.g. scanned/image-only)
    def test_pdf_no_extractable_text_warning(self):
        # Create a PDF with 2 blank pages
        pdf_bytes = create_in_memory_pdf(["", ""])
        result = self.service.load_document(pdf_bytes, filename="scanned_form.pdf")

        self.assertTrue(result.is_success)
        self.assertEqual(result.total_pages, 2)
        self.assertEqual(result.blank_pages_count, 2)
        self.assertEqual(result.total_characters, 0)
        self.assertTrue(result.is_scanned_pdf)
        self.assertIsNotNone(result.warning_message)
        self.assertIn("scanned", result.warning_message.lower())
        self.assertEqual(len(result.chunks), 0)

    # 10. Text requiring fallback decoding & BOM
    def test_txt_with_utf8_bom(self):
        # UTF-8 with BOM: \xef\xbb\xbf
        bom_bytes = b"\xef\xbb\xbfSynapseAI Knowledge Graph with BOM header."
        result = self.service.load_document(bom_bytes, filename="bom_doc.txt")

        self.assertTrue(result.is_success)
        self.assertNotIn("\ufeff", result.pages[0].text)
        self.assertIn("SynapseAI", result.pages[0].text)

    def test_txt_cp1252_fallback(self):
        # 0x93 and 0x94 are smart quotes in Windows-1252, invalid in UTF-8
        cp1252_bytes = b"This is a test with CP1252 smart quotes: \x93SynapseAI\x94"
        result = self.service.load_document(cp1252_bytes, filename="legacy_notes.txt")

        self.assertTrue(result.is_success)
        self.assertIn("SynapseAI", result.pages[0].text)

    # 11. File size limit enforcement
    def test_file_size_limit_exceeded(self):
        tiny_limit_service = DocumentIngestionService(max_file_size_bytes=100)
        large_bytes = b"A" * 200
        result = tiny_limit_service.load_document(large_bytes, filename="oversized.txt")
        self.assertFalse(result.is_success)
        self.assertIn("exceeds", result.error_message.lower())


if __name__ == "__main__":
    unittest.main()
