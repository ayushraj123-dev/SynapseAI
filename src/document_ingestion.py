"""Document Ingestion & Text Chunking Module for SynapseAI.

Task 2 Implementation:
- In-memory parsing of PDF (via PyMuPDF) and TXT documents.
- Strict validation: supported extensions (.pdf, .txt), 20 MB file size limit, empty file detection.
- Page-by-page PDF extraction preserving 1-based page numbers and detecting scanned/image-only PDFs.
- Robust TXT decoding with UTF-8 BOM (utf-8-sig) and CP1252 fallback handling.
- Boundary-aware, provenance-preserving text chunking (~800 characters target, ~120 characters overlap).
- Graceful per-file error handling (one corrupted file never halts processing of other files).
"""

from dataclasses import dataclass, field
import io
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pymupdf
except ImportError:
    # Fallback to fitz alias if present
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None  # Handled gracefully at runtime if missing


# Maximum allowed upload file size: 20 MB
DEFAULT_MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
DEFAULT_TARGET_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


@dataclass
class TextChunk:
    """Represents an extracted, provenance-tracked chunk of text."""
    chunk_id: str
    source_filename: str
    page_number: Optional[int]  # 1-based page number for PDFs, None for TXT
    text: str
    chunk_index: int            # Index of chunk within document/page
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
        }


@dataclass
class ExtractedPage:
    """Represents extracted content and metadata for a single PDF page."""
    page_number: int  # 1-based index
    text: str
    char_count: int
    is_blank: bool


@dataclass
class DocumentProcessingResult:
    """Represents the complete extraction and chunking outcome for a document."""
    filename: str
    file_type: str
    file_size_bytes: int
    is_success: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    pages: List[ExtractedPage] = field(default_factory=list)
    chunks: List[TextChunk] = field(default_factory=list)
    total_characters: int = 0
    total_pages: int = 0
    blank_pages_count: int = 0
    is_scanned_pdf: bool = False

    @property
    def source_filename(self) -> str:
        """Alias for filename."""
        return self.filename

    @property
    def page_count(self) -> int:
        """Alias for total_pages."""
        return self.total_pages

    @property
    def full_text(self) -> str:
        """Concatenated non-blank text across all pages."""
        return "\n\n".join(p.text for p in self.pages if not p.is_blank)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a lightweight summary dictionary for UI and telemetry."""
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "is_success": self.is_success,
            "error_message": self.error_message,
            "warning_message": self.warning_message,
            "total_characters": self.total_characters,
            "total_pages": self.total_pages,
            "blank_pages_count": self.blank_pages_count,
            "chunks_count": len(self.chunks),
            "is_scanned_pdf": self.is_scanned_pdf,
        }


class DocumentIngestionService:
    """Service for validating, extracting, and chunking documents."""

    def __init__(
        self,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        target_chunk_size: int = DEFAULT_TARGET_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.supported_extensions = [".pdf", ".txt"]
        self.max_file_size_bytes = max_file_size_bytes
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def _read_bytes(self, file_source: Any, filename: Optional[str] = None) -> Tuple[bytes, str]:
        """Safely read raw bytes and resolve filename from various input types.

        Supports:
        - Streamlit UploadedFile or file-like buffer (has .read(), .name, .getvalue())
        - Path string or pathlib.Path
        - Raw bytes
        """
        resolved_name = filename or "document"

        # Streamlit UploadedFile or BytesIO
        if hasattr(file_source, "name") and not filename:
            resolved_name = file_source.name

        # Tuple (filename, bytes)
        if isinstance(file_source, tuple) and len(file_source) == 2:
            resolved_name = filename or str(file_source[0])
            content_bytes = file_source[1] if isinstance(file_source[1], bytes) else bytes(file_source[1])
            return content_bytes, resolved_name

        if hasattr(file_source, "getvalue"):
            content_bytes = file_source.getvalue()
        elif hasattr(file_source, "read"):
            current_pos = file_source.tell() if hasattr(file_source, "tell") else None
            content_bytes = file_source.read()
            # Restore position if possible
            if current_pos is not None and hasattr(file_source, "seek"):
                file_source.seek(current_pos)
        elif isinstance(file_source, (str, Path)):
            path_obj = Path(file_source)
            if not filename:
                resolved_name = path_obj.name
            with open(path_obj, "rb") as f:
                content_bytes = f.read()
        elif isinstance(file_source, bytes):
            content_bytes = file_source
        else:
            raise ValueError(f"Unsupported file source type: {type(file_source)}")

        return content_bytes, resolved_name

    def validate_file(self, content_bytes: bytes, filename: str) -> Optional[str]:
        """Validate file format, size, and non-emptiness.

        Returns an error message string if invalid, or None if valid.
        """
        ext = Path(filename).suffix.lower()
        if ext not in self.supported_extensions:
            return (
                f"Unsupported file format '{ext or 'unknown'}'. "
                f"SynapseAI currently supports: {', '.join(self.supported_extensions)}."
            )

        file_size = len(content_bytes)
        if file_size == 0:
            return "File is empty (0 bytes)."

        if file_size > self.max_file_size_bytes:
            limit_mb = self.max_file_size_bytes / (1024 * 1024)
            size_mb = file_size / (1024 * 1024)
            return f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {limit_mb:.0f} MB."

        return None

    def _extract_pdf(self, content_bytes: bytes, filename: str) -> DocumentProcessingResult:
        """Extract text page-by-page from PDF bytes using PyMuPDF."""
        if pymupdf is None:
            return DocumentProcessingResult(
                filename=filename,
                file_type="pdf",
                file_size_bytes=len(content_bytes),
                is_success=False,
                error_message="PyMuPDF library is not installed or could not be loaded.",
            )

        try:
            doc = pymupdf.open(stream=content_bytes, filetype="pdf")
        except Exception as e:
            return DocumentProcessingResult(
                filename=filename,
                file_type="pdf",
                file_size_bytes=len(content_bytes),
                is_success=False,
                error_message=f"Corrupt or invalid PDF file: {str(e)}",
            )

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return DocumentProcessingResult(
                filename=filename,
                file_type="pdf",
                file_size_bytes=len(content_bytes),
                is_success=False,
                error_message="PDF contains 0 pages.",
            )

        extracted_pages: List[ExtractedPage] = []
        all_chunks: List[TextChunk] = []
        blank_pages_count = 0
        total_chars = 0
        global_chunk_idx = 0

        try:
            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1  # 1-based page indexing
                page_text = page.get_text() or ""
                trimmed = page_text.strip()

                if not trimmed:
                    blank_pages_count += 1
                    extracted_pages.append(
                        ExtractedPage(page_number=page_num, text="", char_count=0, is_blank=True)
                    )
                    continue

                page_chars = len(trimmed)
                total_chars += page_chars
                extracted_pages.append(
                    ExtractedPage(page_number=page_num, text=trimmed, char_count=page_chars, is_blank=False)
                )

                # Chunk this page
                page_chunks = self.chunk_text(
                    text=trimmed,
                    source_filename=filename,
                    page_number=page_num,
                    starting_chunk_index=global_chunk_idx,
                )
                all_chunks.extend(page_chunks)
                global_chunk_idx += len(page_chunks)
        except Exception as e:
            doc.close()
            return DocumentProcessingResult(
                filename=filename,
                file_type="pdf",
                file_size_bytes=len(content_bytes),
                is_success=False,
                error_message=f"Failed to extract text from PDF: {str(e)}",
            )

        doc.close()

        # Check if PDF contains no extractable text (likely scanned)
        is_scanned = (total_chars == 0 and total_pages > 0)
        warning_msg = None
        if is_scanned:
            warning_msg = (
                "No extractable text detected in this PDF. It may be a scanned document "
                "or image-only PDF requiring OCR. OCR is not implemented in this phase."
            )

        return DocumentProcessingResult(
            filename=filename,
            file_type="pdf",
            file_size_bytes=len(content_bytes),
            is_success=True,
            warning_message=warning_msg,
            pages=extracted_pages,
            chunks=all_chunks,
            total_characters=total_chars,
            total_pages=total_pages,
            blank_pages_count=blank_pages_count,
            is_scanned_pdf=is_scanned,
        )

    def _extract_txt(self, content_bytes: bytes, filename: str) -> DocumentProcessingResult:
        """Decode TXT file with UTF-8, BOM handling, and CP1252 fallback."""
        decoded_text: Optional[str] = None
        encoding_used: Optional[str] = None

        # Strategy 1: UTF-8 with BOM support (utf-8-sig automatically strips BOM)
        try:
            decoded_text = content_bytes.decode("utf-8-sig")
            encoding_used = "utf-8"
        except UnicodeDecodeError:
            pass

        # Strategy 2: Fallback to CP1252 (standard Windows western encoding)
        if decoded_text is None:
            try:
                decoded_text = content_bytes.decode("cp1252")
                encoding_used = "cp1252"
            except UnicodeDecodeError:
                pass

        # Strategy 3: Fallback to latin-1 (guaranteed to decode any byte stream, but check validity)
        if decoded_text is None:
            try:
                decoded_text = content_bytes.decode("latin-1")
                encoding_used = "latin-1"
            except Exception:
                decoded_text = None

        if decoded_text is None:
            return DocumentProcessingResult(
                filename=filename,
                file_type="txt",
                file_size_bytes=len(content_bytes),
                is_success=False,
                error_message="Unable to decode text file. Ensure it is encoded in UTF-8 or standard ASCII/CP1252.",
            )

        decoded_text = decoded_text.lstrip("\ufeff")

        trimmed = decoded_text.strip()
        if not trimmed:
            return DocumentProcessingResult(
                filename=filename,
                file_type="txt",
                file_size_bytes=len(content_bytes),
                is_success=False,
                error_message="Text file contains only whitespace or is empty.",
            )

        total_chars = len(trimmed)
        chunks = self.chunk_text(
            text=trimmed,
            source_filename=filename,
            page_number=None,
            starting_chunk_index=0,
        )

        return DocumentProcessingResult(
            filename=filename,
            file_type="txt",
            file_size_bytes=len(content_bytes),
            is_success=True,
            pages=[ExtractedPage(page_number=1, text=trimmed, char_count=total_chars, is_blank=False)],
            chunks=chunks,
            total_characters=total_chars,
            total_pages=1,
            blank_pages_count=0,
            is_scanned_pdf=False,
        )

    def chunk_text(
        self,
        text: str,
        source_filename: str,
        page_number: Optional[int] = None,
        target_size: Optional[int] = None,
        overlap: Optional[int] = None,
        starting_chunk_index: int = 0,
    ) -> List[TextChunk]:
        """Split text into manageable chunks respecting word boundaries.

        Args:
            text: Normalized text segment to chunk.
            source_filename: Name of the originating file.
            page_number: 1-based page number for PDFs, or None for TXT files.
            target_size: Target characters per chunk (~800 chars).
            overlap: Overlap characters between consecutive chunks (~120 chars).
            starting_chunk_index: Starting index for sequential chunk numbering.

        Returns:
            List of TextChunk instances with provenance metadata.
        """
        target = target_size or self.target_chunk_size
        ovlp = overlap or self.chunk_overlap

        # Ensure valid chunking parameters to avoid negative steps or infinite loops
        if target <= 0:
            target = DEFAULT_TARGET_CHUNK_SIZE
        if ovlp < 0 or ovlp >= target:
            ovlp = max(0, min(120, target // 4))

        chunks: List[TextChunk] = []
        cleaned_text = text.strip()
        if not cleaned_text:
            return chunks

        text_len = len(cleaned_text)
        start = 0
        chunk_idx = starting_chunk_index

        # Sanitize filename for deterministic, clean chunk IDs
        safe_fname = re.sub(r"[^\w\-.]", "_", Path(source_filename).name)
        page_tag = f"p{page_number}" if page_number is not None else "txt"

        while start < text_len:
            end = start + target

            if end >= text_len:
                # Last remaining piece of text
                chunk_str = cleaned_text[start:].strip()
                if chunk_str:
                    chunk_id = f"{safe_fname}_{page_tag}_c{chunk_idx}"
                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            source_filename=source_filename,
                            page_number=page_number,
                            text=chunk_str,
                            chunk_index=chunk_idx,
                            char_count=len(chunk_str),
                        )
                    )
                break

            # Find a word boundary near 'end' (space, newline, or punctuation)
            boundary = -1
            # Search backward within a reasonable window for a break character
            search_window = min(120, target // 3)
            for i in range(end, max(start, end - search_window), -1):
                if cleaned_text[i] in ("\n", " ", "\t", "—", "-", ";", ".", "!", "?"):
                    boundary = i
                    break

            if boundary != -1 and boundary > start:
                chunk_end = boundary + 1  # Include the separator or break after it
            else:
                chunk_end = end

            chunk_str = cleaned_text[start:chunk_end].strip()
            if chunk_str:
                chunk_id = f"{safe_fname}_{page_tag}_c{chunk_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        source_filename=source_filename,
                        page_number=page_number,
                        text=chunk_str,
                        chunk_index=chunk_idx,
                        char_count=len(chunk_str),
                    )
                )
                chunk_idx += 1

            # Advance start position with overlap
            next_start = chunk_end - ovlp

            # Boundary alignment for overlap: try starting on a word boundary
            if next_start > start and next_start < text_len:
                # Walk forward to find next whitespace / word start to avoid cutting words
                for j in range(next_start, min(text_len, next_start + 40)):
                    if cleaned_text[j] in (" ", "\n", "\t"):
                        next_start = j + 1
                        break

            # Strict guard against zero or negative forward progress (prevents infinite loops)
            if next_start <= start:
                next_start = max(start + 1, chunk_end)

            start = next_start

        return chunks

    def load_document(
        self,
        file_source: Any,
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> DocumentProcessingResult:
        """Validate, parse, extract, and chunk a single document.

        Args:
            file_source: Streamlit UploadedFile, file-like object, Path, or raw bytes.
            filename: Name of the file (optional if file_source has a .name attribute).
            file_type: File extension ('pdf' or 'txt'). Inferred from filename if None.

        Returns:
            DocumentProcessingResult with extraction status, pages, and chunks.
        """
        try:
            content_bytes, resolved_name = self._read_bytes(file_source, filename)
        except Exception as e:
            return DocumentProcessingResult(
                filename=filename or "unknown",
                file_type=file_type or "unknown",
                file_size_bytes=0,
                is_success=False,
                error_message=f"Failed to read file: {str(e)}",
            )

        # Validate file
        validation_error = self.validate_file(content_bytes, resolved_name)
        ext = Path(resolved_name).suffix.lower()
        clean_ext = ext.lstrip(".")

        if validation_error:
            return DocumentProcessingResult(
                filename=resolved_name,
                file_type=clean_ext or "unknown",
                file_size_bytes=len(content_bytes),
                is_success=False,
                error_message=validation_error,
            )

        # Process based on type
        if ext == ".pdf":
            return self._extract_pdf(content_bytes, resolved_name)
        elif ext == ".txt":
            return self._extract_txt(content_bytes, resolved_name)
        else:
            return DocumentProcessingResult(
                filename=resolved_name,
                file_type=clean_ext,
                file_size_bytes=len(content_bytes),
                is_success=False,
                error_message=f"Unsupported file extension: {ext}",
            )

    def process_files(self, files: List[Any]) -> List[DocumentProcessingResult]:
        """Process multiple files in batch; isolated errors never fail the batch."""
        results: List[DocumentProcessingResult] = []
        for f in files:
            result = self.load_document(f)
            results.append(result)
        return results

    # Backward-compatible alias
    process_uploaded_files = process_files

    @staticmethod
    def get_batch_summary(results: List[DocumentProcessingResult]) -> Dict[str, Any]:
        """Compute aggregate statistics over a list of processing results."""
        successful = [r for r in results if r.is_success]
        failed = [r for r in results if not r.is_success]

        total_chars = sum(r.total_characters for r in successful)
        total_pages = sum(r.total_pages for r in successful)
        total_chunks = sum(len(r.chunks) for r in successful)
        blank_pages = sum(r.blank_pages_count for r in successful)
        scanned_count = sum(1 for r in successful if r.is_scanned_pdf)

        return {
            "total_files": len(results),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "total_characters": total_chars,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "blank_pages_count": blank_pages,
            "scanned_pdfs_count": scanned_count,
        }
