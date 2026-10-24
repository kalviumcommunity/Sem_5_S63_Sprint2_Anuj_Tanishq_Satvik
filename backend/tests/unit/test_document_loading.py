"""Unit tests for Concept #10: Document Loading & Multi-Format Intake."""

import io
import os
import pytest
import docx

from backend.app.models.document import DocumentType, LoadedDocument
from backend.app.services.documents.chunker import DocumentChunker
from backend.app.services.documents.exceptions import (
    CorruptedDocumentError,
    DocumentIntakeError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from backend.app.services.documents.extractor import TextExtractor
from backend.app.services.documents.loader import DocumentLoader


def _generate_minimal_pdf_bytes(text: str) -> bytes:
    """Generate a valid single-page PDF with extractable text."""
    stream_content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET"
    stream_len = len(stream_content)
    pdf = f"""%PDF-1.4
1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj
2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj
3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R>> endobj
4 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj
5 0 obj <</Length {stream_len}>> stream
{stream_content}
endstream endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000248 00000 n 
0000000315 00000 n 
trailer <</Size 6 /Root 1 0 R>>
startxref
410
%%EOF"""
    return pdf.encode("latin-1")


def test_load_txt_single_page(tmp_path):
    """Verify loading a single-page plain text file."""
    txt_file = tmp_path / "research_notes.txt"
    content = "Federated optimization enables distributed training on edge devices."
    txt_file.write_text(content, encoding="utf-8")

    doc = DocumentLoader.load(str(txt_file))

    assert isinstance(doc, LoadedDocument)
    assert doc.filename == "research_notes.txt"
    assert doc.document_type == DocumentType.TXT
    assert doc.page_count == 1
    assert "Federated optimization" in doc.raw_text
    assert doc.pages[0].page_number == 1
    assert doc.file_hash is not None


def test_load_txt_multi_page_with_page_dividers(tmp_path):
    """Verify loading multi-page text document with form feed delimiters."""
    txt_file = tmp_path / "thesis_chapter.txt"
    page1 = "Page 1: Introduction to Neural Networks."
    page2 = "Page 2: Backpropagation and Gradient Descent."
    page3 = "Page 3: Empirical Validation and Results."
    txt_file.write_text(f"{page1}\f{page2}\f{page3}", encoding="utf-8")

    doc = DocumentLoader.load(str(txt_file))

    assert doc.page_count == 3
    assert doc.pages[0].page_number == 1
    assert "Introduction to Neural Networks" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Backpropagation" in doc.pages[1].text
    assert doc.pages[2].page_number == 3
    assert "Empirical Validation" in doc.pages[2].text


def test_load_markdown_document(tmp_path):
    """Verify loading markdown document preserving section and page splits."""
    md_file = tmp_path / "survey_paper.md"
    content = """# Section 1: Abstract
This paper surveys graph neural networks.

---

# Section 2: Methodology
We evaluate spectral and spatial graph convolutions.
"""
    md_file.write_text(content, encoding="utf-8")

    doc = DocumentLoader.load(str(md_file))

    assert doc.document_type == DocumentType.MARKDOWN
    assert doc.filename == "survey_paper.md"
    assert doc.page_count == 2
    assert "graph neural networks" in doc.pages[0].text
    assert "graph convolutions" in doc.pages[1].text


def test_load_docx_document(tmp_path):
    """Verify loading a Microsoft Word DOCX document."""
    docx_file = tmp_path / "academic_manuscript.docx"

    doc = docx.Document()
    doc.add_heading("Quantum Annealing Survey", level=1)
    doc.add_paragraph("Quantum annealing solves combinatorial optimization problems.")
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Algorithm"
    table.rows[0].cells[1].text = "Qubits"
    table.rows[1].cells[0].text = "D-Wave Advantage"
    table.rows[1].cells[1].text = "5000+"
    doc.save(str(docx_file))

    loaded = DocumentLoader.load(str(docx_file))

    assert loaded.document_type == DocumentType.DOCX
    assert loaded.filename == "academic_manuscript.docx"
    assert "Quantum Annealing Survey" in loaded.raw_text
    assert "D-Wave Advantage" in loaded.raw_text
    assert loaded.page_count >= 1


def test_load_pdf_document(tmp_path):
    """Verify loading a valid PDF file and extracting text per page."""
    pdf_file = tmp_path / "sample_paper.pdf"
    pdf_bytes = _generate_minimal_pdf_bytes("Quantum Computing Principles")
    pdf_file.write_bytes(pdf_bytes)

    doc = DocumentLoader.load(str(pdf_file))

    assert doc.document_type == DocumentType.PDF
    assert doc.filename == "sample_paper.pdf"
    assert "Quantum Computing Principles" in doc.raw_text
    assert doc.page_count == 1
    assert doc.pages[0].page_number == 1


def test_load_bytes_in_memory():
    """Verify in-memory loading from raw byte streams."""
    raw_md = b"# Title\nIn-memory Markdown test content."
    doc = DocumentLoader.load_bytes(raw_md, filename="memory_test.md")

    assert doc.document_type == DocumentType.MARKDOWN
    assert doc.source_path == "memory://memory_test.md"
    assert "In-memory Markdown" in doc.raw_text


def test_unsupported_format_raises_error(tmp_path):
    """Verify that unsupported file extensions raise UnsupportedFormatError."""
    bad_file = tmp_path / "archive.zip"
    bad_file.write_bytes(b"dummy zip data")

    with pytest.raises(UnsupportedFormatError) as exc_info:
        DocumentLoader.load(str(bad_file))

    err = exc_info.value
    assert isinstance(err, DocumentIntakeError)
    assert "archive.zip" in str(err)
    assert ".zip" in str(err)


def test_corrupted_pdf_raises_error(tmp_path):
    """Verify damaged or unreadable PDF raises CorruptedDocumentError."""
    corrupted_pdf = tmp_path / "broken.pdf"
    corrupted_pdf.write_bytes(b"%PDF-1.4\nGARBAGE_BINARY_CORRUPTED_STREAM")

    with pytest.raises(CorruptedDocumentError) as exc_info:
        DocumentLoader.load(str(corrupted_pdf))

    err = exc_info.value
    assert "broken.pdf" in str(err)


def test_corrupted_docx_raises_error(tmp_path):
    """Verify corrupted DOCX (invalid zip) raises CorruptedDocumentError."""
    corrupted_docx = tmp_path / "broken.docx"
    corrupted_docx.write_bytes(b"NOT_A_VALID_ZIP_ARCHIVE")

    with pytest.raises(CorruptedDocumentError) as exc_info:
        DocumentLoader.load(str(corrupted_docx))

    err = exc_info.value
    assert "broken.docx" in str(err)


def test_empty_document_raises_error(tmp_path):
    """Verify empty document file raises EmptyDocumentError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\t  ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError) as exc_info:
        DocumentLoader.load(str(empty_file))

    err = exc_info.value
    assert "empty.txt" in str(err)


def test_non_existent_file_raises_filenotfound():
    """Verify non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        DocumentLoader.load("non_existent_path/fake_file.txt")


def test_safe_load_graceful_error_handling(tmp_path):
    """Verify safe_load returns (None, error_str) without unhandled exceptions."""
    bad_file = tmp_path / "unsupported.csv"
    bad_file.write_text("col1,col2", encoding="utf-8")

    doc, error = DocumentLoader.safe_load(str(bad_file))
    assert doc is None
    assert error is not None
    assert "unsupported.csv" in error

    # Test success case with safe_load
    good_file = tmp_path / "valid.txt"
    good_file.write_text("Valid text content", encoding="utf-8")
    loaded_doc, err = DocumentLoader.safe_load(str(good_file))
    assert err is None
    assert loaded_doc is not None
    assert loaded_doc.filename == "valid.txt"


def test_document_chunker_integrates_with_loaded_document(tmp_path):
    """Verify DocumentChunker.chunk_document preserves multi-page sequence metadata."""
    txt_file = tmp_path / "multi_page_lecture.txt"
    page1 = "Page 1: Deep learning architectures such as convolutional neural networks."
    page2 = "Page 2: Recurrent neural networks and long short-term memory networks."
    txt_file.write_text(f"{page1}\f{page2}", encoding="utf-8")

    loaded_doc = DocumentLoader.load(str(txt_file))
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_document(loaded_doc)

    assert len(chunks) >= 2
    # Check that chunks track their originating pages
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert chunks[0].metadata["filename"] == "multi_page_lecture.txt"
    assert chunks[0].metadata["document_type"] == "txt"


def test_text_extractor_extract_from_file(tmp_path):
    """Verify TextExtractor.extract_from_file delegates to DocumentLoader."""
    txt_file = tmp_path / "extractor_test.txt"
    txt_file.write_text("Text extractor bridge test content.", encoding="utf-8")

    loaded = TextExtractor.extract_from_file(str(txt_file))
    assert isinstance(loaded, LoadedDocument)
    assert loaded.filename == "extractor_test.txt"
    assert "extractor bridge" in loaded.raw_text
