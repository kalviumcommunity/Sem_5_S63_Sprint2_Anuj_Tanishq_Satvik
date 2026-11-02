"""Unit tests for Concept #11: Document Chunking & Text Splitting."""

import asyncio
import io
import pytest
from pypdf import PdfReader, PdfWriter
from fastapi import UploadFile

from backend.app.api.v1.documents import (
    compare_chunking_strategies,
    get_document_chunks,
    upload_document,
)
from backend.app.models.document import DocumentPage, DocumentType, LoadedDocument
from backend.app.services.documents.chunker import ChunkingStrategy, DocumentChunker
from backend.app.services.documents.loader import DocumentLoader


def _generate_multipage_pdf(pages: list[str]) -> bytes:
    """Helper to synthesize a multi-page PDF document."""
    writer = PdfWriter()
    for text in pages:
        stream = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET"
        stream_len = len(stream)
        pdf = f"""%PDF-1.4
1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj
2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj
3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R>> endobj
4 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj
5 0 obj <</Length {stream_len}>> stream
{stream}
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
        reader = PdfReader(io.BytesIO(pdf.encode("latin-1")))
        writer.add_page(reader.pages[0])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_chunker_initialization_validation():
    """Verify chunk_size and chunk_overlap validation during initialization."""
    # Valid initialization
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    assert chunker.chunk_size == 200
    assert chunker.chunk_overlap == 50

    # Negative or zero chunk size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        DocumentChunker(chunk_size=0, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        DocumentChunker(chunk_size=-10, chunk_overlap=5)

    # Negative overlap
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        DocumentChunker(chunk_size=100, chunk_overlap=-1)

    # Overlap >= chunk size
    with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
        DocumentChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
        DocumentChunker(chunk_size=100, chunk_overlap=120)


def test_chunk_text_empty_and_whitespace():
    """Verify chunking empty or whitespace-only text returns empty list."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_text("", document_id="doc-1") == []
    assert chunker.chunk_text("   \n\t  ", document_id="doc-1") == []
    assert chunker.chunk_text(None, document_id="doc-1") == []


def test_chunk_text_single_chunk_under_limit():
    """Verify text shorter than chunk size returns a single chunk."""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "Short academic abstract summarizing federated learning."
    chunks = chunker.chunk_text(text, document_id="doc-101", page_number=2, section="abstract")

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-101"
    assert chunks[0].page_number == 2
    assert chunks[0].section == "abstract"
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count == len(text.split())


def test_fixed_word_chunking_with_overlap():
    """Verify fixed-word chunking preserves configured overlap between adjacent chunks."""
    words = [f"word_{i}" for i in range(1, 26)]
    text = " ".join(words)

    chunker = DocumentChunker(chunk_size=10, chunk_overlap=3, strategy=ChunkingStrategy.FIXED_WORD)
    chunks = chunker.chunk_text(text, document_id="doc-1")

    assert len(chunks) >= 3
    # First chunk has 10 words: word_1 .. word_10
    c0_words = chunks[0].text.split()
    assert len(c0_words) == 10
    assert c0_words[0] == "word_1"
    assert c0_words[-1] == "word_10"

    # Step is 10 - 3 = 7. Second chunk starts at word_8
    c1_words = chunks[1].text.split()
    assert c1_words[0] == "word_8"
    assert c1_words[1] == "word_9"
    assert c1_words[2] == "word_10"
    # Overlap contains word_8, word_9, word_10
    assert set(c0_words[-3:]) == set(c1_words[:3])


def test_recursive_character_chunking():
    """Verify recursive character splitting respects paragraph and sentence boundaries."""
    text = (
        "Paragraph 1 discusses transformer attention mechanisms in natural language.\n\n"
        "Paragraph 2 evaluates self-attention complexity and memory constraints.\n\n"
        "Paragraph 3 introduces linear attention approximations for long contexts."
    )

    chunker = DocumentChunker(
        chunk_size=110,
        chunk_overlap=20,
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
    )
    chunks = chunker.chunk_text(text, document_id="doc-rec")

    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.text) <= 150
        assert chunk.metadata["strategy"] == ChunkingStrategy.RECURSIVE_CHARACTER.value


def test_markdown_header_chunking():
    """Verify markdown-aware chunking splits cleanly on header boundaries."""
    md_text = """# Introduction
Large language models have shown remarkable few-shot learning capabilities.

## Architecture
The decoder-only transformer architecture uses causal self-attention masks.

## Evaluation
Empirical benchmark results demonstrate superior perplexity on academic texts.
"""
    chunker = DocumentChunker(chunk_size=15, chunk_overlap=3, strategy=ChunkingStrategy.MARKDOWN_HEADER)
    chunks = chunker.chunk_text(md_text, document_id="doc-md")

    assert len(chunks) >= 3
    headers = [c.text for c in chunks if "#" in c.text]
    assert len(headers) >= 2


def test_academic_metadata_enrichment():
    """Verify detection of citations and mathematical expressions in chunks."""
    text_with_cite = "Recent advances in distributed training [12] show notable speedups."
    text_with_math = "The loss function is given by $L = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2$."

    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    chunks_cite = chunker.chunk_text(text_with_cite, document_id="doc-cite")
    chunks_math = chunker.chunk_text(text_with_math, document_id="doc-math")

    assert chunks_cite[0].metadata["has_citation"] is True
    assert chunks_math[0].metadata["has_math"] is True


def test_chunk_document_multi_page_provenance():
    """Verify chunk_document preserves per-page numbers, filenames, and global sequence indexes."""
    doc = LoadedDocument(
        filename="distributed_systems.pdf",
        source_path="/storage/distributed_systems.pdf",
        document_type=DocumentType.PDF,
        pages=[
            DocumentPage(
                page_number=1,
                text="Page 1 introduction to consensus protocols and Paxos fundamentals.",
                metadata={"section": "intro"},
            ),
            DocumentPage(
                page_number=2,
                text="Page 2 Raft consensus protocol leader election and log replication guarantees.",
                metadata={"section": "protocols"},
            ),
        ],
    )

    chunker = DocumentChunker(chunk_size=5, chunk_overlap=1)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.document_id == doc.id
        assert c.metadata["filename"] == "distributed_systems.pdf"
        assert c.metadata["document_type"] == "pdf"

    page_nums = {c.page_number for c in chunks}
    assert 1 in page_nums
    assert 2 in page_nums


def test_academic_chunker_factory():
    """Verify academic chunker factory provides recommended defaults for academic PDFs."""
    chunker = DocumentChunker.create_academic_chunker()
    assert chunker.strategy == ChunkingStrategy.RECURSIVE_CHARACTER
    assert chunker.chunk_size == 600
    assert chunker.chunk_overlap == 120
    assert DocumentChunker.RECOMMENDED_ACADEMIC_STRATEGY == ChunkingStrategy.RECURSIVE_CHARACTER


def test_compare_chunking_strategies_evaluation():
    """Verify compare_strategies outputs valid comparative metrics and selects recursive chunking."""
    academic_sample = (
        "Transformer architectures revolutionized sequence modeling by replacing recurrence with attention [1].\n\n"
        "The self-attention mechanism allows all positions to interact with each other in $O(1)$ operations.\n\n"
        "Empirical evaluations across translation benchmarks confirm significant BLEU improvements (Vaswani et al., 2017)."
    )
    result = DocumentChunker.compare_strategies(academic_sample, chunk_size=200, chunk_overlap=40)

    assert "fixed_size" in result
    assert "recursive" in result
    assert result["chosen_strategy"] == ChunkingStrategy.RECURSIVE_CHARACTER.value
    assert "rationale" in result
    assert result["fixed_size"]["total_chunks"] >= 1
    assert result["recursive"]["total_chunks"] >= 1
    assert "sentence_boundary_integrity" in result["recursive"]


def test_chunk_multiple_pdf_documents():
    """Verify extracting text and chunking multiple multi-page academic PDFs preserves page references."""
    pdf_docs_data = {
        "transformer_paper.pdf": [
            "Attention Is All You Need abstract introducing self-attention [1].",
            "Scaled Dot-Product Attention computes Attention(Q,K,V) with dimension d_k.",
            "Conclusion and citations referencing multi-head attention (Vaswani et al., 2017).",
        ],
        "consensus_raft.pdf": [
            "In Search of an Understandable Consensus Protocol introducing Raft [2].",
            "Leader election and log consistency invariants guarantee linearizable state machine replication.",
        ],
    }

    chunker = DocumentChunker.create_academic_chunker(chunk_size=150, chunk_overlap=30)

    for filename, pages in pdf_docs_data.items():
        pdf_bytes = _generate_multipage_pdf(pages)
        loaded_doc = DocumentLoader.load_bytes(pdf_bytes, filename=filename)

        assert loaded_doc.document_type == DocumentType.PDF
        assert loaded_doc.page_count == len(pages)

        chunks = chunker.chunk_document(loaded_doc)
        assert len(chunks) >= len(pages)

        # Verify page provenance and document references on every chunk
        for chunk in chunks:
            assert chunk.document_id == loaded_doc.id
            assert chunk.metadata["filename"] == filename
            assert chunk.metadata["document_type"] == "pdf"
            assert chunk.page_number in range(1, len(pages) + 1)
            assert len(chunk.text) > 0

        # Verify all pages are represented in chunks
        represented_pages = {c.page_number for c in chunks}
        assert represented_pages == set(range(1, len(pages) + 1))


def test_document_upload_api_pipeline_integration():
    """Verify document upload endpoint extracts, chunks, and persists academic PDF documents."""
    pdf_pages = [
        "Abstract: Deep residual networks ease the training of substantially deeper neural networks [1].",
        "Formulation: We consider building blocks of the form y = F(x, {W_i}) + x where x and y are vectors.",
        "Results: ImageNet evaluation demonstrates 3.57% top-5 error on the test set (He et al., 2016).",
    ]
    pdf_bytes = _generate_multipage_pdf(pdf_pages)

    upload_file = UploadFile(
        file=io.BytesIO(pdf_bytes),
        filename="resnet_survey.pdf",
    )

    # 1. Test upload and pipeline chunking
    response = upload_document(file=upload_file, chunk_size=200, chunk_overlap=40)
    assert response.status == "processed"
    assert response.filename == "resnet_survey.pdf"
    assert response.chunks >= 3
    doc_id = response.document_id

    # 2. Test chunk retrieval
    retrieved_chunks = get_document_chunks(document_id=doc_id)
    assert len(retrieved_chunks) == response.chunks

    # Verify page numbers and metadata preserved end-to-end
    page_numbers = {c.page_number for c in retrieved_chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers

    for c in retrieved_chunks:
        assert c.document_id == doc_id
        assert c.metadata["filename"] == "resnet_survey.pdf"

    # 3. Test strategy comparison endpoint
    comp_file = UploadFile(file=io.BytesIO(pdf_bytes), filename="resnet_survey.pdf")
    comp_resp = compare_chunking_strategies(file=comp_file, chunk_size=200, chunk_overlap=40)
    assert comp_resp.chosen_strategy == ChunkingStrategy.RECURSIVE_CHARACTER.value
    assert comp_resp.recursive["total_chunks"] >= 1
