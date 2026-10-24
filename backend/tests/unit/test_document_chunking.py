"""Unit tests for Concept #11: Document Chunking & Text Splitting."""

import pytest
from backend.app.models.document import DocumentPage, DocumentType, LoadedDocument
from backend.app.services.documents.chunker import ChunkingStrategy, DocumentChunker


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
    # Check monotonic chunk_index
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.document_id == doc.id
        assert c.metadata["filename"] == "distributed_systems.pdf"
        assert c.metadata["document_type"] == "pdf"

    # Check that both page 1 and page 2 are represented
    page_nums = {c.page_number for c in chunks}
    assert 1 in page_nums
    assert 2 in page_nums
