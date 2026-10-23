"""Unit tests for metadata-aware chunking, storage, retrieval, and source citations.

Verifies:
1. A chunk always has a source identifier.
2. Chunk indices are assigned correctly within each document.
3. Character offsets are preserved when supported.
4. Page numbers and section headings are preserved when available.
5. Multiple documents retain separate source identities.
6. Metadata survives embedding-storage and retrieval operations.
7. Retrieved chunks expose their original metadata (including dict subscripting).
8. Generated citations use actual retrieved source information.
9. Missing optional metadata does not cause failures.
10. Existing functionality continues to work without regressions.
"""

import pytest
from backend.app.models.chunk import Chunk, ChunkMetadata
from backend.app.repositories.chunks import ChunkRepository
from backend.app.services.documents.chunker import DocumentChunker
from backend.app.services.documents.extractor import TextExtractor
from backend.app.services.documents.loader import DocumentLoader
from backend.app.services.documents.ingestion import DocumentIngestionPipeline
from backend.app.services.embeddings.service import MockEmbeddingService
from backend.app.services.retrieval.search import SearchResult
from backend.app.services.retrieval.filters import MetadataFilter
from backend.app.services.retrieval.reranker import Reranker
from backend.app.services.rag.context import ContextBuilder
from backend.app.services.rag.citations import CitationService
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.services.llm.parser import ParsedResponse


# 1. A chunk always has a source identifier
def test_chunk_always_has_source_identifier():
    # Explicit source
    chunk1 = Chunk(
        document_id="doc-123",
        chunk_index=0,
        text="Sample text content",
        source="refund-policy.pdf",
    )
    assert chunk1.source == "refund-policy.pdf"
    assert chunk1.metadata["source"] == "refund-policy.pdf"
    assert chunk1["metadata"]["source"] == "refund-policy.pdf"

    # Default to document_id if source not specified
    chunk2 = Chunk(
        document_id="doc-456",
        chunk_index=1,
        text="Another piece of text",
    )
    assert chunk2.source == "doc-456"
    assert chunk2.metadata["source"] == "doc-456"
    assert chunk2["metadata"]["source"] == "doc-456"

    # In chunker output
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_text(
        text="This is a test of source tracking in chunking.",
        document_id="doc-789",
        source="manual.pdf",
    )
    assert len(chunks) > 0
    for c in chunks:
        assert c.source == "manual.pdf"
        assert c.metadata["source"] == "manual.pdf"


# 2. Chunk indices are assigned correctly within each document
def test_chunk_indices_assigned_correctly_within_document():
    chunker = DocumentChunker(chunk_size=4, chunk_overlap=1)
    # 10 words -> step = 3 -> indices 0, 1, 2, 3
    sample_text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    chunks = chunker.chunk_text(
        text=sample_text,
        document_id="doc-index-test",
        source="doc-index.txt",
    )
    assert len(chunks) == 3
    for expected_idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == expected_idx
        assert chunk.metadata["chunk_index"] == expected_idx
        assert chunk["metadata"]["chunk_index"] == expected_idx

    # Multi-page/section continuous indexing
    pages = [
        {"text": "page one content here now", "page_number": 1},
        {"text": "page two content continues here", "page_number": 2},
    ]
    doc_chunks = chunker.chunk_document(
        document_id="doc-multi-page",
        pages_or_sections=pages,
        source="multi.pdf",
    )
    for i, c in enumerate(doc_chunks):
        assert c.chunk_index == i
        assert c.metadata["chunk_index"] == i


# 3. Character offsets are preserved when supported
def test_character_offsets_preserved():
    chunker = DocumentChunker(chunk_size=3, chunk_overlap=1)
    text = "Alpha beta gamma delta epsilon"
    chunks = chunker.chunk_text(
        text=text,
        document_id="doc-offsets",
        source="greek.txt",
        char_offset_base=100,
    )
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert first_chunk.char_start is not None
    assert first_chunk.char_end is not None
    assert first_chunk.metadata["char_start"] == 100
    # First chunk: "Alpha beta gamma" -> len is 16 -> end is 100 + 16 = 116
    assert first_chunk.metadata["char_end"] == 116

    # Verify extracted offsets match raw substring
    rel_start = first_chunk.char_start - 100
    rel_end = first_chunk.char_end - 100
    assert text[rel_start:rel_end] == first_chunk.text

    # Verify TextExtractor offset preservation
    raw = "Page 1 text\fPage 2 text"
    extracted = TextExtractor.extract_from_text(raw)
    assert len(extracted) == 2
    assert extracted[0]["char_start"] == 0
    assert extracted[0]["char_end"] == len("Page 1 text")
    assert extracted[1]["char_start"] == len("Page 1 text") + 1


# 4. Page numbers and section headings are preserved when available
def test_page_numbers_and_sections_preserved():
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=1)
    chunks = chunker.chunk_text(
        text="The refund policy requires original receipt within thirty days.",
        document_id="doc-policy",
        source="refund-policy.pdf",
        page_number=3,
        section="Refund Policy",
        url="https://example.com/refund",
        document_version="2024",
    )
    assert len(chunks) > 0
    for c in chunks:
        assert c.page_number == 3
        assert c.section == "Refund Policy"
        assert c.metadata["page_number"] == 3
        assert c.metadata["section"] == "Refund Policy"
        assert c.metadata["url"] == "https://example.com/refund"
        assert c.metadata["document_version"] == "2024"

    # Markdown extraction preserves section headers
    md = "# Overview\nThis is overview.\n## Architecture\nThis is architecture."
    sections = TextExtractor.extract_from_markdown(md)
    assert len(sections) == 2
    assert sections[0]["section"] == "Overview"
    assert sections[1]["section"] == "Architecture"


# 5. Multiple documents retain separate source identities
def test_multiple_documents_retain_separate_sources():
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=1)
    docs = [
        {
            "document_id": "doc-a",
            "source": "terms-of-service.pdf",
            "text": "User agrees to terms and conditions upon registration.",
            "page_number": 1,
            "section": "Terms",
        },
        {
            "document_id": "doc-b",
            "source": "privacy-policy.pdf",
            "text": "We respect user privacy and do not sell data.",
            "page_number": 2,
            "section": "Privacy",
        },
    ]
    grouped = chunker.chunk_documents(docs)
    assert "doc-a" in grouped
    assert "doc-b" in grouped

    chunks_a = grouped["doc-a"]
    chunks_b = grouped["doc-b"]

    assert all(c.source == "terms-of-service.pdf" for c in chunks_a)
    assert all(c.metadata["source"] == "terms-of-service.pdf" for c in chunks_a)
    assert all(c.document_id == "doc-a" for c in chunks_a)

    assert all(c.source == "privacy-policy.pdf" for c in chunks_b)
    assert all(c.metadata["source"] == "privacy-policy.pdf" for c in chunks_b)
    assert all(c.document_id == "doc-b" for c in chunks_b)

    # Indices restart at 0 for each document
    assert chunks_a[0].chunk_index == 0
    assert chunks_b[0].chunk_index == 0


# 6. Metadata survives embedding-storage and retrieval operations
@pytest.mark.asyncio
async def test_metadata_survives_embedding_and_storage():
    repo = ChunkRepository()
    embedding_service = MockEmbeddingService(dimension=64)
    pipeline = DocumentIngestionPipeline(
        embedding_service=embedding_service,
        chunk_repository=repo,
    )

    doc_text = "Federated learning enables collaborative training without sharing raw data."
    chunks = await pipeline.ingest_document(
        text=doc_text,
        document_id="doc-fl-01",
        source="fl-paper.pdf",
        url="https://arxiv.org/abs/fl-paper",
        document_version="v2",
        title="Federated Learning Survey",
    )
    assert len(chunks) > 0
    stored_chunk = repo.get_by_id(chunks[0].id)
    assert stored_chunk is not None
    assert stored_chunk.embedding is not None
    assert stored_chunk.metadata["source"] == "fl-paper.pdf"
    assert stored_chunk.metadata["url"] == "https://arxiv.org/abs/fl-paper"
    assert stored_chunk.metadata["document_version"] == "v2"
    assert stored_chunk.metadata["chunk_index"] == 0

    # Retrieve via vector search
    q_vec = await embedding_service.embed_text("collaborative training")
    hits = repo.search_vector(query_vector=q_vec, top_k=1)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.chunk_id == stored_chunk.id
    assert hit.metadata["source"] == "fl-paper.pdf"
    assert hit.metadata["url"] == "https://arxiv.org/abs/fl-paper"
    assert hit.metadata["document_version"] == "v2"


# 7. Retrieved chunks expose their original metadata (dict subscripting and attributes)
def test_retrieved_chunks_expose_original_metadata():
    cand = SearchResult(
        chunk_id="c-99",
        document_id="doc-ret-test",
        document_title="Employee Handbook",
        text="Vacation days accrue at two days per working month.",
        page_number=4,
        section="Leave Policy",
        score=0.92,
        metadata={
            "source": "employee-handbook.pdf",
            "chunk_index": 3,
            "char_start": 1200,
            "char_end": 1255,
            "url": "https://company.org/handbook",
            "document_version": "2024.1",
        },
    )

    # Dict subscripting
    assert cand["text"] == "Vacation days accrue at two days per working month."
    assert cand["metadata"]["source"] == "employee-handbook.pdf"
    assert cand["metadata"]["chunk_index"] == 3
    assert cand["metadata"]["char_start"] == 1200
    assert cand["metadata"]["char_end"] == 1255
    assert cand["metadata"]["url"] == "https://company.org/handbook"
    assert cand["metadata"]["page_number"] == 4
    assert cand["metadata"]["section"] == "Leave Policy"

    # Property access
    assert cand.source == "employee-handbook.pdf"
    assert cand.chunk_id == "c-99"


# 8. Generated citations use actual retrieved source information
def test_generated_citations_use_actual_metadata():
    # 1. Source with URL and page
    cand1 = SearchResult(
        chunk_id="c1",
        document_id="d1",
        document_title="Official Documentation",
        text="API requests must include a Bearer token.",
        page_number=2,
        section="Authentication",
        score=0.88,
        metadata={
            "source": "Official Documentation",
            "chunk_index": 0,
            "page_number": 2,
            "section": "Authentication",
            "url": "https://example.com/documentation",
        },
    )
    # 2. Source with page only
    cand2 = SearchResult(
        chunk_id="c2",
        document_id="d2",
        document_title="Refund Policy Document",
        text="Refunds are processed within 14 business days.",
        page_number=3,
        section="Refunds",
        score=0.85,
        metadata={
            "source": "refund-policy.pdf",
            "chunk_index": 1,
            "page_number": 3,
            "section": "Refunds",
        },
    )
    # 3. Source with section only
    cand3 = SearchResult(
        chunk_id="c3",
        document_id="d3",
        document_title="Employee Handbook",
        text="Employees receive 20 days annual leave.",
        page_number=1,
        section="Leave Policy",
        score=0.82,
        metadata={
            "source": "Employee Handbook",
            "chunk_index": 5,
            "page_number": None,
            "section": "Leave Policy",
        },
    )

    parsed = ParsedResponse(answer="Here are the answers.", is_refusal=False)
    citations = CitationService.validate(parsed, [cand1, cand2, cand3])
    assert len(citations) == 3

    # Check URL markdown link
    assert "https://example.com/documentation" in citations[0].citation_text
    assert "Official Documentation" in citations[0].citation_text
    assert citations[0].source == "Official Documentation"

    # Check document + page citation
    assert "refund-policy.pdf" in citations[1].citation_text
    assert "page 3" in citations[1].citation_text

    # Check document + section citation
    assert "Employee Handbook" in citations[2].citation_text
    assert "Leave Policy section" in citations[2].citation_text

    # Unknown source case
    unknown_cand = SearchResult(
        chunk_id="c-unk",
        document_id="unknown_doc",
        document_title="",
        text="Some orphaned chunk.",
        page_number=1,
        score=0.5,
        metadata={"source": "unknown_doc"},
    )
    unk_citations = CitationService.validate(parsed, [unknown_cand])
    assert "Unknown Source - Unverified" in unk_citations[0].citation_text


# 9. Missing optional metadata does not cause failures
def test_missing_optional_metadata_does_not_fail():
    # Chunk with absolute bare minimum
    chunk = Chunk(
        document_id="bare-doc",
        chunk_index=0,
        text="Bare text content.",
    )
    assert chunk.metadata["source"] == "bare-doc"
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["char_start"] is None
    assert chunk.metadata["char_end"] is None
    assert chunk.metadata["url"] is None
    assert chunk.metadata["document_version"] is None

    # ContextBuilder handles missing optional fields gracefully
    res = SearchResult.from_chunk(chunk, score=0.7)
    context_str = ContextBuilder.build([res])
    assert "bare-doc" in context_str
    assert "Bare text content" in context_str

    # CitationService handles missing optional fields
    parsed = ParsedResponse(answer="Answer here", is_refusal=False)
    citations = CitationService.validate(parsed, [res])
    assert len(citations) == 1
    assert citations[0].citation_text == "[bare-doc, page 1]"

    # Legacy migration handles missing fields safely
    legacy_dict = {"document_id": "legacy-01", "text": "Old chunk without schema."}
    migrated = Chunk.migrate_legacy_chunk(legacy_dict)
    assert migrated.source == "legacy-01"
    assert migrated.chunk_index == 0
    assert migrated.metadata["char_start"] is None


# 10. Metadata-based filtering functionality
def test_metadata_based_retrieval_filtering():
    c1 = SearchResult(
        chunk_id="c1",
        document_id="doc-1",
        document_title="Title 1",
        text="Text 1",
        page_number=1,
        metadata={"source": "paper-v1.pdf", "document_version": "2023", "document_type": "pdf", "section": "Intro"},
    )
    c2 = SearchResult(
        chunk_id="c2",
        document_id="doc-2",
        document_title="Title 2",
        text="Text 2",
        page_number=2,
        metadata={"source": "paper-v2.pdf", "document_version": "2024", "document_type": "pdf", "section": "Methods"},
    )
    c3 = SearchResult(
        chunk_id="c3",
        document_id="doc-3",
        document_title="Title 3",
        text="Text 3",
        page_number=1,
        metadata={"source": "notes.md", "document_version": "2024", "document_type": "markdown", "section": "Intro"},
    )
    results = [c1, c2, c3]

    # Filter by source
    filtered_src = MetadataFilter.apply(results, {"source": "paper-v1.pdf"})
    assert len(filtered_src) == 1
    assert filtered_src[0].chunk_id == "c1"

    # Filter by version
    filtered_ver = MetadataFilter.apply(results, {"document_version": "2024"})
    assert len(filtered_ver) == 2
    assert {c.chunk_id for c in filtered_ver} == {"c2", "c3"}

    # Filter by section
    filtered_sec = MetadataFilter.apply(results, {"section": "Intro"})
    assert len(filtered_sec) == 2
    assert {c.chunk_id for c in filtered_sec} == {"c1", "c3"}

    # Filter by list of sources
    filtered_multi = MetadataFilter.apply(results, {"source": ["paper-v1.pdf", "notes.md"]})
    assert len(filtered_multi) == 2
