"""Integration tests for end-to-end metadata-aware RAG pipeline flow."""

import pytest
from backend.app.repositories.chunks import ChunkRepository
from backend.app.services.documents.ingestion import DocumentIngestionPipeline
from backend.app.services.embeddings.service import MockEmbeddingService
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.services.retrieval.reranker import Reranker


@pytest.mark.asyncio
async def test_end_to_end_metadata_ingestion_and_answering():
    # Setup repository and services
    repo = ChunkRepository()
    embedding_service = MockEmbeddingService(dimension=64)
    pipeline = DocumentIngestionPipeline(
        embedding_service=embedding_service,
        chunk_repository=repo,
    )

    # Ingest document with metadata
    document_content = (
        "ResearchMate provides source citation tracking.\f"
        "Students can verify claims using original page numbers and section references."
    )
    chunks = await pipeline.ingest_document(
        text=document_content,
        document_id="doc-researchmate-intro",
        source="researchmate-architecture.pdf",
        url="https://researchmate.org/docs/architecture",
        document_version="1.0",
        title="ResearchMate Architecture",
    )
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[0].source == "researchmate-architecture.pdf"
    assert chunks[1].source == "researchmate-architecture.pdf"

    # Vector search retrieval
    q_vec = await embedding_service.embed_text("source citation tracking")
    candidates = repo.search_vector(query_vector=q_vec, top_k=5)
    assert len(candidates) == 2

    # Check candidate metadata preservation and subscript access
    top_cand = candidates[0]
    assert top_cand["metadata"]["source"] == "researchmate-architecture.pdf"
    assert top_cand["metadata"]["url"] == "https://researchmate.org/docs/architecture"
    assert top_cand.chunk_id == chunks[0].id or top_cand.chunk_id == chunks[1].id

    # Execute full RAG pipeline answering
    rag = RAGPipeline()
    response = await rag.answer(
        query="How does citation tracking work in ResearchMate?",
        candidates=candidates,
    )

    assert response.grounded is True
    assert len(response.citations) > 0
    assert len(response.source_references) > 0

    first_citation = response.citations[0]
    assert first_citation.source == "researchmate-architecture.pdf"
    assert first_citation.url == "https://researchmate.org/docs/architecture"
    assert "https://researchmate.org/docs/architecture" in first_citation.citation_text
    assert "researchmate-architecture.pdf" in first_citation.citation_text

    first_ref = response.source_references[0]
    assert first_ref.source == "researchmate-architecture.pdf"
    assert first_ref.url == "https://researchmate.org/docs/architecture"
