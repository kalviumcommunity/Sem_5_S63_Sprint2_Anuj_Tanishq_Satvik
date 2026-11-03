"""Tests for embedding-backed retrieval in the query API."""

import pytest
from fastapi import HTTPException

from backend.app.api.v1 import query as query_api
from backend.app.schemas.query import QueryRequest
from backend.app.schemas.response import QueryResponse
from backend.app.services.embeddings.service import EmbeddingServiceError
from backend.app.services.retrieval.search import SearchResult


@pytest.mark.asyncio
async def test_query_embeds_text_and_passes_filtered_vector_results_to_rag(monkeypatch):
    observed = {}
    candidate = SearchResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Neural Networks",
        text="Neural networks learn patterns.",
        page_number=2,
    )

    class EmbeddingService:
        dimension = 3

        async def embed_text(self, text):
            observed["query"] = text
            return [1.0, 0.0, 0.0]

    class VectorStore:
        def search(self, **kwargs):
            observed["search"] = kwargs
            return [candidate]

    async def answer(query, candidates, conversation_id):
        observed["answer_candidates"] = candidates
        return QueryResponse(answer="Retrieved answer")

    monkeypatch.setattr(query_api, "get_embedding_service", lambda: EmbeddingService())
    monkeypatch.setattr(query_api, "get_vector_store", lambda: VectorStore())
    monkeypatch.setattr(query_api.pipeline, "answer", answer)

    response = await query_api.query_academic_sources(
        QueryRequest(query="How do neural networks learn?", top_k=3, filters={"course": "CS"})
    )

    assert response.answer == "Retrieved answer"
    assert observed["query"] == "How do neural networks learn?"
    assert observed["search"]["query_vector"] == [1.0, 0.0, 0.0]
    assert observed["search"]["top_k"] == 3
    assert observed["search"]["score_threshold"] == query_api.settings.SIMILARITY_THRESHOLD
    assert observed["search"]["filters"] == {"course": "CS"}
    assert observed["answer_candidates"] == [candidate]


@pytest.mark.asyncio
async def test_query_hides_embedding_provider_error_details(monkeypatch):
    class EmbeddingService:
        dimension = 3

        async def embed_text(self, text):
            raise EmbeddingServiceError("provider response contained secret-token")

    monkeypatch.setattr(query_api, "get_embedding_service", lambda: EmbeddingService())

    with pytest.raises(HTTPException) as error:
        await query_api.query_academic_sources(QueryRequest(query="question"))

    assert error.value.status_code == 502
    assert error.value.detail == "Embedding provider request failed."
    assert "secret-token" not in error.value.detail


@pytest.mark.asyncio
async def test_mock_embeddings_use_offline_threshold_without_changing_provider_thresholds(monkeypatch):
    observed = {}

    class EmbeddingService:
        dimension = 3
        uses_mock_embeddings = True

        async def embed_text(self, text):
            return [1.0, 0.0, 0.0]

    class VectorStore:
        def search(self, **kwargs):
            observed["search"] = kwargs
            return []

    async def answer(query, candidates, conversation_id, grounding_threshold, mock_embeddings):
        observed["grounding_threshold"] = grounding_threshold
        observed["mock_embeddings"] = mock_embeddings
        return QueryResponse(answer="No matching evidence")

    monkeypatch.setattr(query_api, "get_embedding_service", lambda: EmbeddingService())
    monkeypatch.setattr(query_api, "get_vector_store", lambda: VectorStore())
    monkeypatch.setattr(query_api.pipeline, "answer", answer)

    await query_api.query_academic_sources(QueryRequest(query="Offline question"))

    assert observed["search"]["score_threshold"] == query_api.settings.MOCK_SIMILARITY_THRESHOLD
    assert observed["search"]["top_k"] == query_api.settings.MOCK_TOP_K_RETRIEVAL
    assert observed["grounding_threshold"] == query_api.settings.MOCK_SIMILARITY_THRESHOLD
    assert observed["mock_embeddings"] is True
