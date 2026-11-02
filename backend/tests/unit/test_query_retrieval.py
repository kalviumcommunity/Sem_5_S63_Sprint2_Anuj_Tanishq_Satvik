"""Tests for embedding-backed retrieval in the query API."""

import pytest

from backend.app.api.v1 import query as query_api
from backend.app.schemas.query import QueryRequest
from backend.app.schemas.response import QueryResponse
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
    assert observed["search"]["filters"] == {"course": "CS"}
    assert observed["answer_candidates"] == [candidate]
