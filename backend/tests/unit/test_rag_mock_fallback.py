import pytest

from backend.app.services.llm import OpenAILLMClient
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.services.retrieval.search import SearchResult


@pytest.mark.asyncio
async def test_mock_llm_fallback_answers_with_retrieved_text_and_real_citation():
    pipeline = RAGPipeline()
    pipeline.llm = OpenAILLMClient(api_key="your-openai-api-key-here")
    candidate = SearchResult(
        chunk_id="chunk-rag-definition",
        document_id="doc-rag",
        document_title="Retrieval-Augmented Generation",
        text=(
            "We explore retrieval-augmented generation (RAG), a model that combines "
            "pre-trained parametric and non-parametric memory for language generation."
        ),
        page_number=2,
        score=0.3,
    )

    response = await pipeline.answer(
        query="What is Retrieval-Augmented Generation?",
        candidates=[candidate],
        grounding_threshold=0.15,
    )

    assert response.grounded
    assert "retrieval-augmented generation (RAG)" in response.answer
    assert "federated learning" not in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].document_id == "doc-rag"
    assert response.citations[0].page == 2
    assert response.citations[0].chunk_id == "chunk-rag-definition"
    assert response.citations[0].snippet == candidate.text
    assert response.citations[0].similarity_score == candidate.score
