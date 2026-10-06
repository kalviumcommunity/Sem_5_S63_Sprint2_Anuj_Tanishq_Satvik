"""Unit tests for token accounting and cost estimation."""

import pytest
from backend.app.services.llm.cost import (
    TokenCostEstimator,
    TokenUsage,
    token_cost_estimator,
)
from backend.app.services.llm import (
    LLMConfig,
    MockLLMClient,
)
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.services.retrieval.search import SearchResult


def test_token_estimation_across_providers():
    text = "Federated learning enables privacy-preserving collaborative model training."

    # OpenAI estimation
    openai_tokens = token_cost_estimator.estimate_tokens(text, model="gpt-4o-mini", provider="openai")
    assert openai_tokens > 0

    # Gemini estimation
    gemini_tokens = token_cost_estimator.estimate_tokens(text, model="gemini-1.5-flash", provider="gemini")
    assert gemini_tokens > 0

    # Generic / mock estimation
    mock_tokens = token_cost_estimator.estimate_tokens(text, model="mock", provider="mock")
    assert mock_tokens > 0

    # Empty text
    assert token_cost_estimator.estimate_tokens("", provider="openai") == 0


def test_cost_calculation_gpt4o_mini():
    estimator = TokenCostEstimator()
    # gpt-4o-mini: 1000 prompt tokens @ $0.15/1M ($0.00015), 500 comp tokens @ $0.60/1M ($0.00030) => $0.00045
    cost = estimator.calculate_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        model="gpt-4o-mini",
        provider="openai",
    )
    expected = (1000 * 0.15 / 1_000_000.0) + (500 * 0.60 / 1_000_000.0)
    assert abs(cost - expected) < 1e-7


def test_cost_calculation_gemini():
    estimator = TokenCostEstimator()
    # gemini-1.5-flash: 1000 prompt tokens @ $0.075/1M, 500 comp tokens @ $0.30/1M
    cost = estimator.calculate_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        model="gemini-1.5-flash",
        provider="gemini",
    )
    expected = (1000 * 0.075 / 1_000_000.0) + (500 * 0.30 / 1_000_000.0)
    assert abs(cost - expected) < 1e-7


def test_cost_calculation_mock():
    estimator = TokenCostEstimator()
    cost = estimator.calculate_cost(
        prompt_tokens=5000,
        completion_tokens=2000,
        model="mock-v1",
        provider="mock",
    )
    assert cost == 0.0


def test_token_usage_model():
    usage = token_cost_estimator.create_usage(
        prompt_tokens=200,
        completion_tokens=100,
        model="gpt-4o-mini",
        provider="openai",
    )
    assert isinstance(usage, TokenUsage)
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens == 100
    assert usage.total_tokens == 300
    assert usage.estimated_cost_usd > 0.0

    d = usage.to_dict()
    assert d["prompt_tokens"] == 200
    assert d["completion_tokens"] == 100
    assert d["total_tokens"] == 300
    assert "estimated_cost_usd" in d


@pytest.mark.asyncio
async def test_llm_client_token_cost_integration():
    client = MockLLMClient()
    response = await client.complete("Explain transfer learning in computer vision.")

    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens
    assert response.estimated_cost_usd >= 0.0
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == response.total_tokens


@pytest.mark.asyncio
async def test_rag_pipeline_propagates_token_usage():
    pipeline = RAGPipeline()
    candidate = SearchResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Deep Learning",
        text="Convolutional networks extract spatial hierarchies.",
        page_number=1,
        score=0.9,
    )

    query_response = await pipeline.answer(
        query="What do convolutional networks extract?",
        candidates=[candidate],
    )

    assert query_response.grounded is True
    assert query_response.token_usage is not None
    assert "prompt_tokens" in query_response.token_usage
    assert "completion_tokens" in query_response.token_usage
    assert "total_tokens" in query_response.token_usage
    assert query_response.estimated_cost_usd is not None
