"""Unit tests for configurable tokenization, message token counting, and cost estimation."""

from unittest.mock import patch
import pytest

from backend.app.models.conversation import Message
from backend.app.services.tokenizer import (
    TokenizerConfig,
    TokenizerService,
    get_tokenizer,
    default_tokenizer,
)
from backend.app.services.llm.cost import (
    CostBreakdown,
    TokenCostEstimator,
    TokenUsage,
    token_cost_estimator,
)


def test_tokenizer_empty_and_whitespace_text():
    """Empty and whitespace strings should result in 0 tokens without error."""
    tokenizer = TokenizerService()
    assert tokenizer.count_tokens("") == 0
    assert tokenizer.count_tokens("   ") == 0
    assert tokenizer.count_tokens("\n\t  \r") == 0
    assert tokenizer.count_tokens(None) == 0


def test_tokenizer_different_models_and_providers():
    """Tokenizer produces deterministic, non-zero token counts across model families."""
    text = "Retrieval-augmented generation grounds language model responses in verified documents."

    openai_tokens = default_tokenizer.count_tokens(text, model="gpt-4o-mini", provider="openai")
    gemini_tokens = default_tokenizer.count_tokens(text, model="gemini-1.5-flash", provider="gemini")
    claude_tokens = default_tokenizer.count_tokens(text, model="claude-3-5-sonnet", provider="anthropic")
    mock_tokens = default_tokenizer.count_tokens(text, model="mock", provider="mock")

    assert openai_tokens > 0
    assert gemini_tokens > 0
    assert claude_tokens > 0
    assert mock_tokens > 0


def test_tokenizer_custom_char_ratio_override():
    """Custom character-to-token ratio configuration should override default heuristics."""
    config = TokenizerConfig(chars_per_token_ratio=4.0)
    tokenizer = TokenizerService(config=config)

    text = "12345678901234567890"  # 20 chars
    # 20 / 4.0 = 5 tokens
    assert tokenizer.count_tokens(text) == 5


def test_count_single_message_tokens():
    """Single message token calculation includes content, role, and standard framing tokens."""
    tokenizer = TokenizerService()
    msg = Message(conversation_id="conv-1", role="user", content="Explain diffusion models.")

    tokens = tokenizer.count_single_message_tokens(msg)
    content_tokens = tokenizer.count_tokens(msg.content)
    # Must be greater than pure content tokens due to role and framing overhead (+3 framing + role)
    assert tokens > content_tokens
    assert tokens >= content_tokens + 4


def test_count_message_tokens_sequence():
    """Sequence of chat messages includes individual message overhead and assistant priming tokens."""
    tokenizer = TokenizerService()
    messages = [
        Message(conversation_id="conv-1", role="system", content="You are an academic research assistant."),
        Message(conversation_id="conv-1", role="user", content="Summarize the core contribution."),
    ]

    tokens = tokenizer.count_message_tokens(messages)
    m1_tokens = tokenizer.count_single_message_tokens(messages[0])
    m2_tokens = tokenizer.count_single_message_tokens(messages[1])

    # Total should equal sum of message tokens plus 3 assistant reply priming tokens
    assert tokens == m1_tokens + m2_tokens + 3

    # Empty list returns 0
    assert tokenizer.count_message_tokens([]) == 0


def test_estimate_usage_separates_prompt_and_completion():
    """Estimate usage should provide prompt, completion, total tokens, and cost separately."""
    tokenizer = TokenizerService()
    prompt = "What is the primary objective of contrastive learning?"
    completion = "The objective is to pull positive pairs closer and push negative pairs apart."

    usage = tokenizer.estimate_usage(
        prompt=prompt,
        completion=completion,
        model="gpt-4o-mini",
        provider="openai",
    )

    assert isinstance(usage, TokenUsage)
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.is_estimated is True
    assert usage.usage_type == "estimated"
    assert usage.estimated_cost_usd > 0.0

    d = usage.to_dict()
    assert d["is_estimated"] is True
    assert d["usage_type"] == "estimated"


def test_estimate_usage_with_max_completion_tokens():
    """When completion text is not yet generated, max_completion_tokens acts as budget estimate."""
    tokenizer = TokenizerService()
    usage = tokenizer.estimate_usage(
        prompt="Explain attention mechanism.",
        max_completion_tokens=500,
        model="gpt-4o-mini",
        provider="openai",
    )

    assert usage.prompt_tokens > 0
    assert usage.completion_tokens == 500
    assert usage.total_tokens == usage.prompt_tokens + 500
    assert usage.is_estimated is True


def test_estimate_embedding_usage_single_and_batch():
    """Estimate embedding usage calculates input tokens and model pricing with zero completion tokens."""
    tokenizer = TokenizerService()
    texts = [
        "First document paragraph describing methodology.",
        "Second document paragraph detailing benchmark results.",
    ]

    usage = tokenizer.estimate_embedding_usage(texts, model="text-embedding-3-small")
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == usage.prompt_tokens
    assert usage.is_estimated is True
    assert usage.estimated_cost_usd > 0.0


def test_calculate_embedding_cost_exact():
    """Embedding cost should calculate accurately based on published rates ($0.02/1M for text-embedding-3-small)."""
    estimator = TokenCostEstimator()
    # 1,000,000 tokens @ $0.02 / 1M = $0.02
    cost_1m = estimator.calculate_embedding_cost(1_000_000, model="text-embedding-3-small")
    assert abs(cost_1m - 0.02) < 1e-8

    # 50,000 tokens @ $0.02 / 1M = $0.001
    cost_50k = estimator.calculate_embedding_cost(50_000, model="text-embedding-3-small")
    assert abs(cost_50k - 0.001) < 1e-8

    # Zero tokens = $0.0
    assert estimator.calculate_embedding_cost(0, model="text-embedding-3-small") == 0.0


def test_calculate_cost_breakdown():
    """Cost breakdown provides separate prompt, completion, embedding, and total items."""
    estimator = TokenCostEstimator()
    breakdown = estimator.calculate_breakdown(
        prompt_tokens=1000,
        completion_tokens=500,
        embedding_tokens=2000,
        model="gpt-4o-mini",
        provider="openai",
    )

    assert isinstance(breakdown, CostBreakdown)
    assert breakdown.prompt_cost_usd > 0
    assert breakdown.completion_cost_usd > 0
    assert breakdown.embedding_cost_usd > 0
    assert abs(breakdown.total_cost_usd - (breakdown.prompt_cost_usd + breakdown.completion_cost_usd + breakdown.embedding_cost_usd)) < 1e-7
    assert breakdown.is_approximate is False

    d = breakdown.to_dict()
    assert "prompt_cost_usd" in d
    assert "completion_cost_usd" in d
    assert "embedding_cost_usd" in d
    assert "total_cost_usd" in d


def test_distinguish_estimated_vs_actual_usage():
    """TokenUsage must explicitly differentiate between estimated and API-reported usage."""
    estimator = TokenCostEstimator()

    # Actual usage reported by API
    actual_usage = estimator.create_actual_usage(
        prompt_tokens=150,
        completion_tokens=75,
        model="gpt-4o-mini",
        provider="openai",
    )
    assert actual_usage.is_estimated is False
    assert actual_usage.usage_type == "actual"
    assert actual_usage.to_dict()["is_estimated"] is False
    assert actual_usage.to_dict()["usage_type"] == "actual"

    # Locally estimated usage
    estimated_usage = estimator.create_estimated_usage(
        prompt_tokens=150,
        completion_tokens=75,
        model="gpt-4o-mini",
        provider="openai",
    )
    assert estimated_usage.is_estimated is True
    assert estimated_usage.usage_type == "estimated"
    assert estimated_usage.to_dict()["is_estimated"] is True
    assert estimated_usage.to_dict()["usage_type"] == "estimated"


def test_unknown_models_and_missing_prices():
    """Unknown models and missing prices should use fallback rates gracefully without raising exceptions."""
    estimator = TokenCostEstimator()
    tokenizer = TokenizerService()

    # Tokenizer handles unknown model
    tokens = tokenizer.count_tokens("Academic paper abstract content.", model="unknown-academic-model-v999")
    assert tokens > 0

    # Pricing query handles unknown model
    assert estimator.has_pricing_for("unknown-academic-model-v999") is False

    # Cost calculation uses fallback rates gracefully
    cost = estimator.calculate_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        model="unknown-academic-model-v999",
        provider="custom-corp",
    )
    assert cost > 0.0


def test_register_custom_model_pricing():
    """Users can dynamically register or override pricing rates for proprietary models."""
    estimator = TokenCostEstimator()
    estimator.register_pricing(
        model="university-llama-3-8b",
        input_rate_per_million=0.20,
        output_rate_per_million=0.80,
    )

    assert estimator.has_pricing_for("university-llama-3-8b") is True

    cost = estimator.calculate_cost(
        prompt_tokens=10_000,
        completion_tokens=5_000,
        model="university-llama-3-8b",
    )
    # 10k * 0.20 / 1M = 0.002; 5k * 0.80 / 1M = 0.004 => 0.006
    assert abs(cost - 0.006) < 1e-8


def test_tokenizer_failure_graceful_fallback():
    """When a tokenizer library fails or raises an error, fallback heuristic catches it safely."""
    import sys
    from unittest.mock import MagicMock

    tokenizer = TokenizerService()
    mock_tiktoken = MagicMock()
    mock_tiktoken.encoding_for_model.side_effect = RuntimeError("Corrupt tokenizer index")

    with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
        tokens = tokenizer.count_tokens("Semantic search over scholarly articles.", model="gpt-4o-mini")
        assert tokens > 0


def test_factory_get_tokenizer():
    """get_tokenizer factory returns configured instance."""
    tok = get_tokenizer(model="gemini-1.5-pro", provider="gemini")
    assert tok.config.default_model == "gemini-1.5-pro"
    assert tok.config.default_provider == "gemini"
