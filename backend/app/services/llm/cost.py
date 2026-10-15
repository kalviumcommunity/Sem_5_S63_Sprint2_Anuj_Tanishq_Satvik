"""Provider-aware token accounting and cost estimation service.

Supports tracking input/output/embedding tokens, calculating dollar costs across
multiple model providers (OpenAI, Gemini, Anthropic, Mock), and distinguishing
estimated usage from actual API-reported usage.
"""

import math
import re
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from backend.app.core.logging import logger


class TokenUsage(BaseModel):
    """Structured record of token consumption and associated cost."""

    prompt_tokens: int = Field(default=0, ge=0, description="Tokens consumed by the prompt/input")
    completion_tokens: int = Field(default=0, ge=0, description="Tokens generated in the completion/output")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed (prompt + completion)")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Calculated cost in USD")
    model: str = Field(default="unknown", description="Model identifier used for generation or estimation")
    provider: str = Field(default="unknown", description="Provider identifier (e.g., openai, gemini, mock)")
    is_estimated: bool = Field(
        default=True,
        description="True if tokens are locally estimated; False if reported by official API usage",
    )
    usage_type: str = Field(
        default="estimated",
        description="Descriptor for token source: 'estimated' or 'actual'",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize usage to dictionary format."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "model": self.model,
            "provider": self.provider,
            "is_estimated": self.is_estimated,
            "usage_type": self.usage_type,
        }


class CostBreakdown(BaseModel):
    """Structured breakdown of input, output, embedding, and total costs in USD."""

    prompt_cost_usd: float = Field(default=0.0, ge=0.0, description="Cost for prompt tokens")
    completion_cost_usd: float = Field(default=0.0, ge=0.0, description="Cost for completion tokens")
    embedding_cost_usd: float = Field(default=0.0, ge=0.0, description="Cost for embedding tokens")
    total_cost_usd: float = Field(default=0.0, ge=0.0, description="Total aggregated cost")
    model: str = Field(default="unknown", description="Model identifier")
    provider: str = Field(default="unknown", description="Provider identifier")
    is_approximate: bool = Field(
        default=False,
        description="True if fallback or approximate pricing was used due to missing rates",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize cost breakdown to dictionary format."""
        return {
            "prompt_cost_usd": round(self.prompt_cost_usd, 8),
            "completion_cost_usd": round(self.completion_cost_usd, 8),
            "embedding_cost_usd": round(self.embedding_cost_usd, 8),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "model": self.model,
            "provider": self.provider,
            "is_approximate": self.is_approximate,
        }


# Model Pricing Rates: (input_cost_per_million, output_cost_per_million) in USD
# For embedding models, the first value is the embedding rate per 1M tokens.
MODEL_PRICING_PER_MILLION: Dict[str, Tuple[float, float]] = {
    # OpenAI Chat Models
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # OpenAI Embedding Models
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
    # Google Gemini Models
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "text-embedding-004": (0.00, 0.0),
    # Anthropic Claude Models
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    # Mock / Local
    "mock": (0.0, 0.0),
    "mock-v1": (0.0, 0.0),
    "mock-academic-v1": (0.0, 0.0),
}

DEFAULT_FALLBACK_RATES: Tuple[float, float] = (0.50, 1.50)
DEFAULT_EMBEDDING_FALLBACK_RATE: float = 0.02


class TokenCostEstimator:
    """Provider-aware token counter and dollar cost estimator."""

    def __init__(
        self,
        pricing_table: Optional[Dict[str, Tuple[float, float]]] = None,
        fallback_rates: Tuple[float, float] = DEFAULT_FALLBACK_RATES,
    ):
        self.pricing_table: Dict[str, Tuple[float, float]] = dict(
            pricing_table if pricing_table is not None else MODEL_PRICING_PER_MILLION
        )
        self.fallback_rates = fallback_rates

    def register_pricing(
        self,
        model: str,
        input_rate_per_million: float,
        output_rate_per_million: float = 0.0,
    ) -> None:
        """Register or override pricing for a specific model."""
        self.pricing_table[model.lower()] = (float(input_rate_per_million), float(output_rate_per_million))
        logger.debug(
            "Registered pricing for model '%s': $%.4f input, $%.4f output per 1M tokens",
            model,
            input_rate_per_million,
            output_rate_per_million,
        )

    def has_pricing_for(self, model: str) -> bool:
        """Check if pricing rates are explicitly configured for a model."""
        clean_model = model.lower()
        if clean_model in self.pricing_table:
            return True
        return any(known_model in clean_model for known_model in self.pricing_table)

    def get_model_pricing(self, model: str) -> Optional[Tuple[float, float]]:
        """Retrieve configured pricing tuple (input_rate, output_rate) for a model."""
        clean_model = model.lower()
        if clean_model in self.pricing_table:
            return self.pricing_table[clean_model]
        for known_model, rates in self.pricing_table.items():
            if known_model in clean_model:
                return rates
        return None

    def estimate_tokens(self, text: Optional[str], model: str = "gpt-4o-mini", provider: str = "openai") -> int:
        """Estimate token count for a text string using provider-aware logic."""
        if not text or not text.strip():
            return 0

        target_provider = provider.lower()
        clean_model = model.lower()

        # 1. Try tiktoken for OpenAI if available
        if target_provider == "openai" or clean_model.startswith(("gpt-", "text-embedding-")):
            try:
                import tiktoken

                try:
                    encoding = tiktoken.encoding_for_model(clean_model)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception as exc:
                logger.debug("Tiktoken unavailable for '%s' (%s). Using heuristic.", clean_model, exc)

        # 2. Gemini token heuristic (~4 characters per token or ~0.75 words)
        if target_provider in ("gemini", "google") or "gemini" in clean_model:
            char_count = len(text)
            word_count = len(text.split())
            return max(1, int(math.ceil((char_count / 4.0 + word_count / 0.75) / 2.0)))

        # 3. Claude heuristic
        if target_provider in ("anthropic", "claude") or "claude" in clean_model:
            return max(1, int(math.ceil(len(text) / 3.6)))

        # 4. Universal heuristic fallback: approx 1.2 tokens per word + punctuation
        words = re.findall(r"\S+", text)
        punctuation = len(re.findall(r"[.,!?;:()\[\]{}\"\']", text))
        return max(1, int(len(words) * 1.2 + punctuation * 0.3))

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int = 0,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ) -> float:
        """Calculate total dollar cost for prompt and completion tokens."""
        if prompt_tokens <= 0 and completion_tokens <= 0:
            return 0.0

        clean_model = model.lower()
        rates = self.get_model_pricing(clean_model)

        if not rates:
            if provider.lower() in ("mock", "local"):
                rates = (0.0, 0.0)
            else:
                logger.warning(
                    "Missing pricing for model '%s'. Applying fallback rates $%.2f / $%.2f per 1M tokens.",
                    model,
                    self.fallback_rates[0],
                    self.fallback_rates[1],
                )
                rates = self.fallback_rates

        input_rate_per_million, output_rate_per_million = rates
        cost = (prompt_tokens * input_rate_per_million / 1_000_000.0) + (
            completion_tokens * output_rate_per_million / 1_000_000.0
        )
        return round(cost, 8)

    def calculate_embedding_cost(
        self,
        tokens: int,
        model: str = "text-embedding-3-small",
        provider: str = "openai",
    ) -> float:
        """Calculate dollar cost for generating vector embeddings based on token count."""
        if tokens <= 0:
            return 0.0

        clean_model = model.lower()
        rates = self.get_model_pricing(clean_model)

        if not rates:
            if provider.lower() in ("mock", "local"):
                rate_per_million = 0.0
            else:
                logger.warning(
                    "Missing pricing for embedding model '%s'. Applying fallback embedding rate $%.4f per 1M tokens.",
                    model,
                    DEFAULT_EMBEDDING_FALLBACK_RATE,
                )
                rate_per_million = DEFAULT_EMBEDDING_FALLBACK_RATE
        else:
            rate_per_million = rates[0]

        cost = tokens * rate_per_million / 1_000_000.0
        return round(cost, 8)

    def calculate_breakdown(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        embedding_tokens: int = 0,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ) -> CostBreakdown:
        """Calculate itemized cost breakdown across prompt, completion, and embedding tokens."""
        rates = self.get_model_pricing(model)
        is_approx = rates is None and provider.lower() not in ("mock", "local")

        prompt_cost = self.calculate_cost(prompt_tokens=prompt_tokens, completion_tokens=0, model=model, provider=provider)
        completion_cost = self.calculate_cost(prompt_tokens=0, completion_tokens=completion_tokens, model=model, provider=provider)
        embedding_cost = self.calculate_embedding_cost(tokens=embedding_tokens, model=model, provider=provider)
        total_cost = round(prompt_cost + completion_cost + embedding_cost, 8)

        return CostBreakdown(
            prompt_cost_usd=prompt_cost,
            completion_cost_usd=completion_cost,
            embedding_cost_usd=embedding_cost,
            total_cost_usd=total_cost,
            model=model,
            provider=provider,
            is_approximate=is_approx,
        )

    def create_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        provider: str = "openai",
        is_estimated: bool = True,
        usage_type: Optional[str] = None,
    ) -> TokenUsage:
        """Construct a TokenUsage record with computed total and cost."""
        cost = self.calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            provider=provider,
        )
        resolved_usage_type = usage_type or ("estimated" if is_estimated else "actual")
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=cost,
            model=model,
            provider=provider,
            is_estimated=is_estimated,
            usage_type=resolved_usage_type,
        )

    def create_actual_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        provider: str = "openai",
    ) -> TokenUsage:
        """Construct an actual API-reported TokenUsage record (is_estimated=False)."""
        return self.create_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            provider=provider,
            is_estimated=False,
            usage_type="actual",
        )

    def create_estimated_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        provider: str = "openai",
    ) -> TokenUsage:
        """Construct an estimated TokenUsage record (is_estimated=True)."""
        return self.create_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            provider=provider,
            is_estimated=True,
            usage_type="estimated",
        )


# Global singleton instance
token_cost_estimator = TokenCostEstimator()
