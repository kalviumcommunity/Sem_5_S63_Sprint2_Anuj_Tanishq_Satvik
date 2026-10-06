"""Provider-aware token accounting and cost estimation service.

Supports tracking input/output tokens and calculating dollar costs across
multiple model providers (OpenAI, Gemini, Anthropic, Mock).
"""

import math
import re
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field

from backend.app.core.logging import logger


class TokenUsage(BaseModel):
    """Structured record of token consumption and associated cost."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    model: str = "unknown"
    provider: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "model": self.model,
            "provider": self.provider,
        }


# Model Pricing Rates: (input_cost_per_million, output_cost_per_million) in USD
MODEL_PRICING_PER_MILLION: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # Google Gemini
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    # Anthropic Claude
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-5-sonnet": (3.00, 15.00),
    # Mock / Local
    "mock": (0.0, 0.0),
    "mock-v1": (0.0, 0.0),
    "mock-academic-v1": (0.0, 0.0),
}

DEFAULT_FALLBACK_RATES: Tuple[float, float] = (0.50, 1.50)


class TokenCostEstimator:
    """Provider-aware token counter and dollar cost estimator."""

    def __init__(self, pricing_table: Optional[Dict[str, Tuple[float, float]]] = None):
        self.pricing_table = pricing_table or MODEL_PRICING_PER_MILLION

    def estimate_tokens(self, text: str, model: str = "gpt-4o-mini", provider: str = "openai") -> int:
        """Estimate token count for a text string using provider-aware logic."""
        if not text:
            return 0

        target_provider = provider.lower()

        # 1. Try tiktoken for OpenAI if available
        if target_provider == "openai":
            try:
                import tiktoken

                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except ImportError:
                pass

        # 2. Gemini token heuristic (~4 characters per token or ~0.75 words)
        if target_provider in ("gemini", "google"):
            char_count = len(text)
            word_count = len(text.split())
            return max(1, int(math.ceil((char_count / 4.0 + word_count / 0.75) / 2.0)))

        # 3. Universal heuristic fallback: approx 1.3 tokens per whitespace word
        words = re.findall(r"\S+", text)
        punctuation = len(re.findall(r"[.,!?;:()\[\]{}\"\']", text))
        return max(1, int(len(words) * 1.2 + punctuation * 0.3))

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        provider: str = "openai",
    ) -> float:
        """Calculate total dollar cost for a request."""
        clean_model = model.lower()

        # Find best matching rate
        rates = self.pricing_table.get(clean_model)
        if not rates:
            for known_model, r in self.pricing_table.items():
                if known_model in clean_model:
                    rates = r
                    break
        if not rates:
            if provider.lower() == "mock":
                rates = (0.0, 0.0)
            else:
                rates = DEFAULT_FALLBACK_RATES

        input_rate_per_million, output_rate_per_million = rates
        cost = (prompt_tokens * input_rate_per_million / 1_000_000.0) + (
            completion_tokens * output_rate_per_million / 1_000_000.0
        )
        return round(cost, 8)

    def create_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        provider: str = "openai",
    ) -> TokenUsage:
        """Construct a TokenUsage record with computed total and cost."""
        cost = self.calculate_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
            provider=provider,
        )
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=cost,
            model=model,
            provider=provider,
        )


# Global singleton instance
token_cost_estimator = TokenCostEstimator()
