"""Configurable tokenization service for ResearchMate.

Provides model-appropriate token counting and estimation for:
- Raw text strings
- Individual chat messages with role framing overhead
- Multi-message conversation histories and prompt bundles
- Prompt, completion, and embedding token usage estimation

Supports graceful fallback when offline or when model-specific tokenizer
libraries are unavailable.
"""

import math
import re 
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, Field

from backend.app.core.logging import logger
from backend.app.models.conversation import Message
from backend.app.services.llm.cost import (
    TokenCostEstimator,
    TokenUsage,
    token_cost_estimator,
)


class TokenizerConfig(BaseModel):
    """Configuration for tokenization behavior."""

    default_model: str = Field(default="gpt-4o-mini", description="Default model for tokenization")
    default_provider: str = Field(default="openai", description="Default model provider")
    chars_per_token_ratio: Optional[float] = Field(
        default=None,
        description="Optional fixed characters-per-token ratio override (e.g., 4.0)",
    )
    fallback_to_heuristic: bool = Field(
        default=True,
        description="Whether to fall back to heuristic estimation on tokenizer failure",
    )
    enable_tiktoken: bool = Field(
        default=True,
        description="Whether to attempt using tiktoken for OpenAI models",
    )


class TokenizerService:
    """Configurable tokenization service supporting multiple model families."""

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        cost_estimator: Optional[TokenCostEstimator] = None,
    ):
        self.config = config or TokenizerConfig()
        self.cost_estimator = cost_estimator or token_cost_estimator

    def count_tokens(
        self,
        text: Optional[str],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> int:
        """Count or estimate tokens in a raw text string.

        Handles empty text, unknown models, and tokenizer library failures
        gracefully with deterministic fallback heuristics.
        """
        if not text or not text.strip():
            return 0

        target_model = (model or self.config.default_model).lower()
        target_provider = (provider or self.config.default_provider).lower()

        # Custom character ratio override if configured
        if self.config.chars_per_token_ratio and self.config.chars_per_token_ratio > 0:
            return max(1, int(math.ceil(len(text) / self.config.chars_per_token_ratio)))

        # 1. OpenAI / Tiktoken
        if target_provider == "openai" or target_model.startswith(("gpt-", "text-embedding-")):
            if self.config.enable_tiktoken:
                try:
                    import tiktoken

                    try:
                        encoding = tiktoken.encoding_for_model(target_model)
                    except KeyError:
                        encoding = tiktoken.get_encoding("cl100k_base")
                    return len(encoding.encode(text))
                except Exception as exc:
                    if not self.config.fallback_to_heuristic:
                        raise
                    logger.debug(
                        "Tiktoken unavailable or failed for model '%s' (%s). Using deterministic heuristic.",
                        target_model,
                        exc,
                    )

            # Heuristic for OpenAI-like tokenizers: ~1.25 tokens/word + punctuation
            return self._heuristic_openai_tokens(text)

        # 2. Google Gemini
        if target_provider in ("gemini", "google") or "gemini" in target_model:
            char_count = len(text)
            word_count = len(text.split())
            return max(1, int(math.ceil((char_count / 4.0 + word_count / 0.75) / 2.0)))

        # 3. Anthropic Claude
        if target_provider in ("anthropic", "claude") or "claude" in target_model:
            return max(1, int(math.ceil(len(text) / 3.6)))

        # 4. Mock / Local / Generic Fallback
        return self._heuristic_universal_tokens(text)

    def _heuristic_openai_tokens(self, text: str) -> int:
        """Deterministic heuristic for OpenAI-style token counting."""
        words = re.findall(r"\S+", text)
        punctuation = len(re.findall(r"[.,!?;:()\[\]{}\"\']", text))
        return max(1, int(len(words) * 1.25 + punctuation * 0.35))

    def _heuristic_universal_tokens(self, text: str) -> int:
        """Universal deterministic token heuristic based on whitespace and punctuation."""
        words = re.findall(r"\S+", text)
        punctuation = len(re.findall(r"[.,!?;:()\[\]{}\"\']", text))
        return max(1, int(len(words) * 1.2 + punctuation * 0.3))

    def count_single_message_tokens(
        self,
        message: Union[Message, Dict[str, Any], str],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> int:
        """Count tokens for a single chat message including role and framing overhead."""
        if isinstance(message, str):
            return self.count_tokens(message, model=model, provider=provider)

        if isinstance(message, Message):
            role = message.role
            content = message.content or ""
        elif isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "") or ""
        else:
            content = str(message)
            role = "user"

        content_tokens = self.count_tokens(content, model=model, provider=provider) if content.strip() else 0
        role_tokens = self.count_tokens(role, model=model, provider=provider) if role else 1
        framing_tokens = 3  # Standard ChatML delimiter overhead: <|im_start|>{role}\n{content}<|im_end|>\n

        return role_tokens + content_tokens + framing_tokens

    def count_message_tokens(
        self,
        messages: Sequence[Union[Message, Dict[str, Any], str]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> int:
        """Count tokens across a sequence of chat messages including conversational framing."""
        if not messages:
            return 0

        total = sum(
            self.count_single_message_tokens(m, model=model, provider=provider)
            for m in messages
        )
        # Assistant priming tokens (<|im_start|>assistant<|message|>)
        assistant_priming_tokens = 3
        return total + assistant_priming_tokens

    def estimate_usage(
        self,
        prompt: Union[str, Sequence[Union[Message, Dict[str, Any], str]]],
        completion: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> TokenUsage:
        """Estimate prompt, completion, and total token usage separately with cost estimation.

        Always marks the resulting TokenUsage record as is_estimated=True.
        """
        target_model = model or self.config.default_model
        target_provider = provider or self.config.default_provider

        # 1. Prompt tokens
        if isinstance(prompt, (list, tuple)):
            prompt_tokens = self.count_message_tokens(prompt, model=target_model, provider=target_provider)
        else:
            prompt_tokens = self.count_tokens(str(prompt), model=target_model, provider=target_provider)

        # 2. Completion tokens
        if completion is not None:
            completion_tokens = self.count_tokens(completion, model=target_model, provider=target_provider)
        elif max_completion_tokens is not None and max_completion_tokens > 0:
            completion_tokens = int(max_completion_tokens)
        else:
            completion_tokens = 0

        # 3. Create TokenUsage record
        return self.cost_estimator.create_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=target_model,
            provider=target_provider,
            is_estimated=True,
            usage_type="estimated",
        )

    def estimate_embedding_usage(
        self,
        texts: Union[str, Sequence[str]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> TokenUsage:
        """Estimate embedding token consumption and dollar cost for single or batch inputs."""
        target_model = model or "text-embedding-3-small"
        target_provider = provider or "openai"

        if isinstance(texts, str):
            input_texts = [texts]
        else:
            input_texts = list(texts)

        total_tokens = sum(
            self.count_tokens(t, model=target_model, provider=target_provider)
            for t in input_texts
        )
        cost = self.cost_estimator.calculate_embedding_cost(
            tokens=total_tokens,
            model=target_model,
            provider=target_provider,
        )

        return TokenUsage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            model=target_model,
            provider=target_provider,
            is_estimated=True,
            usage_type="estimated",
        )


# Global default instance and factory
default_tokenizer = TokenizerService()


def get_tokenizer(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    chars_per_token_ratio: Optional[float] = None,
) -> TokenizerService:
    """Factory function for retrieving a configured TokenizerService."""
    config = TokenizerConfig(
        default_model=model or "gpt-4o-mini",
        default_provider=provider or "openai",
        chars_per_token_ratio=chars_per_token_ratio,
    )
    return TokenizerService(config=config)
