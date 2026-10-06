"""Data structures and configuration types for LLM completions."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from backend.app.services.llm.cost import TokenUsage


class LLMConfig(BaseModel):
    """Configuration parameters for LLM client execution."""

    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    api_key: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    """Standardized response object returned by all LLM providers."""

    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    token_usage: Optional[TokenUsage] = None
    finish_reason: Optional[str] = "stop"
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
