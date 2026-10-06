"""Data structures and configuration types for LLM completions."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.services.llm.cost import TokenUsage


class LLMConfig(BaseModel):
    """Centralized configuration parameters for LLM client execution."""

    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    seed: Optional[int] = 42
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    response_format: str = Field(default="text", description="'text' or 'json_object'")
    max_retries: int = Field(default=2, ge=0)
    stop_sequences: Optional[List[str]] = None
    api_key: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_settings(cls, custom_settings=None) -> "LLMConfig":
        """Instantiate LLMConfig dynamically from environment settings."""
        if custom_settings is None:
            from backend.app.core.config import settings
            cfg_source = settings
        else:
            cfg_source = custom_settings

        return cls(
            model=getattr(cfg_source, "LLM_MODEL", "gpt-4o-mini"),
            temperature=getattr(cfg_source, "LLM_TEMPERATURE", 0.1),
            max_tokens=getattr(cfg_source, "LLM_MAX_TOKENS", 1024),
            top_p=getattr(cfg_source, "LLM_TOP_P", 1.0),
            seed=getattr(cfg_source, "LLM_SEED", 42),
            timeout_seconds=getattr(cfg_source, "LLM_TIMEOUT_SECONDS", 30.0),
            response_format=getattr(cfg_source, "LLM_RESPONSE_FORMAT", "text"),
            max_retries=getattr(cfg_source, "LLM_MAX_RETRIES", 2),
            api_key=getattr(cfg_source, "LLM_API_KEY", None),
        )

    @classmethod
    def for_environment(cls, environment: str) -> "LLMConfig":
        """Generate preset configurations tailored for specific environments."""
        env = environment.lower()
        if env == "production":
            return cls(
                model="gpt-4o-mini",
                temperature=0.0,  # Zero temperature for maximum factual consistency
                max_tokens=1024,
                top_p=1.0,
                seed=42,
                timeout_seconds=25.0,
                response_format="text",
                max_retries=3,
            )
        elif env in ("testing", "test"):
            return cls(
                model="mock-academic-v1",
                temperature=0.0,
                max_tokens=256,
                top_p=1.0,
                seed=123,
                timeout_seconds=5.0,
                response_format="text",
                max_retries=0,
            )
        elif env == "staging":
            return cls(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1024,
                top_p=1.0,
                seed=42,
                timeout_seconds=30.0,
                response_format="text",
                max_retries=2,
            )
        else:  # development default
            return cls.from_settings()


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
