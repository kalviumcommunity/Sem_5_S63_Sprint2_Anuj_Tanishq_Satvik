"""Data structures, configuration types, and parameter validation for LLM completions."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.services.llm.cost import TokenUsage
from backend.app.services.llm.exceptions import InvalidModelParameterError


def validate_model_parameters(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    response_format: Optional[str] = None,
    stop_sequences: Optional[List[str]] = None,
    timeout_seconds: Optional[float] = None,
    provider: Optional[str] = None,
) -> None:
    """Validate model parameters against supported ranges and reject incompatible combinations."""
    # 1. Temperature validation [0.0, 2.0]
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature < 0.0 or temperature > 2.0:
            raise InvalidModelParameterError(
                f"Invalid temperature '{temperature}'. Must be a float between 0.0 and 2.0.",
                parameter="temperature",
                value=temperature,
                provider=provider or "llm",
            )

    # 2. Maximum tokens validation (> 0)
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise InvalidModelParameterError(
                f"Invalid max_tokens '{max_tokens}'. Must be a positive integer greater than 0.",
                parameter="max_tokens",
                value=max_tokens,
                provider=provider or "llm",
            )

    # 3. Top-p nucleus sampling validation [0.0, 1.0]
    if top_p is not None:
        if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or top_p < 0.0 or top_p > 1.0:
            raise InvalidModelParameterError(
                f"Invalid top_p '{top_p}'. Must be a float between 0.0 and 1.0.",
                parameter="top_p",
                value=top_p,
                provider=provider or "llm",
            )

    # 4. Top-k sampling validation (>= 1)
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise InvalidModelParameterError(
                f"Invalid top_k '{top_k}'. Must be an integer >= 1.",
                parameter="top_k",
                value=top_k,
                provider=provider or "llm",
            )

    # 5. Presence penalty [-2.0, 2.0]
    if presence_penalty is not None:
        if (
            not isinstance(presence_penalty, (int, float))
            or isinstance(presence_penalty, bool)
            or presence_penalty < -2.0
            or presence_penalty > 2.0
        ):
            raise InvalidModelParameterError(
                f"Invalid presence_penalty '{presence_penalty}'. Must be between -2.0 and 2.0.",
                parameter="presence_penalty",
                value=presence_penalty,
                provider=provider or "llm",
            )

    # 6. Frequency penalty [-2.0, 2.0]
    if frequency_penalty is not None:
        if (
            not isinstance(frequency_penalty, (int, float))
            or isinstance(frequency_penalty, bool)
            or frequency_penalty < -2.0
            or frequency_penalty > 2.0
        ):
            raise InvalidModelParameterError(
                f"Invalid frequency_penalty '{frequency_penalty}'. Must be between -2.0 and 2.0.",
                parameter="frequency_penalty",
                value=frequency_penalty,
                provider=provider or "llm",
            )

    # 7. Response format
    if response_format is not None:
        if response_format not in ("text", "json_object"):
            raise InvalidModelParameterError(
                f"Unsupported response_format '{response_format}'. Supported formats: 'text', 'json_object'.",
                parameter="response_format",
                value=response_format,
                provider=provider or "llm",
            )

    # 8. Stop sequences
    if stop_sequences is not None:
        if not isinstance(stop_sequences, (list, tuple)):
            raise InvalidModelParameterError(
                "Stop sequences must be provided as a list or tuple of strings.",
                parameter="stop_sequences",
                value=stop_sequences,
                provider=provider or "llm",
            )
        if any(seq == "" for seq in stop_sequences):
            raise InvalidModelParameterError(
                "Stop sequences cannot contain empty strings.",
                parameter="stop_sequences",
                value=stop_sequences,
                provider=provider or "llm",
            )
        norm_prov = (provider or "").lower()
        if norm_prov == "openai" and len(stop_sequences) > 4:
            raise InvalidModelParameterError(
                f"OpenAI supports a maximum of 4 stop sequences, got {len(stop_sequences)}.",
                parameter="stop_sequences",
                value=stop_sequences,
                provider="openai",
            )
        if norm_prov in ("gemini", "google") and len(stop_sequences) > 5:
            raise InvalidModelParameterError(
                f"Gemini supports a maximum of 5 stop sequences, got {len(stop_sequences)}.",
                parameter="stop_sequences",
                value=stop_sequences,
                provider="gemini",
            )

    # 9. Timeout
    if timeout_seconds is not None:
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise InvalidModelParameterError(
                f"Invalid timeout_seconds '{timeout_seconds}'. Must be > 0.",
                parameter="timeout_seconds",
                value=timeout_seconds,
                provider=provider or "llm",
            )

    # 10. Incompatible parameter combinations:
    # Deterministic greedy decoding (temperature=0.0) combined with non-trivial nucleus cutoff (top_p < 1.0)
    if temperature is not None and temperature == 0.0 and top_p is not None and top_p < 1.0:
        raise InvalidModelParameterError(
            f"Incompatible parameter combination: temperature=0.0 (greedy argmax decoding) cannot be combined with top_p={top_p} (< 1.0). "
            "Set top_p=1.0 or temperature > 0.0.",
            parameter="top_p",
            value=top_p,
            provider=provider or "llm",
        )


class ProviderCapabilities(BaseModel):
    """Specification of capabilities, limits, and supported parameters for an LLM provider."""

    provider: str
    supported_models: List[str]
    supported_parameters: List[str]
    max_output_tokens: int
    max_stop_sequences: int
    supports_seed: bool = True
    supports_top_k: bool = False
    supports_penalties: bool = False
    supports_json_object: bool = True
    parameter_mappings: Dict[str, str] = Field(default_factory=dict)


PROVIDER_CAPABILITIES: Dict[str, ProviderCapabilities] = {
    "openai": ProviderCapabilities(
        provider="openai",
        supported_models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        supported_parameters=[
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            "seed",
            "response_format",
            "stop",
            "presence_penalty",
            "frequency_penalty",
        ],
        max_output_tokens=16384,
        max_stop_sequences=4,
        supports_seed=True,
        supports_top_k=False,
        supports_penalties=True,
        supports_json_object=True,
        parameter_mappings={"stop_sequences": "stop"},
    ),
    "gemini": ProviderCapabilities(
        provider="gemini",
        supported_models=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        supported_parameters=[
            "model",
            "temperature",
            "maxOutputTokens",
            "topP",
            "topK",
            "responseMimeType",
            "stopSequences",
        ],
        max_output_tokens=8192,
        max_stop_sequences=5,
        supports_seed=False,
        supports_top_k=True,
        supports_penalties=False,
        supports_json_object=True,
        parameter_mappings={
            "max_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
            "stop_sequences": "stopSequences",
            "stop": "stopSequences",
        },
    ),
    "mock": ProviderCapabilities(
        provider="mock",
        supported_models=["mock-academic-v1", "mock-v1"],
        supported_parameters=[
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "seed",
            "response_format",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
        ],
        max_output_tokens=4096,
        max_stop_sequences=5,
        supports_seed=True,
        supports_top_k=True,
        supports_penalties=True,
        supports_json_object=True,
        parameter_mappings={},
    ),
}


def get_provider_capabilities(provider: str) -> ProviderCapabilities:
    """Retrieve capability definitions for a given provider."""
    normalized = (provider or "mock").lower()
    if normalized in ("google", "gemini"):
        return PROVIDER_CAPABILITIES["gemini"]
    return PROVIDER_CAPABILITIES.get(normalized, PROVIDER_CAPABILITIES["mock"])


class LLMConfig(BaseModel):
    """Centralized configuration parameters for LLM client execution."""

    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    seed: Optional[int] = 42
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    response_format: str = Field(default="text", description="'text' or 'json_object'")
    max_retries: int = Field(default=2, ge=0)
    stop_sequences: Optional[List[str]] = None
    fallback_to_mock: bool = True
    api_key: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        # Validate parameters and reject out-of-range or incompatible values
        validate_model_parameters(
            model=data.get("model"),
            temperature=data.get("temperature", 0.1),
            max_tokens=data.get("max_tokens", 1024),
            top_p=data.get("top_p", 1.0),
            top_k=data.get("top_k"),
            presence_penalty=data.get("presence_penalty"),
            frequency_penalty=data.get("frequency_penalty"),
            response_format=data.get("response_format", "text"),
            stop_sequences=data.get("stop_sequences"),
            timeout_seconds=data.get("timeout_seconds", 30.0),
        )
        super().__init__(**data)

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
            top_k=getattr(cfg_source, "LLM_TOP_K", None),
            presence_penalty=getattr(cfg_source, "LLM_PRESENCE_PENALTY", None),
            frequency_penalty=getattr(cfg_source, "LLM_FREQUENCY_PENALTY", None),
            seed=getattr(cfg_source, "LLM_SEED", 42),
            timeout_seconds=getattr(cfg_source, "LLM_TIMEOUT_SECONDS", 30.0),
            response_format=getattr(cfg_source, "LLM_RESPONSE_FORMAT", "text"),
            max_retries=getattr(cfg_source, "LLM_MAX_RETRIES", 2),
            fallback_to_mock=getattr(cfg_source, "LLM_FALLBACK_TO_MOCK", True),
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

    @classmethod
    def for_academic_explanation(
        cls,
        model: Optional[str] = None,
        provider: str = "openai",
    ) -> "LLMConfig":
        """Sensible preset for concise, source-grounded academic explanations.

        - Low temperature (0.1) suppresses hallucinations and maintains strict grounding to cited sources.
        - Bound max_tokens (512) prevents verbose runaway explanations while reserving ample room for key findings.
        - Focused top_p (0.9) prunes low-probability tokens without hurting technical academic vocabulary.
        - Fixed seed (42) ensures reproducible academic evaluation results.
        """
        default_model = "gemini-1.5-flash" if provider in ("gemini", "google") else "gpt-4o-mini"
        return cls(
            model=model or default_model,
            temperature=0.1,
            max_tokens=512,
            top_p=0.9,
            seed=42 if provider not in ("gemini", "google") else None,
            timeout_seconds=30.0,
            response_format="text",
        )

    @classmethod
    def for_creative_exploration(
        cls,
        model: Optional[str] = None,
    ) -> "LLMConfig":
        """Preset for brainstormed research hypotheses or broad literature exploration."""
        return cls(
            model=model or "gpt-4o-mini",
            temperature=0.7,
            max_tokens=1500,
            top_p=0.95,
            seed=None,
            timeout_seconds=45.0,
            response_format="text",
        )

    @classmethod
    def for_deterministic_extraction(
        cls,
        model: Optional[str] = None,
    ) -> "LLMConfig":
        """Preset for structured data and metadata extraction with zero randomness."""
        return cls(
            model=model or "gpt-4o-mini",
            temperature=0.0,
            max_tokens=1024,
            top_p=1.0,
            seed=42,
            timeout_seconds=25.0,
            response_format="json_object",
        )


ACADEMIC_EXPLANATION_CONFIG = LLMConfig.for_academic_explanation()


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
