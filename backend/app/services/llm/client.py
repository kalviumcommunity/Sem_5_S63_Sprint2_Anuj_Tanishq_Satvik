"""Reusable LLM client abstraction layer supporting multiple providers."""

from abc import ABC, abstractmethod
import json
import os
import time
from typing import Any, Dict, Optional, Tuple, Type, TypeVar
import httpx
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.response import StructuredResearchResponse
from backend.app.services.llm.exceptions import (
    InvalidModelParameterError,
    LLMAuthenticationError,
    LLMException,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    StructuredOutputValidationError,
)
from backend.app.services.llm.cost import TokenUsage, token_cost_estimator
from backend.app.services.llm.prompts import ACADEMIC_RAG_SYSTEM_PROMPT
from backend.app.services.llm.types import (
    CompletionResponse,
    LLMConfig,
    get_provider_capabilities,
    validate_model_parameters,
)
from backend.app.services.llm.parser import StructuredOutputParser

T = TypeVar("T", bound=BaseModel)


class LLMClientInterface(ABC):
    """Abstract interface defining standard LLM completion contracts."""

    config: LLMConfig

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute completion call and return standardized CompletionResponse."""
        pass

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method returning raw text response."""
        resp = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp.text

    async def complete_structured(
        self,
        prompt: str,
        schema_cls: Type[T] = StructuredResearchResponse,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[T, CompletionResponse]:
        """Execute completion requesting structured JSON output and strictly validate against schema_cls.

        Never silently accepts invalid or malformed structured outputs.
        """
        # Enforce json_object mode in provider payload where supported
        kwargs["response_format"] = "json_object"

        # Append schema formatting guidelines to system prompt
        schema_instruction = StructuredOutputParser.generate_schema_prompt(schema_cls)
        effective_system = (
            f"{system_prompt}\n\n{schema_instruction}"
            if system_prompt
            else schema_instruction
        )

        resp = await self.complete(
            prompt=prompt,
            system_prompt=effective_system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        validated_model = StructuredOutputParser.parse_and_validate(
            raw_text=resp.text,
            schema_cls=schema_cls,
            provider=getattr(self, "provider_name", "llm"),
        )
        return validated_model, resp

    def _validate_call_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Validate per-call parameter overrides against ranges and combinations."""
        effective_temp = temperature if temperature is not None else getattr(self.config, "temperature", 0.1)
        effective_max_tokens = max_tokens if max_tokens is not None else getattr(self.config, "max_tokens", 1024)
        effective_top_p = kwargs.get("top_p", getattr(self.config, "top_p", 1.0))
        effective_top_k = kwargs.get("top_k", getattr(self.config, "top_k", None))
        effective_pres_pen = kwargs.get("presence_penalty", getattr(self.config, "presence_penalty", None))
        effective_freq_pen = kwargs.get("frequency_penalty", getattr(self.config, "frequency_penalty", None))
        effective_resp_format = kwargs.get("response_format", getattr(self.config, "response_format", "text"))
        effective_stop = kwargs.get("stop", kwargs.get("stop_sequences", getattr(self.config, "stop_sequences", None)))
        effective_timeout = kwargs.get("timeout_seconds", getattr(self.config, "timeout_seconds", 30.0))

        validate_model_parameters(
            model=kwargs.get("model", getattr(self.config, "model", None)),
            temperature=effective_temp,
            max_tokens=effective_max_tokens,
            top_p=effective_top_p,
            top_k=effective_top_k,
            presence_penalty=effective_pres_pen,
            frequency_penalty=effective_freq_pen,
            response_format=effective_resp_format,
            stop_sequences=effective_stop,
            timeout_seconds=effective_timeout,
            provider=provider or getattr(self, "provider_name", "llm"),
        )
        return {
            "temperature": effective_temp,
            "max_tokens": effective_max_tokens,
            "top_p": effective_top_p,
            "top_k": effective_top_k,
            "presence_penalty": effective_pres_pen,
            "frequency_penalty": effective_freq_pen,
            "response_format": effective_resp_format,
            "stop_sequences": effective_stop,
            "timeout_seconds": effective_timeout,
        }


class MockLLMClient(LLMClientInterface):
    """Deterministic mock LLM client for offline development and testing."""

    provider_name: str = "mock"

    def __init__(self, config: Optional[LLMConfig] = None, simulate_error: Optional[str] = None):
        self.config = config or LLMConfig(model="mock-academic-v1")
        self.simulate_error = simulate_error

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        params = self._validate_call_params(
            temperature=temperature,
            max_tokens=max_tokens,
            provider="mock",
            **kwargs,
        )
        start_time = time.perf_counter()

        if self.simulate_error == "timeout":
            raise LLMTimeoutError("Mock request timed out", provider="mock")
        elif self.simulate_error == "auth":
            raise LLMAuthenticationError("Mock invalid credentials", provider="mock", status_code=401)
        elif self.simulate_error == "rate_limit":
            raise LLMRateLimitError("Mock rate limit exceeded", provider="mock", status_code=429)

        resp_format = params["response_format"]

        if self.simulate_error == "invalid_json":
            text = "Here is the answer: { invalid_json: missing_quotes, "
        elif self.simulate_error == "invalid_schema":
            text = json.dumps({"citations": [], "evidence_status": "sufficient"})  # Missing required 'answer'
        elif resp_format == "json_object":
            text = json.dumps({
                "answer": (
                    "Based on the provided academic sources, federated learning is a distributed "
                    "machine learning technique where model training occurs across decentralized edge devices "
                    "without centralizing raw user data."
                ),
                "citations": [
                    {
                        "source": "Federated Learning Survey",
                        "page": 8,
                        "chunk_id": "chunk_001",
                        "quote": "Model training occurs across decentralized edge devices without centralizing raw user data.",
                        "relevance_explanation": "Defines core architecture of federated learning.",
                    }
                ],
                "evidence_status": "sufficient",
                "confidence": 0.98,
                "source_references": [
                    {
                        "document_id": "doc_fl_survey_2024",
                        "title": "Federated Learning Survey",
                        "page": 8,
                        "chunk_id": "chunk_001",
                        "section": "Architecture Overview",
                        "snippet": "Model training occurs across decentralized edge devices without centralizing raw user data.",
                    }
                ],
                "limitations": None,
                "refusal_reason": None,
            })
        else:
            text = (
                "Based on the provided academic sources, federated learning is a distributed machine learning "
                "technique where model training occurs across decentralized edge devices without centralizing "
                "raw user data [Doc: Federated Learning Survey, Page: 8]."
            )

        latency = (time.perf_counter() - start_time) * 1000
        prompt_tokens = token_cost_estimator.estimate_tokens(prompt, model=self.config.model, provider="mock")
        comp_tokens = token_cost_estimator.estimate_tokens(text, model=self.config.model, provider="mock")
        usage = token_cost_estimator.create_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            model=self.config.model,
            provider="mock",
        )

        logger.info(
            "LLM Completion [mock/%s]: %d prompt + %d comp = %d total tokens | Cost: $%.6f | Latency: %.2fms",
            self.config.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.estimated_cost_usd,
            latency,
        )

        return CompletionResponse(
            text=text,
            model=self.config.model,
            provider="mock",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            token_usage=usage,
            finish_reason="stop",
            latency_ms=round(latency, 2),
            raw_response={"mock": True},
        )


class OpenAILLMClient(LLMClientInterface):
    """OpenAI Chat Completion API client."""

    provider_name: str = "openai"

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        api_key: Optional[str] = None,
        fallback_to_mock: Optional[bool] = None,
    ):
        self.config = config or LLMConfig.from_settings()
        self.fallback_to_mock = (
            fallback_to_mock
            if fallback_to_mock is not None
            else getattr(self.config, "fallback_to_mock", True)
        )
        self.api_key = self._resolve_api_key(api_key)
        self.base_url = "https://api.openai.com/v1"

    def _resolve_api_key(self, explicit_key: Optional[str]) -> Optional[str]:
        """Resolve API key securely: explicit -> config -> settings/env."""
        key = (
            explicit_key
            or (self.config.api_key if self.config else None)
            or getattr(settings, "OPENAI_API_KEY", None)
            or getattr(settings, "LLM_API_KEY", None)
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if key:
            key = key.strip()
        return key or None

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        # Validate per-call parameters
        params = self._validate_call_params(
            temperature=temperature,
            max_tokens=max_tokens,
            provider="openai",
            **kwargs,
        )

        is_missing_or_placeholder = not self.api_key or self.api_key.startswith("your-")
        if is_missing_or_placeholder:
            if self.fallback_to_mock:
                logger.warning("Valid OpenAI API key not detected; falling back to MockLLMClient.")
                return await MockLLMClient(config=self.config).complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    **kwargs,
                )
            else:
                raise LLMAuthenticationError(
                    "Missing or invalid OpenAI API credentials. Configure OPENAI_API_KEY or LLM_API_KEY.",
                    provider="openai",
                    status_code=401,
                )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        api_messages = kwargs.get("messages")
        if not api_messages:
            api_messages = [
                {"role": "system", "content": system_prompt or ACADEMIC_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

        # Assemble provider-aware parameters
        model_name = kwargs.get("model", self.config.model)
        seed_val = kwargs.get("seed", self.config.seed)
        timeout = params["timeout_seconds"]

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": api_messages,
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
        }
        if params["top_p"] is not None:
            payload["top_p"] = params["top_p"]
        if seed_val is not None:
            payload["seed"] = seed_val
        if params["response_format"] == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if params["stop_sequences"]:
            payload["stop"] = params["stop_sequences"]
        if params["presence_penalty"] is not None:
            payload["presence_penalty"] = params["presence_penalty"]
        if params["frequency_penalty"] is not None:
            payload["frequency_penalty"] = params["frequency_penalty"]
        # Note: top_k is intentionally excluded as OpenAI does not support it

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"OpenAI request timed out after {timeout}s: {exc}",
                provider="openai",
            ) from exc
        except httpx.RequestError as exc:
            raise LLMException(
                f"Network communication failure: {exc}",
                provider="openai",
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code == 401:
            raise LLMAuthenticationError(
                "Invalid OpenAI API credentials.",
                provider="openai",
                status_code=401,
            )
        elif response.status_code == 429:
            raise LLMRateLimitError(
                "OpenAI rate limit or credit quota exceeded.",
                provider="openai",
                status_code=429,
            )
        elif response.status_code >= 500:
            raise LLMProviderError(
                f"OpenAI server returned error {response.status_code}.",
                provider="openai",
                status_code=response.status_code,
            )
        elif response.is_error:
            raise LLMException(
                f"OpenAI request failed: {response.text}",
                provider="openai",
                status_code=response.status_code,
            )

        try:
            data = response.json()
            choice = data["choices"][0]
            raw_usage = data.get("usage", {})
            model_name = data.get("model", self.config.model)
            text_content = choice["message"]["content"]

            is_actual = bool(raw_usage and "prompt_tokens" in raw_usage and "completion_tokens" in raw_usage)
            p_tokens = raw_usage.get("prompt_tokens") if is_actual else token_cost_estimator.estimate_tokens(prompt, model=model_name, provider="openai")
            c_tokens = raw_usage.get("completion_tokens") if is_actual else token_cost_estimator.estimate_tokens(text_content, model=model_name, provider="openai")
            usage_record = token_cost_estimator.create_usage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                model=model_name,
                provider="openai",
                is_estimated=not is_actual,
                usage_type="actual" if is_actual else "estimated",
            )

            logger.info(
                "LLM Completion [openai/%s]: %d prompt + %d comp = %d total tokens | Cost: $%.6f | Latency: %.2fms",
                model_name,
                usage_record.prompt_tokens,
                usage_record.completion_tokens,
                usage_record.total_tokens,
                usage_record.estimated_cost_usd,
                latency_ms,
            )

            return CompletionResponse(
                text=text_content,
                model=model_name,
                provider="openai",
                prompt_tokens=usage_record.prompt_tokens,
                completion_tokens=usage_record.completion_tokens,
                total_tokens=usage_record.total_tokens,
                estimated_cost_usd=usage_record.estimated_cost_usd,
                token_usage=usage_record,
                finish_reason=choice.get("finish_reason", "stop"),
                latency_ms=round(latency_ms, 2),
                raw_response=data,
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMResponseError(
                f"Failed to parse OpenAI completion payload: {exc}",
                provider="openai",
            ) from exc


class GeminiLLMClient(LLMClientInterface):
    """Google Gemini REST API client."""

    provider_name: str = "gemini"

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        api_key: Optional[str] = None,
        fallback_to_mock: Optional[bool] = None,
    ):
        self.config = config or LLMConfig.from_settings()
        if not config:
            self.config.model = "gemini-1.5-flash"
        self.fallback_to_mock = (
            fallback_to_mock
            if fallback_to_mock is not None
            else getattr(self.config, "fallback_to_mock", True)
        )
        self.api_key = self._resolve_api_key(api_key)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def _resolve_api_key(self, explicit_key: Optional[str]) -> Optional[str]:
        """Resolve Gemini API key securely: explicit -> config -> settings/env."""
        key = (
            explicit_key
            or (self.config.api_key if self.config else None)
            or getattr(settings, "GEMINI_API_KEY", None)
            or getattr(settings, "LLM_API_KEY", None)
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if key:
            key = key.strip()
        return key or None

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        # Validate per-call parameters
        params = self._validate_call_params(
            temperature=temperature,
            max_tokens=max_tokens,
            provider="gemini",
            **kwargs,
        )

        is_missing_or_placeholder = not self.api_key or self.api_key.startswith("your-")
        if is_missing_or_placeholder:
            if self.fallback_to_mock:
                logger.warning("Valid Gemini API key not detected; falling back to MockLLMClient.")
                return await MockLLMClient(config=self.config).complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    **kwargs,
                )
            else:
                raise LLMAuthenticationError(
                    "Missing or invalid Gemini API credentials. Configure GEMINI_API_KEY or LLM_API_KEY.",
                    provider="gemini",
                    status_code=401,
                )

        model_name = kwargs.get("model", self.config.model)
        url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"

        gen_config: Dict[str, Any] = {
            "temperature": params["temperature"],
            "maxOutputTokens": params["max_tokens"],
        }
        if params["top_p"] is not None:
            gen_config["topP"] = params["top_p"]
        if params["top_k"] is not None:
            gen_config["topK"] = params["top_k"]
        if params["response_format"] == "json_object":
            gen_config["responseMimeType"] = "application/json"
        if params["stop_sequences"]:
            gen_config["stopSequences"] = params["stop_sequences"]

        timeout = params["timeout_seconds"]

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": gen_config,
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"Gemini API request timed out after {timeout}s: {exc}",
                provider="gemini",
            ) from exc
        except httpx.RequestError as exc:
            raise LLMException(
                f"Network communication failure: {exc}",
                provider="gemini",
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code in (401, 403):
            raise LLMAuthenticationError(
                "Invalid Gemini API key.",
                provider="gemini",
                status_code=response.status_code,
            )
        elif response.status_code == 429:
            raise LLMRateLimitError(
                "Gemini quota exceeded.",
                provider="gemini",
                status_code=429,
            )
        elif response.status_code >= 500:
            raise LLMProviderError(
                f"Gemini server returned error {response.status_code}.",
                provider="gemini",
                status_code=response.status_code,
            )
        elif response.is_error:
            raise LLMException(
                f"Gemini request failed: {response.text}",
                provider="gemini",
                status_code=response.status_code,
            )

        try:
            data = response.json()
            candidate = data["candidates"][0]
            part_text = candidate["content"]["parts"][0]["text"]
            raw_usage = data.get("usageMetadata", {})

            is_actual = bool(raw_usage and "promptTokenCount" in raw_usage and "candidatesTokenCount" in raw_usage)
            p_tokens = raw_usage.get("promptTokenCount") if is_actual else token_cost_estimator.estimate_tokens(prompt, model=self.config.model, provider="gemini")
            c_tokens = raw_usage.get("candidatesTokenCount") if is_actual else token_cost_estimator.estimate_tokens(part_text, model=self.config.model, provider="gemini")
            usage_record = token_cost_estimator.create_usage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                model=self.config.model,
                provider="gemini",
                is_estimated=not is_actual,
                usage_type="actual" if is_actual else "estimated",
            )

            logger.info(
                "LLM Completion [gemini/%s]: %d prompt + %d comp = %d total tokens | Cost: $%.6f | Latency: %.2fms",
                self.config.model,
                usage_record.prompt_tokens,
                usage_record.completion_tokens,
                usage_record.total_tokens,
                usage_record.estimated_cost_usd,
                latency_ms,
            )

            return CompletionResponse(
                text=part_text,
                model=self.config.model,
                provider="gemini",
                prompt_tokens=usage_record.prompt_tokens,
                completion_tokens=usage_record.completion_tokens,
                total_tokens=usage_record.total_tokens,
                estimated_cost_usd=usage_record.estimated_cost_usd,
                token_usage=usage_record,
                finish_reason=candidate.get("finishReason", "STOP"),
                latency_ms=round(latency_ms, 2),
                raw_response=data,
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMResponseError(
                f"Failed to parse Gemini response: {exc}",
                provider="gemini",
            ) from exc


def get_llm_client(
    provider: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    fallback_to_mock: Optional[bool] = None,
) -> LLMClientInterface:
    """Factory to instantiate the appropriate LLM client based on configuration."""
    target_provider = (provider or settings.LLM_PROVIDER).lower()
    effective_config = config or LLMConfig.from_settings()

    if target_provider == "openai":
        return OpenAILLMClient(config=effective_config, fallback_to_mock=fallback_to_mock)
    elif target_provider in ("gemini", "google"):
        return GeminiLLMClient(config=effective_config, fallback_to_mock=fallback_to_mock)
    elif target_provider == "mock":
        return MockLLMClient(config=effective_config)
    else:
        logger.warning("Unrecognized provider '%s', defaulting to MockLLMClient", target_provider)
        return MockLLMClient(config=effective_config)
