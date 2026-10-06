"""Reusable LLM client abstraction layer supporting multiple providers."""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Optional
import httpx

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMException,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from backend.app.services.llm.cost import TokenUsage, token_cost_estimator
from backend.app.services.llm.prompts import ACADEMIC_RAG_SYSTEM_PROMPT
from backend.app.services.llm.types import CompletionResponse, LLMConfig


class LLMClientInterface(ABC):
    """Abstract interface defining standard LLM completion contracts."""

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


class MockLLMClient(LLMClientInterface):
    """Deterministic mock LLM client for offline development and testing."""

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
        start_time = time.perf_counter()

        if self.simulate_error == "timeout":
            raise LLMTimeoutError("Mock request timed out", provider="mock")
        elif self.simulate_error == "auth":
            raise LLMAuthenticationError("Mock invalid credentials", provider="mock", status_code=401)
        elif self.simulate_error == "rate_limit":
            raise LLMRateLimitError("Mock rate limit exceeded", provider="mock", status_code=429)

        resp_format = kwargs.get("response_format", getattr(self.config, "response_format", "text"))
        if resp_format == "json_object":
            text = (
                '{"answer": "Based on the provided academic sources, federated learning is a distributed '
                'machine learning technique where model training occurs across decentralized edge devices '
                'without centralizing raw user data.", "citations": [{"document": "Federated Learning Survey", "page": 8}]}'
            )
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

    def __init__(self, config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        self.config = config or LLMConfig.from_settings()
        self.api_key = api_key or self.config.api_key or settings.LLM_API_KEY
        self.base_url = "https://api.openai.com/v1"

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        if not self.api_key or self.api_key.startswith("your-"):
            logger.warning("Valid OpenAI API key not detected; falling back to MockLLMClient.")
            return await MockLLMClient(config=self.config).complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
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
        temp = temperature if temperature is not None else self.config.temperature
        max_toks = max_tokens if max_tokens is not None else self.config.max_tokens
        top_p_val = kwargs.get("top_p", self.config.top_p)
        seed_val = kwargs.get("seed", self.config.seed)
        resp_format_val = kwargs.get("response_format", self.config.response_format)
        stop_seqs = kwargs.get("stop", kwargs.get("stop_sequences", self.config.stop_sequences))
        timeout = kwargs.get("timeout_seconds", self.config.timeout_seconds)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": api_messages,
            "temperature": temp,
            "max_tokens": max_toks,
        }
        if top_p_val is not None:
            payload["top_p"] = top_p_val
        if seed_val is not None:
            payload["seed"] = seed_val
        if resp_format_val == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if stop_seqs:
            payload["stop"] = stop_seqs

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
                f"OpenAI request timed out after {self.config.timeout_seconds}s: {exc}",
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

            p_tokens = raw_usage.get("prompt_tokens") or token_cost_estimator.estimate_tokens(prompt, model=model_name, provider="openai")
            c_tokens = raw_usage.get("completion_tokens") or token_cost_estimator.estimate_tokens(text_content, model=model_name, provider="openai")
            usage_record = token_cost_estimator.create_usage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                model=model_name,
                provider="openai",
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

    def __init__(self, config: Optional[LLMConfig] = None, api_key: Optional[str] = None):
        self.config = config or LLMConfig.from_settings()
        if not config:
            self.config.model = "gemini-1.5-flash"
        self.api_key = api_key or self.config.api_key or settings.LLM_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        if not self.api_key or self.api_key.startswith("your-"):
            logger.warning("Valid Gemini API key not detected; falling back to MockLLMClient.")
            return await MockLLMClient(config=self.config).complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        model_name = kwargs.get("model", self.config.model)
        url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"

        gen_config: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "maxOutputTokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        top_p_val = kwargs.get("top_p", self.config.top_p)
        if top_p_val is not None:
            gen_config["topP"] = top_p_val

        resp_format = kwargs.get("response_format", self.config.response_format)
        if resp_format == "json_object":
            gen_config["responseMimeType"] = "application/json"

        stop_seqs = kwargs.get("stop", kwargs.get("stop_sequences", self.config.stop_sequences))
        if stop_seqs:
            gen_config["stopSequences"] = stop_seqs

        timeout = kwargs.get("timeout_seconds", self.config.timeout_seconds)

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
                f"Gemini API request timed out: {exc}",
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

            p_tokens = raw_usage.get("promptTokenCount") or token_cost_estimator.estimate_tokens(prompt, model=self.config.model, provider="gemini")
            c_tokens = raw_usage.get("candidatesTokenCount") or token_cost_estimator.estimate_tokens(part_text, model=self.config.model, provider="gemini")
            usage_record = token_cost_estimator.create_usage(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                model=self.config.model,
                provider="gemini",
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
) -> LLMClientInterface:
    """Factory to instantiate the appropriate LLM client based on configuration."""
    target_provider = (provider or settings.LLM_PROVIDER).lower()
    effective_config = config or LLMConfig.from_settings()

    if target_provider == "openai":
        return OpenAILLMClient(config=effective_config)
    elif target_provider in ("gemini", "google"):
        return GeminiLLMClient(config=effective_config)
    elif target_provider == "mock":
        return MockLLMClient(config=effective_config)
    else:
        logger.warning("Unrecognized provider '%s', defaulting to MockLLMClient", target_provider)
        return MockLLMClient(config=effective_config)
