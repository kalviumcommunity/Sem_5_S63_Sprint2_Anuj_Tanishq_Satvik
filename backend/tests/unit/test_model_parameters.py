"""Unit tests for model parameters, output control, environment presets, and provider capabilities."""

import json
import os
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.services.llm import (
    GeminiLLMClient,
    InvalidModelParameterError,
    LLMAuthenticationError,
    LLMConfig,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    MockLLMClient,
    OpenAILLMClient,
    get_provider_capabilities,
    validate_model_parameters,
)


def test_llm_config_from_settings():
    """Verify LLMConfig loads centralized defaults from settings."""
    custom_settings = Settings(
        LLM_MODEL="gpt-4o",
        LLM_TEMPERATURE=0.5,
        LLM_MAX_TOKENS=2048,
        LLM_TOP_P=0.9,
        LLM_SEED=99,
        LLM_TIMEOUT_SECONDS=45.0,
        LLM_RESPONSE_FORMAT="json_object",
        LLM_MAX_RETRIES=3,
    )
    config = LLMConfig.from_settings(custom_settings)

    assert config.model == "gpt-4o"
    assert config.temperature == 0.5
    assert config.max_tokens == 2048
    assert config.top_p == 0.9
    assert config.seed == 99
    assert config.timeout_seconds == 45.0
    assert config.response_format == "json_object"
    assert config.max_retries == 3


def test_llm_config_environment_presets():
    """Verify distinct configurations for production, testing, and staging environments."""
    prod_cfg = LLMConfig.for_environment("production")
    assert prod_cfg.temperature == 0.0  # Deterministic in production
    assert prod_cfg.max_retries == 3
    assert prod_cfg.timeout_seconds == 25.0

    test_cfg = LLMConfig.for_environment("testing")
    assert test_cfg.model == "mock-academic-v1"
    assert test_cfg.timeout_seconds == 5.0
    assert test_cfg.max_retries == 0

    staging_cfg = LLMConfig.for_environment("staging")
    assert staging_cfg.model == "gpt-4o-mini"
    assert staging_cfg.temperature == 0.1


@pytest.mark.asyncio
async def test_openai_payload_parameters():
    """Verify OpenAI client includes top_p, seed, and json_object response_format."""
    config = LLMConfig(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=500,
        top_p=0.85,
        seed=1234,
        response_format="json_object",
        stop_sequences=["### END"],
    )
    client = OpenAILLMClient(config=config, api_key="sk-test-valid-key")

    mock_resp_data = {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": '{"answer": "Grounded answer"}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }
    mock_http = httpx.Response(status_code=200, json=mock_resp_data, request=httpx.Request("POST", "http://test"))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http
        await client.complete("Test prompt")

        call_args, call_kwargs = mock_post.call_args
        payload = call_kwargs["json"]

        assert payload["model"] == "gpt-4o-mini"
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 500
        assert payload["top_p"] == 0.85
        assert payload["seed"] == 1234
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["stop"] == ["### END"]


@pytest.mark.asyncio
async def test_gemini_payload_parameters():
    """Verify Gemini client sets topP and responseMimeType in generationConfig."""
    config = LLMConfig(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=800,
        top_p=0.92,
        response_format="json_object",
    )
    client = GeminiLLMClient(config=config, api_key="test-key")

    mock_resp_data = {
        "candidates": [
            {"content": {"parts": [{"text": '{"answer": "Gemini JSON"}'}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20},
    }
    mock_http = httpx.Response(status_code=200, json=mock_resp_data, request=httpx.Request("POST", "http://test"))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http
        await client.complete("Test prompt")

        call_args, call_kwargs = mock_post.call_args
        payload = call_kwargs["json"]
        gen_config = payload["generationConfig"]

        assert gen_config["temperature"] == 0.3
        assert gen_config["maxOutputTokens"] == 800
        assert gen_config["topP"] == 0.92
        assert gen_config["responseMimeType"] == "application/json"


@pytest.mark.asyncio
async def test_mock_llm_json_response_format():
    """Verify MockLLMClient outputs valid JSON when response_format is json_object."""
    client = MockLLMClient()
    response = await client.complete("Query", response_format="json_object")

    parsed_json = json.loads(response.text)
    assert "answer" in parsed_json
    assert "citations" in parsed_json


@pytest.mark.asyncio
async def test_per_call_parameter_overrides():
    """Verify per-call parameters override LLMConfig defaults."""
    config = LLMConfig(temperature=0.1, max_tokens=100)
    client = OpenAILLMClient(config=config, api_key="sk-test-key")

    mock_resp_data = {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "Overridden"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    mock_http = httpx.Response(status_code=200, json=mock_resp_data, request=httpx.Request("POST", "http://test"))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http
        await client.complete("Prompt", temperature=0.9, max_tokens=2048, top_p=0.5)

        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 2048
        assert payload["top_p"] == 0.5


# ==============================================================================
# Assignment 3.16 Tests: Validation, Capabilities, Presets, Error Handling
# ==============================================================================


def test_invalid_temperature_out_of_range():
    """Verify temperature outside [0.0, 2.0] is rejected with InvalidModelParameterError."""
    with pytest.raises(InvalidModelParameterError) as exc_low:
        LLMConfig(temperature=-0.1)
    assert "temperature" in str(exc_low.value).lower()

    with pytest.raises(InvalidModelParameterError) as exc_high:
        LLMConfig(temperature=2.1)
    assert "temperature" in str(exc_high.value).lower()


def test_invalid_max_tokens_non_positive():
    """Verify max_tokens <= 0 is rejected with InvalidModelParameterError."""
    with pytest.raises(InvalidModelParameterError):
        LLMConfig(max_tokens=0)

    with pytest.raises(InvalidModelParameterError):
        LLMConfig(max_tokens=-50)


def test_invalid_top_p_out_of_range():
    """Verify top_p outside [0.0, 1.0] is rejected with InvalidModelParameterError."""
    with pytest.raises(InvalidModelParameterError):
        LLMConfig(top_p=-0.05)

    with pytest.raises(InvalidModelParameterError):
        LLMConfig(top_p=1.05)


def test_invalid_top_k():
    """Verify top_k < 1 is rejected with InvalidModelParameterError."""
    with pytest.raises(InvalidModelParameterError):
        LLMConfig(top_k=0)

    with pytest.raises(InvalidModelParameterError):
        LLMConfig(top_k=-2)


def test_invalid_penalties_range():
    """Verify presence and frequency penalties outside [-2.0, 2.0] are rejected."""
    with pytest.raises(InvalidModelParameterError):
        LLMConfig(presence_penalty=-2.5)

    with pytest.raises(InvalidModelParameterError):
        LLMConfig(frequency_penalty=2.5)


def test_incompatible_greedy_and_nucleus_sampling():
    """Verify incompatible combination of greedy temperature (0.0) with nucleus cutoff (top_p < 1.0) is rejected."""
    with pytest.raises(InvalidModelParameterError) as exc_info:
        LLMConfig(temperature=0.0, top_p=0.8)
    assert "incompatible" in str(exc_info.value).lower()
    assert "top_p" in str(exc_info.value).lower()

    # temperature=0.0 with top_p=1.0 is compatible and valid
    valid_cfg = LLMConfig(temperature=0.0, top_p=1.0)
    assert valid_cfg.temperature == 0.0
    assert valid_cfg.top_p == 1.0


def test_invalid_response_format():
    """Verify unsupported response_format raises InvalidModelParameterError."""
    with pytest.raises(InvalidModelParameterError) as exc:
        LLMConfig(response_format="yaml")
    assert "response_format" in str(exc.value).lower()


def test_stop_sequences_validation():
    """Verify empty stop sequences or provider limits are strictly enforced."""
    with pytest.raises(InvalidModelParameterError):
        LLMConfig(stop_sequences=["VALID", ""])

    # OpenAI maximum of 4 stop sequences
    with pytest.raises(InvalidModelParameterError):
        validate_model_parameters(stop_sequences=["1", "2", "3", "4", "5"], provider="openai")

    # Gemini maximum of 5 stop sequences
    with pytest.raises(InvalidModelParameterError):
        validate_model_parameters(stop_sequences=["1", "2", "3", "4", "5", "6"], provider="gemini")


def test_academic_explanation_preset():
    """Verify sensible defaults for concise academic explanations."""
    academic_cfg = LLMConfig.for_academic_explanation()
    assert academic_cfg.temperature == 0.1
    assert academic_cfg.max_tokens == 512
    assert academic_cfg.top_p == 0.9
    assert academic_cfg.seed == 42
    assert academic_cfg.response_format == "text"
    assert academic_cfg.model == "gpt-4o-mini"

    # Gemini-specific academic explanation preset
    gemini_academic_cfg = LLMConfig.for_academic_explanation(provider="gemini")
    assert gemini_academic_cfg.model == "gemini-1.5-flash"
    assert gemini_academic_cfg.temperature == 0.1
    assert gemini_academic_cfg.max_tokens == 512
    assert gemini_academic_cfg.top_p == 0.9
    assert gemini_academic_cfg.seed is None  # Gemini v1beta does not take seed


@pytest.mark.asyncio
async def test_provider_capability_openai_filters_top_k():
    """Verify OpenAI client does not send unsupported top_k in its HTTP payload."""
    config = LLMConfig(model="gpt-4o-mini", temperature=0.2, top_k=40)
    client = OpenAILLMClient(config=config, api_key="sk-test-valid-key")

    mock_resp_data = {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "Answer without top_k"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    mock_http = httpx.Response(status_code=200, json=mock_resp_data, request=httpx.Request("POST", "http://test"))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http
        await client.complete("Prompt", top_k=50)

        payload = mock_post.call_args[1]["json"]
        assert "top_k" not in payload
        assert "topK" not in payload
        assert payload["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_provider_capability_openai_sends_penalties():
    """Verify OpenAI client sends supported presence and frequency penalties."""
    config = LLMConfig(
        model="gpt-4o-mini",
        presence_penalty=0.5,
        frequency_penalty=0.3,
    )
    client = OpenAILLMClient(config=config, api_key="sk-test-valid-key")

    mock_resp_data = {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "Penalized text"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }
    mock_http = httpx.Response(status_code=200, json=mock_resp_data, request=httpx.Request("POST", "http://test"))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http
        await client.complete("Prompt")

        payload = mock_post.call_args[1]["json"]
        assert payload["presence_penalty"] == 0.5
        assert payload["frequency_penalty"] == 0.3


@pytest.mark.asyncio
async def test_provider_capability_gemini_maps_top_k_and_omits_unsupported():
    """Verify Gemini client maps top_k to topK and omits seed/penalties from generationConfig."""
    config = LLMConfig(
        model="gemini-1.5-flash",
        temperature=0.2,
        seed=42,  # Unsupported by Gemini v1beta
        presence_penalty=0.5,  # Unsupported by Gemini
    )
    client = GeminiLLMClient(config=config, api_key="test-gemini-key")

    mock_resp_data = {
        "candidates": [
            {"content": {"parts": [{"text": "Gemini response"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5, "totalTokenCount": 10},
    }
    mock_http = httpx.Response(status_code=200, json=mock_resp_data, request=httpx.Request("POST", "http://test"))

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http
        await client.complete("Prompt", top_k=30)

        payload = mock_post.call_args[1]["json"]
        gen_config = payload["generationConfig"]

        assert gen_config["topK"] == 30
        assert "seed" not in gen_config
        assert "presence_penalty" not in gen_config
        assert "frequency_penalty" not in gen_config


def test_provider_capabilities_registry():
    """Verify capability definitions and metadata for known providers."""
    openai_caps = get_provider_capabilities("openai")
    assert openai_caps.provider == "openai"
    assert openai_caps.supports_top_k is False
    assert openai_caps.supports_penalties is True
    assert openai_caps.max_stop_sequences == 4

    gemini_caps = get_provider_capabilities("gemini")
    assert gemini_caps.provider == "gemini"
    assert gemini_caps.supports_top_k is True
    assert gemini_caps.supports_seed is False
    assert gemini_caps.max_stop_sequences == 5

    mock_caps = get_provider_capabilities("mock")
    assert mock_caps.provider == "mock"
    assert mock_caps.supports_json_object is True


def test_credential_loading_order(monkeypatch):
    """Verify credential precedence: explicit -> config -> provider env -> generic env."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai-env")
    monkeypatch.setenv("LLM_API_KEY", "sk-from-generic-env")

    # Explicit takes top priority
    client_explicit = OpenAILLMClient(api_key="sk-explicit")
    assert client_explicit.api_key == "sk-explicit"

    # Config takes next priority
    client_config = OpenAILLMClient(config=LLMConfig(api_key="sk-from-config"))
    assert client_config.api_key == "sk-from-config"

    # Provider env takes priority over generic LLM_API_KEY
    client_env = OpenAILLMClient()
    assert client_env.api_key == "sk-from-openai-env"


@pytest.mark.asyncio
async def test_missing_credentials_fallback_disabled(monkeypatch):
    """Verify that when fallback_to_mock=False, missing credentials raise LLMAuthenticationError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    client = OpenAILLMClient(api_key="", fallback_to_mock=False)
    with pytest.raises(LLMAuthenticationError) as exc_info:
        await client.complete("Test prompt")
    assert exc_info.value.status_code == 401
    assert "OpenAI" in str(exc_info.value)

    gemini_client = GeminiLLMClient(api_key="", fallback_to_mock=False)
    with pytest.raises(LLMAuthenticationError) as gemini_exc:
        await gemini_client.complete("Test prompt")
    assert gemini_exc.value.status_code == 401
    assert "Gemini" in str(gemini_exc.value)


@pytest.mark.asyncio
async def test_per_call_override_validation():
    """Verify per-call parameter overrides are strictly validated before execution."""
    client = MockLLMClient()

    # Out-of-range temperature override
    with pytest.raises(InvalidModelParameterError):
        await client.complete("Prompt", temperature=2.5)

    # Incompatible combination in override
    with pytest.raises(InvalidModelParameterError):
        await client.complete("Prompt", temperature=0.0, top_p=0.4)

    # Negative max_tokens override
    with pytest.raises(InvalidModelParameterError):
        await client.complete("Prompt", max_tokens=-10)


@pytest.mark.asyncio
async def test_client_timeout_handling():
    """Verify timeout exception mapping."""
    client = OpenAILLMClient(api_key="sk-test", fallback_to_mock=False)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")
        with pytest.raises(LLMTimeoutError) as exc:
            await client.complete("Prompt")
        assert "timed out" in str(exc.value).lower()
        assert exc.value.provider == "openai"


@pytest.mark.asyncio
async def test_client_provider_errors():
    """Verify HTTP status code mappings for rate limits and server errors."""
    client = OpenAILLMClient(api_key="sk-test", fallback_to_mock=False)

    # 429 Rate Limit
    rate_resp = httpx.Response(status_code=429, text="Quota exceeded", request=httpx.Request("POST", "http://test"))
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = rate_resp
        with pytest.raises(LLMRateLimitError):
            await client.complete("Prompt")

    # 503 Provider Error
    serv_resp = httpx.Response(status_code=503, text="Service unavailable", request=httpx.Request("POST", "http://test"))
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = serv_resp
        with pytest.raises(LLMProviderError):
            await client.complete("Prompt")
