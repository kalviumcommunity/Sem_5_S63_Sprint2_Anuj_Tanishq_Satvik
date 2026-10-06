"""Unit tests for model parameters, output control, and environment presets."""

import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.services.llm import (
    GeminiLLMClient,
    LLMConfig,
    MockLLMClient,
    OpenAILLMClient,
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
