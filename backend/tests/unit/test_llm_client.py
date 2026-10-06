"""Unit tests for the LLM client abstraction layer and providers."""

import pytest
from unittest.mock import AsyncMock, patch
import httpx

from backend.app.services.llm import (
    CompletionResponse,
    GeminiLLMClient,
    LLMAuthenticationError,
    LLMConfig,
    LLMException,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    MockLLMClient,
    OpenAILLMClient,
    get_llm_client,
)


@pytest.mark.asyncio
async def test_mock_llm_client_complete():
    config = LLMConfig(model="mock-v1", temperature=0.2, max_tokens=256)
    client = MockLLMClient(config=config)

    response = await client.complete("What is federated learning?")

    assert isinstance(response, CompletionResponse)
    assert response.provider == "mock"
    assert response.model == "mock-v1"
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens
    assert response.finish_reason == "stop"
    assert response.latency_ms >= 0.0
    assert "federated learning" in response.text.lower()


@pytest.mark.asyncio
async def test_mock_llm_client_generate_helper():
    client = MockLLMClient()
    text = await client.generate("Summarize research findings.")
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_mock_llm_client_error_simulation():
    # Timeout
    timeout_client = MockLLMClient(simulate_error="timeout")
    with pytest.raises(LLMTimeoutError) as exc_info:
        await timeout_client.complete("Test query")
    assert "timed out" in str(exc_info.value).lower()

    # Authentication
    auth_client = MockLLMClient(simulate_error="auth")
    with pytest.raises(LLMAuthenticationError) as exc_info:
        await auth_client.complete("Test query")
    assert exc_info.value.status_code == 401

    # Rate limit
    rate_client = MockLLMClient(simulate_error="rate_limit")
    with pytest.raises(LLMRateLimitError) as exc_info:
        await rate_client.complete("Test query")
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_openai_client_fallback_without_key():
    # When api_key is placeholder, fallback to mock is triggered
    client = OpenAILLMClient(api_key="your-api-key-here")
    response = await client.complete("Test prompt")
    assert response.provider == "mock"
    assert len(response.text) > 0


@pytest.mark.asyncio
async def test_openai_client_success_mocked_http():
    config = LLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=100)
    client = OpenAILLMClient(config=config, api_key="sk-test-valid-key")

    mock_resp_data = {
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {"content": "Decentralized privacy-preserving ML."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23,
        },
    }

    mock_http_response = httpx.Response(
        status_code=200,
        json=mock_resp_data,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http_response
        response = await client.complete("Explain federated learning")

        assert response.provider == "openai"
        assert response.model == "gpt-4o-mini"
        assert response.text == "Decentralized privacy-preserving ML."
        assert response.prompt_tokens == 15
        assert response.completion_tokens == 8
        assert response.total_tokens == 23


@pytest.mark.asyncio
async def test_openai_client_error_mappings():
    config = LLMConfig(model="gpt-4o-mini", timeout_seconds=1.0)
    client = OpenAILLMClient(config=config, api_key="sk-test-key")

    # 401 Authentication
    auth_resp = httpx.Response(
        status_code=401,
        text="Invalid API Key",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = auth_resp
        with pytest.raises(LLMAuthenticationError):
            await client.complete("Hello")

    # 429 Rate Limit
    rate_resp = httpx.Response(
        status_code=429,
        text="Rate limit exceeded",
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = rate_resp
        with pytest.raises(LLMRateLimitError):
            await client.complete("Hello")

    # Timeout
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ReadTimeout("Timeout connecting to OpenAI")
        with pytest.raises(LLMTimeoutError):
            await client.complete("Hello")


@pytest.mark.asyncio
async def test_gemini_client_success_mocked_http():
    config = LLMConfig(model="gemini-1.5-flash", temperature=0.2)
    client = GeminiLLMClient(config=config, api_key="test-gemini-key")

    mock_resp_data = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Federated learning answer from Gemini."}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 10,
            "totalTokenCount": 30,
        },
    }

    mock_http_response = httpx.Response(
        status_code=200,
        json=mock_resp_data,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http_response
        response = await client.complete("Explain federated learning")

        assert response.provider == "gemini"
        assert response.model == "gemini-1.5-flash"
        assert response.text == "Federated learning answer from Gemini."
        assert response.prompt_tokens == 20
        assert response.completion_tokens == 10


def test_get_llm_client_factory():
    openai_client = get_llm_client("openai")
    assert isinstance(openai_client, OpenAILLMClient)

    gemini_client = get_llm_client("gemini")
    assert isinstance(gemini_client, GeminiLLMClient)

    mock_client = get_llm_client("mock")
    assert isinstance(mock_client, MockLLMClient)

    default_client = get_llm_client("unknown_provider")
    assert isinstance(default_client, MockLLMClient)
