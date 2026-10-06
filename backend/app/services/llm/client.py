"""LLM client interface and providers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.llm.prompts import ACADEMIC_RAG_SYSTEM_PROMPT


class LLMClientInterface(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from the LLM."""
        pass


class MockLLMClient(LLMClientInterface):
    """Development/Testing LLM client returning deterministic responses."""

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        logger.info("MockLLMClient generating response")
        return (
            "ResearchMate grounded answer based on academic sources. "
            "Federated learning preserves privacy through decentralized training [Doc: Survey, Page: 1]."
        )


class OpenAILLMClient(LLMClientInterface):
    """OpenAI API client implementation."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # If API key is not configured, fall back gracefully in development
        if not self.api_key or self.api_key.startswith("your-"):
            logger.warning("OpenAI API key not configured, falling back to mock response.")
            return await MockLLMClient().generate(prompt, system_prompt, temperature, max_tokens)

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": system_prompt or ACADEMIC_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or settings.LLM_TEMPERATURE,
            "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


def get_llm_client() -> LLMClientInterface:
    """Factory to instantiate the configured LLM client."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai":
        return OpenAILLMClient(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
        )
    return MockLLMClient()
