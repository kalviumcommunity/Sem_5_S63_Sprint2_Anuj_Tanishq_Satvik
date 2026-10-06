"""Embedding service interface and implementations."""

from abc import ABC, abstractmethod
import math
from typing import List
from backend.app.core.config import settings
from backend.app.core.logging import logger


class EmbeddingServiceInterface(ABC):
    """Abstract interface for generating vector embeddings."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts."""
        pass


class MockEmbeddingService(EmbeddingServiceInterface):
    """Deterministic mock embedding service for development and testing."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def _generate_vector(self, text: str) -> List[float]:
        """Generate pseudo-normalized deterministic vector from text hash."""
        seed = sum(ord(c) for c in text)
        vec = [math.sin(seed + i) for i in range(self.dimension)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed_text(self, text: str) -> List[float]:
        return self._generate_vector(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]


class OpenAIEmbeddingService(EmbeddingServiceInterface):
    """OpenAI Embeddings API client."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def embed_text(self, text: str) -> List[float]:
        batch = await self.embed_batch([text])
        return batch[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key or self.api_key.startswith("your-"):
            logger.warning("OpenAI API key missing for embeddings, using mock service.")
            return await MockEmbeddingService().embed_batch(texts)

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]


def get_embedding_service() -> EmbeddingServiceInterface:
    """Factory to retrieve configured embedding service."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "openai":
        return OpenAIEmbeddingService(
            api_key=settings.EMBEDDING_API_KEY,
            model=settings.EMBEDDING_MODEL,
        )
    return MockEmbeddingService(dimension=settings.EMBEDDING_DIMENSION)
