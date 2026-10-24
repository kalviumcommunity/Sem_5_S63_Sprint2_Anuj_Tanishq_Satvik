"""Embedding service interface and implementations."""

from abc import ABC, abstractmethod
import math
from typing import List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.embeddings.batcher import EmbeddingBatcher
from backend.app.services.embeddings.cache import EmbeddingCache


class EmbeddingServiceInterface(ABC):
    """Abstract interface for generating vector embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality of the embedding model."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts."""
        pass

    @staticmethod
    def normalize_vector(vec: List[float]) -> List[float]:
        """Normalize vector to unit length (L2 norm = 1.0)."""
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]


class MockEmbeddingService(EmbeddingServiceInterface):
    """Deterministic mock embedding service for development, CI, and testing."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        """Generate pseudo-normalized deterministic vector from text hash."""
        seed = sum(ord(c) for c in text) if text else 42
        vec = [math.sin(seed + i * 0.1) for i in range(self._dimension)]
        return self.normalize_vector(vec)

    async def embed_text(self, text: str) -> List[float]:
        return self._generate_vector(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]


class OpenAIEmbeddingService(EmbeddingServiceInterface):
    """OpenAI Embeddings API client with fallback to mock service when keys are absent."""

    def __init__(self, api_key: str, model: str, dimension: int = 1536):
        self.api_key = api_key
        self.model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        batch = await self.embed_batch([text])
        return batch[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key or self.api_key.startswith("your-") or "placeholder" in self.api_key:
            logger.warning("OpenAI API key missing for embeddings, using mock service.")
            return await MockEmbeddingService(dimension=self._dimension).embed_batch(texts)

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
            return [self.normalize_vector(item["embedding"]) for item in data["data"]]


class CachedEmbeddingService(EmbeddingServiceInterface):
    """Decorator around EmbeddingServiceInterface that transparently caches vectors."""

    def __init__(self, base_service: EmbeddingServiceInterface, cache: Optional[EmbeddingCache] = None):
        self.base_service = base_service
        self.cache = cache or EmbeddingCache(max_size=2000)

    @property
    def dimension(self) -> int:
        return self.base_service.dimension

    async def embed_text(self, text: str) -> List[float]:
        cached = self.cache.get(text, model_or_dim=str(self.dimension))
        if cached is not None:
            return cached

        vector = await self.base_service.embed_text(text)
        self.cache.set(text, vector, model_or_dim=str(self.dimension))
        return vector

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        cached_results, missing_indices = self.cache.get_batch(texts, model_or_dim=str(self.dimension))

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            computed_vectors = await self.base_service.embed_batch(missing_texts)
            self.cache.set_batch(missing_texts, computed_vectors, model_or_dim=str(self.dimension))

            for idx, vec in zip(missing_indices, computed_vectors):
                cached_results[idx] = vec

        return [res for res in cached_results if res is not None]


_default_cache = EmbeddingCache(max_size=2000)


def get_embedding_service(use_cache: bool = True) -> EmbeddingServiceInterface:
    """Factory to retrieve configured embedding service, optionally wrapped with caching."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    dimension = settings.EMBEDDING_DIMENSION

    if provider == "openai":
        svc = OpenAIEmbeddingService(
            api_key=settings.EMBEDDING_API_KEY,
            model=settings.EMBEDDING_MODEL,
            dimension=dimension,
        )
    else:
        svc = MockEmbeddingService(dimension=dimension)

    if use_cache:
        return CachedEmbeddingService(base_service=svc, cache=_default_cache)
    return svc
