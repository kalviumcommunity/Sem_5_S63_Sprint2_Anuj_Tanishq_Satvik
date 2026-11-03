"""Embedding service interface and implementations."""

from abc import ABC, abstractmethod
import hashlib
import math
import re
from typing import List, Optional

import httpx

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.embeddings.batcher import chunk_list
from backend.app.services.embeddings.cache import EmbeddingCache


class EmbeddingServiceError(RuntimeError):
    """Safe error raised when the embedding provider cannot return valid vectors."""


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
        scale = max((abs(value) for value in vec), default=0.0)
        if scale == 0.0:
            return vec
        scaled = [value / scale for value in vec]
        norm = math.sqrt(sum(value * value for value in scaled))
        return [value / norm for value in scaled]


class MockEmbeddingService(EmbeddingServiceInterface):
    """Deterministic lexical embedding for offline development, CI, and testing."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        """Hash word features into a normalized vector; this is not a semantic model."""
        vec = [0.0] * self._dimension
        for token in re.findall(r"\w+", text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimension
            vec[index] += 1.0 if digest[8] & 1 else -1.0
        return self.normalize_vector(vec)

    async def embed_text(self, text: str) -> List[float]:
        return self._generate_vector(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(text) for text in texts]


class OpenAIEmbeddingService(EmbeddingServiceInterface):
    """OpenAI Embeddings API client with bounded batches and response validation."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int = 1536,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.api_key = api_key
        self.model = model
        self._dimension = dimension
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        batch = await self.embed_batch([text])
        return batch[0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.api_key or self.api_key.startswith("your-") or "placeholder" in self.api_key:
            logger.warning("OpenAI API key missing for embeddings, using mock service.")
            return await MockEmbeddingService(dimension=self._dimension).embed_batch(texts)

        vectors: List[List[float]] = []
        for batch in chunk_list(texts, self.batch_size):
            vectors.extend(await self._embed_api_batch(batch))
        return vectors

    async def _embed_api_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": texts},
                )
        except httpx.HTTPError:
            raise EmbeddingServiceError("Embedding provider request failed.") from None

        if response.status_code >= 400:
            raise EmbeddingServiceError(
                f"Embedding provider request failed with status {response.status_code}."
            )

        try:
            body = response.json()
        except (TypeError, ValueError):
            raise EmbeddingServiceError("Embedding provider returned invalid JSON.") from None

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingServiceError("Embedding provider returned an invalid response.")

        indexed_vectors = []
        for item in data:
            if not isinstance(item, dict):
                raise EmbeddingServiceError("Embedding provider returned an invalid response.")
            index = item.get("index")
            vector = item.get("embedding")
            if isinstance(index, bool) or not isinstance(index, int):
                raise EmbeddingServiceError("Embedding provider returned invalid vector indexes.")
            if not isinstance(vector, list):
                raise EmbeddingServiceError("Embedding provider returned an invalid vector.")
            try:
                validate_embedding(vector, self._dimension)
            except (OverflowError, TypeError, ValueError):
                raise EmbeddingServiceError("Embedding provider returned an invalid vector.") from None
            indexed_vectors.append((index, self.normalize_vector(vector)))

        if sorted(index for index, _ in indexed_vectors) != list(range(len(texts))):
            raise EmbeddingServiceError("Embedding provider returned invalid vector indexes.")

        vectors = [vector for _, vector in sorted(indexed_vectors)]
        for vector in vectors:
            validate_embedding(vector, self._dimension)
        return vectors


def validate_embedding(vector: List[float], expected_dimension: int) -> None:
    """Reject malformed vectors before indexing or retrieval."""
    if not isinstance(vector, list) or len(vector) != expected_dimension:
        actual_dimension = len(vector) if isinstance(vector, list) else "invalid"
        raise ValueError(
            f"Embedding dimension {actual_dimension} does not match configured dimension {expected_dimension}."
        )
    try:
        invalid_value = any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in vector
        )
    except OverflowError:
        invalid_value = True
    if not vector or invalid_value:
        raise ValueError("Embedding values must be finite and the vector must not be empty.")


class CachedEmbeddingService(EmbeddingServiceInterface):
    """Decorator around EmbeddingServiceInterface that transparently caches vectors."""

    def __init__(self, base_service: EmbeddingServiceInterface, cache: Optional[EmbeddingCache] = None):
        self.base_service = base_service
        self.cache = cache or EmbeddingCache(max_size=2000)
        self._cache_model = (
            f"{getattr(base_service, 'model', type(base_service).__name__)}:{base_service.dimension}"
        )

    @property
    def dimension(self) -> int:
        return self.base_service.dimension

    async def embed_text(self, text: str) -> List[float]:
        cached = self.cache.get(text, model_or_dim=self._cache_model)
        if cached is not None:
            return cached

        vector = await self.base_service.embed_text(text)
        validate_embedding(vector, self.dimension)
        self.cache.set(text, vector, model_or_dim=self._cache_model)
        return vector

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        cached_results, missing_indices = self.cache.get_batch(texts, model_or_dim=self._cache_model)

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            computed_vectors = await self.base_service.embed_batch(missing_texts)
            if len(computed_vectors) != len(missing_texts):
                raise ValueError("Embedding count does not match the number of input texts.")
            for vector in computed_vectors:
                validate_embedding(vector, self.dimension)
            self.cache.set_batch(missing_texts, computed_vectors, model_or_dim=self._cache_model)

            for idx, vec in zip(missing_indices, computed_vectors):
                cached_results[idx] = vec

        if any(result is None for result in cached_results):
            raise ValueError("Embedding cache returned incomplete results.")
        return [result for result in cached_results if result is not None]


_default_cache = EmbeddingCache(max_size=2000)


def get_embedding_service(use_cache: bool = True) -> EmbeddingServiceInterface:
    """Factory to retrieve configured embedding service, optionally wrapped with caching."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    dimension = settings.EMBEDDING_DIMENSION

    if provider == "openai":
        service = OpenAIEmbeddingService(
            api_key=settings.EMBEDDING_API_KEY,
            model=settings.EMBEDDING_MODEL,
            dimension=dimension,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    else:
        service = MockEmbeddingService(dimension=dimension)

    if use_cache:
        return CachedEmbeddingService(base_service=service, cache=_default_cache)
    return service
