"""Embedding caching layer to prevent redundant vector generation."""

from collections import OrderedDict
import hashlib
from typing import Dict, List, Optional, Tuple


class EmbeddingCache:
    """Thread-safe LRU cache for text embeddings."""

    def __init__(self, max_size: int = 1000):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def compute_key(text: str, model_or_dim: str = "default") -> str:
        """Compute deterministic hash key for text and model/dimension."""
        hasher = hashlib.sha256()
        hasher.update(model_or_dim.encode("utf-8"))
        hasher.update(b":")
        hasher.update(text.strip().encode("utf-8"))
        return hasher.hexdigest()

    def get(self, text: str, model_or_dim: str = "default") -> Optional[List[float]]:
        """Retrieve cached embedding vector if present."""
        key = self.compute_key(text, model_or_dim)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, text: str, vector: List[float], model_or_dim: str = "default") -> None:
        """Store embedding vector in cache, evicting oldest item if max_size exceeded."""
        key = self.compute_key(text, model_or_dim)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = vector
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def get_batch(
        self, texts: List[str], model_or_dim: str = "default"
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """Look up batch of texts. Returns list of cached vectors (or None) and indices of misses."""
        results: List[Optional[List[float]]] = []
        missing_indices: List[int] = []

        for idx, text in enumerate(texts):
            cached = self.get(text, model_or_dim)
            results.append(cached)
            if cached is None:
                missing_indices.append(idx)

        return results, missing_indices

    def set_batch(self, texts: List[str], vectors: List[List[float]], model_or_dim: str = "default") -> None:
        """Store batch of embeddings."""
        for text, vec in zip(texts, vectors):
            self.set(text, vec, model_or_dim)

    def clear(self) -> None:
        """Clear cache and reset hit/miss counters."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
