"""Batching and pipeline execution helper for embedding workflows."""

import asyncio
from typing import Callable, Coroutine, List, Optional, TypeVar
from backend.app.core.logging import logger

T = TypeVar("T")


def chunk_list(items: List[T], batch_size: int) -> List[List[T]]:
    """Divide a list into fixed size chunks for API batching."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


class EmbeddingBatcher:
    """Orchestrates batch generation of embeddings with token limits, retries, and rate limiting."""

    def __init__(self, default_batch_size: int = 16, max_retries: int = 3, retry_delay: float = 0.5):
        if default_batch_size <= 0:
            raise ValueError("default_batch_size must be positive")
        self.default_batch_size = default_batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def create_batches(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[str]]:
        """Split a list of text items into batches."""
        size = batch_size or self.default_batch_size
        return chunk_list(texts, size)

    async def execute_batch_pipeline(
        self,
        texts: List[str],
        embed_fn: Callable[[List[str]], Coroutine[any, any, List[List[float]]]],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Process texts through embed_fn in batches with exponential backoff on transient errors."""
        if not texts:
            return []

        batches = self.create_batches(texts, batch_size)
        all_embeddings: List[List[float]] = []

        for batch_idx, batch in enumerate(batches):
            attempt = 0
            success = False
            last_err = None

            while attempt < self.max_retries and not success:
                try:
                    batch_vectors = await embed_fn(batch)
                    if len(batch_vectors) != len(batch):
                        raise ValueError(
                            f"embed_fn returned {len(batch_vectors)} vectors for batch of {len(batch)} items"
                        )
                    all_embeddings.extend(batch_vectors)
                    success = True
                except Exception as exc:
                    attempt += 1
                    last_err = exc
                    logger.warning(
                        f"Embedding batch {batch_idx + 1}/{len(batches)} failed (attempt {attempt}/{self.max_retries}): {exc}"
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))

            if not success:
                raise RuntimeError(
                    f"Failed to generate embeddings for batch {batch_idx + 1} after {self.max_retries} attempts: {last_err}"
                )

        return all_embeddings
