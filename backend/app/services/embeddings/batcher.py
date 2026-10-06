"""Batching helper for embedding pipelines."""

from typing import List, TypeVar

T = TypeVar("T")


def chunk_list(items: List[T], batch_size: int) -> List[List[T]]:
    """Divide a list into fixed size chunks for API batching."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
