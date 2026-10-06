"""Vector similarity search implementation."""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class SearchResult(BaseModel):
    """Result item returned by retrieval search."""

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page_number: int
    section: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = {}


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two numeric vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
