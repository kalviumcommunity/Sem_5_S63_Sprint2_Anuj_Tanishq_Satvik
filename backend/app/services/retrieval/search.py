"""Vector similarity search implementation and metadata-preserving SearchResult."""

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
from backend.app.models.chunk import Chunk


class SearchResult(BaseModel):
    """Result item returned by retrieval search preserving complete attribution metadata."""

    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page_number: int = 1
    section: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_metadata_consistency(self) -> "SearchResult":
        """Guarantee canonical metadata keys exist and are populated."""
        src = self.metadata.get("source") or self.document_title or self.document_id
        chunk_idx = self.metadata.get("chunk_index", 0)

        meta: Dict[str, Any] = {
            "source": src,
            "chunk_index": chunk_idx,
            "char_start": self.metadata.get("char_start"),
            "char_end": self.metadata.get("char_end"),
            "page_number": self.page_number if self.page_number is not None else self.metadata.get("page_number"),
            "section": self.section if self.section is not None else self.metadata.get("section"),
            "url": self.metadata.get("url"),
            "document_version": self.metadata.get("document_version"),
            "document_id": self.document_id,
        }

        # Preserve other arbitrary keys
        for k, v in self.metadata.items():
            if k not in meta:
                meta[k] = v

        self.metadata = meta
        return self

    @property
    def source(self) -> str:
        """Convenience property for source identifier."""
        return self.metadata.get("source") or self.document_title

    def __getitem__(self, item: str) -> Any:
        """Allow subscript indexing like hit['text'], hit['metadata'], hit['score']."""
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.metadata:
            return self.metadata[item]
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        """Dictionary-like safe get method."""
        if hasattr(self, item):
            return getattr(self, item)
        return self.metadata.get(item, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result into dictionary structure."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "text": self.text,
            "page_number": self.page_number,
            "section": self.section,
            "score": self.score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        score: float = 0.0,
        document_title: Optional[str] = None,
    ) -> "SearchResult":
        """Construct SearchResult from a stored Chunk, preserving full metadata."""
        title = document_title or chunk.metadata.get("document_title") or chunk.metadata.get("source") or chunk.document_id
        return cls(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            text=chunk.text,
            page_number=chunk.page_number,
            section=chunk.section,
            score=score,
            metadata=dict(chunk.metadata),
        )


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
