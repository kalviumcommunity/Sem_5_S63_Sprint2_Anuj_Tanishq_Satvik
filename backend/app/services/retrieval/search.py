"""Vector similarity search and in-memory vector store implementations."""

from enum import Enum
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from backend.app.models.chunk import Chunk


class SimilarityMetric(str, Enum):
    """Supported vector distance and similarity calculation methods."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


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
    """Calculate cosine similarity between two numeric vectors in [-1.0, 1.0]."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def dot_product(v1: List[float], v2: List[float]) -> float:
    """Calculate raw dot product between two numeric vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Calculate Euclidean distance (L2 distance) between two numeric vectors."""
    if len(v1) != len(v2) or not v1:
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


class InMemoryVectorStore:
    """In-memory vector store supporting fast similarity search, thresholding, and metadata filtering."""

    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}

    def add_item(
        self,
        chunk_id: str,
        vector: List[float],
        document_id: str,
        document_title: str,
        text: str,
        page_number: int = 1,
        section: Optional[str] = "main",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update a vector record in the store."""
        self._items[chunk_id] = {
            "chunk_id": chunk_id,
            "vector": vector,
            "document_id": document_id,
            "document_title": document_title,
            "text": text,
            "page_number": page_number,
            "section": section,
            "metadata": metadata or {},
        }

    def add_chunks(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
        document_title: str = "Research Document",
    ) -> int:
        """Bulk index a list of chunks with their corresponding embedding vectors."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) does not match number of vectors ({len(vectors)})"
            )

        added = 0
        for chunk, vec in zip(chunks, vectors):
            self.add_item(
                chunk_id=chunk.id,
                vector=vec,
                document_id=chunk.document_id,
                document_title=chunk.metadata.get("filename", document_title),
                text=chunk.text,
                page_number=chunk.page_number,
                section=chunk.section,
                metadata=chunk.metadata,
            )
            added += 1
        return added

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> List[SearchResult]:
        """Search store for nearest neighbors matching query vector, filters, and score threshold."""
        from backend.app.services.retrieval.filters import MetadataFilter

        if not self._items or not query_vector:
            return []

        candidates: List[SearchResult] = []

        for item in self._items.values():
            vec = item["vector"]
            if len(vec) != len(query_vector):
                continue

            if metric == SimilarityMetric.COSINE:
                score = cosine_similarity(query_vector, vec)
            elif metric == SimilarityMetric.DOT_PRODUCT:
                score = dot_product(query_vector, vec)
            elif metric == SimilarityMetric.EUCLIDEAN:
                dist = euclidean_distance(query_vector, vec)
                # Convert distance to a bounded similarity score in [0.0, 1.0]
                score = 1.0 / (1.0 + dist)
            else:
                score = cosine_similarity(query_vector, vec)

            if score < score_threshold:
                continue

            res = SearchResult(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                document_title=item["document_title"],
                text=item["text"],
                page_number=item["page_number"],
                section=item["section"],
                score=round(score, 4),
                metadata=item["metadata"],
            )
            candidates.append(res)

        # Apply metadata filters
        if filters:
            candidates = MetadataFilter.apply(candidates, filters)

        # Sort descending by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    def delete(self, chunk_id: str) -> bool:
        """Remove a single chunk from the store."""
        if chunk_id in self._items:
            del self._items[chunk_id]
            return True
        return False

    def delete_by_document(self, document_id: str) -> int:
        """Remove all chunks associated with a specific document ID."""
        to_delete = [
            cid for cid, item in self._items.items() if item["document_id"] == document_id
        ]
        for cid in to_delete:
            del self._items[cid]
        return len(to_delete)

    def count(self) -> int:
        """Return total number of vectors in the store."""
        return len(self._items)

    def clear(self) -> None:
        """Clear all stored vectors."""
        self._items.clear()


_global_vector_store = InMemoryVectorStore()


def get_vector_store() -> InMemoryVectorStore:
    """Retrieve shared in-memory vector store instance."""
    return _global_vector_store
