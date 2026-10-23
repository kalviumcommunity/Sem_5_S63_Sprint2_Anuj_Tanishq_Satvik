"""Repository for storing and querying text chunks and vector embeddings."""

from typing import Any, Dict, List, Optional
from backend.app.models.chunk import Chunk
from backend.app.services.retrieval.search import SearchResult, cosine_similarity
from backend.app.services.retrieval.filters import MetadataFilter


class ChunkRepository:
    """In-memory repository and vector store for document chunks with metadata preservation."""

    def __init__(self):
        self._storage: Dict[str, Chunk] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def save(self, chunk: Chunk) -> Chunk:
        """Store chunk and record embedding if present."""
        self._storage[chunk.id] = chunk
        if chunk.embedding is not None:
            self._embeddings[chunk.id] = chunk.embedding
        return chunk

    def save_many(self, chunks: List[Chunk]) -> List[Chunk]:
        """Batch store chunks and embeddings."""
        for c in chunks:
            self.save(c)
        return chunks

    def store_embedding(self, chunk_id: str, embedding: List[float]) -> None:
        """Associate embedding vector directly with stored chunk."""
        self._embeddings[chunk_id] = embedding
        if chunk_id in self._storage:
            self._storage[chunk_id].embedding = embedding

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch chunk by its unique identifier."""
        return self._storage.get(chunk_id)

    def get_by_document(self, doc_id: str) -> List[Chunk]:
        """Retrieve all chunks belonging to a document."""
        return [c for c in self._storage.values() if c.document_id == doc_id]

    def list_all(self) -> List[Chunk]:
        """List all stored chunks."""
        return list(self._storage.values())

    def search_vector(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform vector similarity search returning SearchResults with full metadata."""
        if not self._embeddings or not query_vector:
            return []

        candidates: List[SearchResult] = []
        for chunk_id, emb in self._embeddings.items():
            chunk = self._storage.get(chunk_id)
            if not chunk:
                continue

            sim = cosine_similarity(query_vector, emb)
            res = SearchResult.from_chunk(chunk=chunk, score=round(sim, 4))
            candidates.append(res)

        # Apply metadata filters if provided
        if filters:
            candidates = MetadataFilter.apply(candidates, filters)

        # Sort descending by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    def migrate_legacy_records(self) -> int:
        """Normalize existing stored chunks to the consistent metadata schema."""
        migrated_count = 0
        for chunk_id, chunk in list(self._storage.items()):
            migrated = Chunk.migrate_legacy_chunk(chunk)
            self._storage[chunk_id] = migrated
            migrated_count += 1
        return migrated_count

    def clear(self) -> None:
        """Clear all stored chunks and embeddings."""
        self._storage.clear()
        self._embeddings.clear()

    def count(self) -> int:
        """Return total number of stored chunks."""
        return len(self._storage)
