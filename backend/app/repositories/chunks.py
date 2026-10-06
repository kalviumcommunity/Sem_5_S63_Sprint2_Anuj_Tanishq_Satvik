"""Repository for storing and querying text chunks."""

from typing import Dict, List
from backend.app.models.chunk import Chunk


class ChunkRepository:
    """In-memory repository for document chunks."""

    def __init__(self):
        self._storage: Dict[str, Chunk] = {}

    def save(self, chunk: Chunk) -> Chunk:
        self._storage[chunk.id] = chunk
        return chunk

    def save_many(self, chunks: List[Chunk]) -> List[Chunk]:
        for c in chunks:
            self._storage[c.id] = c
        return chunks

    def get_by_document(self, doc_id: str) -> List[Chunk]:
        return [c for c in self._storage.values() if c.document_id == doc_id]

    def list_all(self) -> List[Chunk]:
        return list(self._storage.values())
