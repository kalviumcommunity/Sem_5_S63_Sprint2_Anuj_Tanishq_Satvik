"""Repository for storing and querying documents."""

from threading import RLock
from typing import Dict, List, Optional
from backend.app.models.document import Document


class DocumentRepository:
    """In-memory repository for academic documents."""

    def __init__(self):
        self._storage: Dict[str, Document] = {}
        self._hash_index: Dict[str, str] = {}
        self._lock = RLock()

    def save(self, doc: Document) -> Document:
        with self._lock:
            previous = self._storage.get(doc.id)
            if previous and previous.file_hash != doc.file_hash:
                self._hash_index.pop(previous.file_hash, None)
            self._storage[doc.id] = doc
            self._hash_index[doc.file_hash] = doc.id
        return doc

    def reserve(self, doc: Document) -> tuple[Document, bool]:
        """Atomically register a new file hash or return the existing document."""
        with self._lock:
            existing_id = self._hash_index.get(doc.file_hash)
            if existing_id:
                return self._storage[existing_id], False
            self._storage[doc.id] = doc
            self._hash_index[doc.file_hash] = doc.id
            return doc, True

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            return self._storage.get(doc_id)

    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        with self._lock:
            doc_id = self._hash_index.get(file_hash)
            return self._storage.get(doc_id) if doc_id else None

    def list_all(self) -> List[Document]:
        with self._lock:
            return list(self._storage.values())
