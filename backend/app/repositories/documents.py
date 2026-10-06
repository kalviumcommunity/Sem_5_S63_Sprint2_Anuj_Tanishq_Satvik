"""Repository for storing and querying documents."""

from typing import Dict, List, Optional
from backend.app.models.document import Document


class DocumentRepository:
    """In-memory repository for academic documents."""

    def __init__(self):
        self._storage: Dict[str, Document] = {}

    def save(self, doc: Document) -> Document:
        self._storage[doc.id] = doc
        return doc

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        return self._storage.get(doc_id)

    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        for doc in self._storage.values():
            if doc.file_hash == file_hash:
                return doc
        return None

    def list_all(self) -> List[Document]:
        return list(self._storage.values())
