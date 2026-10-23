"""Chunk data model and metadata schemas."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
import uuid


class ChunkMetadata(BaseModel):
    """Consistent metadata schema for document chunks."""

    source: str = Field(..., description="Original document identifier, filename, or canonical source")
    chunk_index: int = Field(..., description="Chunk's position within its source document")
    char_start: Optional[int] = Field(default=None, description="Starting character offset in source text")
    char_end: Optional[int] = Field(default=None, description="Ending character offset in source text")
    page_number: Optional[int] = Field(default=None, description="Original page number")
    section: Optional[str] = Field(default=None, description="Relevant section heading")
    url: Optional[str] = Field(default=None, description="Canonical document URL")
    document_version: Optional[str] = Field(default=None, description="Version or effective date")
    document_id: Optional[str] = Field(default=None, description="Stable document identifier")

    model_config = {"extra": "allow"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary including extra fields."""
        base = {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_number": self.page_number,
            "section": self.section,
            "url": self.url,
            "document_version": self.document_version,
            "document_id": self.document_id,
        }
        # Add any extra attributes
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            base.update(self.__pydantic_extra__)
        return base


class Chunk(BaseModel):
    """Represents a discrete text chunk extracted from a document with full metadata tracking."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int
    text: str
    page_number: int = 1
    section: Optional[str] = None
    token_count: int = 0
    embedding_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    @model_validator(mode="after")
    def populate_metadata_schema(self) -> "Chunk":
        """Ensure consistent metadata schema is present on every chunk."""
        # Determine source
        src = self.source or self.metadata.get("source") or self.document_id
        self.source = src

        # Sync character offsets
        c_start = self.char_start if self.char_start is not None else self.metadata.get("char_start")
        c_end = self.char_end if self.char_end is not None else self.metadata.get("char_end")
        self.char_start = c_start
        self.char_end = c_end

        # Build consistent metadata dict
        consistent_meta: Dict[str, Any] = {
            "source": src,
            "chunk_index": self.metadata.get("chunk_index", self.chunk_index),
            "char_start": c_start,
            "char_end": c_end,
            "page_number": self.page_number if self.page_number is not None else self.metadata.get("page_number"),
            "section": self.section if self.section is not None else self.metadata.get("section"),
            "url": self.metadata.get("url"),
            "document_version": self.metadata.get("document_version"),
            "document_id": self.document_id or self.metadata.get("document_id"),
        }

        # Preserve any existing extra metadata fields
        for k, v in self.metadata.items():
            if k not in consistent_meta:
                consistent_meta[k] = v

        self.metadata = consistent_meta
        return self

    def __getitem__(self, item: str) -> Any:
        """Allow dict-style subscripting (e.g. chunk['text'], chunk['metadata'])."""
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.metadata:
            return self.metadata[item]
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        """Safe retrieval matching dict.get interface."""
        if hasattr(self, item):
            return getattr(self, item)
        return self.metadata.get(item, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into conceptual {text, metadata} structure."""
        return {
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def migrate_legacy_chunk(cls, legacy_record: Union[Dict[str, Any], "Chunk"]) -> "Chunk":
        """Safely migrate legacy chunk records without losing identity or fabricating metadata."""
        if isinstance(legacy_record, Chunk):
            return legacy_record

        doc_id = legacy_record.get("document_id") or legacy_record.get("doc_id", "unknown_doc")
        chunk_idx = legacy_record.get("chunk_index", 0)
        text = legacy_record.get("text", "")
        raw_meta = legacy_record.get("metadata", {})

        return cls(
            id=legacy_record.get("id") or str(uuid.uuid4()),
            document_id=doc_id,
            chunk_index=chunk_idx,
            text=text,
            page_number=legacy_record.get("page_number", raw_meta.get("page_number", 1)),
            section=legacy_record.get("section", raw_meta.get("section")),
            token_count=legacy_record.get("token_count", 0),
            embedding_id=legacy_record.get("embedding_id"),
            embedding=legacy_record.get("embedding"),
            source=legacy_record.get("source") or raw_meta.get("source") or doc_id,
            char_start=legacy_record.get("char_start") or raw_meta.get("char_start"),
            char_end=legacy_record.get("char_end") or raw_meta.get("char_end"),
            metadata=dict(raw_meta),
        )
