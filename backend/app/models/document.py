"""Document data model."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class ProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported academic document formats."""

    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

    @classmethod
    def from_filename(cls, filename: str) -> "DocumentType":
        """Determine document type from file extension."""
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return cls.PDF
        elif lower.endswith(".txt"):
            return cls.TXT
        elif lower.endswith(".docx"):
            return cls.DOCX
        elif lower.endswith((".md", ".markdown")):
            return cls.MARKDOWN
        return cls.UNKNOWN


class DocumentPage(BaseModel):
    """Represents a page or discrete logical section in an ingested document."""

    page_number: int = Field(..., ge=1, description="1-indexed page or section sequence number")
    text: str = Field(..., description="Extracted textual content of the page")
    char_count: int = Field(default=0, description="Character count for this page")
    metadata: dict = Field(default_factory=dict, description="Page-specific metadata")

    def model_post_init(self, __context) -> None:
        if not self.char_count:
            self.char_count = len(self.text)


class LoadedDocument(BaseModel):
    """Common document representation independent of source format."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = Field(..., description="Original filename")
    source_path: str = Field(..., description="Filesystem path or URI of source file")
    document_type: DocumentType = Field(..., description="Detected format category")
    raw_text: str = Field(default="", description="Consolidated full-text content")
    pages: List[DocumentPage] = Field(default_factory=list, description="Extracted per-page text units")
    page_count: int = Field(default=0, description="Total number of pages or sections")
    char_count: int = Field(default=0, description="Total character count")
    file_hash: Optional[str] = Field(default=None, description="SHA256 checksum of original file")
    metadata: dict = Field(default_factory=dict, description="Additional document metadata")

    def model_post_init(self, __context) -> None:
        if not self.page_count:
            self.page_count = len(self.pages)
        if not self.char_count:
            self.char_count = sum(p.char_count for p in self.pages) if self.pages else len(self.raw_text)


class Document(BaseModel):
    """Represents an ingested academic document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    filename: str
    document_type: str = "research_paper"
    author: Optional[str] = None
    course: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_hash: str
    page_count: int = 0
    processing_status: ProcessingStatus = ProcessingStatus.UPLOADED
