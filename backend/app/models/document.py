"""Document data model."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class ProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


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
