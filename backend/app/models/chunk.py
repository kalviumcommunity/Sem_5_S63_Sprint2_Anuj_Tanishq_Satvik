"""Chunk data model."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import uuid


class Chunk(BaseModel):
    """Represents a discrete text chunk extracted from a document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int
    text: str
    page_number: int = 1
    section: Optional[str] = None
    token_count: int = 0
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
