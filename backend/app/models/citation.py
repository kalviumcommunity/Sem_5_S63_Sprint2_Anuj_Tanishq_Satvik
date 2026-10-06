"""Citation data model."""

from typing import Optional
from pydantic import BaseModel, Field
import uuid


class Citation(BaseModel):
    """Represents a validated citation linking an answer to a source chunk."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer_id: Optional[str] = None
    document_id: str
    chunk_id: str
    page_number: int
    document_title: str
    relevance_score: float = 0.0
    citation_text: str
    section: Optional[str] = None
