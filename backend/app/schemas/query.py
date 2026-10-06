"""Pydantic schemas for queries."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming user query payload."""

    query: str = Field(..., min_length=1, description="Student's research question")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    top_k: Optional[int] = Field(None, description="Number of source chunks to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters (course, author, document_type)")
