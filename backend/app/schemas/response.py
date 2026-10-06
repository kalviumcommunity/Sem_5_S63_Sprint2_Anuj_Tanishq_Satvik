"""Pydantic schemas for API responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CitationItem(BaseModel):
    """Citation item in query response."""

    document_id: str
    title: str
    page: int
    chunk_id: str
    section: Optional[str] = None
    relevance_score: Optional[float] = None
    citation_text: Optional[str] = None


class QueryResponse(BaseModel):
    """Response returned for student query."""

    answer: str
    citations: List[CitationItem] = Field(default_factory=list)
    grounded: bool = True
    conversation_id: Optional[str] = None
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health status response."""

    status: str
    app_name: str
    version: str
    environment: str
    services: Dict[str, str] = Field(default_factory=dict)
