"""Pydantic schemas for document operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned upon document upload."""

    document_id: str
    filename: str
    status: str
    chunks: int
    message: str


class DocumentInfo(BaseModel):
    """Document metadata information schema."""

    id: str
    title: str
    filename: str
    document_type: str
    author: Optional[str] = None
    course: Optional[str] = None
    page_count: int
    processing_status: str
    uploaded_at: datetime


class ChunkResponse(BaseModel):
    """Schema representing an individual document chunk."""

    id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int
    section: Optional[str] = None
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrategyComparisonResponse(BaseModel):
    """Response schema comparing document chunking strategies."""

    fixed_size: Dict[str, Any]
    recursive: Dict[str, Any]
    chosen_strategy: str
    rationale: str
