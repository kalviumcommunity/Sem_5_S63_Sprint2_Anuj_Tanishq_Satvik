"""Pydantic schemas for document operations."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


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
