"""Documents API router."""

from typing import List
from fastapi import APIRouter
from backend.app.schemas.document import DocumentInfo

router = APIRouter(tags=["Documents"])


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents() -> List[DocumentInfo]:
    """List indexed documents."""
    return []
