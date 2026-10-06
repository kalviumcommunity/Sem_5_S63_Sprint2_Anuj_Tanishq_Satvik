"""Query API router."""

from fastapi import APIRouter, HTTPException
from backend.app.schemas.query import QueryRequest
from backend.app.schemas.response import QueryResponse
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.services.retrieval.search import SearchResult

router = APIRouter(tags=["Query"])
pipeline = RAGPipeline()


@router.post("/query", response_model=QueryResponse)
async def query_academic_sources(request: QueryRequest) -> QueryResponse:
    """Submit a research question to get a grounded answer with citations."""
    try:
        # In Concept 1, we provide the real pipeline with candidate structure
        # (Subsequent concepts will wire up live vector retrieval)
        candidates: list[SearchResult] = []
        return await pipeline.answer(
            query=request.query,
            candidates=candidates,
            conversation_id=request.conversation_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
