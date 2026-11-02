"""Query API router."""

from fastapi import APIRouter, HTTPException
from backend.app.core.config import settings
from backend.app.schemas.query import QueryRequest
from backend.app.schemas.response import QueryResponse
from backend.app.services.embeddings.service import get_embedding_service, validate_embedding
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.services.retrieval.search import get_vector_store

router = APIRouter(tags=["Query"])
pipeline = RAGPipeline()


@router.post("/query", response_model=QueryResponse)
async def query_academic_sources(request: QueryRequest) -> QueryResponse:
    """Submit a research question to get a grounded answer with citations."""
    try:
        embedding_service = get_embedding_service()
        query_vector = await embedding_service.embed_text(request.query)
        validate_embedding(query_vector, embedding_service.dimension)
        candidates = get_vector_store().search(
            query_vector=query_vector,
            top_k=request.top_k or settings.TOP_K_RETRIEVAL,
            score_threshold=settings.SIMILARITY_THRESHOLD,
            filters=request.filters,
        )
        return await pipeline.answer(
            query=request.query,
            candidates=candidates,
            conversation_id=request.conversation_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
