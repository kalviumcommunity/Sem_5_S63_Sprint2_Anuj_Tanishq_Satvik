"""Query API router."""

from fastapi import APIRouter, HTTPException
from backend.app.core.config import settings
from backend.app.schemas.query import QueryRequest
from backend.app.schemas.response import QueryResponse
from backend.app.core.logging import logger
from backend.app.services.embeddings.service import (
    EmbeddingServiceError,
    get_embedding_service,
    validate_embedding,
)
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
        uses_mock_embeddings = bool(getattr(embedding_service, "uses_mock_embeddings", False))
        similarity_threshold = (
            settings.MOCK_SIMILARITY_THRESHOLD
            if uses_mock_embeddings
            else settings.SIMILARITY_THRESHOLD
        )
        vector_store = get_vector_store()
        candidates = vector_store.search(
            query_vector=query_vector,
            top_k=(
                max(
                    request.top_k or settings.TOP_K_RETRIEVAL,
                    settings.MOCK_TOP_K_RETRIEVAL,
                )
                if uses_mock_embeddings
                else request.top_k or settings.TOP_K_RETRIEVAL
            ),
            score_threshold=similarity_threshold,
            filters=request.filters,
        )
        top_score = candidates[0].score if candidates else 0.0
        logger.info(
            "Retrieval completed: embedding_mode=%s candidates=%d top_score=%.4f threshold=%.4f",
            "mock" if uses_mock_embeddings else "provider",
            len(candidates),
            top_score,
            similarity_threshold,
        )
        if uses_mock_embeddings:
            return await pipeline.answer(
                query=request.query,
                candidates=candidates,
                conversation_id=request.conversation_id,
                grounding_threshold=settings.MOCK_SIMILARITY_THRESHOLD,
                mock_embeddings=True,
            )
        return await pipeline.answer(
            query=request.query,
            candidates=candidates,
            conversation_id=request.conversation_id,
        )
    except EmbeddingServiceError:
        logger.warning("Embedding provider failed while processing a query.")
        raise HTTPException(status_code=502, detail="Embedding provider request failed.") from None
    except Exception:
        logger.exception("Query processing failed.")
        raise HTTPException(status_code=500, detail="Unable to process query.") from None
