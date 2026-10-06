"""Health check API endpoint."""

from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.response import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system health status and configuration info."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version="0.1.0",
        environment=settings.APP_ENV,
        services={
            "api": "online",
            "llm_provider": settings.LLM_PROVIDER,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "vector_store": settings.VECTOR_DB_TYPE,
        },
    )
