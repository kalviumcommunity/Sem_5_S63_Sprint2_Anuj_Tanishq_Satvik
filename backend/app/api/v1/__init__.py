"""API v1 router composition."""

from fastapi import APIRouter
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.query import router as query_router
from backend.app.api.v1.documents import router as documents_router
from backend.app.api.v1.conversations import router as conversations_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(query_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(conversations_router)

__all__ = ["api_v1_router"]
