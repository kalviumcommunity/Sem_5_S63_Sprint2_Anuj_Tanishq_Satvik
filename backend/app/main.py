"""ResearchMate FastAPI Application Entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.api.v1 import api_v1_router


def create_app() -> FastAPI:
    """Application factory for ResearchMate."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="AI-powered academic research assistant with citation traceability.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root health endpoint
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "status": "online",
            "version": "0.1.0",
            "docs": "/docs",
            "principle": "No unsupported answer. No hidden source.",
        }

    # Include API routers
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)
    # Also mount health directly at root /health for convenience
    from backend.app.api.v1.health import health_check

    app.add_api_route("/health", health_check, methods=["GET"], tags=["Health"])

    logger.info("ResearchMate application initialized successfully.")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
