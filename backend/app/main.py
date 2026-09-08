"""InsightRAG FastAPI Main Application Entrypoint."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1.router import api_v1_router
from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging

settings = get_settings()
logger = setup_logging(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown events."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} in [{settings.ENVIRONMENT}] mode")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


def create_app() -> FastAPI:
    """Application factory for InsightRAG backend."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "InsightRAG: Advanced Knowledge Intelligence System.\n\n"
            "An enterprise-grade, portfolio-level RAG platform featuring hybrid search (Dense + BM25), "
            "Reciprocal Rank Fusion (RRF), Cross-Encoder Reranking, strict grounded generation, and citation tracking."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API v1 router
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    # Root welcome endpoint
    @app.get("/", tags=["Root"])
    async def root():
        return JSONResponse(
            content={
                "message": f"Welcome to {settings.APP_NAME} API",
                "version": settings.APP_VERSION,
                "docs_url": "/docs",
                "health_url": f"{settings.API_V1_PREFIX}/health",
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )

