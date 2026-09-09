"""API v1 Central Router."""
from fastapi import APIRouter
from backend.app.api.v1.endpoints import documents, health, retrieval

api_v1_router = APIRouter()

# Register endpoint routers
api_v1_router.include_router(health.router, tags=["Health & System"])
api_v1_router.include_router(documents.router)
api_v1_router.include_router(retrieval.router)
