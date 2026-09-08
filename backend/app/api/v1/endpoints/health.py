"""Health check and readiness probe endpoints."""
from datetime import datetime, timezone
from typing import Dict
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.app.core.config import Settings, get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Schema for health status response."""
    status: str = Field(default="ok", description="Overall health status")
    app_name: str = Field(description="Name of the application")
    version: str = Field(description="Application version")
    environment: str = Field(description="Active runtime environment")
    timestamp: datetime = Field(description="Current UTC timestamp")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Health status of downstream components (db, vector_store, etc.)"
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness Probe",
    description="Returns the current operational status of the InsightRAG backend and downstream subsystems.",
)
async def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Check health status of the application."""
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
        components={
            "api": "healthy",
            "database": "unconfigured (milestone 2)",
            "vector_store": "unconfigured (milestone 2)",
        },
    )

