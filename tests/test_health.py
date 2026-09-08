"""Tests for system health endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint provides discovery links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "InsightRAG" in data["message"]
    assert "version" in data
    assert data["health_url"] == "/api/v1/health"


def test_health_endpoint():
    """Verify health probe endpoint returns status ok and component metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "InsightRAG"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data
    assert "components" in data
    assert data["components"]["api"] == "healthy"

