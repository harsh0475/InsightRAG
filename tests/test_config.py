"""Tests for application settings and configuration."""
import os
from backend.app.core.config import Settings, get_settings


def test_default_settings():
    """Verify default configuration attributes."""
    settings = Settings()
    assert settings.APP_NAME == "InsightRAG"
    assert settings.APP_VERSION == "0.1.0"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.DEFAULT_CHUNK_SIZE == 500
    assert settings.DEFAULT_CHUNK_OVERLAP == 50
    assert settings.DEFAULT_TOP_K == 5
    assert settings.RRF_K == 60


def test_settings_override(monkeypatch):
    """Verify that environment variables correctly override defaults."""
    monkeypatch.setenv("APP_NAME", "CustomRAG")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("DEFAULT_CHUNK_SIZE", "1000")

    settings = Settings()
    assert settings.APP_NAME == "CustomRAG"
    assert settings.API_PORT == 9000
    assert settings.DEFAULT_CHUNK_SIZE == 1000


def test_get_settings_cached():
    """Verify that get_settings provides a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

