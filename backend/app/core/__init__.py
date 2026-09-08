"""Core configuration and utilities for InsightRAG."""
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import setup_logging

__all__ = ["Settings", "get_settings", "setup_logging"]

