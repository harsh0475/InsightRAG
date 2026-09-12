"""Reranker subsystem providers and factory."""
from typing import Optional

from backend.app.core.config import get_settings
from backend.app.services.reranker.base import BaseRerankerProvider
from backend.app.services.reranker.cross_encoder import CrossEncoderReranker
from backend.app.services.reranker.mock_reranker import MockRerankerProvider


def get_reranker_provider(provider_type: Optional[str] = None) -> BaseRerankerProvider:
    """Factory creating configured reranker provider instance.
    
    Args:
        provider_type: Optional override ('cross-encoder', 'mock', 'none').
        
    Returns:
        Instance conforming to BaseRerankerProvider.
    """
    settings = get_settings()
    selected = (provider_type or settings.RERANKER_PROVIDER).lower().replace("_", "-")

    if selected in ("mock", "none", "test"):
        return MockRerankerProvider()
    elif selected in ("cross-encoder", "huggingface", "sentence-transformers"):
        return CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
    else:
        # Default to CrossEncoder with its internal resilient fallback
        return CrossEncoderReranker(model_name=settings.RERANKER_MODEL)


__all__ = [
    "BaseRerankerProvider",
    "CrossEncoderReranker",
    "MockRerankerProvider",
    "get_reranker_provider",
]

