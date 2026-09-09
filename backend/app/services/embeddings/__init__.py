"""Embedding providers package and factory."""
from typing import Optional
from backend.app.core.config import get_settings
from backend.app.services.embeddings.base import BaseEmbeddingProvider
from backend.app.services.embeddings.mock_provider import MockEmbeddingProvider
from backend.app.services.embeddings.openai_provider import OpenAIEmbeddingProvider

__all__ = [
    "BaseEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "MockEmbeddingProvider",
    "get_embedding_provider",
]


def get_embedding_provider(provider_type: Optional[str] = None) -> BaseEmbeddingProvider:
    """Factory function returning the configured embedding provider."""
    settings = get_settings()
    ptype = (provider_type or settings.EMBEDDING_PROVIDER).lower()

    if ptype == "openai":
        # If API key is present or base_url is custom, use OpenAI provider
        if settings.EMBEDDING_API_KEY:
            return OpenAIEmbeddingProvider()
        # If no API key is provided, log warning and fallback to deterministic mock
        return MockEmbeddingProvider(
            dimension=settings.EMBEDDING_DIMENSION,
            model_name="mock-" + settings.EMBEDDING_MODEL,
        )
    elif ptype == "mock":
        return MockEmbeddingProvider(
            dimension=settings.EMBEDDING_DIMENSION,
            model_name="mock-" + settings.EMBEDDING_MODEL,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: '{ptype}'. Supported: 'openai', 'mock'")

