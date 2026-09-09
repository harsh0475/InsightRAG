"""Vector store factory and package exports."""
import logging
from typing import Optional
from backend.app.core.config import get_settings
from backend.app.services.vector_store.base import BaseVectorStore
from backend.app.services.vector_store.in_memory_store import InMemoryVectorStore
from backend.app.services.vector_store.pgvector_store import PGVectorStore

logger = logging.getLogger("insightrag.vector_store")

__all__ = [
    "BaseVectorStore",
    "InMemoryVectorStore",
    "PGVectorStore",
    "get_vector_store",
]

# Process-level singleton for InMemoryVectorStore so chunks persist across requests in a session
_SHARED_IN_MEMORY_STORE: Optional[InMemoryVectorStore] = None


def get_vector_store(store_type: Optional[str] = None, force_new: bool = False) -> BaseVectorStore:
    """Factory returning configured vector store (pgvector with in-memory fallback)."""
    global _SHARED_IN_MEMORY_STORE
    settings = get_settings()
    stype = (store_type or ("in_memory" if settings.ENVIRONMENT == "test" else "pgvector")).lower()

    if stype == "pgvector":
        try:
            store = PGVectorStore(
                connection_url=settings.DATABASE_URL,
                dimension=settings.EMBEDDING_DIMENSION,
            )
            # Test connection and ensure schema
            store.initialize_schema()
            return store
        except Exception as e:
            logger.warning(
                f"PostgreSQL pgvector is not available ({e}). "
                "Gracefully falling back to InMemoryVectorStore."
            )
            if _SHARED_IN_MEMORY_STORE is None or force_new:
                _SHARED_IN_MEMORY_STORE = InMemoryVectorStore()
            return _SHARED_IN_MEMORY_STORE

    elif stype == "in_memory":
        if _SHARED_IN_MEMORY_STORE is None or force_new:
            _SHARED_IN_MEMORY_STORE = InMemoryVectorStore()
        return _SHARED_IN_MEMORY_STORE
    else:
        raise ValueError(f"Unknown vector store type: '{stype}'. Supported: 'pgvector', 'in_memory'")

