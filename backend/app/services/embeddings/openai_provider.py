"""OpenAI-compatible embedding provider implementation."""
import logging
from typing import List, Optional
from openai import OpenAI
from backend.app.core.config import get_settings
from backend.app.services.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger("insightrag.embeddings.openai")


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Generates dense embeddings via OpenAI or any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        settings = get_settings()
        self._api_key = api_key or settings.EMBEDDING_API_KEY
        self._base_url = base_url or settings.EMBEDDING_BASE_URL
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._dimension = dimension or settings.EMBEDDING_DIMENSION

        if not self._api_key:
            logger.warning("No EMBEDDING_API_KEY configured for OpenAIEmbeddingProvider.")

        self.client = OpenAI(
            api_key=self._api_key or "sk-dummy-key-for-local-compatibility",
            base_url=self._base_url,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Batch embed a list of texts into dense vectors."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                # Text-embedding-3 supports dimensions parameter
                kwargs = {"model": self._model_name, "input": batch}
                if "text-embedding-3" in self._model_name and self._dimension:
                    kwargs["dimensions"] = self._dimension

                response = self.client.embeddings.create(**kwargs)
                batch_vectors = [item.embedding for item in response.data]
                all_embeddings.extend(batch_vectors)
            except Exception as e:
                logger.error(f"Failed to generate embeddings batch [{i}:{i+batch_size}]: {e}")
                raise RuntimeError(f"Embedding API error: {str(e)}") from e

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        embeddings = self.embed_texts([query])
        return embeddings[0]

