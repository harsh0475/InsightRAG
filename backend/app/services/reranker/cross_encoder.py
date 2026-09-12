"""Cross-Encoder Reranker using sentence-transformers or graceful fallback."""
import logging
from typing import List, Optional

from backend.app.services.reranker.base import BaseRerankerProvider
from backend.app.services.reranker.mock_reranker import MockRerankerProvider

logger = logging.getLogger("insightrag.reranker.cross_encoder")


class CrossEncoderReranker(BaseRerankerProvider):
    """Production Cross-Encoder reranker using HuggingFace models (e.g. ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None
        self._fallback_provider: Optional[MockRerankerProvider] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        """Lazy load the sentence_transformers CrossEncoder or initialize fallback."""
        if self._model is not None or self._fallback_provider is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder model '{self._model_name}'...")
            self._model = CrossEncoder(self._model_name)
            logger.info(f"Successfully loaded CrossEncoder model '{self._model_name}'")
            return self._model
        except (ImportError, Exception) as e:
            logger.warning(
                f"sentence-transformers not available or failed to load ({str(e)}). "
                f"Falling back to high-performance deterministic MockRerankerProvider."
            )
            self._fallback_provider = MockRerankerProvider(model_name=f"{self._model_name}-fallback")
            return None

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """Compute cross-encoder relevance scores for (query, text) pairs."""
        if not texts:
            return []

        model = self._get_model()
        if model is not None:
            pairs = [[query, text] for text in texts]
            try:
                raw_scores = model.predict(pairs)
                # If model outputs 1D numpy array of logits, convert to Python floats
                return [float(score) for score in raw_scores]
            except Exception as e:
                logger.error(f"CrossEncoder prediction failed: {str(e)}. Using fallback scoring.")
                if self._fallback_provider is None:
                    self._fallback_provider = MockRerankerProvider()
                return self._fallback_provider.score_pairs(query, texts)

        # Use fallback provider
        return self._fallback_provider.score_pairs(query, texts)

