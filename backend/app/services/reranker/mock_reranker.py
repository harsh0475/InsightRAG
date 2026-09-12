"""Deterministic Mock Reranker Provider for testing and lightweight environments."""
import re
from typing import List

from backend.app.services.reranker.base import BaseRerankerProvider


class MockRerankerProvider(BaseRerankerProvider):
    """Deterministic reranker simulating cross-attention interaction without PyTorch dependencies."""

    def __init__(self, model_name: str = "mock-cross-encoder-minilm"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase alphanumeric tokenizer."""
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def _compute_relevance(self, query: str, text: str) -> float:
        """Compute deterministic cross-interaction score simulating cross-attention."""
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(text)

        if not q_tokens or not d_tokens:
            return 0.0

        q_lower = query.lower().strip()
        t_lower = text.lower()

        # 1. Exact phrase match bonus
        exact_bonus = 0.35 if q_lower in t_lower else 0.0

        # 2. Unigram coverage ratio
        unique_q = set(q_tokens)
        matched_unigrams = sum(1 for tok in unique_q if tok in d_tokens)
        unigram_coverage = matched_unigrams / len(unique_q)

        # 3. Bigram sequence match ratio
        if len(q_tokens) >= 2:
            q_bigrams = [f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(len(q_tokens) - 1)]
            matched_bigrams = sum(1 for bg in q_bigrams if bg in t_lower)
            bigram_coverage = matched_bigrams / len(q_bigrams)
        else:
            bigram_coverage = unigram_coverage

        # 4. Proximity bonus (early occurrence in document)
        early_occurrence_bonus = 0.0
        for tok in unique_q:
            pos = t_lower.find(tok)
            if 0 <= pos < 200:  # Occurs within first 200 characters
                early_occurrence_bonus = 0.1
                break

        raw_score = (
            (unigram_coverage * 0.40)
            + (bigram_coverage * 0.25)
            + exact_bonus
            + early_occurrence_bonus
        )

        # Clamp between 0.0 and 1.0
        return max(0.0, min(1.0, raw_score))

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """Score (query, text) pairs deterministically."""
        return [self._compute_relevance(query, text) for text in texts]

