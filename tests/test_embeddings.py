"""Unit tests for embedding providers."""
import math
import pytest
from backend.app.services.embeddings import (
    BaseEmbeddingProvider,
    MockEmbeddingProvider,
    get_embedding_provider,
)


def test_mock_embedding_dimensions_and_norm():
    """Verify mock provider generates vectors of correct dimension and unit L2 norm."""
    dim = 256
    provider = MockEmbeddingProvider(dimension=dim, model_name="test-mock")
    assert provider.dimension == dim
    assert provider.model_name == "test-mock"

    vec = provider.embed_query("Kubernetes control plane architecture")
    assert len(vec) == dim

    # Check L2 norm is ~1.0
    norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_mock_embedding_determinism():
    """Verify identical text produces identical embedding vector."""
    provider = MockEmbeddingProvider(dimension=128)
    text = "OAuth 2.0 PKCE authorization flow"
    v1 = provider.embed_query(text)
    v2 = provider.embed_query(text)
    assert v1 == v2


def test_mock_embedding_batch():
    """Verify batch embedding produces matching number of vectors."""
    provider = MockEmbeddingProvider(dimension=64)
    texts = ["Chunk one", "Chunk two", "Chunk three"]
    vectors = provider.embed_texts(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 64


def test_mock_embedding_semantic_correlation():
    """Verify texts with shared vocabulary exhibit higher cosine similarity than unrelated texts."""
    provider = MockEmbeddingProvider(dimension=512)
    q = provider.embed_query("kubernetes cluster nodes")
    doc_related = provider.embed_query("kubernetes worker nodes running pods")
    doc_unrelated = provider.embed_query("culinary recipes for baking bread")

    # Dot product between normalized vectors equals cosine similarity
    score_related = sum(a * b for a, b in zip(q, doc_related))
    score_unrelated = sum(a * b for a, b in zip(q, doc_unrelated))

    assert score_related > score_unrelated, "Related document must have higher similarity than unrelated document."


def test_embedding_factory():
    """Verify factory returns valid BaseEmbeddingProvider."""
    p = get_embedding_provider("mock")
    assert isinstance(p, BaseEmbeddingProvider)

