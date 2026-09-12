"""Tests for Evaluation Suite, metrics computation, and benchmarking endpoints."""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.document import ChunkMetadata
from backend.app.schemas.evaluation import EvalSample
from backend.app.schemas.rag import Citation
from backend.app.schemas.retrieval import RetrievalMode, VectorSearchResult
from backend.app.services.evaluation.evaluator import EvaluationRunner
from backend.app.services.evaluation.metrics import (
    compute_generation_metrics,
    compute_retrieval_metrics,
)

client = TestClient(app)


def _make_chunk(chunk_id: str, doc_name: str, text: str) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        document_id="d1",
        document_name=doc_name,
        metadata=ChunkMetadata(
            document_id="d1",
            document_name=doc_name,
            source=f"knowledge_base/{doc_name}",
            chunk_id=chunk_id,
            chunk_position=0,
            token_count=15,
            char_count=len(text),
        ),
        score=0.8,
    )


def test_compute_retrieval_metrics_hit():
    """Verify hit rate, recall, and reciprocal rank on successful retrieval."""
    sample = EvalSample(
        query_id="q1",
        query="What is Redis eviction?",
        expected_document="redis.md",
        expected_keywords=["redis", "eviction", "policy"],
        ground_truth_answer="Redis eviction policies include allkeys-lru and allkeys-lfu.",
        is_answerable=True,
    )

    chunks = [
        _make_chunk("c1", "other.txt", "Irrelevant context."),
        _make_chunk("c2", "redis.md", "Redis eviction policy removes least recently used items."),
    ]

    metrics = compute_retrieval_metrics(sample, chunks)
    assert metrics.hit_rate == 1.0
    assert metrics.reciprocal_rank == 0.5  # Found at rank 2: 1/2 = 0.5
    assert metrics.recall_at_k > 0.0


def test_compute_retrieval_metrics_miss():
    """Verify zero hit rate and reciprocal rank when target doc is absent."""
    sample = EvalSample(
        query_id="q2",
        query="What is OAuth PKCE?",
        expected_document="oauth.txt",
        expected_keywords=["pkce", "oauth"],
        ground_truth_answer="PKCE uses a code verifier.",
        is_answerable=True,
    )
    chunks = [_make_chunk("c1", "other.txt", "Irrelevant context.")]

    metrics = compute_retrieval_metrics(sample, chunks)
    assert metrics.hit_rate == 0.0
    assert metrics.reciprocal_rank == 0.0


def test_compute_generation_metrics_faithful():
    """Verify faithfulness is high when answer words are grounded in context."""
    sample = EvalSample(
        query_id="q3",
        query="How to avoid cache stampedes?",
        expected_document="redis.md",
        expected_keywords=["jitter", "stampede"],
        ground_truth_answer="Use jittered expiration.",
        is_answerable=True,
    )
    context_chunks = [_make_chunk("c1", "redis.md", "Avoid cache stampedes using jittered expiration times.")]
    answer = "We avoid cache stampedes using jittered expiration times."
    citations = [
        Citation(
            chunk_id="c1",
            document_id="d1",
            document_name="redis.md",
            snippet="Avoid cache stampedes",
        )
    ]

    metrics = compute_generation_metrics(
        sample=sample,
        answer=answer,
        retrieved_chunks=context_chunks,
        citations=citations,
        is_refusal=False,
    )
    assert metrics.faithfulness > 0.8
    assert metrics.refusal_correctness is True
    assert metrics.citation_precision == 1.0


def test_compute_generation_metrics_unanswerable_refusal():
    """Verify unanswerable queries correctly handled by refusal earn full faithfulness."""
    sample = EvalSample(
        query_id="q4",
        query="How to bake sourdough bread?",
        expected_document="none",
        expected_keywords=[],
        ground_truth_answer="I do not have sufficient information in the provided context to answer this question.",
        is_answerable=False,
    )
    metrics = compute_generation_metrics(
        sample=sample,
        answer="I do not have sufficient information in the provided context to answer this question.",
        retrieved_chunks=[],
        citations=[],
        is_refusal=True,
    )
    assert metrics.faithfulness == 1.0
    assert metrics.answer_relevance == 1.0
    assert metrics.refusal_correctness is True


def test_evaluator_load_dataset():
    """Verify evaluation runner loads dataset successfully."""
    runner = EvaluationRunner()
    samples = runner.load_dataset()
    assert len(samples) >= 10
    assert all(isinstance(s, EvalSample) for s in samples)


def test_evaluator_evaluate_configuration():
    """Verify full configuration evaluation run generates valid aggregate report."""
    runner = EvaluationRunner()
    report = runner.evaluate_configuration(
        experiment_name="Test Run",
        retrieval_mode=RetrievalMode.HYBRID,
        enable_reranking=True,
        top_k=2,
        candidate_pool_size=5,
    )
    assert report.total_queries == len(runner.load_dataset())
    assert 0.0 <= report.mean_hit_rate <= 1.0
    assert 0.0 <= report.mean_mrr <= 1.0
    assert len(report.sample_results) == report.total_queries


def test_api_evaluation_dataset_endpoint():
    """Verify GET /api/v1/evaluation/dataset returns dataset items."""
    res = client.get("/api/v1/evaluation/dataset")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 10
    assert "query_id" in data[0]
    assert "query" in data[0]


def test_api_evaluation_run_and_latest():
    """Verify POST /api/v1/evaluation/run and GET /api/v1/evaluation/latest."""
    payload = {
        "experiment_name": "API Test Run",
        "retrieval_mode": "hybrid",
        "enable_reranking": True,
        "top_k": 2,
        "candidate_pool_size": 5,
    }
    run_res = client.post("/api/v1/evaluation/run", json=payload)
    assert run_res.status_code == 200
    report_data = run_res.json()
    assert report_data["experiment_name"] == "API Test Run"
    assert "mean_hit_rate" in report_data
    assert "mean_mrr" in report_data

    # Test latest endpoint
    latest_res = client.get("/api/v1/evaluation/latest")
    assert latest_res.status_code == 200
    assert latest_res.json()["experiment_name"] == "API Test Run"

