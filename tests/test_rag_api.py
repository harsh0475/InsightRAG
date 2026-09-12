"""Integration tests for RAG API endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_rag_query_endpoint():
    """Verify RAG query REST endpoint returns grounded answer and citations."""
    # First index a sample chunk
    chunk_payload = {
        "chunks": [
            {
                "chunk_id": "doc_redis_c0001",
                "text": "Redis caching uses jittered expiration between 5% and 15% to mitigate cache stampedes.",
                "metadata": {
                    "document_id": "doc_redis",
                    "document_name": "redis_architecture.md",
                    "source": "knowledge_base/redis_architecture.md",
                    "page_number": 2,
                    "section": "Cache Stampede",
                    "chunk_id": "doc_redis_c0001",
                    "chunk_position": 0,
                    "token_count": 18,
                    "char_count": 92,
                },
            }
        ]
    }
    idx_res = client.post("/api/v1/retrieval/index", json=chunk_payload)
    assert idx_res.status_code == 201

    # Query RAG
    query_payload = {
        "query": "How do we prevent cache stampedes in Redis?",
        "top_k": 2,
    }
    response = client.post("/api/v1/rag/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == query_payload["query"]
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "citations" in data
    assert "retrieved_chunks" in data
    assert data["execution_time_ms"] > 0
    assert "model_name" in data


def test_api_rag_query_empty():
    """Verify validation on empty query."""
    response = client.post("/api/v1/rag/query", json={"query": ""})
    assert response.status_code == 422


def test_api_rag_query_with_chat_history_and_reranking():
    """Verify RAG query endpoint with multi-turn history and reranking flags."""
    query_payload = {
        "query": "Does it prevent stampedes?",
        "top_k": 2,
        "enable_reranking": True,
        "candidate_pool_size": 5,
        "chat_history": [
            {"role": "user", "content": "What is Redis jittered expiration?"},
            {"role": "assistant", "content": "Redis jitter varies expiration times."},
        ],
    }
    response = client.post("/api/v1/rag/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rewritten_query"] is not None
    assert "redis" in data["rewritten_query"].lower()
    assert data["reranker_applied"] is True


def test_api_retrieval_rerank_endpoint():
    """Verify standalone /api/v1/retrieval/rerank endpoint."""
    payload = {
        "query": "Redis jittered expiration",
        "top_n": 2,
        "candidates": [
            {
                "chunk_id": "c1",
                "text": "Irrelevant database logs.",
                "document_id": "d1",
                "document_name": "db.log",
                "metadata": {
                    "document_id": "d1",
                    "document_name": "db.log",
                    "source": "knowledge_base/db.log",
                    "chunk_id": "c1",
                    "chunk_position": 0,
                    "token_count": 10,
                    "char_count": 26,
                },
                "score": 0.8,
            },
            {
                "chunk_id": "c2",
                "text": "Redis jittered expiration prevents simultaneous key expiry.",
                "document_id": "d2",
                "document_name": "redis.md",
                "metadata": {
                    "document_id": "d2",
                    "document_name": "redis.md",
                    "source": "knowledge_base/redis.md",
                    "chunk_id": "c2",
                    "chunk_position": 0,
                    "token_count": 10,
                    "char_count": 59,
                },
                "score": 0.5,
            },
        ],
    }
    response = client.post("/api/v1/retrieval/rerank", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    # c2 must be promoted to rank 1
    assert data["results"][0]["chunk_id"] == "c2"
    assert data["results"][0]["final_rank"] == 1
    assert data["results"][0]["initial_rank"] == 2
    assert data["results"][0]["rank_delta"] == 1


