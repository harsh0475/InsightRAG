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

