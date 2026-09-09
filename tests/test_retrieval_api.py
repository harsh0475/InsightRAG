"""Integration tests for retrieval REST API endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_index_and_search():
    """Verify end-to-end indexing and semantic search via API."""
    chunk_data = {
        "chunks": [
            {
                "chunk_id": "test_chunk_001",
                "text": "PostgreSQL with pgvector supports HNSW indexing for rapid cosine similarity retrieval.",
                "metadata": {
                    "document_id": "doc_pgvector",
                    "document_name": "pgvector_guide.md",
                    "source": "knowledge_base/pgvector_guide.md",
                    "page_number": 1,
                    "section": "HNSW Indexes",
                    "chunk_id": "test_chunk_001",
                    "chunk_position": 0,
                    "token_count": 15,
                    "char_count": 88,
                },
            },
            {
                "chunk_id": "test_chunk_002",
                "text": "FastAPI is a modern asynchronous web framework for Python with automatic OpenAPI docs.",
                "metadata": {
                    "document_id": "doc_fastapi",
                    "document_name": "fastapi_guide.md",
                    "source": "knowledge_base/fastapi_guide.md",
                    "page_number": 1,
                    "section": "Introduction",
                    "chunk_id": "test_chunk_002",
                    "chunk_position": 0,
                    "token_count": 14,
                    "char_count": 86,
                },
            },
        ]
    }

    # 1. Index chunks
    index_res = client.post("/api/v1/retrieval/index", json=chunk_data)
    assert index_res.status_code == 201
    assert index_res.json()["indexed_count"] == 2

    # 2. Search for pgvector
    search_payload = {
        "query": "HNSW cosine vector search in PostgreSQL",
        "top_k": 2,
    }
    search_res = client.post("/api/v1/retrieval/search", json=search_payload)
    assert search_res.status_code == 200
    results = search_res.json()
    assert len(results) >= 1
    # Top result should match chunk_001
    assert results[0]["chunk_id"] == "test_chunk_001"
    assert "pgvector" in results[0]["text"]
    assert results[0]["score"] > 0.0


def test_api_upload_and_index():
    """Verify combined upload and index endpoint."""
    with open("knowledge_base/sample_oauth.txt", "rb") as f:
        res = client.post(
            "/api/v1/retrieval/upload-and-index",
            files={"file": ("sample_oauth.txt", f, "text/plain")},
        )
    assert res.status_code == 201
    data = res.json()
    assert data["indexed_count"] > 0
    assert data["document_id"].startswith("doc_")

    # Search for OAuth refresh tokens
    search_res = client.post(
        "/api/v1/retrieval/search",
        json={"query": "refresh token expiration window and rotation", "top_k": 3},
    )
    assert search_res.status_code == 200
    results = search_res.json()
    assert len(results) > 0
    assert any("refresh token" in r["text"].lower() for r in results)

