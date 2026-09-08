"""Integration and API tests for document ingestion service."""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.document import SourceType
from backend.app.services.ingestion_service import IngestionService

client = TestClient(app)


def test_detect_source_type():
    """Verify file extension to SourceType mapping."""
    assert IngestionService.detect_source_type("guide.pdf") == SourceType.PDF
    assert IngestionService.detect_source_type("doc.md") == SourceType.MARKDOWN
    assert IngestionService.detect_source_type("notes.txt") == SourceType.TXT

    with pytest.raises(ValueError, match="Unsupported file format"):
        IngestionService.detect_source_type("archive.zip")


def test_deterministic_document_id():
    """Verify identical bytes yield identical document IDs."""
    data = b"Enterprise knowledge text."
    id1 = IngestionService.generate_document_id(data, "file.txt")
    id2 = IngestionService.generate_document_id(data, "file.txt")
    id3 = IngestionService.generate_document_id(b"Different data", "file.txt")

    assert id1 == id2
    assert id1.startswith("doc_")
    assert id1 != id3


def test_end_to_end_markdown_ingestion():
    """Verify full ingestion pipeline for Markdown."""
    service = IngestionService(chunk_size=100, chunk_overlap=20)
    with open("knowledge_base/sample_architecture.md", "rb") as f:
        file_bytes = f.read()

    result = service.ingest_document(file_bytes, "sample_architecture.md")
    assert result.document_id.startswith("doc_")
    assert result.document_name == "sample_architecture.md"
    assert result.source_type == SourceType.MARKDOWN
    assert result.total_chunks > 0
    assert result.total_tokens > 0

    # Inspect first chunk metadata
    first_chunk = result.chunks[0]
    assert first_chunk.metadata.chunk_position == 0
    assert first_chunk.metadata.chunk_id.endswith("_c0000")
    assert first_chunk.metadata.document_name == "sample_architecture.md"
    assert first_chunk.metadata.token_count > 0


def test_end_to_end_pdf_ingestion_with_page_provenance():
    """Verify PDF chunks retain accurate page provenance."""
    service = IngestionService(chunk_size=150, chunk_overlap=30)
    with open("knowledge_base/sample_k8s_guide.pdf", "rb") as f:
        file_bytes = f.read()

    result = service.ingest_document(file_bytes, "sample_k8s_guide.pdf")
    assert result.source_type == SourceType.PDF
    assert result.total_chunks >= 3

    # Ensure page numbers 1, 2, and 3 are present across the chunks
    page_numbers = {chunk.metadata.page_number for chunk in result.chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert 3 in page_numbers


def test_api_parse_raw_text():
    """Verify REST API endpoint for raw text parsing."""
    payload = {
        "text": "OAuth 2.0 uses authorization servers to issue access tokens and refresh tokens.",
        "filename": "oauth_overview.txt",
        "source_type": "txt",
        "chunk_size": 50,
        "chunk_overlap": 10,
    }
    response = client.post("/api/v1/documents/parse-raw", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_name"] == "oauth_overview.txt"
    assert data["total_chunks"] == 1
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["metadata"]["chunk_position"] == 0


def test_api_upload_document():
    """Verify REST API endpoint for multipart file upload."""
    with open("knowledge_base/sample_oauth.txt", "rb") as f:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("sample_oauth.txt", f, "text/plain")},
        )
    assert response.status_code == 201
    data = response.json()
    assert data["document_name"] == "sample_oauth.txt"
    assert data["source_type"] == "txt"
    assert data["total_chunks"] > 0


def test_api_upload_empty_document():
    """Verify REST API rejects empty uploads with 400 Bad Request."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]

