"""Unit tests for ContextBuilder and token budgeting."""
from backend.app.schemas.document import ChunkMetadata
from backend.app.schemas.retrieval import VectorSearchResult
from backend.app.services.rag.context_builder import ContextBuilder


def make_result(chunk_id: str, text: str, page: int = 1, section: str = "Intro") -> VectorSearchResult:
    meta = ChunkMetadata(
        document_id="doc1",
        document_name="test.pdf",
        source="knowledge_base/test.pdf",
        page_number=page,
        section=section,
        chunk_id=chunk_id,
        chunk_position=0,
        token_count=len(text.split()),
        char_count=len(text),
    )
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        document_id="doc1",
        document_name="test.pdf",
        metadata=meta,
        score=0.9,
    )


def test_context_builder_formatting():
    """Verify context builder outputs structured chunk tags with metadata."""
    builder = ContextBuilder(max_context_tokens=1000)
    chunks = [
        make_result("c1", "First chunk content.", page=2, section="Architecture"),
        make_result("c2", "Second chunk content.", page=3, section="Security"),
    ]

    context_str, chunk_map = builder.build_context(chunks)

    assert "--- START CHUNK [ID: c1] ---" in context_str
    assert "Source: test.pdf | Page: 2 | Section: Architecture" in context_str
    assert "First chunk content." in context_str
    assert "--- END CHUNK [ID: c1] ---" in context_str

    assert "c1" in chunk_map
    assert "c2" in chunk_map


def test_context_builder_empty():
    """Verify empty chunk list outputs fallback string."""
    builder = ContextBuilder()
    context_str, chunk_map = builder.build_context([])
    assert "No relevant context" in context_str
    assert chunk_map == {}


def test_context_builder_token_budget_truncation():
    """Verify builder truncates chunks when token budget is exceeded."""
    # Strict budget of ~25 tokens
    builder = ContextBuilder(max_context_tokens=30)
    long_text = "This is a long sentence repeated multiple times to ensure it occupies many tokens in the budget. " * 3
    chunks = [
        make_result("c1", long_text),
        make_result("c2", long_text),
        make_result("c3", long_text),
    ]

    context_str, chunk_map = builder.build_context(chunks)
    # Should only fit 1 chunk, truncating the rest
    assert len(chunk_map) == 1
    assert "c1" in chunk_map
    assert "c2" not in chunk_map

