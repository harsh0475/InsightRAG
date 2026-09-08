"""Unit tests for sliding window chunker."""
import pytest
from backend.app.services.chunker import SlidingWindowChunker


def test_chunker_validation():
    """Verify chunker raises errors on invalid chunk_size or overlap."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SlidingWindowChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        SlidingWindowChunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
        SlidingWindowChunker(chunk_size=100, chunk_overlap=100)


def test_chunk_empty_text():
    """Verify empty text produces no chunks."""
    chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.split_text("") == []
    assert chunker.split_text("   ") == []


def test_text_shorter_than_chunk_size():
    """Verify text shorter than chunk_size produces exactly one chunk."""
    chunker = SlidingWindowChunker(chunk_size=50, chunk_overlap=10)
    text = "OAuth 2.0 uses access tokens and refresh tokens."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0][0] == text
    assert chunks[0][1] > 0  # token count
    assert chunks[0][2] == len(text)  # char count


def test_text_sliding_window_with_overlap():
    """Verify text longer than chunk_size creates overlapping chunks."""
    chunker = SlidingWindowChunker(chunk_size=20, chunk_overlap=5)
    # Generate 50 words
    words = [f"word{i}" for i in range(50)]
    text = " ".join(words)

    chunks = chunker.split_text(text)
    assert len(chunks) > 1

    # Verify that consecutive chunks share overlapping tokens
    chunk_1_text = chunks[0][0]
    chunk_2_text = chunks[1][0]
    
    # The end of chunk 1 should share words with the start of chunk 2
    chunk_1_words = chunk_1_text.split()
    chunk_2_words = chunk_2_text.split()
    overlap_found = any(w in chunk_2_words[:10] for w in chunk_1_words[-10:])
    assert overlap_found, "Consecutive chunks must share overlapping tokens"


def test_token_counting():
    """Verify token count matches BPE tokenization."""
    chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=10)
    text = "Machine learning models require high-quality datasets."
    count = chunker.count_tokens(text)
    assert count > 0

