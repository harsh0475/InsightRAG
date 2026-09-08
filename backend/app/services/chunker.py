"""Configurable sliding-window chunking engine with token and overlap control."""
from typing import List, Tuple
import tiktoken


class SlidingWindowChunker:
    """Chunks text using a sliding window of tokens with configurable overlap.
    
    Attributes:
        chunk_size: Target number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between adjacent chunks.
        tokenizer_name: Name of the tiktoken encoding (default 'cl100k_base').
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        tokenizer_name: str = "cl100k_base",
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.stride = chunk_size - chunk_overlap

        try:
            self._tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Return the exact number of BPE tokens in the text."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    def split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into overlapping chunks.
        
        Args:
            text: Normalized text to segment.
            
        Returns:
            List of tuples: (chunk_text, token_count, char_count)
        """
        if not text or not text.strip():
            return []

        tokens = self._tokenizer.encode(text)
        total_tokens = len(tokens)

        # If text fits inside a single chunk, return immediately
        if total_tokens <= self.chunk_size:
            return [(text.strip(), total_tokens, len(text.strip()))]

        chunks: List[Tuple[str, int, int]] = []
        start_idx = 0

        while start_idx < total_tokens:
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode token slice back to clean string
            chunk_text = self._tokenizer.decode(chunk_tokens).strip()
            
            if chunk_text:
                actual_token_count = len(chunk_tokens)
                chunks.append((chunk_text, actual_token_count, len(chunk_text)))

            if end_idx >= total_tokens:
                break

            start_idx += self.stride

        return chunks

