"""Context Builder for formatting retrieved chunks and enforcing token budgets."""
import logging
from typing import Dict, List, Tuple
import tiktoken

from backend.app.schemas.retrieval import VectorSearchResult

logger = logging.getLogger("insightrag.rag.context_builder")


class ContextBuilder:
    """Formats retrieved chunks into a clean, provenance-annotated context block."""

    def __init__(
        self,
        max_context_tokens: int = 2500,
        tokenizer_name: str = "cl100k_base",
    ):
        self.max_context_tokens = max_context_tokens
        try:
            self._tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens using BPE tokenizer."""
        return len(self._tokenizer.encode(text))

    def build_context(
        self,
        chunks: List[VectorSearchResult],
    ) -> Tuple[str, Dict[str, VectorSearchResult]]:
        """Format chunks into a structured context string respecting the token budget.
        
        Args:
            chunks: List of retrieved candidates sorted by relevance.
            
        Returns:
            Tuple of:
                - context_string: Formatted text to insert into the prompt.
                - chunk_map: Dictionary mapping chunk_id -> VectorSearchResult for citation resolution.
        """
        if not chunks:
            return "[No relevant context retrieved from knowledge base.]", {}

        chunk_map: Dict[str, VectorSearchResult] = {}
        formatted_blocks: List[str] = []
        accumulated_tokens = 0

        for chunk in chunks:
            meta = chunk.metadata
            page_str = f"Page: {meta.page_number}" if meta.page_number else "Page: N/A"
            section_str = f"Section: {meta.section}" if meta.section else "Section: N/A"

            block = (
                f"--- START CHUNK [ID: {chunk.chunk_id}] ---\n"
                f"Source: {chunk.document_name} | {page_str} | {section_str}\n"
                f"Content:\n{chunk.text}\n"
                f"--- END CHUNK [ID: {chunk.chunk_id}] ---"
            )

            block_tokens = self.count_tokens(block)
            if accumulated_tokens + block_tokens > self.max_context_tokens and formatted_blocks:
                logger.warning(
                    f"Context budget reached ({accumulated_tokens} tokens). "
                    f"Truncating remaining {len(chunks) - len(formatted_blocks)} chunks."
                )
                break

            formatted_blocks.append(block)
            chunk_map[chunk.chunk_id] = chunk
            accumulated_tokens += block_tokens

        context_string = "\n\n".join(formatted_blocks)
        return context_string, chunk_map

