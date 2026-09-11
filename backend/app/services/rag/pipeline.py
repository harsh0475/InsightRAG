"""Baseline End-to-End RAG Pipeline."""
import logging
import re
import time
from typing import List, Optional

from backend.app.schemas.rag import Citation, RAGResponse
from backend.app.schemas.retrieval import HybridSearchResult, RetrievalMode, VectorSearchResult
from backend.app.services.llm import BaseLLMProvider, get_llm_provider
from backend.app.services.rag.context_builder import ContextBuilder
from backend.app.services.rag.prompts import SYSTEM_GROUNDING_PROMPT, USER_QUERY_TEMPLATE
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.retrieval_service import RetrievalService

logger = logging.getLogger("insightrag.rag.pipeline")


class BaselineRAGPipeline:
    """Baseline RAG Pipeline: Query -> Hybrid/Vector/BM25 Retrieval -> Context -> Grounded Prompt -> LLM -> Citations."""

    # Standard refusal phrase indicating insufficient information
    REFUSAL_PHRASE = "I do not have sufficient information in the provided context to answer this question."

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        self.retrieval_service = retrieval_service or RetrievalService()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(dense_service=self.retrieval_service)
        self.llm_provider = llm_provider or get_llm_provider()
        self.context_builder = context_builder or ContextBuilder()

    def _extract_citations(
        self, answer: str, chunk_map: dict[str, VectorSearchResult]
    ) -> List[Citation]:
        """Extract [chunk_id] citation tags from answer and resolve to provenance metadata."""
        raw_tags = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", answer)
        citations: List[Citation] = []
        seen_chunk_ids = set()

        for tag in raw_tags:
            if tag in chunk_map and tag not in seen_chunk_ids:
                chunk = chunk_map[tag]
                meta = chunk.metadata
                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        document_id=meta.document_id,
                        document_name=meta.document_name,
                        page_number=meta.page_number,
                        section=meta.section,
                        snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    )
                )
                seen_chunk_ids.add(tag)

        return citations

    def run(
        self,
        query: str,
        top_k: int = 5,
        retrieval_mode: Optional[RetrievalMode] = None,
        document_ids: Optional[List[str]] = None,
    ) -> RAGResponse:
        """Execute end-to-end question answering pipeline with configurable retrieval."""
        start_time = time.perf_counter()
        mode_used = retrieval_mode or self.hybrid_retriever.default_mode
        logger.info(f"Executing RAG for query: '{query}' [Mode: {mode_used.value}]")

        # 1. Retrieve candidate chunks via configured retrieval mode (Hybrid, Vector, or BM25)
        hybrid_chunks = self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            mode=mode_used,
            document_ids=document_ids,
        )

        # Convert to VectorSearchResult for ContextBuilder compatibility
        retrieved_chunks = [
            VectorSearchResult(
                chunk_id=hc.chunk_id,
                text=hc.text,
                document_id=hc.document_id,
                document_name=hc.document_name,
                metadata=hc.metadata,
                score=hc.score,
            )
            for hc in hybrid_chunks
        ]

        # 2. Build structured context block with token budgeting
        context_str, chunk_map = self.context_builder.build_context(retrieved_chunks)

        # 3. Format grounded prompt
        prompt = USER_QUERY_TEMPLATE.format(context=context_str, query=query)

        # 4. Generate grounded completion from LLM
        raw_answer = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=SYSTEM_GROUNDING_PROMPT,
        )

        # 5. Extract and resolve citations from generated text
        citations = self._extract_citations(raw_answer, chunk_map)

        # 6. Check for refusal / insufficient evidence
        is_refusal = (
            self.REFUSAL_PHRASE.lower() in raw_answer.lower()
            or "sufficient information" in raw_answer.lower()
            or not retrieved_chunks
        )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"RAG query completed in {duration_ms}ms (Citations: {len(citations)})")

        return RAGResponse(
            query=query,
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            has_sufficient_context=not is_refusal,
            execution_time_ms=duration_ms,
            model_name=self.llm_provider.model_name,
            retrieval_mode=mode_used.value,
        )

