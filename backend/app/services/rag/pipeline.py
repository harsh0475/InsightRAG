"""Baseline End-to-End RAG Pipeline."""
import logging
import re
import time
from typing import List, Optional

from backend.app.schemas.chat import ChatMessage
from backend.app.schemas.rag import Citation, RAGResponse
from backend.app.schemas.reranker import RerankedResult
from backend.app.schemas.retrieval import HybridSearchResult, RetrievalMode, VectorSearchResult
from backend.app.services.llm import BaseLLMProvider, get_llm_provider
from backend.app.services.rag.context_builder import ContextBuilder
from backend.app.services.rag.prompts import SYSTEM_GROUNDING_PROMPT, USER_QUERY_TEMPLATE
from backend.app.services.rag.query_rewriter import QueryRewriter
from backend.app.services.reranker import BaseRerankerProvider, get_reranker_provider
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.retrieval_service import RetrievalService

logger = logging.getLogger("insightrag.rag.pipeline")


class BaselineRAGPipeline:
    """Two-Stage RAG Pipeline: Conversational Rewrite -> Hybrid Retrieval (Top-20) -> Cross-Encoder Rerank (Top-5) -> Grounded Generation -> Citations."""

    # Standard refusal phrase indicating insufficient information
    REFUSAL_PHRASE = "I do not have sufficient information in the provided context to answer this question."

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        context_builder: Optional[ContextBuilder] = None,
        reranker_provider: Optional[BaseRerankerProvider] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ):
        self.retrieval_service = retrieval_service or RetrievalService()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(dense_service=self.retrieval_service)
        self.llm_provider = llm_provider or get_llm_provider()
        self.context_builder = context_builder or ContextBuilder()
        self.reranker_provider = reranker_provider or get_reranker_provider()
        self.query_rewriter = query_rewriter or QueryRewriter(llm_provider=self.llm_provider)

    def _extract_citations(
        self, answer: str, chunk_map: dict[str, VectorSearchResult]
    ) -> List[Citation]:
        """Extract [chunk_id] citation tags from answer and resolve to provenance metadata."""
        # Find all brackets containing alphanumeric and underscore characters
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
        enable_reranking: bool = True,
        candidate_pool_size: int = 20,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> RAGResponse:
        """Execute end-to-end question answering pipeline with query rewriting and two-stage reranking.
        
        Args:
            query: The user's natural language question.
            top_k: Number of candidate chunks to use for final generation.
            retrieval_mode: Optional retrieval mode (vector, bm25, or hybrid).
            document_ids: Optional document filters.
            enable_reranking: Whether to apply cross-encoder reranking to candidate pool.
            candidate_pool_size: Number of candidate chunks retrieved prior to reranking.
            chat_history: Optional conversation turns for conversational query rewriting.
            
        Returns:
            RAGResponse containing answer, citations, retrieved chunks, rerank metrics, and latency.
        """
        start_time = time.perf_counter()
        mode_used = retrieval_mode or self.hybrid_retriever.default_mode

        # Step 1: Conversational Query Rewriting
        rewritten_query: Optional[str] = None
        effective_query = query
        if chat_history and len(chat_history) > 0:
            rewritten_query = self.query_rewriter.rewrite(query=query, chat_history=chat_history)
            if rewritten_query and rewritten_query.strip() != query.strip():
                logger.info(f"Query rewritten for retrieval: '{query}' -> '{rewritten_query}'")
                effective_query = rewritten_query
            else:
                rewritten_query = None

        logger.info(f"Executing RAG for query: '{effective_query}' [Mode: {mode_used.value}]")

        # Step 2: Stage 1 Candidate Pool Retrieval
        fetch_k = max(candidate_pool_size, top_k) if enable_reranking else top_k
        hybrid_chunks = self.hybrid_retriever.retrieve(
            query=effective_query,
            top_k=fetch_k,
            mode=mode_used,
            document_ids=document_ids,
        )

        # Convert to VectorSearchResult objects
        candidate_chunks = [
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

        # Step 3: Stage 2 Cross-Encoder Reranking
        reranked_results: Optional[List[RerankedResult]] = None
        rerank_latency: Optional[float] = None
        reranker_applied = False

        if enable_reranking and len(candidate_chunks) > 0:
            rerank_start = time.perf_counter()
            reranked_results = self.reranker_provider.rerank(
                query=effective_query,
                candidates=candidate_chunks,
                top_n=top_k,
            )
            rerank_latency = round((time.perf_counter() - rerank_start) * 1000, 2)
            reranker_applied = True

            # Use reranked chunks for context generation
            final_chunks = [
                VectorSearchResult(
                    chunk_id=rr.chunk_id,
                    text=rr.text,
                    document_id=rr.document_id,
                    document_name=rr.document_name,
                    metadata=rr.metadata,
                    score=rr.rerank_score,
                )
                for rr in reranked_results
            ]
        else:
            final_chunks = candidate_chunks[:top_k]

        # Step 4: Build structured context block with token budgeting
        context_str, chunk_map = self.context_builder.build_context(final_chunks)

        # Step 5: Format grounded prompt
        prompt = USER_QUERY_TEMPLATE.format(context=context_str, query=effective_query)

        # Step 6: Generate grounded completion from LLM
        raw_answer = self.llm_provider.generate(
            prompt=prompt,
            system_prompt=SYSTEM_GROUNDING_PROMPT,
        )

        # Step 7: Extract and resolve citations from generated text
        citations = self._extract_citations(raw_answer, chunk_map)

        # Step 8: Check for refusal / insufficient evidence
        is_refusal = (
            self.REFUSAL_PHRASE.lower() in raw_answer.lower()
            or "sufficient information" in raw_answer.lower()
            or not final_chunks
        )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            f"RAG query completed in {duration_ms}ms "
            f"(Rerank: {rerank_latency}ms, Citations: {len(citations)})"
        )

        return RAGResponse(
            query=query,
            rewritten_query=rewritten_query,
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=final_chunks,
            reranked_chunks=reranked_results,
            has_sufficient_context=not is_refusal,
            execution_time_ms=duration_ms,
            model_name=self.llm_provider.model_name,
            retrieval_mode=mode_used.value,
            reranker_applied=reranker_applied,
            reranker_latency_ms=rerank_latency,
        )

