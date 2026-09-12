"""REST API endpoints for Baseline RAG question answering."""
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.rag import RAGQueryRequest, RAGResponse
from backend.app.services.rag.pipeline import BaselineRAGPipeline

router = APIRouter(prefix="/rag", tags=["Grounded Generation & RAG"])


@router.post(
    "/query",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Baseline RAG Pipeline",
    description=(
        "Executes the full Baseline RAG pipeline: retrieves semantically relevant context chunks, "
        "builds a grounded prompt, queries the LLM, extracts verified citations, and returns the response."
    ),
)
async def query_rag(payload: RAGQueryRequest) -> RAGResponse:
    """Execute baseline RAG query."""
    try:
        pipeline = BaselineRAGPipeline()
        response = pipeline.run(
            query=payload.query,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
            document_ids=payload.document_ids,
            enable_reranking=payload.enable_reranking,
            candidate_pool_size=payload.candidate_pool_size,
            chat_history=payload.chat_history,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG execution failed: {str(e)}",
        )

