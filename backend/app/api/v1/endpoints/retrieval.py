"""REST API endpoints for vector search, indexing, and semantic retrieval."""
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from backend.app.schemas.retrieval import (
    HybridSearchResult,
    IndexChunksRequest,
    IndexChunksResponse,
    RetrievalMode,
    VectorSearchRequest,
    VectorSearchResult,
)
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/retrieval", tags=["Semantic Retrieval & Vector Search"])


@router.post(
    "/search",
    response_model=List[HybridSearchResult],
    status_code=status.HTTP_200_OK,
    summary="Multi-Strategy Retrieval Search (Vector, BM25, or Hybrid)",
    description="Retrieves nearest neighbor chunks using dense vector search, sparse BM25, or Reciprocal Rank Fusion (RRF).",
)
async def retrieval_search(payload: VectorSearchRequest) -> List[HybridSearchResult]:
    """Execute retrieval search under vector, bm25, or hybrid mode."""
    try:
        retriever = HybridRetriever()
        results = retriever.retrieve(
            query=payload.query,
            top_k=payload.top_k,
            mode=payload.mode,
            document_ids=payload.document_ids,
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )


@router.post(
    "/index",
    response_model=IndexChunksResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Index Document Chunks",
    description="Generates dense vector embeddings for input chunks and stores them in the vector database.",
)
async def index_chunks(payload: IndexChunksRequest) -> IndexChunksResponse:
    """Index an array of DocumentChunk objects."""
    if not payload.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chunks list cannot be empty.",
        )

    try:
        service = RetrievalService()
        count = service.index_chunks(payload.chunks)
        doc_id = payload.chunks[0].metadata.document_id

        return IndexChunksResponse(
            indexed_count=count,
            document_id=doc_id,
            model=service.embedding_provider.model_name,
            dimension=service.embedding_provider.dimension,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )


@router.post(
    "/upload-and-index",
    response_model=IndexChunksResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload, Chunk, and Index Document",
    description="Single-step pipeline: uploads document, parses, chunks, generates embeddings, and indexes into vector store.",
)
async def upload_and_index_document(
    file: UploadFile = File(..., description="Document file (.pdf, .md, .txt)"),
    chunk_size: Optional[int] = Query(default=None, gt=0),
    chunk_overlap: Optional[int] = Query(default=None, ge=0),
) -> IndexChunksResponse:
    """End-to-end ingestion and vector indexing."""
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        # 1. Ingestion & Chunking
        ingestion_service = IngestionService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ingest_result = ingestion_service.ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "unknown_document",
        )

        if not ingest_result.chunks:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No text extracted from document.")

        # 2. Embedding & Vector Indexing
        retrieval_service = RetrievalService()
        indexed_count = retrieval_service.index_chunks(ingest_result.chunks)

        return IndexChunksResponse(
            indexed_count=indexed_count,
            document_id=ingest_result.document_id,
            model=retrieval_service.embedding_provider.model_name,
            dimension=retrieval_service.embedding_provider.dimension,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload and index failed: {str(e)}",
        )

