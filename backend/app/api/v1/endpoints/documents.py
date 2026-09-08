"""Document ingestion endpoints for file upload and raw text chunking."""
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from backend.app.schemas.document import IngestionResult, SourceType
from backend.app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


class RawTextIngestRequest(BaseModel):
    """Schema for direct raw text ingestion."""
    text: str = Field(description="Raw text content to ingest and chunk")
    filename: str = Field(default="document.txt", description="Identifier filename for document")
    source_type: SourceType = Field(default=SourceType.TXT, description="Format type: pdf, markdown, txt")
    chunk_size: Optional[int] = Field(default=None, gt=0, description="Optional custom chunk size in tokens")
    chunk_overlap: Optional[int] = Field(default=None, ge=0, description="Optional custom chunk overlap in tokens")


@router.post(
    "/upload",
    response_model=IngestionResult,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest Document",
    description="Upload a PDF, Markdown, or TXT document to parse, clean, segment into chunks, and attach provenance metadata.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (.pdf, .md, .txt)"),
    chunk_size: Optional[int] = Query(default=None, gt=0, description="Custom chunk size"),
    chunk_overlap: Optional[int] = Query(default=None, ge=0, description="Custom chunk overlap"),
) -> IngestionResult:
    """Upload and parse a document into chunks."""
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        ingestion_service = IngestionService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        result = ingestion_service.ingest_document(
            file_bytes=file_bytes,
            filename=file.filename or "unknown_document",
        )
        return result

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )


@router.post(
    "/parse-raw",
    response_model=IngestionResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest Raw Text Payload",
    description="Directly parse, clean, and chunk a string payload with simulated document metadata.",
)
async def ingest_raw_text(payload: RawTextIngestRequest) -> IngestionResult:
    """Process raw text into chunks without requiring a multipart upload."""
    try:
        ingestion_service = IngestionService(
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
        file_bytes = payload.text.encode("utf-8")
        result = ingestion_service.ingest_document(
            file_bytes=file_bytes,
            filename=payload.filename,
            source_type=payload.source_type,
        )
        return result
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process text: {str(e)}",
        )
