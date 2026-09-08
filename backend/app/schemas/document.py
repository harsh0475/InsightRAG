"""Document and Chunk schemas with provenance metadata."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Supported document source formats."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TXT = "txt"


class ParsedSection(BaseModel):
    """Intermediate parsed text unit before chunking (e.g. a PDF page or Markdown section)."""
    text: str = Field(description="Raw extracted text content")
    page_number: Optional[int] = Field(default=None, description="1-based page number if applicable")
    section_title: Optional[str] = Field(default=None, description="Heading/section title if applicable")


class DocumentMetadata(BaseModel):
    """Metadata describing a source document."""
    document_id: str = Field(description="Unique identifier (e.g. SHA-256 hash of content)")
    document_name: str = Field(description="Original file name or document title")
    source: str = Field(description="Source origin path or URL")
    source_type: SourceType = Field(description="Type/format of source document")
    total_pages: Optional[int] = Field(default=None, description="Total number of pages if PDF")
    file_size_bytes: int = Field(default=0, description="Size of file in bytes")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChunkMetadata(BaseModel):
    """Rich provenance metadata associated with each individual chunk."""
    document_id: str = Field(description="Parent document identifier")
    document_name: str = Field(description="Parent document name")
    source: str = Field(description="Origin source path")
    page_number: Optional[int] = Field(default=None, description="1-based page number where chunk originated")
    section: Optional[str] = Field(default=None, description="Section title or Markdown header")
    chunk_id: str = Field(description="Unique deterministic chunk ID (e.g. {document_id}_c{chunk_position})")
    chunk_position: int = Field(description="0-based sequential position of chunk in the document")
    token_count: int = Field(description="Number of tokens in the chunk")
    char_count: int = Field(description="Number of characters in the chunk")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """An individual text chunk enriched with provenance metadata."""
    chunk_id: str = Field(description="Globally unique chunk identifier")
    text: str = Field(description="Cleaned text content of the chunk")
    metadata: ChunkMetadata = Field(description="Provenance metadata for citations & filtering")


class IngestionResult(BaseModel):
    """Summary of the document ingestion pipeline result."""
    document_id: str
    document_name: str
    source_type: SourceType
    total_chunks: int
    total_tokens: int
    chunks: List[DocumentChunk]

