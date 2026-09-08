"""Orchestration service for document parsing, cleaning, chunking, and metadata generation."""
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from backend.app.core.config import get_settings
from backend.app.schemas.document import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    ParsedSection,
    SourceType,
)
from backend.app.services.cleaner import DocumentCleaner
from backend.app.services.chunker import SlidingWindowChunker
from backend.app.services.parsers.base import BaseParser
from backend.app.services.parsers.markdown_parser import MarkdownParser
from backend.app.services.parsers.pdf_parser import PDFParser
from backend.app.services.parsers.txt_parser import TextParser

logger = logging.getLogger("insightrag.services.ingestion")


class IngestionService:
    """End-to-end ingestion pipeline coordinator."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP

        # Initialize chunker
        self.chunker = SlidingWindowChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # Register format parsers
        self._parsers: Dict[SourceType, BaseParser] = {
            SourceType.PDF: PDFParser(),
            SourceType.MARKDOWN: MarkdownParser(),
            SourceType.TXT: TextParser(),
        }

    @staticmethod
    def detect_source_type(filename: str) -> SourceType:
        """Infer source format from file extension."""
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return SourceType.PDF
        elif suffix in (".md", ".markdown"):
            return SourceType.MARKDOWN
        elif suffix in (".txt", ".text", ".log"):
            return SourceType.TXT
        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Supported: .pdf, .md, .txt")

    @staticmethod
    def generate_document_id(file_bytes: bytes, filename: str) -> str:
        """Generate a deterministic 16-character SHA-256 identifier."""
        hasher = hashlib.sha256()
        hasher.update(file_bytes)
        hasher.update(filename.encode("utf-8"))
        return f"doc_{hasher.hexdigest()[:16]}"

    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        source_type: Optional[SourceType] = None,
        source_path: Optional[str] = None,
    ) -> IngestionResult:
        """Execute full ingestion pipeline: Parse -> Clean -> Chunk -> Metadata.
        
        Args:
            file_bytes: Raw binary content of the file.
            filename: Name of the file.
            source_type: Optional explicit source format; auto-detected if omitted.
            source_path: Optional path or URI describing where the file came from.
            
        Returns:
            IngestionResult containing all generated chunks with provenance metadata.
        """
        resolved_type = source_type or self.detect_source_type(filename)
        parser = self._parsers.get(resolved_type)
        if not parser:
            raise ValueError(f"No parser registered for source type '{resolved_type}'")

        doc_id = self.generate_document_id(file_bytes, filename)
        source_uri = source_path or filename
        logger.info(f"Ingesting '{filename}' (ID: {doc_id}, Type: {resolved_type.value})")

        # 1. Parse raw content into structured sections
        parsed_sections: List[ParsedSection] = parser.parse(file_bytes, filename)
        if not parsed_sections:
            logger.warning(f"File '{filename}' yielded zero text sections.")
            return IngestionResult(
                document_id=doc_id,
                document_name=filename,
                source_type=resolved_type,
                total_chunks=0,
                total_tokens=0,
                chunks=[],
            )

        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0
        total_tokens = 0

        # 2. Iterate through sections, clean and chunk each section
        for section in parsed_sections:
            cleaned_text = DocumentCleaner.clean(section.text)
            if not cleaned_text:
                continue

            section_chunks = self.chunker.split_text(cleaned_text)

            for chunk_text, token_count, char_count in section_chunks:
                chunk_id = f"{doc_id}_c{global_chunk_idx:04d}"
                metadata = ChunkMetadata(
                    document_id=doc_id,
                    document_name=filename,
                    source=source_uri,
                    page_number=section.page_number,
                    section=section.section_title,
                    chunk_id=chunk_id,
                    chunk_position=global_chunk_idx,
                    token_count=token_count,
                    char_count=char_count,
                )
                chunks.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=metadata))
                total_tokens += token_count
                global_chunk_idx += 1

        logger.info(
            f"Successfully ingested '{filename}': {len(chunks)} chunks, {total_tokens} tokens total"
        )

        return IngestionResult(
            document_id=doc_id,
            document_name=filename,
            source_type=resolved_type,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            chunks=chunks,
        )

