"""PDF document parser extracting page-aware text sections."""
import io
import logging
from typing import List
from pypdf import PdfReader
from backend.app.schemas.document import ParsedSection
from backend.app.services.parsers.base import BaseParser

logger = logging.getLogger("insightrag.parsers.pdf")


class PDFParser(BaseParser):
    """Parses PDF documents page-by-page preserving 1-based page metadata."""

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedSection]:
        """Extract text from PDF pages."""
        sections: List[ParsedSection] = []
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            num_pages = len(reader.pages)
            logger.info(f"Parsing PDF '{filename}' with {num_pages} pages")

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                page_num = page_idx + 1
                if page_text.strip():
                    sections.append(
                        ParsedSection(
                            text=page_text,
                            page_number=page_num,
                            section_title=f"Page {page_num}",
                        )
                    )
                else:
                    logger.debug(f"Skipping empty or image-only page {page_num} in '{filename}'")

            if not sections:
                logger.warning(f"No extractable text found in PDF '{filename}'. File may contain only scanned images.")

        except Exception as e:
            logger.error(f"Failed to parse PDF '{filename}': {str(e)}", exc_info=True)
            raise ValueError(f"Corrupt or unreadable PDF document '{filename}': {str(e)}") from e

        return sections

