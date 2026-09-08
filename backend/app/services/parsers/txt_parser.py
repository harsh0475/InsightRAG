"""Plain text document parser."""
import logging
from typing import List
from backend.app.schemas.document import ParsedSection
from backend.app.services.parsers.base import BaseParser

logger = logging.getLogger("insightrag.parsers.txt")


class TextParser(BaseParser):
    """Parses raw text (.txt) files."""

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedSection]:
        """Decode raw text and return as a parsed section."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Unable to decode TXT file '{filename}': {e}") from e

        if not text.strip():
            return []

        logger.info(f"Parsed plain text file '{filename}' ({len(text)} characters)")
        return [
            ParsedSection(
                text=text,
                page_number=None,
                section_title="General",
            )
        ]

