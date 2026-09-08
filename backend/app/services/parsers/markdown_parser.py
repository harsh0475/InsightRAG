"""Markdown document parser extracting section-aware text."""
import logging
import re
from typing import List
from backend.app.schemas.document import ParsedSection
from backend.app.services.parsers.base import BaseParser

logger = logging.getLogger("insightrag.parsers.markdown")


class MarkdownParser(BaseParser):
    """Parses Markdown documents into sections delineated by ATX headings (#, ##, ###)."""

    # Regex matching Markdown ATX headers (# Header Title)
    _HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedSection]:
        """Parse UTF-8 Markdown text into header-aware sections."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Unable to decode Markdown file '{filename}': {e}") from e

        if not text.strip():
            return []

        lines = text.split("\n")
        sections: List[ParsedSection] = []
        current_title: str = "Introduction"
        current_lines: List[str] = []

        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if header_match:
                # If we have accumulated text under the previous header, store it
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append(
                        ParsedSection(
                            text=section_text,
                            page_number=None,
                            section_title=current_title,
                        )
                    )
                # Update current section title
                current_title = header_match.group(2).strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        # Flush remaining section
        final_text = "\n".join(current_lines).strip()
        if final_text:
            sections.append(
                ParsedSection(
                    text=final_text,
                    page_number=None,
                    section_title=current_title,
                )
            )

        logger.info(f"Parsed Markdown '{filename}' into {len(sections)} sections")
        return sections

