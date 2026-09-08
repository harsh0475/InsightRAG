"""Abstract Base Parser for document ingestion."""
from abc import ABC, abstractmethod
from typing import List
from backend.app.schemas.document import ParsedSection


class BaseParser(ABC):
    """Abstract interface defining contract for format-specific document parsers."""

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> List[ParsedSection]:
        """Parse raw file bytes into a list of structured intermediate sections.
        
        Args:
            file_bytes: Binary content of the uploaded document.
            filename: Name of the file for reference and logging.
            
        Returns:
            List of ParsedSection objects containing text, page numbers, or section headings.
        """
        pass

