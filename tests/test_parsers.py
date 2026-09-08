"""Unit tests for format-specific document parsers."""
import pytest
from backend.app.services.parsers.markdown_parser import MarkdownParser
from backend.app.services.parsers.pdf_parser import PDFParser
from backend.app.services.parsers.txt_parser import TextParser


def test_txt_parser():
    """Verify plain text parser extracts raw content."""
    parser = TextParser()
    raw = b"Simple document line 1.\nSimple document line 2."
    sections = parser.parse(raw, "test.txt")
    assert len(sections) == 1
    assert "Simple document line 1." in sections[0].text
    assert sections[0].page_number is None


def test_markdown_parser():
    """Verify Markdown parser separates content by headings."""
    parser = MarkdownParser()
    md_content = b"""# Main Title
This is introduction text.

## Section 1: Invalidation
Details on cache invalidation.

## Section 2: Replication
Details on Redis replication.
"""
    sections = parser.parse(md_content, "test.md")
    assert len(sections) == 3
    assert sections[0].section_title == "Main Title"
    assert "introduction text" in sections[0].text

    assert sections[1].section_title == "Section 1: Invalidation"
    assert "cache invalidation" in sections[1].text

    assert sections[2].section_title == "Section 2: Replication"
    assert "Redis replication" in sections[2].text


def test_pdf_parser_with_sample():
    """Verify PDF parser extracts each page with accurate 1-based page numbers."""
    parser = PDFParser()
    with open("knowledge_base/sample_k8s_guide.pdf", "rb") as f:
        pdf_bytes = f.read()

    sections = parser.parse(pdf_bytes, "sample_k8s_guide.pdf")
    assert len(sections) == 3

    # Check page 1
    assert sections[0].page_number == 1
    assert "kube-apiserver" in sections[0].text

    # Check page 2
    assert sections[1].page_number == 2
    assert "ClusterIP" in sections[1].text

    # Check page 3
    assert sections[2].page_number == 3
    assert "NetworkPolicies" in sections[2].text


def test_pdf_parser_corrupt_file():
    """Verify PDF parser raises ValueError when given corrupt bytes."""
    parser = PDFParser()
    with pytest.raises(ValueError, match="Corrupt or unreadable PDF document"):
        parser.parse(b"NOT_A_REAL_PDF_STREAM", "corrupt.pdf")

