"""Unit tests for text cleaning and normalization service."""
from backend.app.services.cleaner import DocumentCleaner


def test_clean_empty_text():
    """Verify empty or whitespace strings return empty string."""
    assert DocumentCleaner.clean("") == ""
    assert DocumentCleaner.clean("   \n\t  ") == ""


def test_clean_unicode_normalization():
    """Verify ligature expansion and Unicode normalization."""
    # Ligature 'ﬁ' (U+FB01) should normalize to 'fi'
    raw = "The ﬁrst speciﬁcation requires veriﬁcation."
    cleaned = DocumentCleaner.clean(raw)
    assert cleaned == "The first specification requires verification."


def test_clean_non_breaking_spaces_and_crlf():
    """Verify non-breaking spaces and Windows CRLF are sanitized."""
    raw = "Line 1\r\nLine 2\u00a0with\u00a0spaces\r\nLine 3"
    cleaned = DocumentCleaner.clean(raw)
    assert cleaned == "Line 1\nLine 2 with spaces\nLine 3"


def test_clean_excessive_whitespace_and_newlines():
    """Verify multiple spaces and 3+ newlines are compacted."""
    raw = "Word1    Word2\t\tWord3\n\n\n\n\nParagraph 2   continues."
    cleaned = DocumentCleaner.clean(raw)
    assert cleaned == "Word1 Word2 Word3\n\nParagraph 2 continues."


def test_clean_control_characters():
    """Verify non-printable control characters are stripped."""
    raw = "Header\x00\x07Text\x1bFooter"
    cleaned = DocumentCleaner.clean(raw)
    assert cleaned == "HeaderTextFooter"

