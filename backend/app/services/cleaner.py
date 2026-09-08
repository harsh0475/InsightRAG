"""Text cleaning and normalization service for raw document streams."""
import re
import unicodedata


class DocumentCleaner:
    """Sanitizes and normalizes extracted text across all document types."""

    # Non-printable characters (excluding standard whitespace like \n, \t)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
    # 3 or more newlines to be compacted to 2
    _EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
    # Multiple spaces or tabs on a single line
    _CONSECUTIVE_SPACES_RE = re.compile(r"[ \t]+")

    @classmethod
    def clean(cls, text: str) -> str:
        """Run standard text cleaning pipeline.
        
        Args:
            text: Raw extracted text string.
            
        Returns:
            Normalized, clean text string.
        """
        if not text:
            return ""

        # 1. Unicode NFKC normalization (replaces ligatures like 'fi', resolves accents)
        text = unicodedata.normalize("NFKC", text)

        # 2. Replace non-breaking spaces and carriage returns
        text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")

        # 3. Strip control characters
        text = cls._CONTROL_CHAR_RE.sub("", text)

        # 4. Collapse consecutive spaces/tabs on the same line
        lines = text.split("\n")
        cleaned_lines = [cls._CONSECUTIVE_SPACES_RE.sub(" ", line).strip() for line in lines]
        text = "\n".join(cleaned_lines)

        # 5. Collapse excessive blank lines (keep at most 2 newlines for paragraph breaks)
        text = cls._EXCESSIVE_NEWLINES_RE.sub("\n\n", text)

        return text.strip()

