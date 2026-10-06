"""Text cleaner for document processing."""

import re
import unicodedata


class TextCleaner:
    """Cleans and normalizes extracted text from documents."""

    @staticmethod
    def clean(text: str) -> str:
        """Apply text normalization, whitespace compaction, and character cleanup."""
        if not text:
            return ""

        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Replace excessive whitespace while preserving single paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove control characters (except newline and tab)
        text = "".join(ch for ch in text if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C"))

        return text.strip()
