"""Text extraction interface and basic implementations."""

from typing import Dict, List


class TextExtractor:
    """Extracts raw text and metadata from document files."""

    @staticmethod
    def extract_from_text(raw_text: str) -> List[Dict[str, any]]:
        """Extract pages from plain text content."""
        pages = raw_text.split("\f")  # Form feed represents page breaks
        return [
            {"page_number": idx + 1, "text": page}
            for idx, page in enumerate(pages)
            if page.strip()
        ]
