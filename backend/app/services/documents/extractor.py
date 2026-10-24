"""Text extraction interface and basic implementations."""

from typing import Dict, List
from backend.app.models.document import LoadedDocument
from backend.app.services.documents.loader import DocumentLoader


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

    @staticmethod
    def extract_from_file(file_path: str) -> LoadedDocument:
        """Extract structured document content from file on disk."""
        return DocumentLoader.load(file_path)
