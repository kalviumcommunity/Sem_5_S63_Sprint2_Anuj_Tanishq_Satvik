"""Document file loading and initial validation."""

import os
from typing import Any, Dict, Optional, Tuple


class DocumentLoader:
    """Loads document files from storage paths while capturing source metadata."""

    @staticmethod
    def load_text_file(file_path: str) -> Tuple[str, int]:
        """Load a text-based document from disk returning text and character count."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return content, len(content)

    @classmethod
    def load_document(
        cls,
        file_path: str,
        source: Optional[str] = None,
        url: Optional[str] = None,
        document_version: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a document file and package its fundamental source metadata."""
        content, char_count = cls.load_text_file(file_path)
        filename = os.path.basename(file_path)
        canonical_source = source or filename

        # Infer document type from extension if not specified
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        doc_type = document_type or ext or "text"

        return {
            "source": canonical_source,
            "filename": filename,
            "text": content,
            "char_count": char_count,
            "url": url,
            "document_version": document_version,
            "document_type": doc_type,
            "file_path": file_path,
        }
