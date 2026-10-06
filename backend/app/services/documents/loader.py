"""Document file loading and initial validation."""

import os
from typing import Tuple


class DocumentLoader:
    """Loads document files from storage paths."""

    @staticmethod
    def load_text_file(file_path: str) -> Tuple[str, int]:
        """Load a text-based document from disk returning text and character count."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return content, len(content)
