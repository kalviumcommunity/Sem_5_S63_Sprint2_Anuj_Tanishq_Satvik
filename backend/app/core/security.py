"""Security helpers and validation utilities."""

import hashlib
from typing import List


def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content for deduplication."""
    return hashlib.sha256(content).hexdigest()


def is_file_type_allowed(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension against allowed types."""
    lower_name = filename.lower()
    return any(lower_name.endswith(ext.lower()) for ext in allowed_extensions)
