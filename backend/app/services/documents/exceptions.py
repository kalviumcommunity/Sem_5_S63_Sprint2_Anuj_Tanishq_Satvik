"""Exceptions for document intake, extraction, and loading."""


class DocumentIntakeError(Exception):
    """Base exception for all document loading and processing failures."""

    def __init__(self, message: str, filename: str = "", details: dict = None):
        super().__init__(message)
        self.filename = filename
        self.details = details or {}


class UnsupportedFormatError(DocumentIntakeError):
    """Raised when an uploaded document has an unsupported extension or media type."""

    def __init__(self, filename: str, extension: str = "", supported_formats: list = None):
        msg = f"Unsupported document format for file '{filename}' (extension: '{extension}'). Supported: {supported_formats or ['pdf', 'txt', 'docx', 'md']}"
        super().__init__(msg, filename=filename, details={"extension": extension, "supported": supported_formats})


class CorruptedDocumentError(DocumentIntakeError):
    """Raised when a document is damaged, unparseable, or malformed."""

    def __init__(self, filename: str, reason: str = "Unreadable or corrupted structure"):
        msg = f"Corrupted or unreadable document file '{filename}': {reason}"
        super().__init__(msg, filename=filename, details={"reason": reason})


class EmptyDocumentError(DocumentIntakeError):
    """Raised when a document contains zero non-whitespace characters."""

    def __init__(self, filename: str):
        msg = f"Document file '{filename}' contains no extractable text or content."
        super().__init__(msg, filename=filename)
