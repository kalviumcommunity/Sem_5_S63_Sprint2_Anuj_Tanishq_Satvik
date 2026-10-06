"""Custom exception hierarchy for LLM API operations."""


class LLMException(Exception):
    """Base exception for all LLM client errors."""

    def __init__(self, message: str, provider: str = "unknown", status_code: int = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code

    def __str__(self) -> str:
        code_str = f" [HTTP {self.status_code}]" if self.status_code else ""
        return f"[{self.provider}]{code_str} {self.message}"


class LLMAuthenticationError(LLMException):
    """Raised when authentication fails (invalid or missing API key)."""
    pass


class LLMRateLimitError(LLMException):
    """Raised when API rate limits or quota are exceeded."""
    pass


class LLMTimeoutError(LLMException):
    """Raised when a request to the LLM provider times out."""
    pass


class LLMResponseError(LLMException):
    """Raised when the provider returns an invalid or unexpected response format."""
    pass


class LLMProviderError(LLMException):
    """Raised when the LLM provider experiences internal server errors (5xx)."""
    pass
