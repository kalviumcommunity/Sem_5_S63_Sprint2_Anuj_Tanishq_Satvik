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


class StructuredOutputValidationError(LLMResponseError):
    """Raised when structured LLM output is malformed JSON or fails schema validation."""

    def __init__(
        self,
        message: str,
        raw_output: str = "",
        validation_errors: list = None,
        error_type: str = "validation_error",
        provider: str = "unknown",
        status_code: int = None,
    ):
        super().__init__(message, provider=provider, status_code=status_code)
        self.raw_output = raw_output
        self.validation_errors = validation_errors or []
        self.error_type = error_type

    def __str__(self) -> str:
        base = super().__str__()
        if self.validation_errors:
            err_details = "; ".join(
                f"{'.'.join(str(p) for p in err.get('loc', []))}: {err.get('msg', '')}"
                for err in self.validation_errors[:3]
            )
            return f"{base} | Details: [{err_details}]"
        return base

