"""Custom exceptions for conversation and context window management."""


class ConversationException(Exception):
    """Base exception for conversation management errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConversationNotFoundError(ConversationException):
    """Raised when a specified conversation ID does not exist."""

    def __init__(self, conversation_id: str):
        super().__init__(f"Conversation '{conversation_id}' was not found.")
        self.conversation_id = conversation_id


class OversizedQueryError(ConversationException):
    """Raised when a user query exceeds the allocated token budget under strict policy."""

    def __init__(self, query_tokens: int, max_tokens: int):
        super().__init__(
            f"Query size ({query_tokens} tokens) exceeds allowed budget ({max_tokens} tokens)."
        )
        self.query_tokens = query_tokens
        self.max_tokens = max_tokens


class OversizedContextError(ConversationException):
    """Raised when retrieved context passages exceed the allocated budget under strict policy."""

    def __init__(self, context_tokens: int, max_tokens: int):
        super().__init__(
            f"Retrieved context size ({context_tokens} tokens) exceeds allocated budget ({max_tokens} tokens)."
        )
        self.context_tokens = context_tokens
        self.max_tokens = max_tokens


class ContextWindowExceededError(ConversationException):
    """Raised when total mandatory context exceeds the hard model context window limit."""

    def __init__(self, required_tokens: int, context_window: int):
        super().__init__(
            f"Required tokens ({required_tokens}) exceed model context window limit ({context_window})."
        )
        self.required_tokens = required_tokens
        self.context_window = context_window
