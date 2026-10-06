"""LLM module exports."""

from backend.app.services.llm.types import LLMConfig, CompletionResponse
from backend.app.services.llm.exceptions import (
    LLMException,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    LLMProviderError,
)
from backend.app.services.llm.client import (
    LLMClientInterface,
    MockLLMClient,
    OpenAILLMClient,
    GeminiLLMClient,
    get_llm_client,
)
from backend.app.services.llm.prompts import (
    PromptRole,
    PromptMessage,
    PromptBundle,
    PromptBuilder,
    RESEARCHMATE_SYSTEM_PROMPT,
    ACADEMIC_RAG_SYSTEM_PROMPT,
    build_academic_rag_prompt,
    build_rag_user_prompt,
)
from backend.app.services.llm.parser import ResponseParser, ParsedResponse

__all__ = [
    "LLMConfig",
    "CompletionResponse",
    "LLMException",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMResponseError",
    "LLMProviderError",
    "LLMClientInterface",
    "MockLLMClient",
    "OpenAILLMClient",
    "GeminiLLMClient",
    "get_llm_client",
    "PromptRole",
    "PromptMessage",
    "PromptBundle",
    "PromptBuilder",
    "RESEARCHMATE_SYSTEM_PROMPT",
    "ACADEMIC_RAG_SYSTEM_PROMPT",
    "build_academic_rag_prompt",
    "build_rag_user_prompt",
    "ResponseParser",
    "ParsedResponse",
]
