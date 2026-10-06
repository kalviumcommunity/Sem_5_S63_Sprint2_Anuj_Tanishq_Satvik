"""LLM module exports."""

from backend.app.services.llm.types import LLMConfig, CompletionResponse
from backend.app.services.llm.exceptions import (
    LLMException,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    LLMProviderError,
    StructuredOutputValidationError,
    MissingPromptVariableError,
    PromptTemplateNotFoundError,
)
from backend.app.services.llm.cost import (
    TokenUsage,
    TokenCostEstimator,
    token_cost_estimator,
    MODEL_PRICING_PER_MILLION,
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
    build_structured_rag_prompt,
)
from backend.app.services.llm.templates import (
    PromptTemplate,
    PromptTemplateRegistry,
    prompt_registry,
    ACADEMIC_QA_TEMPLATE,
    QUERY_REWRITING_TEMPLATE,
    SUMMARIZATION_TEMPLATE,
    SOURCE_GROUNDED_ANSWERING_TEMPLATE,
    INSUFFICIENT_CONTEXT_REFUSAL_TEMPLATE,
)
from backend.app.services.llm.parser import (
    ResponseParser,
    ParsedResponse,
    StructuredOutputParser,
)
from backend.app.schemas.response import (
    StructuredResearchResponse,
    EvidenceStatus,
    SourceReference,
    StructuredCitation,
)

__all__ = [
    "LLMConfig",
    "CompletionResponse",
    "LLMException",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMResponseError",
    "LLMProviderError",
    "StructuredOutputValidationError",
    "MissingPromptVariableError",
    "PromptTemplateNotFoundError",
    "TokenUsage",
    "TokenCostEstimator",
    "token_cost_estimator",
    "MODEL_PRICING_PER_MILLION",
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
    "build_structured_rag_prompt",
    "PromptTemplate",
    "PromptTemplateRegistry",
    "prompt_registry",
    "ACADEMIC_QA_TEMPLATE",
    "QUERY_REWRITING_TEMPLATE",
    "SUMMARIZATION_TEMPLATE",
    "SOURCE_GROUNDED_ANSWERING_TEMPLATE",
    "INSUFFICIENT_CONTEXT_REFUSAL_TEMPLATE",
    "ResponseParser",
    "ParsedResponse",
    "StructuredOutputParser",
    "StructuredResearchResponse",
    "EvidenceStatus",
    "SourceReference",
    "StructuredCitation",
]


