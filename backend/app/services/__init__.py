"""Services module packaging document processing, LLM, embeddings, retrieval, RAG, and conversations."""

from backend.app.services.documents import (
    TextCleaner,
    DocumentChunker,
    TextExtractor,
    DocumentLoader,
)
from backend.app.services.llm import (
    LLMClientInterface,
    get_llm_client,
    ResponseParser,
)
from backend.app.services.embeddings import (
    EmbeddingServiceInterface,
    get_embedding_service,
    chunk_list,
)
from backend.app.services.retrieval import (
    SearchResult,
    cosine_similarity,
    MetadataFilter,
    Reranker,
)
from backend.app.services.rag import (
    ContextBuilder,
    GroundingService,
    CitationService,
    RAGPipeline,
)
from backend.app.services.conversations import (
    ConversationManager,
    QueryRewriter,
)

__all__ = [
    "TextCleaner",
    "DocumentChunker",
    "TextExtractor",
    "DocumentLoader",
    "LLMClientInterface",
    "get_llm_client",
    "ResponseParser",
    "EmbeddingServiceInterface",
    "get_embedding_service",
    "chunk_list",
    "SearchResult",
    "cosine_similarity",
    "MetadataFilter",
    "Reranker",
    "ContextBuilder",
    "GroundingService",
    "CitationService",
    "RAGPipeline",
    "ConversationManager",
    "QueryRewriter",
]
