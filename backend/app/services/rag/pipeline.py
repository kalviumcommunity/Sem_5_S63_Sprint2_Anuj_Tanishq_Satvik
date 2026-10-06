"""End-to-end RAG query pipeline orchestration."""

import time
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.response import QueryResponse
from backend.app.services.llm.client import get_llm_client
from backend.app.services.llm.prompts import (
    ACADEMIC_RAG_SYSTEM_PROMPT,
    build_academic_rag_prompt,
    build_rag_user_prompt,
)
from backend.app.services.llm.parser import ResponseParser
from backend.app.models.conversation import MessageRole
from backend.app.services.retrieval.search import SearchResult
from backend.app.services.retrieval.reranker import Reranker
from backend.app.services.rag.context import ContextBuilder
from backend.app.services.rag.grounding import GroundingService
from backend.app.services.rag.citations import CitationService
from backend.app.services.conversations.manager import ConversationContextManager


class RAGPipeline:
    """Coordinates retrieval, reranking, context assembly, LLM answering, and citation validation."""

    def __init__(self, conversation_manager: Optional[ConversationContextManager] = None):
        self.llm = get_llm_client()
        self.grounding = GroundingService()
        self.reranker = Reranker()
        self.context_builder = ContextBuilder()
        self.citation_service = CitationService()
        self.conversation_manager = conversation_manager or ConversationContextManager()

    async def answer(
        self,
        query: str,
        candidates: List[SearchResult],
        conversation_id: Optional[str] = None,
    ) -> QueryResponse:
        """Execute RAG answering flow."""
        start_time = time.time()

        # Check evidence grounding sufficiency
        if not candidates or not self.grounding.is_sufficient(candidates):
            logger.info("Insufficient evidence retrieved; returning refusal response.")
            return QueryResponse(
                answer=settings.REFUSAL_MESSAGE,
                citations=[],
                grounded=False,
                conversation_id=conversation_id,
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )

        # Rerank candidates
        reranked = self.reranker.rank(query, candidates, top_k=settings.TOP_K_RERANKED)

        # Build context preserving metadata
        context_str = self.context_builder.build(reranked)

        # Construct structured prompt bundle respecting context window and history budget
        prompt_bundle = self.conversation_manager.assemble_prompt_bundle(
            current_query=query,
            retrieved_context=context_str,
            conv_id=conversation_id,
        )
        completion = await self.llm.complete(
            prompt=prompt_bundle.user_query,
            system_prompt=prompt_bundle.system_prompt,
            messages=prompt_bundle.to_api_messages(),
        )

        parsed = ResponseParser.parse(completion.text)
        citations = self.citation_service.validate(parsed, reranked)

        # Record conversation turns for follow-up inquiry memory
        if conversation_id:
            self.conversation_manager.add_message(conversation_id, MessageRole.USER, query)
            self.conversation_manager.add_message(conversation_id, MessageRole.ASSISTANT, parsed.answer)

        latency = round((time.time() - start_time) * 1000, 2)
        usage_dict = completion.token_usage.to_dict() if completion.token_usage else None

        return QueryResponse(
            answer=parsed.answer,
            citations=citations,
            grounded=not parsed.is_refusal,
            conversation_id=conversation_id,
            latency_ms=latency,
            token_usage=usage_dict,
            estimated_cost_usd=completion.estimated_cost_usd,
        )
