"""End-to-end RAG query pipeline orchestration."""

import time
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.response import (
    EvidenceStatus,
    QueryResponse,
    SourceReference,
    StructuredResearchResponse,
)
from backend.app.services.llm.client import get_llm_client
from backend.app.services.llm.prompts import (
    ACADEMIC_RAG_SYSTEM_PROMPT,
    build_academic_rag_prompt,
    build_rag_user_prompt,
)
from backend.app.services.llm.parser import ResponseParser, StructuredOutputParser
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
                evidence_status=EvidenceStatus.INSUFFICIENT.value,
                confidence=0.0,
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

        structured_model: Optional[StructuredResearchResponse] = None
        evidence_status: str = EvidenceStatus.SUFFICIENT.value
        confidence_val: float = 1.0
        source_refs: List[SourceReference] = []

        if settings.LLM_RESPONSE_FORMAT == "json_object":
            try:
                structured_model = StructuredOutputParser.parse_and_validate(completion.text)
                answer_text = structured_model.answer
                evidence_status = structured_model.evidence_status.value
                confidence_val = structured_model.confidence
                source_refs = structured_model.source_references
                is_refusal = structured_model.is_refusal
            except Exception as exc:
                logger.warning("Structured parsing fallback to text: %s", exc)
                parsed = ResponseParser.parse(completion.text)
                answer_text = parsed.answer
                is_refusal = parsed.is_refusal
        else:
            parsed = ResponseParser.parse(completion.text)
            answer_text = parsed.answer
            is_refusal = parsed.is_refusal
            if is_refusal:
                evidence_status = EvidenceStatus.INSUFFICIENT.value
                confidence_val = 0.0

        # Validate extracted citations
        parsed_obj = parsed if "parsed" in locals() else ResponseParser.parse(answer_text)
        citations = self.citation_service.validate(parsed_obj, reranked)

        # Record conversation turns for follow-up inquiry memory
        if conversation_id:
            self.conversation_manager.add_message(conversation_id, MessageRole.USER, query)
            self.conversation_manager.add_message(conversation_id, MessageRole.ASSISTANT, answer_text)

        latency = round((time.time() - start_time) * 1000, 2)
        usage_dict = completion.token_usage.to_dict() if completion.token_usage else None

        return QueryResponse(
            answer=answer_text,
            citations=citations,
            grounded=not is_refusal,
            conversation_id=conversation_id,
            latency_ms=latency,
            token_usage=usage_dict,
            estimated_cost_usd=completion.estimated_cost_usd,
            evidence_status=evidence_status,
            confidence=confidence_val,
            source_references=source_refs,
            structured_response=structured_model,
        )

    async def answer_structured(
        self,
        query: str,
        candidates: List[SearchResult],
        conversation_id: Optional[str] = None,
    ) -> StructuredResearchResponse:
        """Execute RAG flow returning strictly validated StructuredResearchResponse."""
        if not candidates or not self.grounding.is_sufficient(candidates):
            return StructuredResearchResponse(
                answer=settings.REFUSAL_MESSAGE,
                citations=[],
                evidence_status=EvidenceStatus.INSUFFICIENT,
                confidence=0.0,
                source_references=[],
                refusal_reason="Insufficient retrieved evidence to answer question reliably.",
            )

        reranked = self.reranker.rank(query, candidates, top_k=settings.TOP_K_RERANKED)
        context_str = self.context_builder.build(reranked)

        prompt_bundle = self.conversation_manager.assemble_prompt_bundle(
            current_query=query,
            retrieved_context=context_str,
            conv_id=conversation_id,
        )

        structured_model, _ = await self.llm.complete_structured(
            prompt=prompt_bundle.user_query,
            schema_cls=StructuredResearchResponse,
            system_prompt=prompt_bundle.system_prompt,
            messages=prompt_bundle.to_api_messages(),
        )

        if conversation_id:
            self.conversation_manager.add_message(conversation_id, MessageRole.USER, query)
            self.conversation_manager.add_message(conversation_id, MessageRole.ASSISTANT, structured_model.answer)

        return structured_model

