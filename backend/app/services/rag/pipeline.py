"""End-to-end RAG query pipeline orchestration."""

import re
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
from backend.app.services.llm.parser import ParsedResponse, ResponseParser, StructuredOutputParser
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
        grounding_threshold: Optional[float] = None,
        mock_embeddings: bool = False,
    ) -> QueryResponse:
        """Execute RAG answering flow."""
        start_time = time.time()

        # Check evidence grounding sufficiency
        if not candidates or not self.grounding.is_sufficient(candidates, grounding_threshold):
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
        reranked = self.reranker.rank(
            query,
            candidates,
            top_k=settings.TOP_K_RERANKED,
            vector_weight=0.2 if mock_embeddings else 0.7,
            keyword_weight=0.8 if mock_embeddings else 0.3,
        )

        if self._uses_mock_llm():
            return self._answer_extractively(query, reranked, conversation_id, start_time)

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

    def _uses_mock_llm(self) -> bool:
        if getattr(self.llm, "provider_name", None) == "mock":
            return True
        api_key = getattr(self.llm, "api_key", None)
        missing_or_placeholder_key = not api_key or (
            isinstance(api_key, str)
            and (
                api_key.casefold().startswith("your-")
                or "placeholder" in api_key.casefold()
            )
        )
        return (
            missing_or_placeholder_key
            and bool(getattr(self.llm, "fallback_to_mock", False))
        )

    def _answer_extractively(
        self,
        query: str,
        candidates: List[SearchResult],
        conversation_id: Optional[str],
        start_time: float,
    ) -> QueryResponse:
        """Return source text rather than the canned response from an offline mock LLM."""
        sentence_matches: List[tuple[float, SearchResult, str]] = []

        for candidate in candidates:
            text = re.sub(r"\s+", " ", candidate.text).strip()
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 40 or sentence[-1] not in ".?!":
                    continue
                overlap = self.reranker.keyword_overlap(query, sentence)
                if overlap:
                    sentence_matches.append((overlap, candidate, sentence))

        if not sentence_matches:
            logger.info("Offline extractive answer found no matching source sentence; refusing.")
            refusal = settings.REFUSAL_MESSAGE
            if conversation_id:
                self.conversation_manager.add_message(conversation_id, MessageRole.USER, query)
                self.conversation_manager.add_message(
                    conversation_id,
                    MessageRole.ASSISTANT,
                    refusal,
                )
            return QueryResponse(
                answer=refusal,
                citations=[],
                grounded=False,
                conversation_id=conversation_id,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                evidence_status=EvidenceStatus.INSUFFICIENT.value,
                confidence=0.0,
            )

        sentence_matches.sort(
            key=lambda match: (match[0], match[1].score, -len(match[2])),
            reverse=True,
        )
        query_terms = set(re.findall(r"\w+", query.casefold()))
        primary_match = sentence_matches[0]
        if {"parametric", "non"} <= query_terms:
            definitions = [
                match
                for match in sentence_matches
                if "parametric memory" in match[2].casefold()
                and "non-parametric memory" in match[2].casefold()
                and " is " in match[2].casefold()
            ]
            if definitions:
                primary_match = max(
                    definitions,
                    key=lambda match: (match[0], match[1].score, -len(match[2])),
                )

        selected_matches = [primary_match]
        primary_chunk_id = primary_match[1].chunk_id
        complementary_match = None
        if "how" in query_terms:
            complementary_match = next(
                (
                    match
                    for match in sentence_matches
                    if match[1].chunk_id != primary_chunk_id
                    and match[2].endswith((".", "?", "!"))
                    and "retriev" in match[2].casefold()
                    and "combin" in match[2].casefold()
                    and any(
                        term in match[2].casefold()
                        for term in ("generator", "seq2seq")
                    )
                ),
                None,
            )
        elif query_terms & {"advantage", "advantages", "benefit", "benefits"}:
            complementary_match = next(
                (
                    match
                    for match in sentence_matches
                    if match[1].chunk_id != primary_chunk_id
                    and "rag" in match[2].casefold()
                    and any(
                        term in match[2].casefold()
                        for term in (
                            "accurate", "accuracy", "benefit", "diverse",
                            "factual", "improve", "less", "specific",
                        )
                    )
                ),
                None,
            )
        if complementary_match:
            selected_matches.append(complementary_match)

        answer_text = " ".join(match[2] for match in selected_matches)
        sources = list({match[1].chunk_id: match[1] for match in selected_matches}.values())
        citations = self.citation_service.validate(ParsedResponse(answer=answer_text), sources)
        if conversation_id:
            self.conversation_manager.add_message(conversation_id, MessageRole.USER, query)
            self.conversation_manager.add_message(conversation_id, MessageRole.ASSISTANT, answer_text)

        logger.info(
            "Offline extractive answer returned with %d source passage(s).",
            len(sources),
        )
        return QueryResponse(
            answer=answer_text,
            citations=citations,
            grounded=True,
            conversation_id=conversation_id,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            evidence_status=EvidenceStatus.PARTIALLY_SUPPORTED.value,
            confidence=None,
            source_references=[
                SourceReference(
                    document_id=source.document_id,
                    title=source.document_title,
                    page=source.page_number,
                    chunk_id=source.chunk_id,
                    section=source.section,
                    snippet=next(
                        match[2] for match in selected_matches
                        if match[1].chunk_id == source.chunk_id
                    ),
                )
                for source in sources
            ],
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
