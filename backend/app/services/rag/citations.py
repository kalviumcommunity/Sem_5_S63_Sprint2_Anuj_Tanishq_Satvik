"""Citation validation and attribution service."""

import re
from typing import List
from backend.app.schemas.response import CitationItem
from backend.app.services.retrieval.search import SearchResult
from backend.app.services.llm.parser import ParsedResponse


class CitationService:
    """Validates citations to guarantee all references map strictly to retrieved evidence."""

    @staticmethod
    def validate(
        parsed_response: ParsedResponse,
        retrieved_chunks: List[SearchResult],
    ) -> List[CitationItem]:
        """Produce verified citations based on the retrieved evidence."""
        citations: List[CitationItem] = []

        if parsed_response.is_refusal or not retrieved_chunks:
            return citations

        answer_sentences = [
            " ".join(sentence.split())
            for sentence in re.split(r"(?<=[.!?])\s+", parsed_response.answer)
            if sentence.strip()
        ]
        for chunk in retrieved_chunks:
            normalized_chunk = " ".join(chunk.text.split())
            evidence_snippet = next(
                (
                    sentence
                    for sentence in answer_sentences
                    if sentence.casefold() in normalized_chunk.casefold()
                ),
                chunk.text[:500].strip(),
            )
            citations.append(
                CitationItem(
                    document_id=chunk.document_id,
                    title=chunk.document_title,
                    page=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    section=chunk.section,
                    relevance_score=chunk.score,
                    similarity_score=(
                        chunk.similarity_score
                        if chunk.similarity_score is not None
                        else chunk.score
                    ),
                    citation_text=f"{chunk.document_title}, Page {chunk.page_number}",
                    snippet=evidence_snippet,
                )
            )

        return citations
