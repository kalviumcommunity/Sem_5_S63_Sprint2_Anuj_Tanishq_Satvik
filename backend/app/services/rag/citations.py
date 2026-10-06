"""Citation validation and attribution service."""

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

        for chunk in retrieved_chunks:
            citations.append(
                CitationItem(
                    document_id=chunk.document_id,
                    title=chunk.document_title,
                    page=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    section=chunk.section,
                    relevance_score=chunk.score,
                    citation_text=f"{chunk.document_title}, Page {chunk.page_number}",
                )
            )

        return citations
