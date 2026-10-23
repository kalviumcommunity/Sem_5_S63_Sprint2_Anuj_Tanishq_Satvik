"""Citation validation and attribution service linking answers to retrieved source metadata."""

from typing import List, Optional
from backend.app.schemas.response import CitationItem
from backend.app.services.retrieval.search import SearchResult
from backend.app.services.llm.parser import ParsedResponse


class CitationService:
    """Validates citations to guarantee all references map strictly to retrieved evidence."""

    @classmethod
    def format_citation_label(
        cls,
        source: Optional[str],
        page: Optional[int] = None,
        section: Optional[str] = None,
        url: Optional[str] = None,
    ) -> str:
        """Format a human-readable citation label based on available location information."""
        if not source or source == "unknown_doc":
            return "[Unknown Source - Unverified]"

        # Section formatting
        sec_str = f", {section} section" if section and section.lower() not in ("main", "n/a", "overview") else ""
        page_str = f", page {page}" if page is not None else ""

        # Build location suffix
        loc_suffix = f"{sec_str}{page_str}"

        # Markdown URL link if canonical URL is present
        if url:
            if loc_suffix:
                return f"[{source}{loc_suffix}]({url})"
            return f"[{source}]({url})"

        if loc_suffix:
            return f"[{source}{loc_suffix}]"

        return f"[{source}]"

    @classmethod
    def validate(
        cls,
        parsed_response: ParsedResponse,
        retrieved_chunks: List[SearchResult],
    ) -> List[CitationItem]:
        """Produce verified citations based strictly on the retrieved evidence."""
        citations: List[CitationItem] = []

        if parsed_response.is_refusal or not retrieved_chunks:
            return citations

        for chunk in retrieved_chunks:
            src = chunk.metadata.get("source") or chunk.document_title or chunk.document_id
            page = chunk.page_number if chunk.page_number is not None else chunk.metadata.get("page_number")
            section = chunk.section or chunk.metadata.get("section")
            url = chunk.metadata.get("url")
            chunk_idx = chunk.metadata.get("chunk_index")
            doc_ver = chunk.metadata.get("document_version")
            char_start = chunk.metadata.get("char_start")
            char_end = chunk.metadata.get("char_end")

            cit_label = cls.format_citation_label(
                source=src,
                page=page,
                section=section,
                url=url,
            )

            citations.append(
                CitationItem(
                    document_id=chunk.document_id,
                    title=chunk.document_title,
                    page=page if page is not None else 1,
                    chunk_id=chunk.chunk_id,
                    section=section,
                    relevance_score=chunk.score,
                    citation_text=cit_label,
                    source=src,
                    chunk_index=chunk_idx,
                    url=url,
                    document_version=doc_ver,
                    char_start=char_start,
                    char_end=char_end,
                    metadata=dict(chunk.metadata),
                )
            )

        return citations
