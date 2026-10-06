"""Context builder for LLM prompts."""

from typing import List
from backend.app.services.retrieval.search import SearchResult


class ContextBuilder:
    """Builds structured context strings from retrieved chunks, preserving attribution metadata."""

    @staticmethod
    def build(chunks: List[SearchResult]) -> str:
        """Format chunks into numbered reference items with document title, page, and section."""
        if not chunks:
            return "No relevant documents found."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            block = (
                f"[{idx}] Source Document: \"{chunk.document_title}\" | "
                f"Page: {chunk.page_number} | "
                f"Section: {chunk.section or 'N/A'} | "
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Content: {chunk.text}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)
