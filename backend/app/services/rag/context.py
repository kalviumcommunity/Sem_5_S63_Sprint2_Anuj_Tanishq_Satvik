"""Context builder for LLM prompts preserving source and location attribution."""

from typing import List
from backend.app.services.retrieval.search import SearchResult


class ContextBuilder:
    """Builds structured context strings from retrieved chunks, preserving attribution metadata."""

    @staticmethod
    def build(chunks: List[SearchResult]) -> str:
        """Format chunks into numbered reference items with document title, page, section, and metadata."""
        if not chunks:
            return "No relevant documents found."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            source_ref = chunk.metadata.get("source") or chunk.document_title
            url_str = f" | URL: {chunk.metadata.get('url')}" if chunk.metadata.get("url") else ""
            ver_str = f" | Version: {chunk.metadata.get('document_version')}" if chunk.metadata.get("document_version") else ""
            chunk_idx_str = f" | Chunk Index: {chunk.metadata.get('chunk_index', 0)}"

            char_offsets_str = ""
            if chunk.metadata.get("char_start") is not None and chunk.metadata.get("char_end") is not None:
                char_offsets_str = f" | Offsets: [{chunk.metadata.get('char_start')}-{chunk.metadata.get('char_end')}]"

            block = (
                f"[{idx}] Source Document: \"{chunk.document_title}\" | "
                f"Source: \"{source_ref}\" | "
                f"Page: {chunk.page_number} | "
                f"Section: {chunk.section or 'N/A'}"
                f"{chunk_idx_str}"
                f"{url_str}"
                f"{ver_str}"
                f"{char_offsets_str} | "
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Content: {chunk.text}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)
