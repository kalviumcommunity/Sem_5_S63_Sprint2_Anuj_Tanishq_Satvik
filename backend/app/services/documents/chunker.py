"""Document chunker implementation."""

from typing import List
from backend.app.models.chunk import Chunk


class DocumentChunker:
    """Splits document text into chunks preserving page and section context."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        document_id: str,
        page_number: int = 1,
        section: str = "main",
    ) -> List[Chunk]:
        """Split text into overlapping chunks with metadata."""
        if not text:
            return []

        chunks: List[Chunk] = []
        words = text.split()
        if not words:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_idx,
                text=chunk_text,
                page_number=page_number,
                section=section,
                token_count=len(chunk_words),  # Approximate word/token estimate
                metadata={
                    "page_number": page_number,
                    "section": section,
                    "word_count": len(chunk_words),
                },
            )
            chunks.append(chunk)
            chunk_idx += 1

            if i + self.chunk_size >= len(words):
                break

        return chunks
