"""Document chunker implementation."""

from typing import List, Optional
from backend.app.models.chunk import Chunk
from backend.app.models.document import LoadedDocument


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

    def chunk_document(self, doc: LoadedDocument) -> List[Chunk]:
        """Split a loaded multi-page document into chunks while preserving page provenance and metadata."""
        all_chunks: List[Chunk] = []
        global_chunk_idx = 0

        pages_to_process = doc.pages if doc.pages else []
        if not pages_to_process and doc.raw_text:
            pages_to_process = [type("Page", (), {"page_number": 1, "text": doc.raw_text, "metadata": {}})()]

        for page in pages_to_process:
            page_chunks = self.chunk_text(
                text=page.text,
                document_id=doc.id,
                page_number=page.page_number,
                section=getattr(page, "metadata", {}).get("section", "main"),
            )
            for chunk in page_chunks:
                chunk.chunk_index = global_chunk_idx
                chunk.metadata.update(
                    {
                        "filename": doc.filename,
                        "document_type": doc.document_type.value if hasattr(doc.document_type, "value") else str(doc.document_type),
                        "source_path": doc.source_path,
                        "page_number": page.page_number,
                    }
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        return all_chunks
