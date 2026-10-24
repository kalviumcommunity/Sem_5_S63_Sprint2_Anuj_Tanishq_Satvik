"""Document chunker and text splitting implementations."""

from enum import Enum
import re
from typing import Any, Dict, List, Optional
from backend.app.models.chunk import Chunk
from backend.app.models.document import LoadedDocument


class ChunkingStrategy(str, Enum):
    """Supported text splitting strategies for academic documents."""

    FIXED_WORD = "fixed_word"
    RECURSIVE_CHARACTER = "recursive_character"
    MARKDOWN_HEADER = "markdown_header"


class DocumentChunker:
    """Splits academic document text into chunks preserving page provenance, section context, and overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_WORD,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def chunk_text(
        self,
        text: str,
        document_id: str,
        page_number: int = 1,
        section: str = "main",
        strategy: Optional[ChunkingStrategy] = None,
    ) -> List[Chunk]:
        """Split text into overlapping chunks with metadata."""
        if not text or not text.strip():
            return []

        active_strategy = strategy or self.strategy

        if active_strategy == ChunkingStrategy.RECURSIVE_CHARACTER:
            chunks_text = self._split_recursive(text, self.chunk_size, self.chunk_overlap, self.separators)
        elif active_strategy == ChunkingStrategy.MARKDOWN_HEADER:
            chunks_text = self._split_markdown(text, self.chunk_size, self.chunk_overlap)
        else:
            chunks_text = self._split_fixed_words(text, self.chunk_size, self.chunk_overlap)

        chunks: List[Chunk] = []
        for idx, c_text in enumerate(chunks_text):
            cleaned = c_text.strip()
            if not cleaned:
                continue

            words = cleaned.split()
            token_est = len(words)
            has_citation = bool(re.search(r"\[\d+\]|\([A-Z][a-zA-Z]+ et al\., \d{4}\)|\([A-Z][a-zA-Z]+, \d{4}\)", cleaned))
            has_math = bool(re.search(r"\$[^$]+\$|\\[a-zA-Z]+|=", cleaned))

            chunk = Chunk(
                document_id=document_id,
                chunk_index=idx,
                text=cleaned,
                page_number=page_number,
                section=section,
                token_count=token_est,
                metadata={
                    "page_number": page_number,
                    "section": section,
                    "word_count": len(words),
                    "char_count": len(cleaned),
                    "has_citation": has_citation,
                    "has_math": has_math,
                    "strategy": active_strategy.value,
                },
            )
            chunks.append(chunk)

        return chunks

    def chunk_document(
        self,
        doc: LoadedDocument,
        strategy: Optional[ChunkingStrategy] = None,
    ) -> List[Chunk]:
        """Split a loaded multi-page document into chunks while preserving page provenance and metadata."""
        all_chunks: List[Chunk] = []
        global_chunk_idx = 0

        pages_to_process = doc.pages if doc.pages else []
        if not pages_to_process and doc.raw_text:
            pages_to_process = [type("Page", (), {"page_number": 1, "text": doc.raw_text, "metadata": {}})()]

        for page in pages_to_process:
            page_section = getattr(page, "metadata", {}).get("section", "main")
            page_chunks = self.chunk_text(
                text=page.text,
                document_id=doc.id,
                page_number=page.page_number,
                section=page_section,
                strategy=strategy,
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

    def _split_fixed_words(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        words = text.split()
        if not words:
            return []

        step = max(1, chunk_size - chunk_overlap)
        chunks: List[str] = []

        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break

        return chunks

    def _split_recursive(self, text: str, chunk_size_chars: int, chunk_overlap_chars: int, separators: List[str]) -> List[str]:
        """Recursively split text by separators down to chunk_size characters with overlap."""
        final_chunks: List[str] = []
        self._recursive_split_helper(text, chunk_size_chars, chunk_overlap_chars, separators, final_chunks)
        return final_chunks

    def _recursive_split_helper(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str],
        result: List[str],
    ) -> None:
        if len(text) <= chunk_size:
            if text.strip():
                result.append(text)
            return

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if sep and sep in text:
            splits = text.split(sep)
            current = ""
            for s in splits:
                candidate = (current + sep + s) if current else s
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    if len(s) > chunk_size and remaining_seps:
                        self._recursive_split_helper(s, chunk_size, chunk_overlap, remaining_seps, result)
                        current = ""
                    else:
                        current = s
            if current and current.strip():
                result.append(current)
        else:
            if remaining_seps:
                self._recursive_split_helper(text, chunk_size, chunk_overlap, remaining_seps, result)
            else:
                # Character-level slicing with overlap
                step = max(1, chunk_size - chunk_overlap)
                for i in range(0, len(text), step):
                    sub = text[i : i + chunk_size]
                    if sub.strip():
                        result.append(sub)

    def _split_markdown(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split markdown by headers while respecting chunk size boundaries."""
        header_pattern = re.compile(r"(?=(?:\r?\n|^)#{1,4}\s+)")
        sections = [s for s in header_pattern.split(text) if s.strip()]

        if not sections:
            return self._split_fixed_words(text, chunk_size, chunk_overlap)

        chunks: List[str] = []
        current = ""
        for sec in sections:
            if len(current.split()) + len(sec.split()) <= chunk_size:
                current = (current + "\n\n" + sec).strip() if current else sec.strip()
            else:
                if current:
                    chunks.append(current)
                if len(sec.split()) > chunk_size:
                    chunks.extend(self._split_fixed_words(sec, chunk_size, chunk_overlap))
                    current = ""
                else:
                    current = sec.strip()
        if current:
            chunks.append(current)
        return chunks
