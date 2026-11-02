"""Document chunker and text splitting implementations."""

from enum import Enum
import os
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

    RECOMMENDED_ACADEMIC_STRATEGY: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER

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
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    @classmethod
    def create_academic_chunker(
        cls,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
    ) -> "DocumentChunker":
        """Factory method producing an academic document chunker with optimal recursive settings."""
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )

    def chunk_text(
        self,
        text: str,
        document_id: str,
        page_number: int = 1,
        section: Optional[str] = None,
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
        document_title = doc.metadata.get("document_title") or os.path.splitext(doc.filename)[0].replace("_", " ").title()
        author = doc.metadata.get("author")

        pages_to_process = doc.pages if doc.pages else []
        if not pages_to_process and doc.raw_text:
            pages_to_process = [type("Page", (), {"page_number": 1, "text": doc.raw_text, "metadata": {}})()]

        for page in pages_to_process:
            page_section = getattr(page, "metadata", {}).get("section")
            page_chunks = self.chunk_text(
                text=page.text,
                document_id=doc.id,
                page_number=page.page_number,
                section=page_section,
                strategy=strategy,
            )
            for chunk in page_chunks:
                chunk.chunk_index = global_chunk_idx
                chunk.document_title = document_title
                chunk.author = author
                chunk.metadata.update(
                    {
                        "filename": doc.filename,
                        "document_title": document_title,
                        "author": author,
                        "document_type": doc.document_type.value if hasattr(doc.document_type, "value") else str(doc.document_type),
                        "source_path": doc.source_path,
                        "page_number": page.page_number,
                        "chunk_id": chunk.id,
                        "chunk_index": global_chunk_idx,
                    }
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        return all_chunks

    @classmethod
    def compare_strategies(
        cls,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> Dict[str, Any]:
        """Compare fixed-size and recursive character chunking strategies on a given text."""
        # Fixed-word uses word counts (~100 words ~ 500 chars)
        fixed_chunker = cls(
            chunk_size=max(20, chunk_size // 5),
            chunk_overlap=max(5, chunk_overlap // 5),
            strategy=ChunkingStrategy.FIXED_WORD,
        )
        rec_chunker = cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
        )

        fixed_chunks = fixed_chunker.chunk_text(text, document_id="cmp_fixed")
        rec_chunks = rec_chunker.chunk_text(text, document_id="cmp_rec")

        def sentence_boundary_integrity(chunks: List[Chunk]) -> float:
            if not chunks:
                return 0.0
            clean_count = sum(
                1 for c in chunks
                if c.text.rstrip().endswith((".", "!", "?", '."', ".'", "\n")) or len(c.text) < 40
            )
            return round((clean_count / len(chunks)) * 100, 2)

        return {
            "fixed_size": {
                "strategy": ChunkingStrategy.FIXED_WORD.value,
                "total_chunks": len(fixed_chunks),
                "avg_chunk_chars": round(sum(len(c.text) for c in fixed_chunks) / max(1, len(fixed_chunks)), 1),
                "sentence_boundary_integrity": sentence_boundary_integrity(fixed_chunks),
                "citations_detected": sum(1 for c in fixed_chunks if c.metadata.get("has_citation")),
            },
            "recursive": {
                "strategy": ChunkingStrategy.RECURSIVE_CHARACTER.value,
                "total_chunks": len(rec_chunks),
                "avg_chunk_chars": round(sum(len(c.text) for c in rec_chunks) / max(1, len(rec_chunks)), 1),
                "sentence_boundary_integrity": sentence_boundary_integrity(rec_chunks),
                "citations_detected": sum(1 for c in rec_chunks if c.metadata.get("has_citation")),
            },
            "chosen_strategy": cls.RECOMMENDED_ACADEMIC_STRATEGY.value,
            "rationale": (
                "Recursive character splitting is selected as the optimal strategy for academic documents "
                "because it respects hierarchical structural boundaries (paragraphs, sentence terminations, "
                "and clauses). This minimizes semantic fragmentation, prevents slicing citations or math formulas "
                "across chunk boundaries, and preserves coherent context for LLM retrieval."
            ),
        }

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
                        overlap_tail = ""
                        if chunk_overlap > 0:
                            tail = current[-chunk_overlap:]
                            space_idx = tail.find(" ")
                            overlap_tail = tail[space_idx + 1:] if space_idx != -1 else tail
                        current = f"{overlap_tail} {s}".strip() if overlap_tail else s
                    else:
                        current = s

                    if len(current) > chunk_size and remaining_seps:
                        self._recursive_split_helper(current, chunk_size, chunk_overlap, remaining_seps, result)
                        current = ""
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
