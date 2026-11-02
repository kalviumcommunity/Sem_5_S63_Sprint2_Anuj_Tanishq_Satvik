"""Document chunker and text splitting implementations."""

from enum import Enum
import os
import re
from typing import Any, Dict, List, Optional
from backend.app.models.chunk import Chunk
from backend.app.models.document import LoadedDocument
from backend.app.core.config import settings


class ChunkingStrategy(str, Enum):
    """Supported text splitting strategies for academic documents."""

    FIXED_WORD = "fixed_word"
    RECURSIVE_CHARACTER = "recursive_character"
    MARKDOWN_HEADER = "markdown_header"


class DocumentChunker:
    """Split documents within token budgets while retaining source-page provenance."""

    RECOMMENDED_ACADEMIC_STRATEGY: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
        max_chunk_tokens: Optional[int] = None,
    ):
        max_chunk_tokens = max_chunk_tokens if max_chunk_tokens is not None else settings.MAX_CHUNK_TOKENS
        if max_chunk_tokens <= 0:
            raise ValueError(f"max_chunk_tokens must be positive, got {max_chunk_tokens}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_size > max_chunk_tokens:
            raise ValueError(
                f"chunk_size ({chunk_size}) cannot exceed max_chunk_tokens ({max_chunk_tokens})"
            )
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_tokens = max_chunk_tokens
        self.strategy = strategy

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
        """Split text into token-limited chunks with configurable overlap and metadata."""
        if not text or not text.strip():
            return []

        active_strategy = strategy or self.strategy
        chunks_text = self._split_token_aware(
            text,
            self.chunk_size,
            self.chunk_overlap,
            sentence_aware=active_strategy != ChunkingStrategy.FIXED_WORD,
        )

        chunks: List[Chunk] = []
        for idx, c_text in enumerate(chunks_text):
            cleaned = c_text.strip()
            if not cleaned:
                continue

            words = cleaned.split()
            token_count = self._token_count(cleaned)
            has_citation = bool(re.search(r"\[\d+\]|\([A-Z][a-zA-Z]+ et al\., \d{4}\)|\([A-Z][a-zA-Z]+, \d{4}\)", cleaned))
            has_math = bool(re.search(r"\$[^$]+\$|\\[a-zA-Z]+|=", cleaned))

            chunk = Chunk(
                document_id=document_id,
                chunk_index=idx,
                text=cleaned,
                page_number=page_number,
                section=section,
                token_count=token_count,
                metadata={
                    "page_number": page_number,
                    "section": section,
                    "token_count": token_count,
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
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ) -> Dict[str, Any]:
        """Compare word-boundary and sentence-aware token chunking on a given text."""
        fixed_chunker = cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
                "avg_chunk_tokens": round(sum(c.token_count for c in fixed_chunks) / max(1, len(fixed_chunks)), 1),
                "sentence_boundary_integrity": sentence_boundary_integrity(fixed_chunks),
                "citations_detected": sum(1 for c in fixed_chunks if c.metadata.get("has_citation")),
            },
            "recursive": {
                "strategy": ChunkingStrategy.RECURSIVE_CHARACTER.value,
                "total_chunks": len(rec_chunks),
                "avg_chunk_chars": round(sum(len(c.text) for c in rec_chunks) / max(1, len(rec_chunks)), 1),
                "avg_chunk_tokens": round(sum(c.token_count for c in rec_chunks) / max(1, len(rec_chunks)), 1),
                "sentence_boundary_integrity": sentence_boundary_integrity(rec_chunks),
                "citations_detected": sum(1 for c in rec_chunks if c.metadata.get("has_citation")),
            },
            "chosen_strategy": cls.RECOMMENDED_ACADEMIC_STRATEGY.value,
            "rationale": (
                "The recursive strategy measures chunk limits in tokens and prefers paragraph and sentence "
                "boundaries; overlap retains nearby context between chunks."
            ),
        }

    @staticmethod
    def _join_units(units: List[tuple[str, str]]) -> str:
        return "".join(
            (separator if index else "") + text
            for index, (text, separator) in enumerate(units)
        ).strip()

    @staticmethod
    def _token_count(text: str) -> int:
        from backend.app.services.tokenizer import default_tokenizer

        return default_tokenizer.count_tokens(text)

    def _split_token_aware(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        sentence_aware: bool,
    ) -> List[str]:
        units: List[tuple[str, str]] = []
        paragraphs = re.split(r"(?m)\n\s*\n|(?=^#{1,4}\s+)", text.strip())
        for paragraph_index, paragraph in enumerate(paragraphs):
            if sentence_aware:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
            else:
                sentences = paragraph.split()
            for sentence_index, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                separator = (
                    "\n\n"
                    if paragraph_index and sentence_index == 0
                    else " " if sentence_index else ""
                )
                if self._token_count(sentence) <= chunk_size:
                    units.append((sentence, separator))
                    continue

                current = ""
                for word in sentence.split():
                    candidate = f"{current} {word}".strip()
                    if self._token_count(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current:
                            units.append((current, separator))
                            separator = " "
                        if self._token_count(word) <= chunk_size:
                            current = word
                        else:
                            fragment = ""
                            for character in word:
                                candidate = fragment + character
                                if self._token_count(candidate) > chunk_size and fragment:
                                    units.append((fragment, separator))
                                    separator = ""
                                    fragment = character
                                else:
                                    fragment = candidate
                            current = fragment
                if current:
                    units.append((current, separator))
        if not units:
            return []

        chunks: List[str] = []
        current_units: List[tuple[str, str]] = []
        for unit in units:
            candidate_units = current_units + [unit]
            if self._token_count(self._join_units(candidate_units)) <= chunk_size:
                current_units = candidate_units
                continue

            if current_units:
                chunks.append(self._join_units(current_units))
            overlap_budget = min(
                chunk_overlap,
                chunk_size - self._token_count(unit[0]),
            )
            current_units = self._overlap_units(current_units, overlap_budget)
            while (
                current_units
                and self._token_count(self._join_units(current_units + [unit])) > chunk_size
            ):
                current_units.pop(0)
            current_units.append(unit)

        if current_units:
            chunks.append(self._join_units(current_units))
        return chunks

    def _overlap_units(
        self,
        units: List[tuple[str, str]],
        chunk_overlap: int,
    ) -> List[tuple[str, str]]:
        if chunk_overlap == 0 or not units:
            return []

        overlap: List[tuple[str, str]] = []
        for text, separator in reversed(units):
            candidate = [(text, separator)] + overlap
            if self._token_count(self._join_units(candidate)) > chunk_overlap:
                break
            overlap = candidate
        if overlap:
            return overlap

        tail_words: List[str] = []
        for word in reversed(units[-1][0].split()):
            candidate = " ".join([word] + tail_words)
            if self._token_count(candidate) > chunk_overlap:
                break
            tail_words.insert(0, word)
        return [(" ".join(tail_words), " ")] if tail_words else []
