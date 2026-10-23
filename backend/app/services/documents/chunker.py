"""Document chunker implementation preserving source and location metadata."""

import re
from typing import Any, Dict, List, Optional
from backend.app.models.chunk import Chunk


class DocumentChunker:
    """Splits document text into chunks preserving page, section, and character location metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        document_id: str,
        page_number: Optional[int] = 1,
        section: Optional[str] = "main",
        source: Optional[str] = None,
        char_offset_base: int = 0,
        url: Optional[str] = None,
        document_version: Optional[str] = None,
        start_chunk_index: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split text into overlapping chunks with precise metadata and character offsets."""
        if not text:
            return []

        # Find all word tokens with their precise character start and end spans
        word_spans = [m for m in re.finditer(r"\S+", text)]
        if not word_spans:
            return []

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = start_chunk_index
        src = source or document_id

        for i in range(0, len(word_spans), step):
            span_slice = word_spans[i : i + self.chunk_size]
            rel_char_start = span_slice[0].start()
            rel_char_end = span_slice[-1].end()

            chunk_text = text[rel_char_start:rel_char_end]
            abs_char_start = char_offset_base + rel_char_start
            abs_char_end = char_offset_base + rel_char_end

            meta: Dict[str, Any] = {
                "source": src,
                "chunk_index": chunk_idx,
                "char_start": abs_char_start,
                "char_end": abs_char_end,
                "page_number": page_number,
                "section": section,
                "url": url,
                "document_version": document_version,
                "document_id": document_id,
                "word_count": len(span_slice),
            }
            if extra_metadata:
                meta.update(extra_metadata)

            chunk = Chunk(
                document_id=document_id,
                chunk_index=chunk_idx,
                text=chunk_text,
                page_number=page_number if page_number is not None else 1,
                section=section,
                token_count=len(span_slice),
                source=src,
                char_start=abs_char_start,
                char_end=abs_char_end,
                metadata=meta,
            )
            chunks.append(chunk)
            chunk_idx += 1

            if i + self.chunk_size >= len(word_spans):
                break

        return chunks

    def chunk_document(
        self,
        document_id: str,
        pages_or_sections: List[Dict[str, Any]],
        source: Optional[str] = None,
        url: Optional[str] = None,
        document_version: Optional[str] = None,
        default_section: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Chunk a multi-page or multi-section document while preserving a continuous chunk index."""
        all_chunks: List[Chunk] = []
        current_chunk_idx = 0
        src = source or document_id

        for item in pages_or_sections:
            item_text = item.get("text", "")
            if not item_text or not item_text.strip():
                continue

            page_num = item.get("page_number")
            sec = item.get("section", default_section)
            char_base = item.get("char_offset_base", 0)
            item_url = item.get("url", url)
            item_ver = item.get("document_version", document_version)

            combined_extra = dict(extra_metadata or {})
            if "extra_metadata" in item:
                combined_extra.update(item["extra_metadata"])

            item_chunks = self.chunk_text(
                text=item_text,
                document_id=document_id,
                page_number=page_num,
                section=sec,
                source=src,
                char_offset_base=char_base,
                url=item_url,
                document_version=item_ver,
                start_chunk_index=current_chunk_idx,
                extra_metadata=combined_extra,
            )
            all_chunks.extend(item_chunks)
            current_chunk_idx += len(item_chunks)

        return all_chunks

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, List[Chunk]]:
        """Chunk multiple documents, keeping their source identities and indices strictly isolated."""
        result: Dict[str, List[Chunk]] = {}
        for doc in documents:
            doc_id = doc["document_id"]
            src = doc.get("source", doc_id)
            url = doc.get("url")
            version = doc.get("document_version")
            pages_or_sections = doc.get("content_items", [])

            if pages_or_sections:
                chunks = self.chunk_document(
                    document_id=doc_id,
                    pages_or_sections=pages_or_sections,
                    source=src,
                    url=url,
                    document_version=version,
                    extra_metadata=doc.get("metadata"),
                )
            else:
                raw_text = doc.get("text", "")
                chunks = self.chunk_text(
                    text=raw_text,
                    document_id=doc_id,
                    page_number=doc.get("page_number", 1),
                    section=doc.get("section", "main"),
                    source=src,
                    url=url,
                    document_version=version,
                    extra_metadata=doc.get("metadata"),
                )
            result[doc_id] = chunks
        return result
