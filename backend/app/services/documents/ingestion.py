"""End-to-end document ingestion, chunking, embedding, and storage pipeline."""

import uuid
from typing import Any, Dict, List, Optional
from backend.app.models.chunk import Chunk
from backend.app.repositories.chunks import ChunkRepository
from backend.app.services.documents.chunker import DocumentChunker
from backend.app.services.documents.extractor import TextExtractor
from backend.app.services.documents.loader import DocumentLoader
from backend.app.services.documents.cleaner import TextCleaner
from backend.app.services.embeddings.service import EmbeddingServiceInterface, get_embedding_service


class DocumentIngestionPipeline:
    """Orchestrates document ingestion: extract -> clean -> chunk -> attach metadata -> embed -> store."""

    def __init__(
        self,
        chunker: Optional[DocumentChunker] = None,
        embedding_service: Optional[EmbeddingServiceInterface] = None,
        chunk_repository: Optional[ChunkRepository] = None,
    ):
        self.chunker = chunker or DocumentChunker()
        self.embedding_service = embedding_service or get_embedding_service()
        self.chunk_repository = chunk_repository or ChunkRepository()

    async def ingest_document(
        self,
        text: Optional[str] = None,
        file_path: Optional[str] = None,
        document_id: Optional[str] = None,
        source: Optional[str] = None,
        url: Optional[str] = None,
        document_version: Optional[str] = None,
        document_type: str = "text",
        title: Optional[str] = None,
        clean_text: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Ingest document through the complete RAG pipeline preserving source metadata."""
        # 1. Ingestion / Loading
        doc_source = source
        doc_url = url
        doc_ver = document_version
        raw_text = text

        if file_path:
            loaded = DocumentLoader.load_document(
                file_path=file_path,
                source=source,
                url=url,
                document_version=document_version,
                document_type=document_type,
            )
            raw_text = loaded["text"]
            doc_source = doc_source or loaded["source"]
            doc_url = doc_url or loaded["url"]
            doc_ver = doc_ver or loaded["document_version"]
            document_type = loaded["document_type"]

        if not raw_text:
            return []

        doc_id = document_id or str(uuid.uuid4())
        canonical_source = doc_source or title or doc_id

        # 2. Text Extraction
        if document_type in ("md", "markdown"):
            content_items = TextExtractor.extract_from_markdown(raw_text)
        elif "\f" in raw_text:
            content_items = TextExtractor.extract_from_text(raw_text)
        else:
            content_items = [{
                "text": raw_text,
                "page_number": 1,
                "section": "main",
                "char_offset_base": 0,
            }]

        # Optional cleaning
        if clean_text:
            for item in content_items:
                item["text"] = TextCleaner.clean(item["text"])

        # 3. Chunking & Metadata Attachment
        doc_extra = dict(extra_metadata or {})
        if title:
            doc_extra["document_title"] = title
        doc_extra["document_type"] = document_type

        chunks = self.chunker.chunk_document(
            document_id=doc_id,
            pages_or_sections=content_items,
            source=canonical_source,
            url=doc_url,
            document_version=doc_ver,
            extra_metadata=doc_extra,
        )
        if not chunks:
            return []

        # 4. Embedding Generation (Embed text only, do NOT embed metadata)
        texts_to_embed = [c.text for c in chunks]
        embeddings = await self.embedding_service.embed_batch(texts_to_embed)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            chunk.embedding_id = str(uuid.uuid4())

        # 5. Storage (Vector Database / Chunk Repository)
        self.chunk_repository.save_many(chunks)

        return chunks
