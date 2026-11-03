"""Documents API router providing PDF upload, chunking, and metadata retrieval."""

import hashlib
import os
from typing import List
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document, LoadedDocument, ProcessingStatus
from backend.app.repositories.chunks import ChunkRepository
from backend.app.repositories.documents import DocumentRepository
from backend.app.schemas.document import (
    ChunkResponse,
    DocumentInfo,
    DocumentUploadResponse,
    StrategyComparisonResponse,
)
from backend.app.services.documents.chunker import ChunkingStrategy, DocumentChunker
from backend.app.services.documents.cleaner import TextCleaner
from backend.app.services.documents.exceptions import (
    CorruptedDocumentError,
    DocumentIntakeError,
    EmptyDocumentError,
)
from backend.app.services.documents.loader import DocumentLoader
from backend.app.services.embeddings.service import (
    EmbeddingServiceError,
    get_embedding_service,
    validate_embedding,
)
from backend.app.services.retrieval.search import get_vector_store

router = APIRouter(tags=["Documents"])

# In-memory document and chunk repository instances
document_repository = DocumentRepository()
chunk_repository = ChunkRepository()


@router.get("/documents", response_model=List[DocumentInfo])
def list_documents() -> List[DocumentInfo]:
    """List all indexed documents."""
    docs = document_repository.list_all()
    return [
        DocumentInfo(
            id=d.id,
            title=d.title,
            filename=d.filename,
            document_type=d.document_type,
            author=d.author,
            course=d.course,
            page_count=d.page_count,
            processing_status=d.processing_status.value if hasattr(d.processing_status, "value") else str(d.processing_status),
            uploaded_at=d.uploaded_at,
        )
        for d in docs
    ]


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="Academic PDF or document file to process and chunk"),
    chunk_size: int = Query(default=settings.CHUNK_SIZE, ge=1, description="Maximum chunk size in tokens"),
    chunk_overlap: int = Query(default=settings.CHUNK_OVERLAP, ge=0, description="Chunk overlap in tokens"),
    strategy: ChunkingStrategy = Query(
        default=ChunkingStrategy.RECURSIVE_CHARACTER,
        description="Chunking strategy (recursive_character recommended for academic PDFs)",
    ),
) -> DocumentUploadResponse:
    """Extract, validate, and chunk one uploaded academic document."""
    return await _process_upload(
        file=file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )


@router.post("/documents/upload/batch", response_model=List[DocumentUploadResponse])
async def upload_documents(
    files: List[UploadFile] = File(..., description="Academic documents to process independently"),
    chunk_size: int = Query(default=settings.CHUNK_SIZE, ge=1, description="Maximum chunk size in tokens"),
    chunk_overlap: int = Query(default=settings.CHUNK_OVERLAP, ge=0, description="Chunk overlap in tokens"),
    strategy: ChunkingStrategy = Query(default=ChunkingStrategy.RECURSIVE_CHARACTER),
) -> List[DocumentUploadResponse]:
    """Process multiple documents independently so one invalid file does not stop the batch."""
    results = []
    for file in files:
        try:
            results.append(await _process_upload(file, chunk_size, chunk_overlap, strategy))
        except HTTPException as exc:
            results.append(
                DocumentUploadResponse(
                    document_id="",
                    filename=file.filename or "",
                    status=ProcessingStatus.FAILED.value,
                    chunks=0,
                    message=str(exc.detail),
                )
            )
    return results


async def _process_upload(
    file: UploadFile,
    chunk_size: int,
    chunk_overlap: int,
    strategy: ChunkingStrategy,
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must have a valid filename.")

    actual_chunk_size = chunk_size if isinstance(chunk_size, int) else settings.CHUNK_SIZE
    actual_chunk_overlap = chunk_overlap if isinstance(chunk_overlap, int) else settings.CHUNK_OVERLAP
    actual_strategy = strategy if isinstance(strategy, ChunkingStrategy) else ChunkingStrategy.RECURSIVE_CHARACTER
    if actual_chunk_size > settings.MAX_CHUNK_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_size cannot exceed {settings.MAX_CHUNK_TOKENS} tokens.",
        )
    if actual_chunk_overlap >= actual_chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if ext not in ["pdf", "txt", "docx", "md", "markdown"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Supported formats: pdf, txt, docx, md.",
        )

    content = file.file.read() if hasattr(file, "file") and hasattr(file.file, "read") else file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    doc_record = Document(
        title=os.path.splitext(file.filename)[0].replace("_", " ").title(),
        filename=file.filename,
        document_type=ext,
        file_hash=file_hash,
        processing_status=ProcessingStatus.PENDING,
    )
    doc_record, is_new = document_repository.reserve(doc_record)
    if not is_new:
        existing_chunks = chunk_repository.get_by_document(doc_record.id)
        return DocumentUploadResponse(
            document_id=doc_record.id,
            filename=doc_record.filename,
            status=doc_record.processing_status.value,
            chunks=len(existing_chunks),
            message=f"Document already registered with status '{doc_record.processing_status.value}'.",
            duplicate=True,
        )

    doc_record.processing_status = ProcessingStatus.PROCESSING
    document_repository.save(doc_record)
    try:
        if not content.strip():
            raise EmptyDocumentError(file.filename)
        loaded_doc = DocumentLoader.load_bytes(data=content, filename=file.filename)
        loaded_doc.id = doc_record.id
        for page in loaded_doc.pages:
            page.text = TextCleaner.clean(page.text)

        chunker = DocumentChunker(
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            strategy=actual_strategy,
            max_chunk_tokens=settings.MAX_CHUNK_TOKENS,
        )
        chunks = chunker.chunk_document(loaded_doc, strategy=actual_strategy)
        _validate_ingested_chunks(loaded_doc, chunks)

        doc_record.title = loaded_doc.metadata.get("document_title") or doc_record.title
        doc_record.document_type = loaded_doc.document_type.value
        doc_record.author = loaded_doc.metadata.get("author")
        doc_record.page_count = loaded_doc.page_count
        embedding_service = get_embedding_service()
        vectors = await embedding_service.embed_batch([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match the number of document chunks.")
        for vector in vectors:
            validate_embedding(vector, embedding_service.dimension)

        get_vector_store().add_chunks(chunks, vectors, document_title=doc_record.title)
        chunk_repository.save_many(chunks)
        doc_record.processing_status = ProcessingStatus.COMPLETED
        document_repository.save(doc_record)
        return DocumentUploadResponse(
            document_id=doc_record.id,
            filename=file.filename,
            status=doc_record.processing_status.value,
            chunks=len(chunks),
            message=f"Processed {len(chunks)} chunks from {loaded_doc.page_count} pages.",
        )
    except EmbeddingServiceError:
        doc_record.processing_status = ProcessingStatus.FAILED
        document_repository.save(doc_record)
        return DocumentUploadResponse(
            document_id=doc_record.id,
            filename=file.filename,
            status=doc_record.processing_status.value,
            chunks=0,
            message="Embedding provider request failed.",
        )
    except (EmptyDocumentError, CorruptedDocumentError, DocumentIntakeError, ValueError) as exc:
        doc_record.processing_status = ProcessingStatus.FAILED
        document_repository.save(doc_record)
        return DocumentUploadResponse(
            document_id=doc_record.id,
            filename=file.filename,
            status=doc_record.processing_status.value,
            chunks=0,
            message=str(exc),
        )
    except Exception as exc:
        logger.exception("Document processing failed for '%s'.", file.filename)
        doc_record.processing_status = ProcessingStatus.FAILED
        document_repository.save(doc_record)
        return DocumentUploadResponse(
            document_id=doc_record.id,
            filename=file.filename,
            status=doc_record.processing_status.value,
            chunks=0,
            message=f"Document processing failed: {exc}",
        )


def _validate_ingested_chunks(loaded_doc: LoadedDocument, chunks: List[Chunk]) -> None:
    """Reject missing pages, invalid provenance, or chunks that lost source text."""
    if loaded_doc.page_count != len(loaded_doc.pages):
        raise ValueError("Extracted page count does not match the document page list.")
    if sorted(page.page_number for page in loaded_doc.pages) != list(range(1, loaded_doc.page_count + 1)):
        raise ValueError("Extracted page numbers are missing, duplicated, or out of sequence.")
    expected_pages = {page.page_number for page in loaded_doc.pages if page.text.strip()}
    actual_pages = {chunk.page_number for chunk in chunks}
    if actual_pages != expected_pages:
        raise ValueError("Chunk page coverage does not match non-empty extracted pages.")
    if not chunks or any(not chunk.text.strip() for chunk in chunks):
        raise ValueError("Document produced no complete, non-empty chunks.")
    if sorted(chunk.chunk_index for chunk in chunks) != list(range(len(chunks))):
        raise ValueError("Chunk indexes are missing or duplicated.")
    if len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk identifiers are not unique.")

    for page in loaded_doc.pages:
        page_chunks = sorted(
            (chunk for chunk in chunks if chunk.page_number == page.page_number),
            key=lambda chunk: chunk.chunk_index,
        )
        if not page.text.strip():
            continue
        if any(chunk.document_id != loaded_doc.id for chunk in page_chunks):
            raise ValueError("A chunk references the wrong source document.")
        extracted = iter("".join(chunk.text for chunk in page_chunks))
        if not all(character.isspace() or any(found == character for found in extracted) for character in page.text):
            raise ValueError(f"Chunks do not preserve all extracted text on page {page.page_number}.")


@router.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str) -> DocumentInfo:
    """Retrieve metadata for a specific document."""
    doc = document_repository.get_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with id '{document_id}' not found.",
        )
    return DocumentInfo(
        id=doc.id,
        title=doc.title,
        filename=doc.filename,
        document_type=doc.document_type,
        author=doc.author,
        course=doc.course,
        page_count=doc.page_count,
        processing_status=doc.processing_status.value if hasattr(doc.processing_status, "value") else str(doc.processing_status),
        uploaded_at=doc.uploaded_at,
    )


@router.get("/documents/{document_id}/chunks", response_model=List[ChunkResponse])
def get_document_chunks(document_id: str) -> List[ChunkResponse]:
    """Retrieve all extracted chunks for a specific document."""
    chunks = chunk_repository.get_by_document(document_id)
    if not chunks:
        # Check if document exists
        if not document_repository.get_by_id(document_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with id '{document_id}' not found.",
            )
    return [
        ChunkResponse(
            id=c.id,
            document_id=c.document_id,
            document_title=c.document_title,
            author=c.author,
            chunk_index=c.chunk_index,
            text=c.text,
            page_number=c.page_number,
            section=c.section,
            token_count=c.token_count,
            metadata=c.metadata,
        )
        for c in chunks
    ]


@router.post("/documents/compare-strategies", response_model=StrategyComparisonResponse)
def compare_chunking_strategies(
    file: UploadFile = File(..., description="Sample academic PDF to benchmark chunking strategies"),
    chunk_size: int = Query(default=settings.CHUNK_SIZE, ge=1, description="Maximum chunk size in tokens"),
    chunk_overlap: int = Query(default=settings.CHUNK_OVERLAP, ge=0, description="Chunk overlap in tokens"),
) -> StrategyComparisonResponse:
    """Compare fixed-size and recursive text chunking on an uploaded academic document."""
    actual_chunk_size = chunk_size if isinstance(chunk_size, int) else settings.CHUNK_SIZE
    actual_chunk_overlap = chunk_overlap if isinstance(chunk_overlap, int) else settings.CHUNK_OVERLAP
    if actual_chunk_size > settings.MAX_CHUNK_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"chunk_size cannot exceed {settings.MAX_CHUNK_TOKENS} tokens.",
        )
    if actual_chunk_overlap >= actual_chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    if hasattr(file, "file") and hasattr(file.file, "read"):
        content = file.file.read()
    else:
        content = file.read()

    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")

    loaded_doc = DocumentLoader.load_bytes(data=content, filename=file.filename)
    comparison = DocumentChunker.compare_strategies(
        text=loaded_doc.raw_text,
        chunk_size=actual_chunk_size,
        chunk_overlap=actual_chunk_overlap,
    )
    return StrategyComparisonResponse(**comparison)
