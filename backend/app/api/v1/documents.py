"""Documents API router providing PDF upload, chunking, and metadata retrieval."""

import os
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from backend.app.models.document import Document, ProcessingStatus
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
    UnsupportedFormatError,
)
from backend.app.services.documents.loader import DocumentLoader

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
def upload_document(
    file: UploadFile = File(..., description="Academic PDF or document file to process and chunk"),
    chunk_size: int = Query(default=600, ge=50, le=4000, description="Chunk size in characters"),
    chunk_overlap: int = Query(default=120, ge=0, le=1000, description="Chunk overlap in characters"),
    strategy: ChunkingStrategy = Query(
        default=ChunkingStrategy.RECURSIVE_CHARACTER,
        description="Chunking strategy (recursive_character recommended for academic PDFs)",
    ),
) -> DocumentUploadResponse:
    """Extract text from uploaded academic PDF/document, split into meaningful chunks, and index."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if ext not in ["pdf", "txt", "docx", "md", "markdown"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Supported formats: pdf, txt, docx, md.",
        )

    if hasattr(file, "file") and hasattr(file.file, "read"):
        content = file.file.read()
    else:
        content = file.read()

    if not content or len(content.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        loaded_doc = DocumentLoader.load_bytes(data=content, filename=file.filename)
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except (CorruptedDocumentError, UnsupportedFormatError, DocumentIntakeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {exc}",
        )

    # Handle direct function invocations where FastAPI Query object is default
    actual_chunk_size = chunk_size if isinstance(chunk_size, int) else 600
    actual_chunk_overlap = chunk_overlap if isinstance(chunk_overlap, int) else 120
    actual_strategy = strategy if isinstance(strategy, ChunkingStrategy) else ChunkingStrategy.RECURSIVE_CHARACTER

    # Clean text per page to remove artifacts while preserving structure
    for page in loaded_doc.pages:
        page.text = TextCleaner.clean(page.text)

    # Chunk the multi-page academic document
    chunker = DocumentChunker(
        chunk_size=actual_chunk_size,
        chunk_overlap=actual_chunk_overlap,
        strategy=actual_strategy,
    )
    chunks = chunker.chunk_document(loaded_doc, strategy=actual_strategy)

    # Save to repositories
    doc_record = Document(
        id=loaded_doc.id,
        title=os.path.splitext(file.filename)[0].replace("_", " ").title(),
        filename=file.filename,
        document_type=loaded_doc.document_type.value if hasattr(loaded_doc.document_type, "value") else str(loaded_doc.document_type),
        file_hash=loaded_doc.file_hash or "",
        page_count=loaded_doc.page_count,
        processing_status=ProcessingStatus.INDEXED,
    )
    document_repository.save(doc_record)
    chunk_repository.save_many(chunks)

    return DocumentUploadResponse(
        document_id=doc_record.id,
        filename=file.filename,
        status="processed",
        chunks=len(chunks),
        message=f"Successfully extracted and chunked {len(chunks)} sections from {file.filename} across {loaded_doc.page_count} pages.",
    )


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
    chunk_size: int = Query(default=600, ge=50, le=4000),
    chunk_overlap: int = Query(default=120, ge=0, le=1000),
) -> StrategyComparisonResponse:
    """Compare fixed-size and recursive text chunking on an uploaded academic document."""
    if hasattr(file, "file") and hasattr(file.file, "read"):
        content = file.file.read()
    else:
        content = file.read()

    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")

    actual_chunk_size = chunk_size if isinstance(chunk_size, int) else 600
    actual_chunk_overlap = chunk_overlap if isinstance(chunk_overlap, int) else 120

    loaded_doc = DocumentLoader.load_bytes(data=content, filename=file.filename)
    comparison = DocumentChunker.compare_strategies(
        text=loaded_doc.raw_text,
        chunk_size=actual_chunk_size,
        chunk_overlap=actual_chunk_overlap,
    )
    return StrategyComparisonResponse(**comparison)
