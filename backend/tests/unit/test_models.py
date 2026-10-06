"""Unit tests for data models."""

from backend.app.models.document import Document, ProcessingStatus
from backend.app.models.chunk import Chunk
from backend.app.models.citation import Citation
from backend.app.models.conversation import Conversation, Message, MessageRole


def test_document_model_creation():
    doc = Document(
        title="Deep Learning Survey",
        filename="survey.pdf",
        file_hash="abc123hash",
        page_count=12,
    )
    assert doc.id is not None
    assert doc.title == "Deep Learning Survey"
    assert doc.processing_status == ProcessingStatus.UPLOADED


def test_chunk_model_creation():
    chunk = Chunk(
        document_id="doc-1",
        chunk_index=0,
        text="Introduction to transformer models.",
        page_number=1,
        section="Introduction",
        token_count=4,
    )
    assert chunk.document_id == "doc-1"
    assert chunk.page_number == 1
    assert chunk.token_count == 4


def test_citation_model_creation():
    citation = Citation(
        document_id="doc-1",
        chunk_id="chunk-1",
        page_number=3,
        document_title="Attention is All You Need",
        relevance_score=0.95,
        citation_text="Self-attention mechanisms calculate dependencies.",
    )
    assert citation.document_id == "doc-1"
    assert citation.page_number == 3
    assert citation.relevance_score == 0.95


def test_conversation_and_message_model():
    conv = Conversation(title="Transformers Inquiry")
    msg = Message(
        conversation_id=conv.id,
        role=MessageRole.USER,
        content="What is self-attention?",
    )
    assert conv.id == msg.conversation_id
    assert msg.role == MessageRole.USER
