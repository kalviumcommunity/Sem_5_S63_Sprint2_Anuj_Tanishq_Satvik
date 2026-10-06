"""Unit tests for core RAG modules."""

import pytest
from backend.app.services.documents.cleaner import TextCleaner
from backend.app.services.documents.chunker import DocumentChunker
from backend.app.services.llm.client import MockLLMClient
from backend.app.services.llm.parser import ResponseParser
from backend.app.services.llm.prompts import build_rag_user_prompt
from backend.app.services.embeddings.service import MockEmbeddingService
from backend.app.services.embeddings.batcher import chunk_list
from backend.app.services.retrieval.search import SearchResult, cosine_similarity
from backend.app.services.retrieval.reranker import Reranker
from backend.app.services.rag.context import ContextBuilder
from backend.app.services.rag.grounding import GroundingService
from backend.app.services.conversations.query_rewriter import QueryRewriter
from backend.app.models.conversation import Message, MessageRole


def test_text_cleaner():
    dirty_text = "This   is   a    sample\n\n\n\ntext   with extra   spaces."
    cleaned = TextCleaner.clean(dirty_text)
    assert "This is a sample" in cleaned
    assert "\n\n\n\n" not in cleaned


def test_document_chunker():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    sample_text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    chunks = chunker.chunk_text(sample_text, document_id="doc-test", page_number=2, section="Intro")
    assert len(chunks) >= 2
    assert chunks[0].document_id == "doc-test"
    assert chunks[0].page_number == 2
    assert chunks[0].section == "Intro"


@pytest.mark.asyncio
async def test_mock_llm_client_and_parser():
    client = MockLLMClient()
    response = await client.generate(prompt="What is federated learning?")
    assert len(response) > 0
    parsed = ResponseParser.parse(response)
    assert parsed.is_refusal is False
    assert len(parsed.answer) > 0


@pytest.mark.asyncio
async def test_mock_embeddings():
    service = MockEmbeddingService(dimension=64)
    vec1 = await service.embed_text("Federated learning")
    vec2 = await service.embed_text("Federated learning")
    assert len(vec1) == 64
    assert vec1 == vec2  # Deterministic

    sim = cosine_similarity(vec1, vec2)
    assert abs(sim - 1.0) < 1e-4


def test_embedding_batcher():
    items = list(range(10))
    batches = chunk_list(items, batch_size=3)
    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[-1] == [9]


def test_retrieval_reranker():
    cand1 = SearchResult(
        chunk_id="c1",
        document_id="d1",
        document_title="Federated Learning Overview",
        text="Federated learning enables decentralized model training.",
        page_number=1,
        score=0.7,
    )
    cand2 = SearchResult(
        chunk_id="c2",
        document_id="d2",
        document_title="Database Systems",
        text="Relational databases use B-trees for indexing tables.",
        page_number=4,
        score=0.8,
    )

    reranked = Reranker.rank(query="federated learning model training", candidates=[cand1, cand2], top_k=2)
    assert len(reranked) == 2
    # cand1 has higher overlap with federated learning query terms
    assert reranked[0].chunk_id == "c1"


def test_context_builder_and_grounding():
    cand = SearchResult(
        chunk_id="c1",
        document_id="d1",
        document_title="AI Research Paper",
        text="Reinforcement learning optimizes cumulative reward.",
        page_number=5,
        section="Methods",
        score=0.85,
    )
    context_str = ContextBuilder.build([cand])
    assert "AI Research Paper" in context_str
    assert "Page: 5" in context_str
    assert "Methods" in context_str

    grounding = GroundingService(threshold=0.60)
    assert grounding.is_sufficient([cand]) is True

    cand_low = SearchResult(
        chunk_id="c2",
        document_id="d2",
        document_title="Irrelevant",
        text="Cooking recipe",
        page_number=1,
        score=0.30,
    )
    assert grounding.is_sufficient([cand_low]) is False


def test_query_rewriter():
    history = [
        Message(conversation_id="conv-1", role=MessageRole.USER, content="What is federated learning?"),
        Message(conversation_id="conv-1", role=MessageRole.ASSISTANT, content="It is decentralized training."),
    ]
    rewritten = QueryRewriter.rewrite("What are its benefits?", history)
    assert "federated learning" in rewritten.lower()
