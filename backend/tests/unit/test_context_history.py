"""Unit tests for conversation context window and message history management."""

import pytest
from backend.app.models.conversation import Message, MessageRole
from backend.app.services.conversations.manager import ConversationContextManager
from backend.app.services.llm.prompts import PromptRole
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.services.retrieval.search import SearchResult


def test_conversation_turn_storage():
    """Verify storing and retrieving conversation turns."""
    manager = ConversationContextManager()
    conv = manager.create_conversation("Research Topic A")

    manager.add_message(conv.id, MessageRole.USER, "What is zero-shot learning?")
    manager.add_message(conv.id, MessageRole.ASSISTANT, "Zero-shot learning classifies unseen classes.")

    history = manager.get_history(conv.id)
    assert len(history) == 2
    assert history[0].role == MessageRole.USER
    assert history[0].content == "What is zero-shot learning?"
    assert history[1].role == MessageRole.ASSISTANT


def test_history_bounded_by_turn_limit():
    """Verify history truncation when turns exceed max_turns."""
    manager = ConversationContextManager(max_turns=2)
    conv = manager.create_conversation()

    # Add 4 turns (turn 1 to 4)
    for i in range(1, 5):
        manager.add_message(conv.id, MessageRole.USER, f"User Question {i}")
        manager.add_message(conv.id, MessageRole.ASSISTANT, f"Assistant Answer {i}")

    # Bounded history with max_turns=2 should keep the latest 2 user turns + their answers
    bounded = manager.get_bounded_history(conv.id, max_turns=2)
    user_questions = [m.content for m in bounded if m.role == MessageRole.USER]

    assert len(user_questions) == 2
    assert "User Question 4" in user_questions
    assert "User Question 3" in user_questions
    assert "User Question 1" not in user_questions


def test_history_bounded_by_token_limit():
    """Verify history truncation when messages exceed the token budget."""
    manager = ConversationContextManager()
    conv = manager.create_conversation()

    # Add messages with large content
    manager.add_message(conv.id, MessageRole.USER, "Ancient question " + "word " * 50)
    manager.add_message(conv.id, MessageRole.ASSISTANT, "Ancient answer " + "word " * 50)
    manager.add_message(conv.id, MessageRole.USER, "Recent question about transformers")
    manager.add_message(conv.id, MessageRole.ASSISTANT, "Transformers use self-attention")

    # Tight token budget (e.g. 50 tokens) should drop ancient messages and preserve the recent ones
    bounded = manager.get_bounded_history(conv.id, max_tokens=50)

    contents = [m.content for m in bounded]
    assert any("Recent question about transformers" in c for c in contents)
    assert not any("Ancient question" in c for c in contents)


def test_current_query_preservation():
    """Verify the current user question is always preserved in full, even under extreme budget constraints."""
    manager = ConversationContextManager(max_context_window_tokens=200)
    conv = manager.create_conversation()

    for i in range(5):
        manager.add_message(conv.id, MessageRole.USER, f"Historical question {i} with long description.")
        manager.add_message(conv.id, MessageRole.ASSISTANT, f"Historical answer {i} with long explanation.")

    current_query = "What is federated averaging?"
    retrieved_context = "Federated averaging aggregates local model weights at the central server."

    bundle = manager.assemble_prompt_bundle(
        current_query=current_query,
        retrieved_context=retrieved_context,
        conv_id=conv.id,
    )

    # Current query MUST be preserved verbatim in bundle and user message
    assert bundle.user_query == current_query
    user_msg = next((m for m in bundle.messages if m.role == PromptRole.USER), None)
    assert user_msg is not None
    assert current_query in user_msg.content


def test_followup_query_rewriting():
    """Verify pronoun-heavy follow-up questions are rewritten with conversational context."""
    manager = ConversationContextManager()
    conv = manager.create_conversation()

    manager.add_message(conv.id, MessageRole.USER, "What is federated learning?")
    manager.add_message(conv.id, MessageRole.ASSISTANT, "It is decentralized privacy-preserving training.")

    rewritten = manager.rewrite_followup_query(conv.id, "What are its main limitations?")
    assert "federated learning" in rewritten.lower()
    assert "limitations" in rewritten.lower()


@pytest.mark.asyncio
async def test_rag_pipeline_multi_turn_history():
    """Verify RAGPipeline records turns and maintains multi-turn context."""
    manager = ConversationContextManager()
    pipeline = RAGPipeline(conversation_manager=manager)
    conv = manager.create_conversation()

    candidate = SearchResult(
        chunk_id="c1",
        document_id="d1",
        document_title="Federated Learning Survey",
        text="Federated learning enables decentralized training.",
        page_number=1,
        score=0.9,
    )

    # Turn 1
    resp1 = await pipeline.answer(
        query="What is federated learning?",
        candidates=[candidate],
        conversation_id=conv.id,
    )
    assert resp1.grounded is True

    # History should now contain Turn 1 (user and assistant)
    history_after_turn1 = manager.get_history(conv.id)
    assert len(history_after_turn1) == 2

    # Turn 2 (Follow-up)
    resp2 = await pipeline.answer(
        query="Does it preserve data privacy?",
        candidates=[candidate],
        conversation_id=conv.id,
    )
    assert resp2.grounded is True

    history_after_turn2 = manager.get_history(conv.id)
    assert len(history_after_turn2) == 4
    assert history_after_turn2[2].content == "Does it preserve data privacy?"
