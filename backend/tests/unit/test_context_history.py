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


def test_empty_history_handling():
    """Verify empty histories operate cleanly across all manager operations."""
    manager = ConversationContextManager()
    conv = manager.create_conversation("Empty Topic")

    assert manager.get_history(conv.id) == []
    assert manager.get_bounded_history(conv.id) == []
    assert manager.get_conversation_token_count(conv.id) == 0

    # Non-existent or None conversation ID
    assert manager.get_history(None) == []
    assert manager.get_history("non-existent-conv-id") == []
    assert manager.get_bounded_history(None) == []

    # Assemble bundle with empty history
    bundle = manager.assemble_prompt_bundle(
        current_query="What is reinforcement learning?",
        retrieved_context="RL optimizes cumulative rewards.",
        conv_id=conv.id,
    )
    assert bundle.user_query == "What is reinforcement learning?"
    # No intermediate history messages should be present
    history_in_bundle = [m for m in bundle.messages if m.role == PromptRole.ASSISTANT]
    assert history_in_bundle == []


def test_long_history_turn_and_token_pruning():
    """Retained history keeps all turns while bounded history strictly retains the newest turns."""
    manager = ConversationContextManager(max_turns=3)
    conv = manager.create_conversation("Long Debate")

    # Add 15 user-assistant turns (30 messages total)
    for i in range(1, 16):
        manager.add_message(conv.id, MessageRole.USER, f"Question {i}")
        manager.add_message(conv.id, MessageRole.ASSISTANT, f"Answer {i}")

    # Full history retains all 30 messages
    full_history = manager.get_history(conv.id)
    assert len(full_history) == 30

    # Bounded history retains exactly the latest 3 user turns + their answers (6 messages)
    bounded = manager.get_bounded_history(conv.id, max_turns=3)
    assert len(bounded) == 6

    # Chronological order must be preserved among retained messages
    user_contents = [m.content for m in bounded if m.role == MessageRole.USER]
    assert user_contents == ["Question 13", "Question 14", "Question 15"]


def test_context_budget_plan_reserves_output_tokens():
    """Budget plan explicitly reserves expected output tokens and allocates remaining space."""
    manager = ConversationContextManager(
        max_context_window_tokens=4000,
        expected_output_tokens=600,
    )
    conv = manager.create_conversation("Budget Topic")

    plan = manager.plan_budget(
        current_query="Explain multi-head attention in transformer networks.",
        retrieved_context="Multi-head attention projects queries, keys, and values into subspaces.",
        conv_id=conv.id,
    )

    assert plan.total_context_limit == 4000
    assert plan.expected_output_reserve == 600
    assert plan.system_tokens > 0
    assert plan.query_tokens > 0
    # Mandatory tokens = system + query + output reserve
    mandatory = plan.system_tokens + plan.query_tokens + plan.expected_output_reserve
    assert plan.available_content_budget == 4000 - mandatory
    # Context budget is 65% of available
    assert plan.context_budget == int(plan.available_content_budget * 0.65)
    # History budget is 35% of available
    assert plan.history_budget == int(plan.available_content_budget * 0.35)


def test_oversized_query_policy_error_and_truncate():
    """Test oversized query policies: error and truncate."""
    from backend.app.services.conversations.exceptions import OversizedQueryError

    # 1. Error policy
    error_manager = ConversationContextManager(
        oversized_query_policy="error",
        max_context_window_tokens=2000,
    )
    error_manager.config.max_query_tokens = 20

    long_query = "What is the detailed mathematical derivation of backpropagation " * 5
    with pytest.raises(OversizedQueryError) as exc_info:
        error_manager.plan_budget(current_query=long_query)
    assert "Query size" in str(exc_info.value)

    # 2. Truncate policy
    trunc_manager = ConversationContextManager(
        oversized_query_policy="truncate",
        max_context_window_tokens=2000,
    )
    trunc_manager.config.max_query_tokens = 15

    bundle = trunc_manager.assemble_prompt_bundle(
        current_query=long_query,
        retrieved_context="Backpropagation computes gradients.",
    )
    assert "[... Truncated query ...]" in bundle.user_query


def test_oversized_context_policy_error_and_drop():
    """Test oversized context policies: error and drop."""
    from backend.app.services.conversations.exceptions import OversizedContextError

    # Context window of 300 tokens, 100 reserved for output => ~200 available content
    # 1. Error policy
    error_manager = ConversationContextManager(
        max_context_window_tokens=400,
        expected_output_tokens=100,
        oversized_context_policy="error",
    )
    huge_context = "Massive paper passage text detailing thousands of lines of experimental results. " * 30

    with pytest.raises(OversizedContextError):
        error_manager.assemble_prompt_bundle(
            current_query="What were the results?",
            retrieved_context=huge_context,
        )

    # 2. Drop policy
    drop_manager = ConversationContextManager(
        max_context_window_tokens=400,
        expected_output_tokens=100,
        oversized_context_policy="drop",
    )
    bundle = drop_manager.assemble_prompt_bundle(
        current_query="What were the results?",
        retrieved_context=huge_context,
    )
    assert not bundle.context_text


def test_conversation_isolation():
    """Different conversation sessions must remain strictly isolated."""
    manager = ConversationContextManager()
    conv_a = manager.create_conversation("Quantum Session")
    conv_b = manager.create_conversation("Biology Session")

    manager.add_message(conv_a.id, MessageRole.USER, "Explain Shor's algorithm.")
    manager.add_message(conv_a.id, MessageRole.ASSISTANT, "Shor's algorithm factors integers in polynomial time.")

    manager.add_message(conv_b.id, MessageRole.USER, "What is CRISPR-Cas9?")
    manager.add_message(conv_b.id, MessageRole.ASSISTANT, "CRISPR-Cas9 is a targeted genome editing tool.")

    # Isolation check: History of A does not contain B
    history_a = manager.get_history(conv_a.id)
    history_b = manager.get_history(conv_b.id)

    assert len(history_a) == 2
    assert len(history_b) == 2
    assert all("quantum" in m.content.lower() or "shor" in m.content.lower() for m in history_a)
    assert all("crispr" in m.content.lower() or "genome" in m.content.lower() for m in history_b)

    # Query rewriting uses only isolated history
    rewritten_b = manager.rewrite_followup_query(conv_b.id, "How does it cut DNA?")
    assert "crispr" in rewritten_b.lower()
    assert "shor" not in rewritten_b.lower()

    # Clearing history of A leaves B intact
    manager.clear_history(conv_a.id)
    assert manager.get_history(conv_a.id) == []
    assert len(manager.get_history(conv_b.id)) == 2

    # Deleting conversation B does not impact A
    assert manager.delete_conversation(conv_b.id) is True
    assert manager.get_conversation(conv_b.id) is None
    assert manager.get_conversation(conv_a.id) is not None


def test_message_model_validation():
    """Message model enforces non-empty conversation_id and valid roles."""
    from pydantic import ValidationError

    # Valid message
    msg = Message(conversation_id="conv-valid", role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.conversation_id == "conv-valid"

    # Empty conversation_id should fail
    with pytest.raises(ValidationError):
        Message(conversation_id="   ", role=MessageRole.USER, content="Hello")

    # None content should fail
    with pytest.raises(ValidationError):
        Message(conversation_id="conv-1", role=MessageRole.USER, content=None)

