"""Unit tests for prompt construction and system/user role separation."""

import pytest
from backend.app.models.conversation import Message, MessageRole
from backend.app.services.llm.prompts import (
    PromptBuilder,
    PromptBundle,
    PromptMessage,
    PromptRole,
    RESEARCHMATE_SYSTEM_PROMPT,
    build_academic_rag_prompt,
    build_rag_user_prompt,
)


def test_researchmate_system_prompt_directives():
    """Verify system prompt contains all required academic research directives."""
    prompt = RESEARCHMATE_SYSTEM_PROMPT

    # 1. Grounding using supplied context
    assert "solely using the supplied academic context" in prompt
    assert "No unsupported answer. No hidden source." in prompt

    # 2. Avoid unsupported claims
    assert "Do NOT extrapolate, speculate, or bring in external knowledge" in prompt
    assert "Never claim unsupported information as fact" in prompt

    # 3. Insufficient evidence indication
    assert "couldn't find enough supporting information" in prompt
    assert "Do NOT attempt to fabricate or guess" in prompt

    # 4. Citation preservation
    assert "citation linking directly to its source" in prompt
    assert "[Doc: <Title>, Page: <Page>]" in prompt

    # 5. Conciseness
    assert "concise" in prompt.lower()
    assert "Avoid generic pleasantries" in prompt

    # 6. Untrusted data boundary
    assert "passive reference data" in prompt


def test_prompt_builder_role_separation():
    """Verify strict separation between system, context, and user messages."""
    builder = PromptBuilder()
    builder.add_context_message("Federated learning protects data privacy.")
    builder.add_user_message("What is federated learning?")
    bundle: PromptBundle = builder.build()

    messages = bundle.messages
    assert len(messages) == 3

    # System message
    assert messages[0].role == PromptRole.SYSTEM
    assert messages[0].content == RESEARCHMATE_SYSTEM_PROMPT

    # Context message
    assert messages[1].role == PromptRole.CONTEXT
    assert "BEGIN RETRIEVED ACADEMIC CONTEXT" in messages[1].content
    assert "Federated learning protects data privacy." in messages[1].content

    # User message
    assert messages[2].role == PromptRole.USER
    assert "Student Question: What is federated learning?" in messages[2].content
    assert "Answer using only the supplied academic context" in messages[2].content


def test_prompt_builder_with_conversation_history():
    """Verify multi-turn conversation history is placed properly before the current query."""
    history = [
        Message(conversation_id="conv-1", role=MessageRole.USER, content="What is distributed learning?"),
        Message(conversation_id="conv-1", role=MessageRole.ASSISTANT, content="It is training across nodes."),
    ]

    bundle = build_academic_rag_prompt(
        query="How does it compare to federated learning?",
        context="Federated learning emphasizes data privacy on edge devices.",
        history=history,
    )

    api_messages = bundle.to_api_messages()
    assert len(api_messages) == 5

    # Check role sequence
    assert api_messages[0]["role"] == "system"
    assert api_messages[1]["role"] == "user"  # context mapped to user data block
    assert api_messages[2]["role"] == "user"  # prior query
    assert api_messages[3]["role"] == "assistant"  # prior answer
    assert api_messages[4]["role"] == "user"  # current query


def test_prompt_bundle_serialization():
    """Verify conversion to API dictionaries and unified string representation."""
    builder = PromptBuilder()
    builder.add_context_message("Sample context.")
    builder.add_user_message("Sample query?")
    bundle = builder.build()

    api_messages = bundle.to_api_messages()
    for item in api_messages:
        assert "role" in item
        assert "content" in item
        assert isinstance(item["role"], str)
        assert isinstance(item["content"], str)

    rendered = bundle.render_full_prompt()
    assert "[SYSTEM]" in rendered
    assert "[CONTEXT]" in rendered
    assert "[USER]" in rendered


def test_prompt_builder_custom_system_prompt():
    """Verify custom system prompt override works as intended."""
    custom_sys = "Custom academic tutor prompt."
    builder = PromptBuilder(system_prompt=custom_sys)
    builder.add_user_message("Hello")
    bundle = builder.build()

    assert bundle.system_prompt == custom_sys
    assert bundle.messages[0].content == custom_sys


def test_build_rag_user_prompt_helper():
    """Verify string helper formats user prompt with context."""
    user_prompt = build_rag_user_prompt(
        query="What are neural networks?",
        context="Neural networks are computational models.",
    )
    assert "Neural networks are computational models." in user_prompt
    assert "Student Question: What are neural networks?" in user_prompt
