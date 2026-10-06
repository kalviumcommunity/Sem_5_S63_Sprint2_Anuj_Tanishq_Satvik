"""Unit tests for Concept #9: Prompt Templates & Reusable Prompt Design."""

import pytest

from backend.app.models.conversation import Message, MessageRole
from backend.app.services.conversations.query_rewriter import QueryRewriter
from backend.app.services.llm.exceptions import (
    MissingPromptVariableError,
    PromptTemplateNotFoundError,
)
from backend.app.services.llm.prompts import PromptRole
from backend.app.services.llm.templates import (
    ACADEMIC_QA_TEMPLATE,
    INSUFFICIENT_CONTEXT_REFUSAL_TEMPLATE,
    QUERY_REWRITING_TEMPLATE,
    SOURCE_GROUNDED_ANSWERING_TEMPLATE,
    SUMMARIZATION_TEMPLATE,
    PromptTemplate,
    PromptTemplateRegistry,
    prompt_registry,
)


def test_academic_qa_template_rendering():
    """Verify academic QA template renders with default and custom parameters."""
    sys_prompt, usr_prompt = ACADEMIC_QA_TEMPLATE.render({"question": "What is the P vs NP problem?"})

    assert "General Academic" in sys_prompt
    assert "Undergraduate" in sys_prompt
    assert "What is the P vs NP problem?" in usr_prompt
    assert "Academic Inquiry:" in usr_prompt

    # Test with overridden optional variables
    custom_sys, _ = ACADEMIC_QA_TEMPLATE.render({
        "question": "Explain quantum entanglement.",
        "discipline": "Theoretical Physics",
        "academic_level": "Postgraduate",
    })
    assert "Theoretical Physics" in custom_sys
    assert "Postgraduate" in custom_sys


def test_query_rewriting_template_rendering():
    """Verify query rewriting template resolves pronouns and conversation context."""
    history = "User: What is federated learning?\nAssistant: It is distributed machine learning."
    follow_up = "How does it protect user privacy?"

    sys_prompt, usr_prompt = QUERY_REWRITING_TEMPLATE.render({
        "conversation_history": history,
        "follow_up_question": follow_up,
        "max_keywords": 6,
    })

    assert "max 6 keywords" in sys_prompt
    assert "What is federated learning?" in usr_prompt
    assert "How does it protect user privacy?" in usr_prompt
    assert "Standalone Academic Search Query:" in usr_prompt


def test_summarization_template_rendering():
    """Verify summarization template interpolates document text and custom parameters."""
    doc_text = (
        "In this study, we evaluate transformer models on low-resource language translation tasks. "
        "Our empirical results demonstrate a 4.2 BLEU score improvement using back-translation."
    )

    sys_prompt, usr_prompt = SUMMARIZATION_TEMPLATE.render({
        "document_text": doc_text,
        "title": "Low-Resource Neural Machine Translation",
        "focus": "empirical findings and translation gains",
        "max_bullet_points": 3,
    })

    assert "empirical findings and translation gains" in sys_prompt
    assert "3 bullet points" in sys_prompt
    assert "Low-Resource Neural Machine Translation" in usr_prompt
    assert "BLEU score improvement" in usr_prompt


def test_source_grounded_answering_template_rendering():
    """Verify source-grounded answering template enforces context boundaries and citations."""
    context = "[Doc: Deep Learning, Page: 12] Convolutional networks process grid-structured topology."
    question = "How do convolutional neural networks process spatial data?"

    sys_prompt, usr_prompt = SOURCE_GROUNDED_ANSWERING_TEMPLATE.render({
        "context": context,
        "question": question,
        "citation_format": "[Doc: <Source>, Page: <Page>]",
    })

    assert "No unsupported answer. No hidden source." in sys_prompt
    assert "--- BEGIN REFERENCE CONTEXT ---" in usr_prompt
    assert "Convolutional networks process grid-structured topology." in usr_prompt
    assert question in usr_prompt
    assert "[Doc: <Source>, Page: <Page>]" in usr_prompt


def test_insufficient_context_refusal_template_rendering():
    """Verify insufficient context template generates principled gap analysis and guidance."""
    question = "What is the capital expenditure of the local library in 1842?"

    sys_prompt, usr_prompt = INSUFFICIENT_CONTEXT_REFUSAL_TEMPLATE.render({
        "question": question,
        "evaluated_topics": "19th century historical records collection",
        "suggested_terms": "municipal library ledger records, civic financial reports",
    })

    assert "never speculate or hallucinate" in sys_prompt
    assert "Identify the specific knowledge gap" in sys_prompt
    assert "19th century historical records collection" in usr_prompt
    assert "municipal library ledger records" in usr_prompt


def test_missing_required_variable_raises_error():
    """Verify MissingPromptVariableError is raised when mandatory variables are absent."""
    with pytest.raises(MissingPromptVariableError) as exc_info:
        # SUMMARIZATION_TEMPLATE requires 'document_text'
        SUMMARIZATION_TEMPLATE.render({})

    err = exc_info.value
    assert err.template_id == "summarization"
    assert "document_text" in err.missing_vars
    assert "missing required variable" in str(err)


def test_missing_multiple_required_variables_identified():
    """Verify multiple missing variables are reported simultaneously."""
    with pytest.raises(MissingPromptVariableError) as exc_info:
        # QUERY_REWRITING_TEMPLATE requires both 'conversation_history' and 'follow_up_question'
        QUERY_REWRITING_TEMPLATE.render({})

    err = exc_info.value
    assert err.template_id == "query_rewriting"
    assert "conversation_history" in err.missing_vars
    assert "follow_up_question" in err.missing_vars


def test_template_versioning_and_metadata():
    """Verify all canonical templates possess explicit identifiers, versions, and descriptions."""
    templates = [
        ACADEMIC_QA_TEMPLATE,
        QUERY_REWRITING_TEMPLATE,
        SUMMARIZATION_TEMPLATE,
        SOURCE_GROUNDED_ANSWERING_TEMPLATE,
        INSUFFICIENT_CONTEXT_REFUSAL_TEMPLATE,
    ]

    for tmpl in templates:
        assert tmpl.template_id != ""
        assert tmpl.name != ""
        assert tmpl.version == "1.0.0"
        assert tmpl.description != ""
        assert len(tmpl.required_variables) > 0
        assert "category" in tmpl.metadata


def test_registry_registration_and_lookup():
    """Verify PromptTemplateRegistry handles template registration, versioning, and lookup."""
    registry = PromptTemplateRegistry()

    # Verify canonical template retrieval
    tmpl = registry.get("academic_qa")
    assert tmpl.template_id == "academic_qa"
    assert tmpl.version == "1.0.0"

    # Specific version query
    tmpl_v1 = registry.get("academic_qa", version="1.0.0")
    assert tmpl_v1.version == "1.0.0"

    # Register updated version
    v2_template = PromptTemplate(
        template_id="academic_qa",
        name="Academic QA v2",
        version="2.0.0",
        description="Next-gen QA template",
        system_template="System v2: {discipline}",
        user_template="User v2: {question}",
        required_variables=["question"],
        optional_variables={"discipline": "Computer Science"},
    )
    registry.register(v2_template)

    # Retrieval without version should return latest (v2.0.0)
    latest = registry.get("academic_qa")
    assert latest.version == "2.0.0"

    # Retrieval of explicit v1 remains available
    v1_recheck = registry.get("academic_qa", version="1.0.0")
    assert v1_recheck.version == "1.0.0"

    # Check available versions list
    versions = registry.list_versions("academic_qa")
    assert "1.0.0" in versions
    assert "2.0.0" in versions


def test_registry_missing_template_raises_error():
    """Verify requesting non-existent template raises PromptTemplateNotFoundError."""
    registry = PromptTemplateRegistry()

    with pytest.raises(PromptTemplateNotFoundError):
        registry.get("non_existent_template")

    with pytest.raises(PromptTemplateNotFoundError):
        registry.get("academic_qa", version="99.9.9")


def test_safe_interpolation_with_json_and_special_characters():
    """Verify safe interpolation leaves literal JSON braces intact without KeyError."""
    template = PromptTemplate(
        template_id="json_safe_test",
        name="JSON Safe Test",
        version="1.0.0",
        system_template='You must output JSON matching schema: {"type": "object", "properties": {"ans": "str"}}',
        user_template='User Question: {question}\nRespond as: {"answer": "{question}"}',
        required_variables=["question"],
    )

    sys_out, usr_out = template.render({"question": "What is entropy?"})

    assert '{"type": "object"' in sys_out
    assert '{"answer": "What is entropy?"}' in usr_out


def test_render_bundle_produces_valid_prompt_bundle():
    """Verify render_bundle constructs an API-ready PromptBundle with role separation."""
    history = [
        Message(conversation_id="conv-1", role=MessageRole.USER, content="Hello"),
        Message(conversation_id="conv-1", role=MessageRole.ASSISTANT, content="Greetings researcher!"),
    ]
    context = "Academic literature context passage."

    bundle = SOURCE_GROUNDED_ANSWERING_TEMPLATE.render_bundle(
        variables={
            "context": context,
            "question": "What is machine learning?",
        },
        history=history,
    )

    assert bundle.system_prompt != ""
    assert bundle.user_query != ""
    assert any(m.role == PromptRole.SYSTEM for m in bundle.messages)
    assert any(m.role == PromptRole.USER for m in bundle.messages)
    assert any(m.role == PromptRole.ASSISTANT for m in bundle.messages)

    api_messages = bundle.to_api_messages()
    assert len(api_messages) >= 4
    assert api_messages[0]["role"] == "system"


def test_query_rewriter_integration_with_template():
    """Verify QueryRewriter.build_rewrite_prompt properly integrates with template system."""
    history = [
        Message(conversation_id="conv-1", role=MessageRole.USER, content="What is attention in NLP?"),
        Message(conversation_id="conv-1", role=MessageRole.ASSISTANT, content="Attention weights input representations."),
    ]
    query = "How does it scale with sequence length?"

    bundle = QueryRewriter.build_rewrite_prompt(query, history)

    assert "max 8 keywords" in bundle.system_prompt
    assert "What is attention in NLP?" in bundle.messages[-1].content
    assert "How does it scale with sequence length?" in bundle.messages[-1].content
