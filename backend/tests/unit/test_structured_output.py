"""Unit tests for Concept #8: Structured Output & JSON Response Handling."""

import json
import pytest

from backend.app.schemas.response import (
    ConfidenceLevel,
    EvidenceStatus,
    SourceReference,
    StructuredCitation,
    StructuredResearchResponse,
)
from backend.app.services.llm.client import MockLLMClient
from backend.app.services.llm.exceptions import (
    LLMResponseError,
    StructuredOutputValidationError,
)
from backend.app.services.llm.parser import (
    ResponseParser,
    StructuredOutputParser,
)
from backend.app.services.llm.prompts import build_structured_rag_prompt
from backend.app.services.rag.pipeline import RAGPipeline
from backend.app.services.retrieval.search import SearchResult


def test_structured_research_response_valid():
    """Verify that a fully populated valid structured response model instantiates correctly."""
    data = {
        "answer": "Quantum annealing is an optimization process for finding the global minimum of an objective function.",
        "citations": [
            {
                "source": "Quantum Computing Principles",
                "page": 42,
                "chunk_id": "qc_chunk_12",
                "quote": "Quantum annealing finds the global minimum of an objective function.",
                "relevance_explanation": "Defines core operation of quantum annealing.",
            }
        ],
        "evidence_status": "sufficient",
        "confidence": 0.95,
        "source_references": [
            {
                "document_id": "doc_qc_2023",
                "title": "Quantum Computing Principles",
                "page": 42,
                "chunk_id": "qc_chunk_12",
                "section": "Optimization Algorithms",
                "snippet": "Quantum annealing finds the global minimum...",
            }
        ],
        "limitations": "Limited to transverse Ising spin glass formulations.",
        "refusal_reason": None,
    }

    model = StructuredResearchResponse.model_validate(data)
    assert model.answer.startswith("Quantum annealing")
    assert len(model.citations) == 1
    assert model.evidence_status == EvidenceStatus.SUFFICIENT
    assert model.confidence == 0.95
    assert len(model.source_references) == 1
    assert not model.is_refusal


def test_structured_output_parser_extracts_markdown_code_fences():
    """Verify parser extracts JSON cleanly when wrapped inside markdown fences."""
    raw_output = """
Here is the structured analysis of the provided academic material:

```json
{
  "answer": "Differential privacy provides mathematically bounded privacy guarantees against membership inference attacks.",
  "citations": [
    {
      "source": "Algorithmic Foundations of DP",
      "page": 17
    }
  ],
  "evidence_status": "sufficient",
  "confidence": 1.0,
  "source_references": []
}
```

Please let me know if you need additional citations.
"""
    parsed = StructuredOutputParser.parse_and_validate(raw_output)
    assert isinstance(parsed, StructuredResearchResponse)
    assert "Differential privacy" in parsed.answer
    assert parsed.confidence == 1.0
    assert len(parsed.citations) == 1


def test_structured_output_parser_repairs_trailing_commas():
    """Verify parser safely cleans minor syntax issues like trailing commas."""
    raw_output = """
{
  "answer": "Knowledge distillation transfers knowledge from a teacher model to a student model.",
  "citations": [
    {
      "source": "Distilling Knowledge in Neural Networks",
      "page": 3,
    },
  ],
  "evidence_status": "sufficient",
  "confidence": 0.9,
  "source_references": [],
}
"""
    parsed = StructuredOutputParser.parse_and_validate(raw_output, allow_repair=True)
    assert isinstance(parsed, StructuredResearchResponse)
    assert "Knowledge distillation" in parsed.answer
    assert len(parsed.citations) == 1


def test_invalid_json_raises_structured_output_validation_error():
    """Verify that unrecoverable malformed JSON explicitly raises StructuredOutputValidationError."""
    malformed_output = '{"answer": "Incomplete string without closing bracket'

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        StructuredOutputParser.parse_and_validate(malformed_output)

    err = exc_info.value
    assert isinstance(err, LLMResponseError)
    assert err.error_type in ("no_json_found", "malformed_json")
    assert err.raw_output == malformed_output


def test_empty_or_no_json_raises_structured_output_validation_error():
    """Verify that plain text with no JSON objects is rejected and never silently accepted."""
    plain_text = "Sorry, I am just an AI language model without JSON formatting."

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        StructuredOutputParser.parse_and_validate(plain_text)

    err = exc_info.value
    assert err.error_type == "no_json_found"


def test_missing_required_answer_field_raises_validation_error():
    """Verify that JSON missing the mandatory 'answer' field fails validation loudly."""
    missing_answer = json.dumps({
        "citations": [],
        "evidence_status": "sufficient",
        "confidence": 0.8,
    })

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        StructuredOutputParser.parse_and_validate(missing_answer)

    err = exc_info.value
    assert err.error_type == "schema_validation_error"
    assert any("answer" in error["loc"] for error in err.validation_errors)


def test_empty_whitespace_answer_raises_validation_error():
    """Verify that an empty string or whitespace-only answer field is rejected."""
    empty_answer = json.dumps({
        "answer": "   \n\t  ",
        "citations": [],
        "evidence_status": "sufficient",
        "confidence": 1.0,
    })

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        StructuredOutputParser.parse_and_validate(empty_answer)

    err = exc_info.value
    assert err.error_type == "schema_validation_error"


def test_invalid_evidence_status_enum_raises_validation_error():
    """Verify that an invalid evidence_status enum value fails validation loudly."""
    invalid_status = json.dumps({
        "answer": "Valid answer text.",
        "citations": [],
        "evidence_status": "unsupported_hallucinated_mode",
        "confidence": 0.5,
    })

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        StructuredOutputParser.parse_and_validate(invalid_status)

    err = exc_info.value
    assert err.error_type == "schema_validation_error"


def test_out_of_range_confidence_raises_validation_error():
    """Verify that confidence out of [0.0, 1.0] fails validation loudly."""
    high_confidence = json.dumps({
        "answer": "Valid answer text.",
        "citations": [],
        "evidence_status": "sufficient",
        "confidence": 1.85,  # Exceeds max 1.0
    })

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        StructuredOutputParser.parse_and_validate(high_confidence)

    err = exc_info.value
    assert err.error_type == "schema_validation_error"


def test_json_array_instead_of_object_raises_validation_error():
    """Verify that top-level JSON array is rejected when an object is expected."""
    json_array = json.dumps([{"answer": "Item 1"}, {"answer": "Item 2"}])

    with pytest.raises(StructuredOutputValidationError) as exc_info:
        StructuredOutputParser.parse_and_validate(json_array)

    err = exc_info.value
    assert err.error_type == "schema_validation_error"


@pytest.mark.asyncio
async def test_mock_client_complete_structured():
    """Verify MockLLMClient.complete_structured returns a valid StructuredResearchResponse."""
    client = MockLLMClient()
    structured_resp, completion = await client.complete_structured("Explain federated learning.")

    assert isinstance(structured_resp, StructuredResearchResponse)
    assert "federated learning" in structured_resp.answer.lower()
    assert structured_resp.evidence_status == EvidenceStatus.SUFFICIENT
    assert structured_resp.confidence == 0.98
    assert len(structured_resp.citations) >= 1
    assert len(structured_resp.source_references) >= 1
    assert completion.total_tokens > 0


@pytest.mark.asyncio
async def test_mock_client_simulated_invalid_json_raises():
    """Verify simulated invalid JSON from LLM raises StructuredOutputValidationError."""
    client = MockLLMClient(simulate_error="invalid_json")

    with pytest.raises(StructuredOutputValidationError):
        await client.complete_structured("Test query")


@pytest.mark.asyncio
async def test_mock_client_simulated_invalid_schema_raises():
    """Verify simulated missing field schema from LLM raises StructuredOutputValidationError."""
    client = MockLLMClient(simulate_error="invalid_schema")

    with pytest.raises(StructuredOutputValidationError):
        await client.complete_structured("Test query")


@pytest.mark.asyncio
async def test_rag_pipeline_answer_structured():
    """Verify RAGPipeline.answer_structured produces a fully valid structured response."""
    pipeline = RAGPipeline()
    candidates = [
        SearchResult(
            chunk_id="chunk_fl_1",
            document_id="doc_fl_1",
            document_title="Federated Learning Survey",
            page_number=8,
            section="Architecture",
            text="Model training occurs across decentralized edge devices without centralizing raw user data.",
            score=0.92,
        )
    ]

    structured_resp = await pipeline.answer_structured(
        query="How does federated learning handle data privacy?",
        candidates=candidates,
    )

    assert isinstance(structured_resp, StructuredResearchResponse)
    assert structured_resp.evidence_status == EvidenceStatus.SUFFICIENT
    assert structured_resp.confidence >= 0.8
    assert len(structured_resp.citations) >= 1


@pytest.mark.asyncio
async def test_rag_pipeline_answer_structured_insufficient_evidence():
    """Verify RAGPipeline.answer_structured correctly sets refusal evidence status when candidates are empty."""
    pipeline = RAGPipeline()
    structured_resp = await pipeline.answer_structured(
        query="What is the square root of a library?",
        candidates=[],
    )

    assert isinstance(structured_resp, StructuredResearchResponse)
    assert structured_resp.evidence_status == EvidenceStatus.INSUFFICIENT
    assert structured_resp.confidence == 0.0
    assert structured_resp.is_refusal
    assert structured_resp.refusal_reason is not None


def test_build_structured_rag_prompt_contains_schema_instructions():
    """Verify build_structured_rag_prompt attaches the JSON schema format requirements."""
    bundle = build_structured_rag_prompt(
        query="Explain transformers.",
        context="Transformers use self-attention mechanisms [Doc: Attention Is All You Need, Page: 3].",
    )

    assert "OUTPUT FORMAT REQUIREMENT" in bundle.system_prompt
    assert "StructuredResearchResponse" in bundle.system_prompt or "properties" in bundle.system_prompt
    assert "answer" in bundle.system_prompt
    assert "evidence_status" in bundle.system_prompt
    assert bundle.user_query == "Explain transformers."
