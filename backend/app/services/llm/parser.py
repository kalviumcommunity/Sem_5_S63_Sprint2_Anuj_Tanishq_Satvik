"""Response parser and structured JSON output validation for LLM outputs."""

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError

from backend.app.core.logging import logger
from backend.app.schemas.response import StructuredResearchResponse
from backend.app.services.llm.exceptions import StructuredOutputValidationError

T = TypeVar("T", bound=BaseModel)


class ParsedResponse(BaseModel):
    """Structured representation of parsed LLM response."""

    answer: str
    inferred_citations: List[str] = []
    is_refusal: bool = False


class StructuredOutputParser:
    """Parser and validator for structured JSON LLM responses."""

    @staticmethod
    def extract_json_string(raw_text: str, provider: str = "llm") -> str:
        """Extract JSON substring from raw model output, handling markdown blocks or commentary."""
        if not raw_text or not raw_text.strip():
            raise StructuredOutputValidationError(
                "Raw LLM output is empty; expected valid JSON object.",
                raw_output=raw_text,
                error_type="no_json_found",
                provider=provider,
            )

        cleaned = raw_text.strip()

        # Check for markdown code fences (```json ... ``` or ``` ... ```)
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            if (candidate.startswith("{") and candidate.endswith("}")) or (
                candidate.startswith("[") and candidate.endswith("]")
            ):
                return candidate

        # If wrapped entirely in array brackets
        if cleaned.startswith("[") and cleaned.endswith("]"):
            return cleaned

        # Locate outermost opening '{' and closing '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            # Check if there is an outer array enclosing it
            arr_start = cleaned.find("[")
            arr_end = cleaned.rfind("]")
            if arr_start != -1 and arr_end != -1 and arr_start < start_idx and arr_end > end_idx:
                return cleaned[arr_start : arr_end + 1]
            return cleaned[start_idx : end_idx + 1]

        # Check if text contains a JSON array
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            return cleaned[arr_start : arr_end + 1]

        raise StructuredOutputValidationError(
            "No JSON object delimiters '{...}' found in LLM output.",
            raw_output=raw_text,
            error_type="no_json_found",
            provider=provider,
        )


    @staticmethod
    def sanitize_json_string(json_candidate: str) -> str:
        """Apply safe syntactic sanitization for common minor LLM formatting errors."""
        text = json_candidate.strip()
        # Remove single-line JS-style comments (// ...)
        text = re.sub(r"//.*?\n", "\n", text)
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r",\s*([\]}])", r"\1", text)
        # Strip trailing semicolons often appended by models
        text = text.rstrip(";")
        return text.strip()

    @classmethod
    def parse_and_validate(
        cls,
        raw_text: str,
        schema_cls: Type[T] = StructuredResearchResponse,
        allow_repair: bool = True,
        provider: str = "llm",
    ) -> T:
        """Extract, parse, and validate JSON from raw LLM output against the target schema.

        Never silently accepts invalid or malformed outputs.
        """
        json_str = cls.extract_json_string(raw_text, provider=provider)

        # Attempt initial JSON parse
        parsed_dict: Optional[Dict[str, Any]] = None
        try:
            parsed_dict = json.loads(json_str)
        except json.JSONDecodeError as decode_err:
            if allow_repair:
                sanitized = cls.sanitize_json_string(json_str)
                try:
                    parsed_dict = json.loads(sanitized)
                    logger.debug("Successfully repaired minor JSON syntax defects in LLM output.")
                except json.JSONDecodeError:
                    pass

            if parsed_dict is None:
                raise StructuredOutputValidationError(
                    f"Malformed JSON syntax returned by LLM: {decode_err}",
                    raw_output=raw_text,
                    error_type="malformed_json",
                    provider=provider,
                ) from decode_err

        # Verify parsed entity is a dictionary (JSON object)
        if not isinstance(parsed_dict, dict):
            raise StructuredOutputValidationError(
                f"Expected top-level JSON object ({schema_cls.__name__}), received {type(parsed_dict).__name__}.",
                raw_output=raw_text,
                error_type="schema_validation_error",
                provider=provider,
            )

        # Validate against Pydantic schema
        try:
            return schema_cls.model_validate(parsed_dict)
        except ValidationError as val_err:
            errors = [
                {
                    "loc": list(err.get("loc", [])),
                    "msg": err.get("msg", ""),
                    "type": err.get("type", ""),
                }
                for err in val_err.errors()
            ]
            raise StructuredOutputValidationError(
                f"Structured response failed schema validation for {schema_cls.__name__}",
                raw_output=raw_text,
                validation_errors=errors,
                error_type="schema_validation_error",
                provider=provider,
            ) from val_err

    @staticmethod
    def generate_schema_prompt(schema_cls: Type[T] = StructuredResearchResponse) -> str:
        """Generate schema formatting instructions to guide the model towards valid JSON."""
        schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
        return (
            "OUTPUT FORMAT REQUIREMENT:\n"
            "Respond strictly with a valid JSON object matching the JSON schema below.\n"
            "Do NOT include conversational introductory remarks or markdown commentary outside the JSON object.\n\n"
            f"```json\n{schema_json}\n```"
        )


class ResponseParser:
    """Parses raw LLM text into answers and citations (supports legacy and structured formats)."""

    REFUSAL_TRIGGERS = [
        "couldn't find enough supporting information",
        "cannot find enough supporting information",
        "insufficient information",
        "not enough information in the provided",
    ]

    @classmethod
    def parse(cls, raw_text: str) -> ParsedResponse:
        """Parse raw response text into plain text answer and bracketed citations."""
        cleaned = raw_text.strip()
        is_refusal = any(trigger in cleaned.lower() for trigger in cls.REFUSAL_TRIGGERS)

        # Extract citation brackets like [Doc: ..., Page: ...]
        citations = re.findall(r"\[([^\]]+)\]", cleaned)

        return ParsedResponse(
            answer=cleaned,
            inferred_citations=citations,
            is_refusal=is_refusal,
        )

    @classmethod
    def parse_structured(
        cls,
        raw_text: str,
        schema_cls: Type[T] = StructuredResearchResponse,
        allow_repair: bool = True,
        provider: str = "llm",
    ) -> T:
        """Parse and strictly validate raw LLM text against a Pydantic schema."""
        return StructuredOutputParser.parse_and_validate(
            raw_text=raw_text,
            schema_cls=schema_cls,
            allow_repair=allow_repair,
            provider=provider,
        )
