"""Response parser for LLM outputs."""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel


class ParsedResponse(BaseModel):
    """Structured representation of parsed LLM response."""

    answer: str
    inferred_citations: List[str] = []
    is_refusal: bool = False


class ResponseParser:
    """Parses raw LLM text into answers and citations."""

    REFUSAL_TRIGGERS = [
        "couldn't find enough supporting information",
        "cannot find enough supporting information",
        "insufficient information",
        "not enough information in the provided",
    ]

    @classmethod
    def parse(cls, raw_text: str) -> ParsedResponse:
        """Parse raw response text."""
        cleaned = raw_text.strip()
        is_refusal = any(trigger in cleaned.lower() for trigger in cls.REFUSAL_TRIGGERS)

        # Extract citation brackets like [Doc: ..., Page: ...]
        citations = re.findall(r"\[([^\]]+)\]", cleaned)

        return ParsedResponse(
            answer=cleaned,
            inferred_citations=citations,
            is_refusal=is_refusal,
        )
