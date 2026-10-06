"""Pydantic schemas for API responses and structured LLM outputs."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class EvidenceStatus(str, Enum):
    """Categorical status of evidence grounding for the query."""

    SUFFICIENT = "sufficient"
    PARTIALLY_SUPPORTED = "partially_supported"
    INSUFFICIENT = "insufficient"
    UNSUPPORTED = "unsupported"


class ConfidenceLevel(str, Enum):
    """Confidence tier for answer groundedness."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceReference(BaseModel):
    """Metadata for an academic source cited in an answer."""

    document_id: Optional[str] = Field(default=None, description="Unique identifier for the document")
    title: str = Field(..., description="Document or academic paper title")
    page: Optional[int] = Field(default=None, description="Page number where evidence appears")
    chunk_id: Optional[str] = Field(default=None, description="Specific chunk identifier")
    section: Optional[str] = Field(default=None, description="Section heading")
    snippet: Optional[str] = Field(default=None, description="Direct supporting passage or snippet")


class StructuredCitation(BaseModel):
    """Citation mapping a statement to an academic source."""

    source: str = Field(..., description="Referenced document title or identifier")
    page: Optional[int] = Field(default=None, description="Referenced page number")
    chunk_id: Optional[str] = Field(default=None, description="Referenced chunk identifier")
    quote: Optional[str] = Field(default=None, description="Direct supporting quote")
    relevance_explanation: Optional[str] = Field(default=None, description="Explanation of relevance to the claim")


class StructuredResearchResponse(BaseModel):
    """Structured response schema for ResearchMate LLM output."""

    answer: str = Field(..., min_length=1, description="Synthesized, grounded answer to the user query")
    citations: List[StructuredCitation] = Field(
        default_factory=list,
        description="Structured citations linking claims to evidence",
    )
    evidence_status: EvidenceStatus = Field(
        default=EvidenceStatus.SUFFICIENT,
        description="Status of supporting academic evidence",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 reflecting factual grounding certainty",
    )
    source_references: List[SourceReference] = Field(
        default_factory=list,
        description="Direct references to source documents and pages",
    )
    limitations: Optional[str] = Field(
        default=None,
        description="Noted caveats or missing aspects in the provided literature",
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Explanation if the question cannot be reliably answered",
    )

    @field_validator("answer")
    @classmethod
    def validate_answer_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("The 'answer' field cannot be empty or purely whitespace.")
        return cleaned

    @property
    def is_refusal(self) -> bool:
        """Check whether this structured response constitutes a refusal to answer."""
        return (
            self.evidence_status in (EvidenceStatus.INSUFFICIENT, EvidenceStatus.UNSUPPORTED)
            or self.refusal_reason is not None
        )


class CitationItem(BaseModel):
    """Citation item in query response (for API compatibility)."""

    document_id: str
    title: str
    page: int
    chunk_id: str
    section: Optional[str] = None
    relevance_score: Optional[float] = None
    citation_text: Optional[str] = None


class QueryResponse(BaseModel):
    """Response returned for student query."""

    answer: str
    citations: List[CitationItem] = Field(default_factory=list)
    grounded: bool = True
    conversation_id: Optional[str] = None
    latency_ms: Optional[float] = None
    token_usage: Optional[Dict[str, Any]] = None
    estimated_cost_usd: Optional[float] = None
    evidence_status: Optional[str] = None
    confidence: Optional[float] = None
    source_references: List[SourceReference] = Field(default_factory=list)
    structured_response: Optional[StructuredResearchResponse] = None


class HealthResponse(BaseModel):
    """Health status response."""

    status: str
    app_name: str
    version: str
    environment: str
    services: Dict[str, str] = Field(default_factory=dict)

