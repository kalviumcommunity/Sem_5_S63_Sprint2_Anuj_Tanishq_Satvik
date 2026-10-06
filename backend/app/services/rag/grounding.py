"""Grounding checks and hallucination guardrails."""

from typing import List
from backend.app.core.config import settings
from backend.app.services.retrieval.search import SearchResult


class GroundingService:
    """Evaluates whether retrieved evidence is sufficient to ground a factual answer."""

    def __init__(self, threshold: float = None):
        self.threshold = threshold or settings.GROUNDING_THRESHOLD

    def is_sufficient(self, results: List[SearchResult]) -> bool:
        """Check if at least one candidate meets the grounding confidence threshold."""
        if not results:
            return False

        max_score = max((r.score for r in results), default=0.0)
        return max_score >= self.threshold
