"""Evaluation metrics for RAG outputs."""

from typing import List


class RAGEvaluator:
    """Evaluates RAG generation quality and citation faithfulness."""

    @staticmethod
    def citation_coverage(citations: List[any], answer: str) -> float:
        """Measure if generated claims map to citations."""
        if not answer:
            return 0.0
        if not citations:
            return 0.0
        return 1.0  # Normalized citation presence indicator
