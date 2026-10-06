"""Evaluation metrics for retrieval."""

from typing import List, Set


class RetrievalEvaluator:
    """Evaluates retrieval quality (Recall@K, Precision@K, Hit Rate)."""

    @staticmethod
    def hit_rate(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
        """Calculate hit rate: 1.0 if at least one ground truth chunk was retrieved, 0.0 otherwise."""
        if not ground_truth_ids or not retrieved_ids:
            return 0.0
        return 1.0 if any(r in ground_truth_ids for r in retrieved_ids) else 0.0

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Calculate Precision@K."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for item in top_k if item in ground_truth_ids)
        return hits / len(top_k)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
        """Calculate Recall@K."""
        if not ground_truth_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for item in top_k if item in ground_truth_ids)
        return hits / len(ground_truth_ids)
