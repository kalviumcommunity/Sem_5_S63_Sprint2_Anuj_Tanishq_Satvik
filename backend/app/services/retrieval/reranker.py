"""Candidate re-ranking for retrieval results."""

from typing import List
from backend.app.services.retrieval.search import SearchResult


class Reranker:
    """Reranks search candidates to prioritize the highest quality evidence."""

    @staticmethod
    def rank(query: str, candidates: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """Rerank candidates based on keyword overlap and semantic relevance."""
        if not candidates:
            return []

        query_terms = set(query.lower().split())

        def score_candidate(cand: SearchResult) -> float:
            text_terms = set(cand.text.lower().split())
            overlap = len(query_terms.intersection(text_terms)) / max(1, len(query_terms))
            # Blend vector score with keyword match score
            return (cand.score * 0.7) + (overlap * 0.3)

        scored = [(cand, score_candidate(cand)) for cand in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        ranked = []
        for cand, new_score in scored[:top_k]:
            cand.score = round(new_score, 4)
            ranked.append(cand)

        return ranked
