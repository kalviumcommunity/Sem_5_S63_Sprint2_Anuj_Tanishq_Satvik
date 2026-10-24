"""Candidate re-ranking for retrieval results."""

from typing import List
from backend.app.services.retrieval.search import SearchResult


class Reranker:
    """Reranks search candidates to prioritize the highest quality evidence combining dense vector scores and keyword match."""

    @staticmethod
    def rank(
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[SearchResult]:
        """Rerank candidates based on keyword overlap and semantic relevance."""
        if not candidates:
            return []

        query_terms = set(query.lower().split())

        def score_candidate(cand: SearchResult) -> float:
            text_terms = set(cand.text.lower().split())
            overlap = (
                len(query_terms.intersection(text_terms)) / max(1, len(query_terms))
                if query_terms
                else 0.0
            )
            # Blend vector score with keyword match score
            blended = (cand.score * vector_weight) + (overlap * keyword_weight)
            return blended

        scored = [(cand, score_candidate(cand)) for cand in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        ranked: List[SearchResult] = []
        for cand, new_score in scored[:top_k]:
            reranked_cand = cand.model_copy(update={"score": round(new_score, 4)})
            ranked.append(reranked_cand)

        return ranked
