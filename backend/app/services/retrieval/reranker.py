"""Candidate re-ranking for retrieval results."""

import re
from typing import List
from backend.app.services.retrieval.search import SearchResult

_STOP_WORDS = {
    "a", "an", "and", "are", "between", "difference", "do", "does", "for",
    "how", "in", "is", "of", "the", "to", "what",
}
_TERM_GROUPS = {
    "retrieval": {"retrieval", "retrieve", "retriever", "retrieving"},
    "generation": {"generation", "generate", "generator", "generative"},
    "combine": {"combine", "combined", "combines", "combining"},
}
_ADVANTAGE_TERMS = {"advantage", "advantages", "benefit", "benefits"}
_BENEFIT_EVIDENCE_TERMS = {
    "accurate", "accuracy", "benefit", "benefits", "diverse", "factual",
    "factually", "improve", "improved", "improvement", "less", "outperform",
    "performance", "specific",
}


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

        def score_candidate(cand: SearchResult) -> float:
            overlap = Reranker.keyword_overlap(query, cand.text)
            # Blend vector score with keyword match score
            blended = (cand.score * vector_weight) + (overlap * keyword_weight)
            return blended

        scored = [(cand, score_candidate(cand)) for cand in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        ranked: List[SearchResult] = []
        for cand, new_score in scored[:top_k]:
            reranked_cand = cand.model_copy(
                update={
                    "score": round(new_score, 4),
                    "similarity_score": (
                        cand.similarity_score
                        if cand.similarity_score is not None
                        else cand.score
                    ),
                }
            )
            ranked.append(reranked_cand)
        return ranked

    @staticmethod
    def keyword_overlap(query: str, text: str) -> float:
        """Score lexical evidence with small synonym groups for common academic phrasing."""
        query_terms = set(re.findall(r"\w+", query.casefold())) - _STOP_WORDS
        text_terms = set(re.findall(r"\w+", text.casefold()))
        if not query_terms:
            return 0.0

        advantage_query = bool(query_terms & _ADVANTAGE_TERMS)
        query_terms -= _ADVANTAGE_TERMS
        groups = [_TERM_GROUPS.get(term, {term}) for term in query_terms]
        if advantage_query:
            groups.append(_BENEFIT_EVIDENCE_TERMS)
        if not groups:
            return 0.0
        return sum(bool(group & text_terms) for group in groups) / len(groups)
