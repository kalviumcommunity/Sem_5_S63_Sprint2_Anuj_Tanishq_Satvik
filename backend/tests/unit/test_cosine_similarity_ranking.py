"""Unit tests for Cosine Similarity & Retrieval Ranking."""

import numpy as np
import pytest

from backend.app.models.chunk import Chunk
from backend.app.services.embeddings.service import get_embedding_service
from backend.app.services.retrieval.demo import main as run_retrieval_demo
from backend.app.services.retrieval.search import (
    InMemoryVectorStore,
    SearchResult,
    SimilarityMetric,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    rank_chunks,
)


def test_cosine_similarity_basic_properties():
    """Verify core mathematical properties of cosine similarity."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v_opposite = [-1.0, 0.0, 0.0]

    # Identical vectors -> 1.0
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0

    # Orthogonal vectors -> 0.0
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0

    # Opposite direction -> -1.0
    assert pytest.approx(cosine_similarity(v1, v_opposite)) == -1.0


def test_cosine_similarity_compares_direction_not_raw_length():
    """Verify cosine similarity is invariant to vector magnitude/scale."""
    a = [1.0, 2.0, 3.0]
    b = [10.0, 20.0, 30.0]  # Same direction, 10x magnitude
    c = [0.001, 0.002, 0.003]  # Same direction, tiny magnitude

    assert pytest.approx(cosine_similarity(a, b)) == 1.0
    assert pytest.approx(cosine_similarity(a, c)) == 1.0


def test_cosine_similarity_numpy_array_and_list_support():
    """Verify function works interchangeably with numpy arrays, lists, and tuples."""
    np_v1 = np.array([0.6, 0.8])
    np_v2 = np.array([0.6, 0.8])
    assert pytest.approx(cosine_similarity(np_v1, np_v2)) == 1.0

    list_v1 = [0.6, 0.8]
    assert pytest.approx(cosine_similarity(np_v1, list_v1)) == 1.0
    assert pytest.approx(cosine_similarity(tuple(list_v1), np_v2)) == 1.0


def test_cosine_similarity_edge_cases():
    """Verify graceful handling of zero vectors, empty vectors, and dimension mismatches."""
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


@pytest.mark.asyncio
async def test_rank_chunks_for_learner_password_query():
    """Verify ranking chunk records by meaning for the assignment password reset query."""
    query = "How can a learner reset their password?"
    service = get_embedding_service(use_cache=False)

    raw_chunks = [
        {
            "text": "Password reset instructions for learner accounts.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0},
        },
        {
            "text": "The cafeteria menu changes every Friday.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 3},
        },
        {
            "text": "Learners can recover access using their registered email.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1},
        },
    ]

    query_embeddings = await service.embed_batch([query])
    query_embedding = query_embeddings[0]

    chunk_texts = [item["text"] for item in raw_chunks]
    chunk_embeddings = await service.embed_batch(chunk_texts)

    chunk_records = [
        {
            "text": item["text"],
            "metadata": item["metadata"],
            "embedding": emb,
        }
        for item, emb in zip(raw_chunks, chunk_embeddings)
    ]

    ranked = rank_chunks(query_embedding, chunk_records)

    # Validate output length and order
    assert len(ranked) == 3

    # Most similar chunk should be relevant to password reset / account recovery
    top_chunk = ranked[0]
    assert "password" in top_chunk["text"].lower() or "recover access" in top_chunk["text"].lower()
    assert top_chunk["metadata"]["source"] == "account-guide.md"
    assert top_chunk["score"] > 0.3

    # Least similar chunk must be the unrelated cafeteria menu
    bottom_chunk = ranked[-1]
    assert "cafeteria menu" in bottom_chunk["text"].lower()
    assert bottom_chunk["metadata"]["source"] == "campus-guide.md"
    assert bottom_chunk["score"] < top_chunk["score"]

    # Verify scores are sorted strictly descending
    scores = [item["score"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_chunks_top_k_limiting():
    """Verify top_k parameter limits number of ranked results."""
    query_vec = [1.0, 0.0]
    records = [
        {"id": "c1", "embedding": [1.0, 0.0], "text": "exact"},
        {"id": "c2", "embedding": [0.8, 0.2], "text": "close"},
        {"id": "c3", "embedding": [0.0, 1.0], "text": "orthogonal"},
    ]

    top_1 = rank_chunks(query_vec, records, top_k=1)
    assert len(top_1) == 1
    assert top_1[0]["id"] == "c1"

    top_2 = rank_chunks(query_vec, records, top_k=2)
    assert len(top_2) == 2
    assert [r["id"] for r in top_2] == ["c1", "c2"]


def test_rank_chunks_with_alternative_metrics():
    """Verify rank_chunks supports dot product and euclidean distance metrics."""
    query_vec = [1.0, 0.0]
    records = [
        {"id": "c1", "embedding": [2.0, 0.0], "text": "scaled match"},
        {"id": "c2", "embedding": [0.0, 1.0], "text": "unrelated"},
    ]

    # Cosine ignores magnitude
    ranked_cos = rank_chunks(query_vec, records, metric=SimilarityMetric.COSINE)
    assert ranked_cos[0]["id"] == "c1"
    assert pytest.approx(ranked_cos[0]["score"]) == 1.0

    # Dot product incorporates magnitude
    ranked_dot = rank_chunks(query_vec, records, metric=SimilarityMetric.DOT_PRODUCT)
    assert ranked_dot[0]["id"] == "c1"
    assert pytest.approx(ranked_dot[0]["score"]) == 2.0

    # Euclidean distance
    ranked_euc = rank_chunks(query_vec, records, metric=SimilarityMetric.EUCLIDEAN)
    assert ranked_euc[0]["id"] == "c1"
    assert ranked_euc[0]["score"] > ranked_euc[1]["score"]


def test_in_memory_vector_store_rank_chunks():
    """Verify rank_chunks convenience method on InMemoryVectorStore."""
    store = InMemoryVectorStore()
    store.add_item("c1", [1.0, 0.0], "doc-1", "Doc 1", "Matches query", page_number=1)
    store.add_item("c2", [0.0, 1.0], "doc-2", "Doc 2", "Unrelated topic", page_number=2)

    query_vec = [1.0, 0.0]
    ranked = store.rank_chunks(query_vec)

    assert len(ranked) == 2
    assert ranked[0].chunk_id == "c1"
    assert ranked[0].score == 1.0
    assert ranked[1].chunk_id == "c2"
    assert ranked[1].score == 0.0


@pytest.mark.asyncio
async def test_retrieval_demo_executes_successfully(capsys):
    """Verify that the retrieval similarity demo script runs end-to-end."""
    await run_retrieval_demo()
    captured = capsys.readouterr()
    assert "query: How can a learner reset their password?" in captured.out
    assert "most similar:" in captured.out
    assert "Password reset instructions" in captured.out
    assert "least similar:" in captured.out
    assert "cafeteria menu" in captured.out
