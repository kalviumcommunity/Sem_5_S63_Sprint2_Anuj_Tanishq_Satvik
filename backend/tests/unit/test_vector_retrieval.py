"""Unit tests for Concept #13: Vector Search & Similarity Retrieval."""

import pytest
from backend.app.models.chunk import Chunk
from backend.app.services.retrieval.filters import MetadataFilter
from backend.app.services.retrieval.reranker import Reranker
from backend.app.services.retrieval.search import (
    InMemoryVectorStore,
    SearchResult,
    SimilarityMetric,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    get_vector_store,
)


def test_similarity_metric_calculations():
    """Verify numeric calculations for cosine similarity, dot product, and euclidean distance."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    # Orthogonal vectors
    assert cosine_similarity(v1, v2) == 1.0
    assert cosine_similarity(v1, v3) == 0.0

    # Dot product
    assert dot_product(v1, v2) == 1.0
    assert dot_product(v1, v3) == 0.0

    # Euclidean distance
    assert euclidean_distance(v1, v2) == 0.0
    assert pytest.approx(euclidean_distance(v1, v3), rel=1e-4) == 1.4142

    # Empty vectors
    assert cosine_similarity([], []) == 0.0
    assert dot_product([], []) == 0.0
    assert euclidean_distance([], []) == float("inf")


def test_vector_store_add_and_count():
    """Verify adding single items and counting items in the store."""
    store = InMemoryVectorStore()
    assert store.count() == 0

    store.add_item(
        chunk_id="c-1",
        vector=[0.6, 0.8],
        document_id="doc-100",
        document_title="Attention Is All You Need",
        text="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        page_number=1,
        section="intro",
        metadata={"year": 2017},
    )

    assert store.count() == 1


def test_vector_store_bulk_add_chunks():
    """Verify indexing Chunks with embedding vectors."""
    store = InMemoryVectorStore()
    chunks = [
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            text="First chunk text discussing self-attention.",
            page_number=1,
            metadata={"filename": "paper.pdf", "course": "CS8803"},
        ),
        Chunk(
            id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            text="Second chunk text discussing positional encodings.",
            page_number=2,
            metadata={"filename": "paper.pdf", "course": "CS8803"},
        ),
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]

    count = store.add_chunks(chunks, vectors)
    assert count == 2
    assert store.count() == 2


def test_vector_store_similarity_search():
    """Verify top-k similarity search returns nearest candidates sorted by score."""
    store = InMemoryVectorStore()
    store.add_item("c1", [1.0, 0.0, 0.0], "doc-1", "Doc 1", "Perfect match on NLP", page_number=1)
    store.add_item("c2", [0.7, 0.7, 0.0], "doc-2", "Doc 2", "Partial match on ML", page_number=2)
    store.add_item("c3", [0.0, 1.0, 0.0], "doc-3", "Doc 3", "Orthogonal unrelated topic", page_number=3)

    query = [1.0, 0.0, 0.0]
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].score == 1.0
    assert results[1].chunk_id == "c2"
    assert results[1].score > 0.6


def test_vector_store_score_thresholding():
    """Verify score_threshold filters out weak retrieval candidates."""
    store = InMemoryVectorStore()
    store.add_item("c1", [1.0, 0.0], "doc-1", "Doc 1", "High similarity", page_number=1)
    store.add_item("c2", [0.2, 0.98], "doc-2", "Doc 2", "Low similarity", page_number=2)

    query = [1.0, 0.0]
    # Set threshold to 0.5; only c1 should pass
    results = store.search(query, top_k=5, score_threshold=0.5)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_vector_store_search_metrics():
    """Verify search using different metrics: cosine, dot product, euclidean."""
    store = InMemoryVectorStore()
    store.add_item("c1", [2.0, 0.0], "d1", "T1", "Text 1")
    store.add_item("c2", [0.0, 2.0], "d2", "T2", "Text 2")

    query = [1.0, 0.0]

    # Cosine ignores magnitude
    res_cos = store.search(query, metric=SimilarityMetric.COSINE)
    assert res_cos[0].chunk_id == "c1"
    assert pytest.approx(res_cos[0].score, rel=1e-3) == 1.0

    # Euclidean
    res_euc = store.search(query, metric=SimilarityMetric.EUCLIDEAN)
    assert res_euc[0].chunk_id == "c1"


def test_metadata_filtering_operators():
    """Verify metadata filtering with exact equality, range ($gte, $lte), and membership ($in)."""
    candidates = [
        SearchResult(chunk_id="1", document_id="doc-A", document_title="T1", text="text1", page_number=1, metadata={"year": 2020, "field": "AI"}),
        SearchResult(chunk_id="2", document_id="doc-B", document_title="T2", text="text2", page_number=3, metadata={"year": 2022, "field": "Bio"}),
        SearchResult(chunk_id="3", document_id="doc-A", document_title="T3", text="text3", page_number=5, metadata={"year": 2024, "field": "AI"}),
    ]

    # Filter by exact match
    ai_only = MetadataFilter.apply(candidates, {"field": "AI"})
    assert len(ai_only) == 2
    assert {c.chunk_id for c in ai_only} == {"1", "3"}

    # Filter by range operator on page_number
    page_filtered = MetadataFilter.apply(candidates, {"page_number": {"$gte": 2, "$lte": 4}})
    assert len(page_filtered) == 1
    assert page_filtered[0].chunk_id == "2"

    # Filter by $in membership
    field_in = MetadataFilter.apply(candidates, {"field": {"$in": ["Bio", "Physics"]}})
    assert len(field_in) == 1
    assert field_in[0].chunk_id == "2"


def test_vector_store_delete_and_delete_by_document():
    """Verify individual chunk deletion and cascading deletion by document ID."""
    store = InMemoryVectorStore()
    store.add_item("c1", [1.0, 0.0], "doc-A", "A", "t1")
    store.add_item("c2", [0.0, 1.0], "doc-A", "A", "t2")
    store.add_item("c3", [0.5, 0.5], "doc-B", "B", "t3")
    assert store.count() == 3

    # Delete single chunk
    assert store.delete("c3") is True
    assert store.delete("c_nonexistent") is False
    assert store.count() == 2

    # Delete by document ID
    deleted_count = store.delete_by_document("doc-A")
    assert deleted_count == 2
    assert store.count() == 0


def test_reranker_hybrid_scoring():
    """Verify Reranker blends vector similarity score with keyword match overlap."""
    c1 = SearchResult(chunk_id="1", document_id="d1", document_title="T1", text="Quantum entanglement and teleportation algorithms.", page_number=1, score=0.8)
    c2 = SearchResult(chunk_id="2", document_id="d2", document_title="T2", text="Classical distributed computing algorithms.", page_number=2, score=0.85)

    # Query specifically about quantum teleportation
    query = "quantum teleportation algorithms"
    reranked = Reranker.rank(query, [c1, c2], top_k=2, vector_weight=0.5, keyword_weight=0.5)

    assert len(reranked) == 2
    # c1 has 3/3 keyword overlap while c2 has only 1/3 (algorithms). So c1 should win despite lower initial vector score!
    assert reranked[0].chunk_id == "1"


def test_get_vector_store_singleton():
    """Verify global vector store instance."""
    store = get_vector_store()
    assert isinstance(store, InMemoryVectorStore)
