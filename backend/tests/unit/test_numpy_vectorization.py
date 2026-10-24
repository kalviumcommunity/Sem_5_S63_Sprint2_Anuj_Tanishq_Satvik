"""Unit tests for Concept #17: NumPy Vectorised Computation Workflow."""

import math
import numpy as np
import pytest
from backend.app.analytics.vectorized import VectorizedAnalytics


def test_batch_cosine_similarities_single_query():
    """Verify single query vs multiple candidate vectors."""
    q = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

    sims = VectorizedAnalytics.batch_cosine_similarities(q, candidates)

    assert sims.shape == (3,)
    assert math.isclose(sims[0], 1.0, abs_tol=1e-5)
    assert math.isclose(sims[1], 0.0, abs_tol=1e-5)
    assert math.isclose(sims[2], -1.0, abs_tol=1e-5)


def test_batch_cosine_similarities_multi_query_matrix():
    """Verify multi-query matrix multiplication produces (M, N) similarity matrix."""
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidates = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)

    sims = VectorizedAnalytics.batch_cosine_similarities(queries, candidates)

    assert sims.shape == (2, 3)
    assert pytest.approx(sims[0, 0], rel=1e-3) == 1.0
    assert pytest.approx(sims[1, 1], rel=1e-3) == 1.0
    assert pytest.approx(sims[0, 2], rel=1e-3) == 0.7071


def test_compute_latency_percentiles_with_nans_and_outliers():
    """Verify robust percentiles and outlier handling with NaNs."""
    latencies = [120.0, 150.0, float("nan"), 200.0, 500.0, 180.0, 110.0]
    stats = VectorizedAnalytics.compute_latency_percentiles(latencies)

    assert stats["count"] == 6
    assert stats["min"] == 110.0
    assert stats["max"] == 500.0
    assert stats["p50"] > 100.0
    assert stats["p99"] >= stats["p95"] >= stats["p90"] >= stats["p50"]


def test_compute_latency_percentiles_empty():
    """Verify empty input returns zeroes without division error."""
    stats = VectorizedAnalytics.compute_latency_percentiles([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


def test_normalize_scores_minmax():
    """Verify min-max scaling to [0, 1] range."""
    scores = np.array([0.2, 0.5, 0.8, 1.0])
    normalized = VectorizedAnalytics.normalize_scores_minmax(scores)

    assert normalized[0] == 0.0
    assert normalized[-1] == 1.0
    assert 0.0 <= normalized[1] <= 1.0

    # Flat scores edge case
    flat = np.array([0.7, 0.7, 0.7])
    norm_flat = VectorizedAnalytics.normalize_scores_minmax(flat)
    assert np.all(norm_flat == 1.0)


def test_compute_mrr_and_hit_rate():
    """Verify MRR and Hit Rate @ K across queries."""
    # 3 queries, 5 retrieved chunks each
    # Query 0: hit at rank 1 (reciprocal rank 1.0)
    # Query 1: hit at rank 2 (reciprocal rank 0.5)
    # Query 2: no hits (reciprocal rank 0.0)
    rankings = np.array([
        [True, False, False, False, False],
        [False, True, False, True, False],
        [False, False, False, False, False],
    ])

    metrics = VectorizedAnalytics.compute_mrr_and_hit_rate(rankings, k=5)

    assert metrics["query_count"] == 3
    # 2 out of 3 had hits in top 5 -> 2/3 = 0.6667
    assert pytest.approx(metrics["hit_rate"], rel=1e-3) == 0.6667
    # MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
    assert pytest.approx(metrics["mrr"], rel=1e-3) == 0.5


def test_compute_ndcg_at_k():
    """Verify NDCG calculation on graded relevance matrix."""
    # Perfect ranking: [2, 1, 0] should yield NDCG = 1.0
    perfect = np.array([[2, 1, 0, 0]])
    ndcg_perfect = VectorizedAnalytics.compute_ndcg_at_k(perfect, k=4)
    assert pytest.approx(ndcg_perfect, rel=1e-3) == 1.0

    # Inverted ranking: [0, 1, 2] should yield NDCG < 1.0
    inverted = np.array([[0, 1, 2, 0]])
    ndcg_inverted = VectorizedAnalytics.compute_ndcg_at_k(inverted, k=4)
    assert 0.0 < ndcg_inverted < 1.0


def test_benchmark_vectorization_speedup():
    """Verify benchmarking function runs and reports metrics."""
    bench = VectorizedAnalytics.benchmark_vectorization_speedup(num_elements=5000)
    assert bench["num_elements"] == 5000
    assert bench["vectorized_duration_sec"] > 0
    assert bench["loop_duration_sec"] > 0
    assert "speedup_factor" in bench
