"""Unit tests for Concept #19: Correlation & Relationship Analysis."""

import math
import numpy as np
import pytest
from backend.app.analytics.correlation import CorrelationAnalyzer


def test_compute_pearson_perfect_and_negative():
    """Verify Pearson correlation coefficient r on linear relationships."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pos = [2.0, 4.0, 6.0, 8.0, 10.0]
    y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]

    pos_res = CorrelationAnalyzer.compute_pearson(x, y_pos)
    assert pytest.approx(pos_res["coefficient"], rel=1e-3) == 1.0
    assert "Positive" in pos_res["interpretation"]

    neg_res = CorrelationAnalyzer.compute_pearson(x, y_neg)
    assert pytest.approx(neg_res["coefficient"], rel=1e-3) == -1.0
    assert "Negative" in neg_res["interpretation"]


def test_compute_spearman_monotonic_relationship():
    """Verify Spearman rank correlation handles non-linear monotonic relationships."""
    # Exponential growth: rank is identical
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 100.0, 1000.0, 10000.0, 100000.0]

    spearman = CorrelationAnalyzer.compute_spearman(x, y)
    assert pytest.approx(spearman["coefficient"], rel=1e-3) == 1.0
    assert spearman["method"] == "spearman"


def test_correlation_zero_variance_and_small_samples():
    """Verify zero variance and sample sizes < 2 return 0.0 with helpful interpretation."""
    res_single = CorrelationAnalyzer.compute_pearson([1.0], [2.0])
    assert res_single["coefficient"] == 0.0
    assert "Insufficient data" in res_single["interpretation"]

    res_flat = CorrelationAnalyzer.compute_pearson([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
    assert res_flat["coefficient"] == 0.0
    assert "Zero variance" in res_flat["interpretation"]


def test_compute_correlation_matrix():
    """Verify multi-feature correlation matrix calculation."""
    data = {
        "retrieval_score": [0.8, 0.85, 0.9, 0.95],
        "latency_ms": [150.0, 160.0, 170.0, 180.0],
        "citations": [2.0, 3.0, 4.0, 5.0],
    }

    matrix_res = CorrelationAnalyzer.compute_correlation_matrix(data, method="pearson")
    assert matrix_res["features"] == ["retrieval_score", "latency_ms", "citations"]
    mat = matrix_res["matrix"]
    assert len(mat) == 3
    # Diagonal should be 1.0
    assert mat[0][0] == 1.0
    assert mat[1][1] == 1.0
    assert mat[2][2] == 1.0


def test_analyze_research_relationships_includes_causation_warning():
    """Verify research relationship analysis explicitly includes causation disclaimer."""
    records = [
        {"retrieval_score": 0.9, "citation_validity": 1.0, "document_size_kb": 500, "processing_duration_sec": 2.0, "context_token_count": 800, "answer_latency_ms": 1200},
        {"retrieval_score": 0.6, "citation_validity": 0.0, "document_size_kb": 1200, "processing_duration_sec": 4.5, "context_token_count": 1500, "answer_latency_ms": 2100},
        {"retrieval_score": 0.85, "citation_validity": 1.0, "document_size_kb": 800, "processing_duration_sec": 3.2, "context_token_count": 1100, "answer_latency_ms": 1600},
    ]

    relationships = CorrelationAnalyzer.analyze_research_relationships(records)

    assert "retrieval_score__vs__citation_validity" in relationships
    assert "document_size_kb__vs__processing_duration_sec" in relationships
    assert "context_token_count__vs__answer_latency_ms" in relationships

    for k, v in relationships.items():
        assert "causation_disclaimer" in v
        assert "does not imply direct causation" in v["causation_disclaimer"]
