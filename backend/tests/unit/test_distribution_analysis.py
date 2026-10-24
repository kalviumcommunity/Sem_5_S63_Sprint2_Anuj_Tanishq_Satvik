"""Unit tests for Concept #18: Distribution Analysis for Business Trends."""

import math
import numpy as np
import pytest
from backend.app.analytics.distribution import DistributionAnalyzer


def test_distribution_summary_standard_dataset():
    """Verify statistical properties (mean, median, quartiles, IQR, skewness) on known dataset."""
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    summary = DistributionAnalyzer.compute_distribution_summary(values, metric_name="latency")

    assert summary["metric_name"] == "latency"
    assert summary["count"] == 7
    assert summary["mean"] == 40.0
    assert summary["median"] == 40.0
    assert summary["min"] == 10.0
    assert summary["max"] == 70.0
    assert summary["q1"] == 25.0
    assert summary["q3"] == 55.0
    assert summary["iqr"] == 30.0
    assert summary["skewness"] == 0.0


def test_distribution_summary_empty_and_nan():
    """Verify empty and all-NaN inputs return zeroed summary gracefully."""
    empty_summary = DistributionAnalyzer.compute_distribution_summary([])
    assert empty_summary["count"] == 0
    assert empty_summary["mean"] == 0.0

    nan_summary = DistributionAnalyzer.compute_distribution_summary([float("nan"), float("nan")])
    assert nan_summary["count"] == 0


def test_distribution_summary_skewed_data():
    """Verify positive skewness detection on right-tailed distribution."""
    # Data with high positive tail
    values = [10.0, 11.0, 12.0, 12.0, 13.0, 14.0, 200.0]
    summary = DistributionAnalyzer.compute_distribution_summary(values, metric_name="query_latency")

    assert summary["mean"] > summary["median"]
    assert summary["skewness"] > 0.0


def test_generate_histogram_bins():
    """Verify histogram bins, intervals, and count frequencies."""
    values = list(range(100))
    hist = DistributionAnalyzer.generate_histogram_bins(values, num_bins=5)

    assert hist["total_samples"] == 100
    assert len(hist["bins"]) == 5
    assert sum(hist["counts"]) == 100
    for b in hist["bins"]:
        assert b["bin_start"] < b["bin_end"]
        assert b["count"] == 20
        assert b["frequency"] == 0.2


def test_analyze_workflow_distributions_end_to_end():
    """Verify workflow records distribution extraction across all 5 key metrics."""
    records = [
        {
            "query_latency_ms": 150.0,
            "processing_duration_sec": 2.5,
            "chunk_count": 24,
            "retrieval_score": 0.88,
            "citation_coverage": 1.0,
        },
        {
            "query_latency_ms": 220.0,
            "processing_duration_sec": 3.1,
            "chunk_count": 30,
            "retrieval_score": 0.72,
            "citation_coverage": 0.8,
        },
        {
            "query_latency_ms": 180.0,
            "processing_duration_sec": 2.8,
            "chunk_count": 28,
            "retrieval_score": 0.91,
            "citation_coverage": 0.9,
        },
    ]

    analysis = DistributionAnalyzer.analyze_workflow_distributions(records)

    assert "query_latency_ms" in analysis
    assert "processing_duration_sec" in analysis
    assert "chunk_count" in analysis
    assert "retrieval_score" in analysis
    assert "citation_coverage" in analysis

    lat_summary = analysis["query_latency_ms"]["summary"]
    assert lat_summary["count"] == 3
    assert lat_summary["min"] == 150.0
    assert lat_summary["max"] == 220.0
