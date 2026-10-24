"""Unit tests for Concept #20: GroupBy Aggregation & Segment Insights."""

import pytest
from backend.app.analytics.segmentation import GroupBySegmenter


def test_aggregate_by_document_type():
    """Verify grouping by document_type yields accurate counts, averages, and success rates."""
    records = [
        {"document_type": "pdf", "processing_duration_sec": 3.0, "chunk_count": 20, "char_count": 5000, "is_success": 1.0},
        {"document_type": "pdf", "processing_duration_sec": 5.0, "chunk_count": 40, "char_count": 9000, "is_success": 1.0},
        {"document_type": "docx", "processing_duration_sec": 1.5, "chunk_count": 15, "char_count": 3000, "is_success": 1.0},
        {"document_type": "docx", "processing_duration_sec": 2.5, "chunk_count": 25, "char_count": 5000, "is_success": 0.0},
        {"document_type": "txt", "processing_duration_sec": 0.5, "chunk_count": 5, "char_count": 1000, "is_success": 1.0},
    ]

    results = GroupBySegmenter.aggregate_by_document_type(records)
    by_type = {r["document_type"]: r for r in results}

    assert len(by_type) == 3
    # PDF: 2 records, avg duration (3+5)/2 = 4.0, success_rate 1.0
    assert by_type["pdf"]["record_count"] == 2
    assert by_type["pdf"]["avg_processing_duration_sec"] == 4.0
    assert by_type["pdf"]["avg_chunk_count"] == 30.0
    assert by_type["pdf"]["success_rate"] == 1.0

    # DOCX: 2 records, avg duration 2.0, success_rate 0.5
    assert by_type["docx"]["record_count"] == 2
    assert by_type["docx"]["success_rate"] == 0.5

    # TXT: 1 record
    assert by_type["txt"]["record_count"] == 1
    assert by_type["txt"]["success_rate"] == 1.0


def test_aggregate_by_query_category():
    """Verify query categorization metrics (latency, retrieval scores, validity)."""
    records = [
        {"query_category": "factual", "latency_ms": 120.0, "retrieval_score": 0.92, "citation_validity": 1.0, "query_success": 1.0},
        {"query_category": "factual", "latency_ms": 140.0, "retrieval_score": 0.88, "citation_validity": 1.0, "query_success": 1.0},
        {"query_category": "comparative", "latency_ms": 350.0, "retrieval_score": 0.75, "citation_validity": 0.5, "query_success": 1.0},
        {"query_category": "summarization", "latency_ms": 500.0, "retrieval_score": 0.65, "citation_validity": 0.0, "query_success": 0.0},
    ]

    results = GroupBySegmenter.aggregate_by_query_category(records)
    by_cat = {r["query_category"]: r for r in results}

    assert len(by_cat) == 3
    assert by_cat["factual"]["query_count"] == 2
    assert by_cat["factual"]["avg_latency_ms"] == 130.0
    assert by_cat["factual"]["citation_validity_rate"] == 1.0

    assert by_cat["comparative"]["avg_latency_ms"] == 350.0
    assert by_cat["summarization"]["query_success_rate"] == 0.0


def test_aggregate_by_status():
    """Verify lifecycle status distribution percentages sum to 100%."""
    records = [
        {"processing_status": "indexed"},
        {"processing_status": "indexed"},
        {"processing_status": "indexed"},
        {"processing_status": "failed"},
    ]

    results = GroupBySegmenter.aggregate_by_status(records)
    by_status = {r["processing_status"]: r for r in results}

    assert by_status["indexed"]["count"] == 3
    assert by_status["indexed"]["percentage"] == 75.0
    assert by_status["failed"]["count"] == 1
    assert by_status["failed"]["percentage"] == 25.0


def test_groupby_empty_and_missing_columns():
    """Verify empty inputs or missing grouping keys return empty lists cleanly."""
    assert GroupBySegmenter.aggregate_by_document_type([]) == []
    assert GroupBySegmenter.aggregate_by_query_category([]) == []
    assert GroupBySegmenter.aggregate_by_status([]) == []

    # Records missing the group column
    records = [{"some_random_key": 123}]
    assert GroupBySegmenter.aggregate_by_document_type(records) == []
    assert GroupBySegmenter.aggregate_by_query_category(records) == []
    assert GroupBySegmenter.aggregate_by_status(records) == []
