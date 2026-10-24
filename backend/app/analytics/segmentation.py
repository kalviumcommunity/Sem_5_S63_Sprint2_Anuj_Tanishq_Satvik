"""GroupBy aggregation and segment insights for ResearchMate analytics."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class GroupBySegmenter:
    """Computes multi-dimensional aggregated metrics across document formats, query categories, and statuses."""

    SEGMENT_DEFINITIONS = {
        "document_type": "Segment based on file format (pdf, docx, txt, markdown) to assess format-specific parsing & chunking.",
        "query_category": "Categorization of academic queries (factual, exploratory, comparative, summarization).",
        "processing_status": "Document ingestion lifecycle status (uploaded, processing, indexed, failed).",
    }

    @classmethod
    def aggregate_by_document_type(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group records by document_type and compute processing duration, chunk counts, and success rates."""
        if not records:
            return []

        df = pd.DataFrame(records)
        required_cols = ["document_type"]
        for col in required_cols:
            if col not in df.columns:
                return []

        # Ensure numeric columns
        for num_col in ["processing_duration_sec", "chunk_count", "char_count", "is_success"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            else:
                df[num_col] = np.nan

        grouped = df.groupby("document_type").agg(
            record_count=("document_type", "count"),
            avg_processing_duration_sec=("processing_duration_sec", "mean"),
            avg_chunk_count=("chunk_count", "mean"),
            avg_char_count=("char_count", "mean"),
            success_rate=("is_success", "mean"),
        ).reset_index()

        # Fill NaNs with sensible defaults for JSON serialization
        grouped = grouped.fillna({
            "avg_processing_duration_sec": 0.0,
            "avg_chunk_count": 0.0,
            "avg_char_count": 0.0,
            "success_rate": 0.0,
        })

        results = []
        for _, row in grouped.iterrows():
            results.append({
                "document_type": str(row["document_type"]),
                "record_count": int(row["record_count"]),
                "avg_processing_duration_sec": round(float(row["avg_processing_duration_sec"]), 3),
                "avg_chunk_count": round(float(row["avg_chunk_count"]), 1),
                "avg_char_count": round(float(row["avg_char_count"]), 1),
                "success_rate": round(float(row["success_rate"]), 4),
            })

        return results

    @classmethod
    def aggregate_by_query_category(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group queries by category and calculate latency, retrieval scores, citation validity, and success rates."""
        if not records:
            return []

        df = pd.DataFrame(records)
        if "query_category" not in df.columns:
            return []

        for num_col in ["latency_ms", "retrieval_score", "citation_validity", "query_success"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            else:
                df[num_col] = np.nan

        grouped = df.groupby("query_category").agg(
            query_count=("query_category", "count"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_retrieval_score=("retrieval_score", "mean"),
            citation_validity_rate=("citation_validity", "mean"),
            query_success_rate=("query_success", "mean"),
        ).reset_index()

        grouped = grouped.fillna({
            "avg_latency_ms": 0.0,
            "avg_retrieval_score": 0.0,
            "citation_validity_rate": 0.0,
            "query_success_rate": 0.0,
        })

        results = []
        for _, row in grouped.iterrows():
            results.append({
                "query_category": str(row["query_category"]),
                "query_count": int(row["query_count"]),
                "avg_latency_ms": round(float(row["avg_latency_ms"]), 2),
                "avg_retrieval_score": round(float(row["avg_retrieval_score"]), 4),
                "citation_validity_rate": round(float(row["citation_validity_rate"]), 4),
                "query_success_rate": round(float(row["query_success_rate"]), 4),
            })

        return results

    @classmethod
    def aggregate_by_status(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group processing records by processing_status and compute counts and distribution percentages."""
        if not records:
            return []

        df = pd.DataFrame(records)
        if "processing_status" not in df.columns:
            return []

        total_records = len(df)
        grouped = df.groupby("processing_status").size().reset_index(name="count")
        grouped["percentage"] = (grouped["count"] / total_records) * 100.0

        results = []
        for _, row in grouped.iterrows():
            results.append({
                "processing_status": str(row["processing_status"]),
                "count": int(row["count"]),
                "percentage": round(float(row["percentage"]), 2),
            })

        return results
