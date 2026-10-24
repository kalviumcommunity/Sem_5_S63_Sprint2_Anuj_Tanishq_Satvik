"""Distribution analysis for ResearchMate workflow and performance metrics."""

from typing import Any, Dict, List, Optional, Union
import numpy as np


class DistributionAnalyzer:
    """Calculates statistical distributions, quartiles, and histograms for research metrics."""

    @staticmethod
    def compute_distribution_summary(
        values: Union[List[float], np.ndarray],
        metric_name: str = "metric",
    ) -> Dict[str, Any]:
        """Compute statistical summary (mean, std, quartiles, IQR, skewness) for a numeric series."""
        arr = np.asarray(values, dtype=np.float64)
        clean = arr[~np.isnan(arr)]

        if len(clean) == 0:
            return {
                "metric_name": metric_name,
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "q1": 0.0,
                "median": 0.0,
                "q3": 0.0,
                "max": 0.0,
                "iqr": 0.0,
                "skewness": 0.0,
            }

        mean = float(np.mean(clean))
        std = float(np.std(clean))
        variance = float(np.var(clean))
        min_val = float(np.min(clean))
        max_val = float(np.max(clean))

        q1, median, q3 = np.percentile(clean, [25, 50, 75])
        iqr = float(q3 - q1)

        # Skewness calculation: 3 * (mean - median) / std (Pearson's second skewness coefficient)
        skewness = float(3.0 * (mean - median) / std) if std > 0 else 0.0

        return {
            "metric_name": metric_name,
            "count": int(len(clean)),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "variance": round(variance, 4),
            "min": round(min_val, 4),
            "q1": round(float(q1), 4),
            "median": round(float(median), 4),
            "q3": round(float(q3), 4),
            "max": round(max_val, 4),
            "iqr": round(iqr, 4),
            "skewness": round(skewness, 4),
        }

    @staticmethod
    def generate_histogram_bins(
        values: Union[List[float], np.ndarray],
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """Generate binned counts and bin intervals for histogram visualization."""
        arr = np.asarray(values, dtype=np.float64)
        clean = arr[~np.isnan(arr)]

        if len(clean) == 0 or num_bins <= 0:
            return {"bins": [], "counts": [], "total_samples": 0}

        counts, bin_edges = np.histogram(clean, bins=num_bins)

        bins_summary = []
        for i in range(len(counts)):
            bins_summary.append({
                "bin_start": round(float(bin_edges[i]), 4),
                "bin_end": round(float(bin_edges[i + 1]), 4),
                "count": int(counts[i]),
                "frequency": round(float(counts[i] / len(clean)), 4),
            })

        return {
            "bins": bins_summary,
            "counts": [int(c) for c in counts],
            "total_samples": int(len(clean)),
        }

    @classmethod
    def analyze_workflow_distributions(
        cls,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract and analyze distributions across core research metrics:
        - query_latency_ms
        - processing_duration_sec
        - chunk_count
        - retrieval_score
        - citation_coverage
        """
        keys = [
            "query_latency_ms",
            "processing_duration_sec",
            "chunk_count",
            "retrieval_score",
            "citation_coverage",
        ]

        metric_data: Dict[str, List[float]] = {k: [] for k in keys}

        for r in records:
            for k in keys:
                val = r.get(k)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        metric_data[k].append(float(val))
                    except (ValueError, TypeError):
                        pass

        results: Dict[str, Any] = {}
        for k in keys:
            summary = cls.compute_distribution_summary(metric_data[k], metric_name=k)
            hist = cls.generate_histogram_bins(metric_data[k], num_bins=8)
            results[k] = {
                "summary": summary,
                "histogram": hist,
            }

        return results
