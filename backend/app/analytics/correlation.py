"""Correlation and relationship analysis for ResearchMate workflow metrics."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class CorrelationAnalyzer:
    """Computes bivariate correlations (Pearson & Spearman) and generates relationship insights."""

    CAUSATION_DISCLAIMER = (
        "Statistical correlation does not imply direct causation. Relationships may be influenced "
        "by confounding variables (e.g. query difficulty, document structure, system load)."
    )

    @classmethod
    def compute_pearson(
        cls,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
    ) -> Dict[str, Any]:
        """Compute Pearson linear correlation coefficient r."""
        arr_x = np.asarray(x, dtype=np.float64)
        arr_y = np.asarray(y, dtype=np.float64)

        mask = ~(np.isnan(arr_x) | np.isnan(arr_y))
        x_clean = arr_x[mask]
        y_clean = arr_y[mask]

        n = len(x_clean)
        if n < 2:
            return {
                "coefficient": 0.0,
                "method": "pearson",
                "sample_size": n,
                "interpretation": "Insufficient data",
                "disclaimer": cls.CAUSATION_DISCLAIMER,
            }

        std_x = np.std(x_clean)
        std_y = np.std(y_clean)

        if std_x == 0.0 or std_y == 0.0:
            return {
                "coefficient": 0.0,
                "method": "pearson",
                "sample_size": n,
                "interpretation": "Zero variance in one or both variables",
                "disclaimer": cls.CAUSATION_DISCLAIMER,
            }

        cov = np.cov(x_clean, y_clean)[0, 1]
        r = float(cov / (std_x * std_y * (n / (n - 1))))
        r_bounded = max(-1.0, min(1.0, r))

        return {
            "coefficient": round(r_bounded, 4),
            "method": "pearson",
            "sample_size": n,
            "interpretation": cls._interpret_strength(r_bounded),
            "disclaimer": cls.CAUSATION_DISCLAIMER,
        }

    @classmethod
    def compute_spearman(
        cls,
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
    ) -> Dict[str, Any]:
        """Compute Spearman rank-order correlation coefficient rho."""
        arr_x = np.asarray(x, dtype=np.float64)
        arr_y = np.asarray(y, dtype=np.float64)

        mask = ~(np.isnan(arr_x) | np.isnan(arr_y))
        x_clean = arr_x[mask]
        y_clean = arr_y[mask]

        n = len(x_clean)
        if n < 2:
            return {
                "coefficient": 0.0,
                "method": "spearman",
                "sample_size": n,
                "interpretation": "Insufficient data",
                "disclaimer": cls.CAUSATION_DISCLAIMER,
            }

        # Rank conversion
        rank_x = np.argsort(np.argsort(x_clean)).astype(np.float64) + 1.0
        rank_y = np.argsort(np.argsort(y_clean)).astype(np.float64) + 1.0

        res = cls.compute_pearson(rank_x, rank_y)
        res["method"] = "spearman"
        return res

    @classmethod
    def compute_correlation_matrix(
        cls,
        data_dict: Dict[str, List[float]],
        method: str = "pearson",
    ) -> Dict[str, Any]:
        """Compute full pairwise correlation matrix across multiple features."""
        features = list(data_dict.keys())
        n_feats = len(features)
        matrix = np.eye(n_feats, dtype=np.float64)

        correlate_fn = cls.compute_pearson if method == "pearson" else cls.compute_spearman

        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                feat_a = features[i]
                feat_b = features[j]
                res = correlate_fn(data_dict[feat_a], data_dict[feat_b])
                matrix[i, j] = res["coefficient"]
                matrix[j, i] = res["coefficient"]

        return {
            "features": features,
            "matrix": matrix.tolist(),
            "method": method,
        }

    @classmethod
    def analyze_research_relationships(
        cls,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Investigate standard ResearchMate relationships:
        1. retrieval_score vs citation_validity
        2. document_size_kb vs processing_duration_sec
        3. context_token_count vs answer_latency_ms
        """
        pairs = [
            ("retrieval_score", "citation_validity", "pearson"),
            ("document_size_kb", "processing_duration_sec", "pearson"),
            ("context_token_count", "answer_latency_ms", "spearman"),
        ]

        insights: Dict[str, Any] = {}

        for var_a, var_b, preferred_method in pairs:
            vals_a: List[float] = []
            vals_b: List[float] = []

            for r in records:
                va = r.get(var_a)
                vb = r.get(var_b)
                if va is not None and vb is not None:
                    try:
                        vals_a.append(float(va))
                        vals_b.append(float(vb))
                    except (ValueError, TypeError):
                        pass

            if preferred_method == "spearman":
                corr_res = cls.compute_spearman(vals_a, vals_b)
            else:
                corr_res = cls.compute_pearson(vals_a, vals_b)

            pair_key = f"{var_a}__vs__{var_b}"
            insights[pair_key] = {
                "variable_x": var_a,
                "variable_y": var_b,
                "correlation": corr_res["coefficient"],
                "method": corr_res["method"],
                "sample_size": corr_res["sample_size"],
                "interpretation": corr_res["interpretation"],
                "causation_disclaimer": cls.CAUSATION_DISCLAIMER,
            }

        return insights

    @staticmethod
    def _interpret_strength(r: float) -> str:
        abs_r = abs(r)
        direction = "Positive" if r >= 0 else "Negative"
        if abs_r >= 0.8:
            return f"Strong {direction}"
        elif abs_r >= 0.5:
            return f"Moderate {direction}"
        elif abs_r >= 0.3:
            return f"Weak {direction}"
        return "Negligible / No linear association"
