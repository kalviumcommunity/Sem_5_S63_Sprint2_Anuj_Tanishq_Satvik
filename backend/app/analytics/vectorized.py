"""NumPy vectorized computation workflow for ResearchMate analytics and metrics."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class VectorizedAnalytics:
    """High-performance vectorized computations for retrieval scores, latencies, and evaluations."""

    @staticmethod
    def batch_cosine_similarities(
        query_vectors: np.ndarray,
        candidate_vectors: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise or batch cosine similarities using vectorized dot product and norms.
        
        Supports:
        - query_vectors of shape (D,) vs candidates of shape (N, D) -> returns (N,)
        - query_vectors of shape (M, D) vs candidates of shape (M, D) -> returns (M,)
        - query_vectors of shape (M, D) vs candidates of shape (N, D) -> returns (M, N)
        """
        q = np.asarray(query_vectors, dtype=np.float32)
        c = np.asarray(candidate_vectors, dtype=np.float32)

        if q.ndim == 1:
            q = q[np.newaxis, :]
            single_query = True
        else:
            single_query = False

        if c.ndim == 1:
            c = c[np.newaxis, :]

        # Normalize rows to unit length with safe epsilon to avoid division by zero
        q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
        c_norm = np.linalg.norm(c, axis=-1, keepdims=True)

        q_safe = np.nan_to_num(q / np.where(q_norm == 0, 1e-12, q_norm), nan=0.0)
        c_safe = np.nan_to_num(c / np.where(c_norm == 0, 1e-12, c_norm), nan=0.0)

        if single_query:
            # Result shape (N,)
            sims = np.dot(c_safe, q_safe.T).squeeze(-1)
            return np.clip(sims, -1.0, 1.0)
        else:
            sims = np.dot(q_safe, c_safe.T)
            return np.clip(sims, -1.0, 1.0)

    @staticmethod
    def compute_latency_percentiles(
        latencies_ms: Union[List[float], np.ndarray],
    ) -> Dict[str, float]:
        """Compute latency statistics and percentiles (p50, p90, p95, p99) avoiding loops."""
        arr = np.asarray(latencies_ms, dtype=np.float64)
        valid = arr[~np.isnan(arr)]

        if len(valid) == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
            }

        p50, p90, p95, p99 = np.percentile(valid, [50, 90, 95, 99])
        return {
            "count": int(len(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "p50": float(p50),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
            "max": float(np.max(valid)),
        }

    @staticmethod
    def normalize_scores_minmax(
        scores: Union[List[float], np.ndarray],
    ) -> np.ndarray:
        """Min-Max normalize retrieval scores to [0.0, 1.0] with zero-spread safety."""
        arr = np.asarray(scores, dtype=np.float64)
        if len(arr) == 0:
            return np.array([], dtype=np.float64)

        clean = np.nan_to_num(arr, nan=0.0)
        min_val = np.min(clean)
        max_val = np.max(clean)

        diff = max_val - min_val
        if diff == 0.0:
            return np.ones_like(clean, dtype=np.float64) if max_val > 0 else np.zeros_like(clean, dtype=np.float64)

        return (clean - min_val) / diff

    @staticmethod
    def compute_mrr_and_hit_rate(
        rankings: np.ndarray,
        k: int = 5,
    ) -> Dict[str, float]:
        """Vectorized computation of Mean Reciprocal Rank (MRR) and Hit Rate @ K.
        
        Args:
            rankings: 2D boolean array of shape (num_queries, num_retrieved), where
                      True indicates a relevant ground-truth chunk.
            k: Top-k cutoff.
        """
        arr = np.asarray(rankings, dtype=bool)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return {"hit_rate": 0.0, "mrr": 0.0, "query_count": 0}

        top_k = arr[:, :k]
        hits = np.any(top_k, axis=1)
        hit_rate = float(np.mean(hits))

        # Find first hit index (1-indexed) for each query
        has_hit = np.any(top_k, axis=1)
        # argmax finds index of first True; if no True, returns 0
        first_hit_idx = np.argmax(top_k, axis=1) + 1
        reciprocal_ranks = np.where(has_hit, 1.0 / first_hit_idx, 0.0)
        mrr = float(np.mean(reciprocal_ranks))

        return {
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "query_count": int(arr.shape[0]),
        }

    @staticmethod
    def compute_ndcg_at_k(
        relevance_matrix: np.ndarray,
        k: int = 5,
    ) -> float:
        """Vectorized Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            relevance_matrix: (num_queries, num_retrieved) numeric relevance labels (0, 1, 2, ...).
        """
        rel = np.asarray(relevance_matrix, dtype=np.float64)
        if rel.ndim != 2 or rel.shape[0] == 0 or rel.shape[1] == 0:
            return 0.0

        rel_k = rel[:, :k]
        num_queries, actual_k = rel_k.shape

        # Discount: log2(rank + 1) -> 1-indexed ranks 1, 2, ..., actual_k
        discounts = 1.0 / np.log2(np.arange(1, actual_k + 1) + 1)

        # DCG
        dcg = np.sum((2.0 ** rel_k - 1.0) * discounts, axis=1)

        # IDCG: sort relevance descending for ideal ranking
        ideal_rel = np.sort(rel, axis=1)[:, ::-1][:, :actual_k]
        idcg = np.sum((2.0 ** ideal_rel - 1.0) * discounts, axis=1)

        ndcg_per_query = np.where(idcg > 0, dcg / idcg, 0.0)
        return float(np.mean(ndcg_per_query))

    @staticmethod
    def benchmark_vectorization_speedup(num_elements: int = 10000) -> Dict[str, Any]:
        """Benchmark comparison: vectorized NumPy vs iterative Python loop."""
        data_a = np.random.randn(num_elements).astype(np.float32)
        data_b = np.random.randn(num_elements).astype(np.float32)
        list_a = data_a.tolist()
        list_b = data_b.tolist()

        # Iterative Python loop
        t0 = time.perf_counter()
        loop_res = [a * b for a, b in zip(list_a, list_b)]
        loop_time = time.perf_counter() - t0

        # Vectorized NumPy
        t1 = time.perf_counter()
        vec_res = data_a * data_b
        vec_time = time.perf_counter() - t1

        speedup = (loop_time / max(vec_time, 1e-9)) if vec_time > 0 else 1.0
        return {
            "num_elements": num_elements,
            "loop_duration_sec": loop_time,
            "vectorized_duration_sec": vec_time,
            "speedup_factor": round(speedup, 2),
        }
