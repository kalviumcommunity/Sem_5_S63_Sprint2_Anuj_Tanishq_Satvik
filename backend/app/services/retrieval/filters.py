"""Metadata filtering for retrieval candidates."""

from typing import Any, Dict, List
from backend.app.services.retrieval.search import SearchResult


class MetadataFilter:
    """Filters search candidates by metadata attributes."""

    @staticmethod
    def apply(results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply key-value filter constraints."""
        if not filters:
            return results

        filtered = []
        for res in results:
            match = True
            for key, val in filters.items():
                res_val = res.metadata.get(key)
                if res_val is None or res_val != val:
                    match = False
                    break
            if match:
                filtered.append(res)
        return filtered
