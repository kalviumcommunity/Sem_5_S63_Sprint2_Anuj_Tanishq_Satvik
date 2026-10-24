"""Metadata filtering for retrieval candidates."""

from typing import Any, Dict, List
from backend.app.services.retrieval.search import SearchResult


class MetadataFilter:
    """Filters search candidates by metadata attributes, tags, and range constraints."""

    @classmethod
    def apply(cls, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply key-value filter constraints with operator support ($eq, $ne, $in, $gte, $lte)."""
        if not filters:
            return results

        filtered: List[SearchResult] = []
        for res in results:
            if cls._matches(res, filters):
                filtered.append(res)
        return filtered

    @classmethod
    def _matches(cls, result: SearchResult, filters: Dict[str, Any]) -> bool:
        for key, expected in filters.items():
            # Check direct attribute on SearchResult or in metadata dict
            actual = getattr(result, key, None)
            if actual is None:
                actual = result.metadata.get(key)

            if isinstance(expected, dict):
                # Operator dict, e.g. {"$gte": 2, "$lte": 5}
                for op, val in expected.items():
                    if op == "$eq" and actual != val:
                        return False
                    elif op == "$ne" and actual == val:
                        return False
                    elif op == "$in" and (actual not in val):
                        return False
                    elif op == "$nin" and (actual in val):
                        return False
                    elif op == "$gt" and (actual is None or actual <= val):
                        return False
                    elif op == "$gte" and (actual is None or actual < val):
                        return False
                    elif op == "$lt" and (actual is None or actual >= val):
                        return False
                    elif op == "$lte" and (actual is None or actual > val):
                        return False
            else:
                if actual is None or actual != expected:
                    return False

        return True
