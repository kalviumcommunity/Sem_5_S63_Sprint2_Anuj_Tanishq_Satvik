"""Metadata filtering for retrieval candidates."""

from typing import Any, Dict, List
from backend.app.services.retrieval.search import SearchResult


class MetadataFilter:
    """Filters search candidates by metadata attributes and document attributes."""

    @staticmethod
    def _extract_candidate_value(res: SearchResult, key: str) -> Any:
        """Extract value for filter key from metadata or top-level properties."""
        val = res.metadata.get(key)
        if val is not None:
            return val

        # Check top-level properties
        if hasattr(res, key):
            return getattr(res, key)

        # Canonical aliases
        if key == "source":
            return res.metadata.get("source") or res.document_title or res.document_id
        if key == "document_id":
            return res.document_id or res.metadata.get("document_id")
        if key == "document_title":
            return res.document_title or res.metadata.get("document_title")

        return None

    @classmethod
    def apply(cls, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply key-value filter constraints with type tolerance and list support."""
        if not filters:
            return results

        filtered: List[SearchResult] = []
        for res in results:
            match = True
            for key, expected_val in filters.items():
                if expected_val is None:
                    continue

                cand_val = cls._extract_candidate_value(res, key)

                if cand_val is None:
                    match = False
                    break

                # Support list/tuple/set of accepted filter values
                if isinstance(expected_val, (list, tuple, set)):
                    if cand_val not in expected_val:
                        match = False
                        break
                else:
                    if cand_val != expected_val:
                        match = False
                        break

            if match:
                filtered.append(res)

        return filtered
