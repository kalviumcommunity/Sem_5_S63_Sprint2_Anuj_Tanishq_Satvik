"""Text extraction interface and multi-format implementations preserving metadata."""

import re
from typing import Any, Dict, List, Optional


class TextExtractor:
    """Extracts raw text and metadata (pages, sections, offsets) from document contents."""

    @staticmethod
    def extract_from_text(raw_text: str) -> List[Dict[str, Any]]:
        """Extract pages from plain text content preserving character boundaries."""
        if not raw_text:
            return []

        pages = raw_text.split("\f")  # Form feed represents page breaks
        results: List[Dict[str, Any]] = []
        current_offset = 0

        for idx, page in enumerate(pages):
            page_len = len(page)
            char_start = current_offset
            char_end = current_offset + page_len

            if page.strip():
                results.append({
                    "page_number": idx + 1,
                    "text": page,
                    "char_start": char_start,
                    "char_end": char_end,
                    "char_offset_base": char_start,
                })

            # Advance offset past the page and the delimiter
            current_offset += page_len + 1  # 1 for '\f'

        return results

    @staticmethod
    def extract_from_markdown(markdown_text: str) -> List[Dict[str, Any]]:
        """Extract sections and character offsets from structured Markdown documents."""
        if not markdown_text:
            return []

        # Find heading markers (# Heading)
        heading_matches = list(re.finditer(r"^(#{1,6})\s+(.+)$", markdown_text, flags=re.MULTILINE))
        if not heading_matches:
            # Single section document
            return [{
                "section": "Overview",
                "text": markdown_text,
                "char_start": 0,
                "char_end": len(markdown_text),
                "char_offset_base": 0,
                "page_number": 1,
            }]

        sections: List[Dict[str, Any]] = []

        # Text before the first heading
        if heading_matches[0].start() > 0:
            preamble = markdown_text[: heading_matches[0].start()]
            if preamble.strip():
                sections.append({
                    "section": "Preamble",
                    "text": preamble,
                    "char_start": 0,
                    "char_end": heading_matches[0].start(),
                    "char_offset_base": 0,
                    "page_number": 1,
                })

        for i, match in enumerate(heading_matches):
            sec_title = match.group(2).strip()
            start_pos = match.start()
            end_pos = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(markdown_text)
            sec_text = markdown_text[start_pos:end_pos]

            sections.append({
                "section": sec_title,
                "text": sec_text,
                "char_start": start_pos,
                "char_end": end_pos,
                "char_offset_base": start_pos,
                "page_number": 1,
            })

        return sections

    @staticmethod
    def extract_from_html(html_text: str, url: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract plain text from web documents preserving canonical source URL."""
        if not html_text:
            return []

        # Remove script and style tags
        cleaned = re.sub(r"<(script|style)[^>]*>[\s\S]*?</\1>", "", html_text, flags=re.IGNORECASE)
        # Strip HTML tags
        plain = re.sub(r"<[^>]+>", " ", cleaned)
        plain = re.sub(r"\s+", " ", plain).strip()

        return [{
            "text": plain,
            "section": "Web Content",
            "url": url,
            "char_start": 0,
            "char_end": len(plain),
            "char_offset_base": 0,
            "page_number": 1,
        }]
