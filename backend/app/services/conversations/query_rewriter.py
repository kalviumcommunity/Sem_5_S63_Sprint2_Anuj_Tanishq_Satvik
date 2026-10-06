"""Conversational query rewriting for multi-turn research queries."""

from typing import List
from backend.app.models.conversation import Message


class QueryRewriter:
    """Rewrites conversational follow-up questions into standalone academic search queries."""

    @staticmethod
    def rewrite(query: str, history: List[Message]) -> str:
        """Combine conversation context with follow-up question if pronouns/references exist."""
        if not history:
            return query

        # Check for pronouns indicating context dependency
        pronouns = ["it", "this", "these", "its", "they", "them", "that"]
        query_words = query.lower().split()

        needs_context = any(p in query_words for p in pronouns) or len(query_words) <= 3

        if needs_context:
            last_user_msg = next((m for m in reversed(history) if m.role.value == "user"), None)
            if last_user_msg:
                return f"{last_user_msg.content} - {query}"

        return query

    @classmethod
    def build_rewrite_prompt(cls, query: str, history: List[Message]):
        """Render query rewriting prompt bundle using canonical template."""
        from backend.app.services.llm.templates import QUERY_REWRITING_TEMPLATE

        history_lines = [f"{m.role.value}: {m.content}" for m in history]
        history_str = "\n".join(history_lines) if history_lines else "None"

        return QUERY_REWRITING_TEMPLATE.render_bundle({
            "conversation_history": history_str,
            "follow_up_question": query,
        })

