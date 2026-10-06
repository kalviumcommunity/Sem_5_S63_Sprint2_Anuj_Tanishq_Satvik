"""Conversation context manager and history budget management.

Maintains conversation history, applies token and turn budgets to prevent
context window overflow, and supports conversational follow-up questions.
"""

from typing import Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.models.conversation import Conversation, Message, MessageRole
from backend.app.services.conversations.query_rewriter import QueryRewriter
from backend.app.services.llm.cost import token_cost_estimator
from backend.app.services.llm.prompts import (
    PromptBuilder,
    PromptBundle,
    RESEARCHMATE_SYSTEM_PROMPT,
    build_academic_rag_prompt,
)


class ConversationContextManager:
    """Manages multi-turn conversation sessions and context window budgets."""

    def __init__(
        self,
        max_turns: Optional[int] = None,
        max_history_tokens: Optional[int] = None,
        max_context_window_tokens: Optional[int] = None,
    ):
        self.max_turns = max_turns or settings.MAX_HISTORY_TURNS
        self.max_history_tokens = max_history_tokens or settings.MAX_HISTORY_TOKENS
        self.max_context_window_tokens = (
            max_context_window_tokens or settings.MAX_CONTEXT_WINDOW_TOKENS
        )
        self._conversations: Dict[str, Conversation] = {}
        self._messages: Dict[str, List[Message]] = {}
        self.query_rewriter = QueryRewriter()

    def create_conversation(self, title: str = "New Inquiry") -> Conversation:
        """Initialize a new conversation session."""
        conv = Conversation(title=title)
        self._conversations[conv.id] = conv
        self._messages[conv.id] = []
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Fetch conversation metadata by ID."""
        return self._conversations.get(conv_id)

    def add_message(self, conv_id: str, role: MessageRole, content: str) -> Message:
        """Record a new message turn in the conversation thread."""
        if conv_id not in self._conversations:
            self.create_conversation()
        msg = Message(conversation_id=conv_id, role=role, content=content)
        self._messages.setdefault(conv_id, []).append(msg)
        return msg

    def get_history(self, conv_id: Optional[str]) -> List[Message]:
        """Retrieve full, raw conversation history."""
        if not conv_id:
            return []
        return list(self._messages.get(conv_id, []))

    def get_bounded_history(
        self,
        conv_id: Optional[str],
        max_tokens: Optional[int] = None,
        max_turns: Optional[int] = None,
    ) -> List[Message]:
        """Retrieve recent conversation history strictly bounded by token and turn limits.
        
        Prioritizes the most recent turns (sliding window), discarding older turns
        when budget is exceeded.
        """
        raw_history = self.get_history(conv_id)
        if not raw_history:
            return []

        token_budget = max_tokens if max_tokens is not None else self.max_history_tokens
        turn_budget = max_turns if max_turns is not None else self.max_turns

        bounded: List[Message] = []
        accumulated_tokens = 0
        turn_count = 0

        # Iterate backward from latest messages to preserve the newest context
        for msg in reversed(raw_history):
            msg_tokens = token_cost_estimator.estimate_tokens(msg.content)

            # Check token budget
            if accumulated_tokens + msg_tokens > token_budget and bounded:
                logger.debug(
                    "History message truncated by token budget: %d + %d > %d",
                    accumulated_tokens,
                    msg_tokens,
                    token_budget,
                )
                break

            # Track turn count on user messages
            if msg.role == MessageRole.USER:
                if turn_count >= turn_budget:
                    logger.debug("History turn limit (%d) reached", turn_budget)
                    break
                turn_count += 1

            bounded.insert(0, msg)
            accumulated_tokens += msg_tokens

        return bounded

    def rewrite_followup_query(self, conv_id: Optional[str], current_query: str) -> str:
        """Rewrite follow-up queries that contain conversational pronouns."""
        history = self.get_bounded_history(conv_id)
        return self.query_rewriter.rewrite(current_query, history)

    def assemble_prompt_bundle(
        self,
        current_query: str,
        retrieved_context: str,
        conv_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> PromptBundle:
        """Assemble a complete prompt bundle while strictly respecting the overall context window.

        Guarantees:
        1. System prompt is preserved.
        2. Current user question is ALWAYS preserved intact (never truncated).
        3. History is truncated first if budget is tight.
        4. Retrieved context is accommodated up to the remaining context budget.
        """
        effective_system_prompt = system_prompt or RESEARCHMATE_SYSTEM_PROMPT
        sys_tokens = token_cost_estimator.estimate_tokens(effective_system_prompt)
        query_tokens = token_cost_estimator.estimate_tokens(current_query)

        # Baseline budget consumed by system prompt and current query
        mandatory_tokens = sys_tokens + query_tokens
        remaining_budget = max(0, self.max_context_window_tokens - mandatory_tokens)

        # Allocate budget: 60% for retrieved evidence, 40% for conversation history
        context_budget = int(remaining_budget * 0.65)
        history_budget = int(remaining_budget * 0.35)

        # 1. Fetch bounded conversation history within history budget
        bounded_history = self.get_bounded_history(
            conv_id=conv_id,
            max_tokens=history_budget,
        )

        # 2. Trim retrieved context if it exceeds context budget
        actual_context = retrieved_context
        context_tokens = token_cost_estimator.estimate_tokens(retrieved_context)
        if context_tokens > context_budget:
            logger.warning(
                "Retrieved context (%d tokens) exceeds budget (%d); applying truncation.",
                context_tokens,
                context_budget,
            )
            # Truncate context by character estimate (~4 chars per token)
            char_limit = context_budget * 4
            actual_context = retrieved_context[:char_limit] + "\n[... Truncated for context budget ...]"

        # 3. Assemble PromptBundle with role separation
        return build_academic_rag_prompt(
            query=current_query,
            context=actual_context,
            history=bounded_history,
            system_prompt=effective_system_prompt,
        )


# Alias for backward compatibility
ConversationManager = ConversationContextManager
