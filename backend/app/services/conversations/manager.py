"""Conversation context manager and history budget management.

Maintains isolated conversation sessions, enforces context window budgets that
reserve tokens for system instructions, retrieved passages, and expected output,
and defines explicit policies for oversized queries and retrieved passages.

Retained History vs Model Context Window:
- Retained History: The persistent, complete log of all conversational turns in a thread.
- Context Window: The model's finite token ceiling for a single inference call.
  This service bridges the two by selecting the most relevant/recent subset of history
  that fits within the dynamically budgeted context window.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.models.conversation import Conversation, Message, MessageRole
from backend.app.services.conversations.exceptions import (
    ContextWindowExceededError,
    ConversationNotFoundError,
    OversizedContextError,
    OversizedQueryError,
)
from backend.app.services.conversations.query_rewriter import QueryRewriter
from backend.app.services.llm.cost import token_cost_estimator
from backend.app.services.llm.prompts import (
    PromptBuilder,
    PromptBundle,
    RESEARCHMATE_SYSTEM_PROMPT,
    build_academic_rag_prompt,
)
from backend.app.services.tokenizer import (
    TokenizerService,
    default_tokenizer,
)


class ConversationHistoryConfig(BaseModel):
    """Configuration for context window budgeting and history retention."""

    max_turns: int = Field(default=5, ge=1, description="Max user turns to retain in sliding history")
    max_history_tokens: int = Field(default=1500, ge=10, description="Max token budget for retained history")
    max_context_window_tokens: int = Field(default=4000, ge=100, description="Hard model context window limit")
    expected_output_tokens: int = Field(default=500, ge=10, description="Tokens reserved for LLM generation")
    retrieved_context_ratio: float = Field(
        default=0.65,
        ge=0.1,
        le=0.9,
        description="Share of available content budget assigned to retrieved evidence",
    )
    history_ratio: float = Field(
        default=0.35,
        ge=0.1,
        le=0.9,
        description="Share of available content budget assigned to conversational history",
    )
    oversized_query_policy: str = Field(
        default="preserve",
        description="Policy for queries exceeding budget: 'preserve', 'truncate', or 'error'",
    )
    oversized_context_policy: str = Field(
        default="truncate",
        description="Policy for context exceeding budget: 'truncate', 'drop', or 'error'",
    )
    max_query_tokens: int = Field(default=1000, ge=50, description="Threshold for oversized query checks")


class ContextBudgetPlan(BaseModel):
    """Itemized token allocation plan for an assembled prompt."""

    total_context_limit: int
    expected_output_reserve: int
    system_tokens: int
    query_tokens: int
    available_content_budget: int
    context_budget: int
    history_budget: int
    actual_context_tokens: int
    actual_history_tokens: int
    history_turns_retained: int
    context_truncated: bool
    query_truncated: bool


class ConversationContextManager:
    """Manages multi-turn conversation sessions and enforces context window budgets."""

    def __init__(
        self,
        config: Optional[ConversationHistoryConfig] = None,
        tokenizer: Optional[TokenizerService] = None,
        max_turns: Optional[int] = None,
        max_history_tokens: Optional[int] = None,
        max_context_window_tokens: Optional[int] = None,
        expected_output_tokens: Optional[int] = None,
        oversized_query_policy: Optional[str] = None,
        oversized_context_policy: Optional[str] = None,
    ):
        # Allow passing config model or individual overrides
        self.config = config or ConversationHistoryConfig(
            max_turns=max_turns or settings.MAX_HISTORY_TURNS,
            max_history_tokens=max_history_tokens or settings.MAX_HISTORY_TOKENS,
            max_context_window_tokens=max_context_window_tokens or settings.MAX_CONTEXT_WINDOW_TOKENS,
            expected_output_tokens=expected_output_tokens or 500,
            oversized_query_policy=oversized_query_policy or "preserve",
            oversized_context_policy=oversized_context_policy or "truncate",
        )
        if max_turns is not None:
            self.config.max_turns = max_turns
        if max_history_tokens is not None:
            self.config.max_history_tokens = max_history_tokens
        if max_context_window_tokens is not None:
            self.config.max_context_window_tokens = max_context_window_tokens
        if expected_output_tokens is not None:
            self.config.expected_output_tokens = expected_output_tokens
        if oversized_query_policy is not None:
            self.config.oversized_query_policy = oversized_query_policy
        if oversized_context_policy is not None:
            self.config.oversized_context_policy = oversized_context_policy

        self.tokenizer = tokenizer or default_tokenizer
        self._conversations: Dict[str, Conversation] = {}
        self._messages: Dict[str, List[Message]] = {}
        self.query_rewriter = QueryRewriter()

    # Convenience properties for backward compatibility
    @property
    def max_turns(self) -> int:
        return self.config.max_turns

    @property
    def max_history_tokens(self) -> int:
        return self.config.max_history_tokens

    @property
    def max_context_window_tokens(self) -> int:
        return self.config.max_context_window_tokens

    def create_conversation(
        self,
        title: str = "New Inquiry",
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """Initialize an isolated conversation session."""
        conv = Conversation(title=title, user_id=user_id, metadata=metadata or {})
        self._conversations[conv.id] = conv
        self._messages[conv.id] = []
        logger.debug("Created conversation '%s' (%s)", conv.id, title)
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Fetch conversation metadata by ID."""
        return self._conversations.get(conv_id)

    def list_conversations(self, user_id: Optional[str] = None) -> List[Conversation]:
        """List all conversations, optionally filtered by user_id."""
        if user_id is not None:
            return [c for c in self._conversations.values() if c.user_id == user_id]
        return list(self._conversations.values())

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation session and all its stored messages."""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            self._messages.pop(conv_id, None)
            logger.debug("Deleted conversation '%s'", conv_id)
            return True
        return False

    def clear_history(self, conv_id: str) -> bool:
        """Clear message history for a conversation while retaining the conversation entity."""
        if conv_id in self._conversations:
            self._messages[conv_id] = []
            logger.debug("Cleared history for conversation '%s'", conv_id)
            return True
        return False

    def add_message(
        self,
        conv_id: str,
        role: Union[MessageRole, str],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Record a validated message turn in the isolated conversation thread."""
        if conv_id not in self._conversations:
            self.create_conversation(title="Automatic Session")

        msg_role = MessageRole(role) if isinstance(role, str) else role
        token_count = self.tokenizer.count_single_message_tokens(
            Message(conversation_id=conv_id, role=msg_role, content=content)
        )

        msg = Message(
            conversation_id=conv_id,
            role=msg_role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )
        self._messages.setdefault(conv_id, []).append(msg)
        return msg

    def get_history(self, conv_id: Optional[str]) -> List[Message]:
        """Retrieve full, raw conversation history for an isolated conversation."""
        if not conv_id:
            return []
        return list(self._messages.get(conv_id, []))

    def get_conversation_token_count(self, conv_id: Optional[str]) -> int:
        """Compute the aggregate token count of all messages stored in this conversation."""
        messages = self.get_history(conv_id)
        if not messages:
            return 0
        return sum(
            m.token_count or self.tokenizer.count_single_message_tokens(m)
            for m in messages
        )

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

        token_budget = max_tokens if max_tokens is not None else self.config.max_history_tokens
        turn_budget = max_turns if max_turns is not None else self.config.max_turns

        # Identify turn boundaries by user messages to prevent dangling assistant messages
        user_indices = [i for i, m in enumerate(raw_history) if m.role == MessageRole.USER]
        if turn_budget is not None and len(user_indices) > turn_budget:
            cutoff_index = user_indices[-turn_budget]
            eligible_history = raw_history[cutoff_index:]
        else:
            eligible_history = raw_history

        bounded: List[Message] = []
        accumulated_tokens = 0

        # Iterate backward from latest messages to respect token budget
        for msg in reversed(eligible_history):
            msg_tokens = msg.token_count or self.tokenizer.count_single_message_tokens(msg)

            # Check token budget
            if accumulated_tokens + msg_tokens > token_budget and bounded:
                logger.debug(
                    "History message truncated by token budget: %d + %d > %d",
                    accumulated_tokens,
                    msg_tokens,
                    token_budget,
                )
                break

            bounded.insert(0, msg)
            accumulated_tokens += msg_tokens

        # Ensure bounded history never starts with an orphan assistant message
        while bounded and bounded[0].role == MessageRole.ASSISTANT:
            bounded.pop(0)

        return bounded

    def rewrite_followup_query(self, conv_id: Optional[str], current_query: str) -> str:
        """Rewrite follow-up queries that contain conversational pronouns using isolated history."""
        history = self.get_bounded_history(conv_id)
        return self.query_rewriter.rewrite(current_query, history)

    def plan_budget(
        self,
        current_query: str,
        retrieved_context: str = "",
        system_prompt: Optional[str] = None,
        conv_id: Optional[str] = None,
    ) -> ContextBudgetPlan:
        """Plan and calculate context budget allocation across system, query, output, context, and history."""
        total_limit = self.config.max_context_window_tokens
        output_reserve = self.config.expected_output_tokens
        effective_system_prompt = system_prompt or RESEARCHMATE_SYSTEM_PROMPT

        sys_tokens = self.tokenizer.count_tokens(effective_system_prompt)
        query_tokens = self.tokenizer.count_tokens(current_query)

        # Check oversized query against threshold
        query_truncated = False
        if query_tokens > self.config.max_query_tokens:
            if self.config.oversized_query_policy == "error":
                raise OversizedQueryError(query_tokens, self.config.max_query_tokens)
            elif self.config.oversized_query_policy == "truncate":
                query_truncated = True

        # Mandatory tokens reserved first
        mandatory_tokens = sys_tokens + query_tokens + output_reserve
        available_content_budget = max(0, total_limit - mandatory_tokens)

        # Allocate between retrieved context and conversational history
        context_budget = int(available_content_budget * self.config.retrieved_context_ratio)
        history_budget = int(available_content_budget * self.config.history_ratio)

        bounded_history = self.get_bounded_history(conv_id=conv_id, max_tokens=history_budget)
        actual_history_tokens = sum(
            m.token_count or self.tokenizer.count_single_message_tokens(m)
            for m in bounded_history
        )
        actual_context_tokens = self.tokenizer.count_tokens(retrieved_context)

        context_truncated = actual_context_tokens > context_budget

        return ContextBudgetPlan(
            total_context_limit=total_limit,
            expected_output_reserve=output_reserve,
            system_tokens=sys_tokens,
            query_tokens=query_tokens,
            available_content_budget=available_content_budget,
            context_budget=context_budget,
            history_budget=history_budget,
            actual_context_tokens=actual_context_tokens,
            actual_history_tokens=actual_history_tokens,
            history_turns_retained=len(bounded_history),
            context_truncated=context_truncated,
            query_truncated=query_truncated,
        )

    def assemble_prompt_bundle(
        self,
        current_query: str,
        retrieved_context: str,
        conv_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> PromptBundle:
        """Assemble a complete prompt bundle while strictly respecting context window reservations.

        Guarantees:
        1. System prompt is preserved.
        2. Expected output generation tokens are reserved before filling context.
        3. Current user question is preserved intact (unless explicit truncate policy is configured).
        4. Older history is trimmed first to respect budget.
        5. Retrieved context is handled according to oversized_context_policy ('truncate', 'drop', 'error').
        """
        budget_plan = self.plan_budget(
            current_query=current_query,
            retrieved_context=retrieved_context,
            system_prompt=system_prompt,
            conv_id=conv_id,
        )

        effective_system_prompt = system_prompt or RESEARCHMATE_SYSTEM_PROMPT
        processed_query = current_query

        # Policy handling for oversized query
        if budget_plan.query_truncated:
            char_limit = self.config.max_query_tokens * 4
            processed_query = current_query[:char_limit] + " [... Truncated query ...]"

        # Policy handling for oversized context
        actual_context = retrieved_context
        if budget_plan.context_truncated:
            if self.config.oversized_context_policy == "error":
                raise OversizedContextError(
                    budget_plan.actual_context_tokens,
                    budget_plan.context_budget,
                )
            elif self.config.oversized_context_policy == "drop":
                logger.warning("Retrieved context exceeded budget; dropping context per policy.")
                actual_context = ""
            else:  # default 'truncate'
                logger.warning(
                    "Retrieved context (%d tokens) exceeds budget (%d); applying truncation.",
                    budget_plan.actual_context_tokens,
                    budget_plan.context_budget,
                )
                char_limit = max(100, budget_plan.context_budget * 4)
                actual_context = retrieved_context[:char_limit] + "\n[... Truncated for context budget ...]"

        # Bounded history within allocated history budget
        bounded_history = self.get_bounded_history(
            conv_id=conv_id,
            max_tokens=budget_plan.history_budget,
        )

        # Assemble PromptBundle with clean role separation
        return build_academic_rag_prompt(
            query=processed_query,
            context=actual_context,
            history=bounded_history,
            system_prompt=effective_system_prompt,
        )


# Alias for backward compatibility
ConversationManager = ConversationContextManager
