# 🧠 Context Windows & Message History Architecture

In **ResearchMate**, conversation management operates across two distinct conceptual layers: **Retained History** and the **Model Context Window**. Understanding the difference between these two layers is critical for building reliable, token-efficient, and grounded RAG applications.

---

## 1. Architectural Distinction

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. RETAINED HISTORY (Persistent Storage Layer)                               │
│                                                                             │
│  [Turn 1: User] ──> [Turn 1: Assistant] ──> [Turn 2: User] ──> ...         │
│  • Storage: Database, In-Memory Repository, Session Cache                   │
│  • Capacity: Unbounded (Hundreds or thousands of turns)                     │
│  • Purpose: Complete chronological transcript for auditing & continuity     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                      Context Governor / Budget Planner
                      (Sliding Window, Pruning & Reservation)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. MODEL CONTEXT WINDOW (Inference Working Memory)                          │
│                                                                             │
│  ┌──────────────────┬─────────────────┬─────────────────┬────────────────┐  │
│  │ System Prompt    │ Retrieved Passages │ Bounded History │ Current Query │  │
│  │ (Mandatory Core) │ (Academic Context) │ (Recent Turns)  │ (Unmodified)  │  │
│  └──────────────────┴─────────────────┴─────────────────┴────────────────┘  │
│                                       +                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Reserved Output Budget (e.g., 500 tokens for generated answer)        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  • Limit: Fixed hard token ceiling (e.g., 4,000 / 8,192 / 32,768 tokens)     │
│  • Purpose: Working memory for a single LLM generation call                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Dimension | Retained History | Model Context Window |
| :--- | :--- | :--- |
| **Primary Role** | Ground truth historical record | Working inference memory buffer |
| **Capacity** | Arbitrarily large / unbounded | Fixed token limit per model (e.g., 4k, 128k) |
| **Persistence** | Long-term (DB / session state) | Ephemeral (single API request lifecycle) |
| **Token Cost** | Zero inference cost (storage only) | Directly determines latency, dollar cost, and API quotas |
| **Retention Policy** | Append-only transcript | Dynamic budget allocation & sliding-window truncation |

---

## 2. Context Window Budget Allocation

ResearchMate strictly enforces a token budget before assembling the model prompt:

$$
\text{Total Context Limit} = \text{System Prompt} + \text{Current Query} + \text{Output Reserve} + \text{Available Content Budget}
$$

Where the **Available Content Budget** is allocated between evidence and conversation history:
1. **Retrieved Evidence Budget**: 65% of available content space.
2. **Conversation History Budget**: 35% of available content space.

### Guarantees
1. **System Prompt Preservation**: Foundational system instructions and academic citation directives are never truncated.
2. **Current Query Preservation**: The student's current question is preserved intact to ensure intent is not corrupted.
3. **Output Generation Reservation**: The configured `expected_output_tokens` (default 500 tokens) are reserved *before* filling prompt context, preventing mid-sentence generation cutoffs.
4. **History Truncation Order**: History is pruned from oldest to newest turns (FIFO sliding window).

---

## 3. Explicit Oversized Input Policies

When inputs exceed allocated budgets, ResearchMate executes explicit, configurable policies:

### Oversized Queries (`oversized_query_policy`)
- **`preserve`** (Default): Keeps the user question intact, sacrificing older history and trimming retrieved passages to fit within the prompt budget.
- **`truncate`**: Bounds the query to `max_query_tokens` with an explicit indicator `[... Truncated query ...]`.
- **`error`**: Raises `OversizedQueryError` rejecting inputs that would overwhelm the context window.

### Oversized Retrieved Context (`oversized_context_policy`)
- **`truncate`** (Default): Truncates retrieved passages at the allocated boundary, appending `[... Truncated for context budget ...]`.
- **`drop`**: Omits context entirely if partial passages would misrepresent source evidence.
- **`error`**: Raises `OversizedContextError` when retrieved material cannot fit within the allocated token budget.

---

## 4. Conversation Isolation

Every conversation session has an isolated identifier (`conv_id`). ResearchMate enforces complete state isolation:
- Turn storage, bounded histories, and token counters are partitioned by conversation ID.
- Adding, clearing, or pruning messages in Conversation A has zero side-effects on Conversation B.
- Follow-up query rewriting resolves pronouns only against the target conversation's isolated history.
