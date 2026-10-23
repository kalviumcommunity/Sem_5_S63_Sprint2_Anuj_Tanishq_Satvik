# ⚙️ Model Parameters & Output Control Architecture

In **ResearchMate**, reliable academic retrieval-augmented generation demands tight governance over large language model (LLM) behavior. This document specifies the centralized configuration patterns, parameter validation rules, provider-specific capabilities, and sensible presets implemented to guarantee deterministic, source-grounded, and concise academic explanations.

---

## 1. Architectural Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. CONFIGURATION SOURCES                                                    │
│                                                                             │
│  .env / Environment Variables ──> backend.app.core.config.Settings          │
│  (OPENAI_API_KEY, GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, etc.)         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         LLMConfig.from_settings()
                         LLMConfig.for_academic_explanation()
                         LLMConfig.for_environment()
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. CENTRALIZED VALIDATION LAYER (types.validate_model_parameters)            │
│                                                                             │
│  • Parameter Range Checking (temperature in [0, 2], max_tokens > 0)          │
│  • Incompatible Combination Rejection (greedy temp=0 with top_p < 1.0)       │
│  • Provider Payload Sanitization & Formatting (strip unsupported keys)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. PROVIDER ABSTRACTION LAYER (LLMClientInterface)                          │
│                                                                             │
│  ┌─────────────────────────┐  ┌────────────────────────┐  ┌───────────────┐ │
│  │   OpenAILLMClient       │  │    GeminiLLMClient     │  │ MockLLMClient │ │
│  │                         │  │                        │  │               │ │
│  │ • payload["messages"]   │  │ • contents[].parts     │  │ • Determin-   │ │
│  │ • max_tokens, top_p     │  │ • generationConfig     │  │   istic text  │ │
│  │ • seed, penalties       │  │ • maxOutputTokens,topP │  │ • Schema test │ │
│  │ • filters out top_k     │  │ • maps top_k -> topK   │  │ • Offline dev │ │
│  └─────────────────────────┘  └────────────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Supported Providers & Capabilities Matrix

Different LLM providers expose heterogeneous REST schemas and parameter constraints. ResearchMate respects provider-specific capabilities and never sends unrecognized parameters that would cause HTTP 400 Bad Request failures.

| Parameter | Type / Range | OpenAI (`chat/completions`) | Gemini (`v1beta:generateContent`) | Mock Client |
| :--- | :--- | :--- | :--- | :--- |
| `model` | `str` | `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo` | `gemini-1.5-flash`, `gemini-1.5-pro` | `mock-academic-v1` |
| `temperature` | `[0.0, 2.0]` | Supported (`payload["temperature"]`) | Supported (`generationConfig["temperature"]`) | Supported |
| `max_tokens` | `int > 0` | Supported (`payload["max_tokens"]`) | Mapped to `maxOutputTokens` | Supported |
| `top_p` | `[0.0, 1.0]` | Supported (`payload["top_p"]`) | Mapped to `topP` | Supported |
| `top_k` | `int >= 1` | **Unsupported** (Filtered out) | Supported (`generationConfig["topK"]`) | Supported |
| `seed` | `int` | Supported (`payload["seed"]`) | **Unsupported** in v1beta (Omitted) | Supported |
| `stop_sequences` | `List[str]` | Max 4 sequences (`payload["stop"]`) | Max 5 sequences (`stopSequences`) | Max 5 sequences |
| `response_format` | `"text"` / `"json_object"` | `{"type": "json_object"}` | `responseMimeType: "application/json"` | Supported |
| `presence_penalty` | `[-2.0, 2.0]` | Supported (`payload["presence_penalty"]`) | **Unsupported** (Omitted) | Supported |
| `frequency_penalty` | `[-2.0, 2.0]` | Supported (`payload["frequency_penalty"]`) | **Unsupported** (Omitted) | Supported |

---

## 3. Parameter Validation & Incompatible Combinations

Validation is performed during `LLMConfig` construction and dynamically during per-call overrides inside `client.complete()`. Invalid parameters immediately raise `InvalidModelParameterError` (inheriting from `LLMException` and `ValueError`).

### Validation Rules
1. **Temperature**: Must be a float within `0.0 <= temperature <= 2.0`. Values outside this range are rejected.
2. **Max Tokens**: Must be an integer strictly greater than 0 (`max_tokens > 0`).
3. **Top-p Nucleus Sampling**: Must be a float within `0.0 <= top_p <= 1.0`.
4. **Top-k Sampling**: When specified, must be an integer `>= 1`.
5. **Penalties**: Presence and frequency penalties must be within `-2.0 <= penalty <= 2.0`.
6. **Stop Sequences**: Empty strings (`""`) are rejected. Exceeding provider limits (OpenAI > 4, Gemini > 5) triggers validation errors.
7. **Response Format**: Must be either `"text"` or `"json_object"`.

### Incompatible Combinations
- **Greedy Decoding with Nucleus Sampling Cutoff**:
  When `temperature=0.0`, the model executes deterministic argmax token selection. Specifying a restrictive `top_p < 1.0` (such as `top_p=0.8`) is an incompatible sampling directive and is rejected with:
  ```text
  InvalidModelParameterError: Incompatible parameter combination: temperature=0.0 (greedy argmax decoding) cannot be combined with top_p=0.8 (< 1.0). Set top_p=1.0 or temperature > 0.0.
  ```

---

## 4. Sensible Settings for Concise Academic Explanations

Academic research question answering requires high factual precision, strict fidelity to retrieved evidence, and low verbosity. ResearchMate centralizes this with `LLMConfig.for_academic_explanation()`:

```python
academic_config = LLMConfig.for_academic_explanation(provider="openai")
```

### Preset Values & Rationale

| Setting | Value | Rationale |
| :--- | :--- | :--- |
| `temperature` | `0.1` | Suppresses speculative hallucination and ensures answers remain strictly grounded in retrieved research literature, while maintaining natural grammatical syntax. |
| `max_tokens` | `512` | Enforces concise explanations (~300-350 words) suitable for academic syntheses, preventing runaway generations and aligning with the 500-token prompt reservation. |
| `top_p` | `0.9` | Concentrates probability mass on high-confidence tokens, pruning rare or errant words while retaining technical domain vocabulary. |
| `seed` | `42` | Ensures reproducible generations for academic benchmarking and evaluation suites. |
| `response_format` | `"text"` | Clean formatted prose with bracketed academic citations `[Doc: ..., Page: ...]`. |

---

## 5. Environment Presets

| Environment | Model | Temperature | Max Tokens | Top-p | Seed | Timeout | Retries |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`production`** | `gpt-4o-mini` | `0.0` | `1024` | `1.0` | `42` | `25.0s` | `3` |
| **`staging`** | `gpt-4o-mini` | `0.1` | `1024` | `1.0` | `42` | `30.0s` | `2` |
| **`testing`** | `mock-academic-v1` | `0.0` | `256` | `1.0` | `123` | `5.0s` | `0` |
| **`academic`** | `gpt-4o-mini` | `0.1` | `512` | `0.9` | `42` | `30.0s` | `2` |

---

## 6. Secure Credential Loading & Hierarchy

API keys are resolved following strict precedence to prevent leaks and support isolated environments:

```text
1. Explicit Argument      -> OpenAILLMClient(api_key="sk-...")
2. Config Field           -> LLMConfig(api_key="sk-...")
3. Provider Env Variable   -> OPENAI_API_KEY / GEMINI_API_KEY
4. Generic Env Variable    -> LLM_API_KEY
```

### Missing Credentials Handling
- **Development / Offline Mode (`LLM_FALLBACK_TO_MOCK=True`)**:
  When API keys are missing or contain placeholder values (`your-llm-api-key-here`), the client logs a warning and falls back to `MockLLMClient`, enabling offline testing.
- **Production Mode (`LLM_FALLBACK_TO_MOCK=False`)**:
  When credentials are missing, the client raises `LLMAuthenticationError(status_code=401)`.

---

## 7. Error Handling & Exception Taxonomy

All LLM errors inherit from `LLMException`:

```text
LLMException
├── LLMAuthenticationError (Missing/invalid key, 401/403)
├── LLMRateLimitError (429 Quota or rate limit exceeded)
├── LLMTimeoutError (httpx.TimeoutException)
├── LLMProviderError (5xx Server errors)
├── LLMResponseError (JSON decode error, unexpected schema)
│   └── StructuredOutputValidationError (Schema mismatch)
└── InvalidModelParameterError (Parameter out of range / incompatible)
```

---

## 8. Usage Examples

### Instantiating Academic Preset
```python
from backend.app.services.llm import LLMConfig, get_llm_client

# Load concise academic explanation configuration
config = LLMConfig.for_academic_explanation(provider="openai")
client = get_llm_client("openai", config=config)

response = await client.complete("Explain federated learning convergence.")
print(response.text)
```

### Per-Call Overrides with Validation
```python
# Temporary override for exploratory query
response = await client.complete(
    prompt="Suggest three future research directions.",
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
)
```
