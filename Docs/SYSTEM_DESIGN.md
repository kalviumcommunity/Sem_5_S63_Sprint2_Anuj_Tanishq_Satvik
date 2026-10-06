# ResearchMate --- System Design

## 1. System Objective

ResearchMate solves this problem:

> A university library holds research papers, theses, and course
> materials, yet students cannot get concise, citation-backed
> explanations and instead scroll through dozens of unrelated documents
> for one answer.

The system must provide:

-   Natural-language academic search
-   Concise grounded answers
-   Citation-backed responses
-   Source/page references
-   Document upload and indexing
-   Conversational follow-up questions
-   Retrieval evaluation
-   Hallucination guardrails
-   Production-ready monitoring and deployment

## 2. End-to-End Architecture

``` text
                         ┌───────────────────────┐
                         │       STUDENT         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────┐
                    │       RESEARCHMATE UI        │
                    │ Chat │ Upload │ Sources      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       FASTAPI BACKEND        │
                    └──────────────┬───────────────┘
                                   │
             ┌─────────────────────┼──────────────────────┐
             ▼                     ▼                      ▼
     ┌──────────────┐      ┌──────────────┐       ┌─────────────┐
     │  Document    │      │ Conversation │       │    RAG      │
     │   Service    │      │   Service    │       │   Service   │
     └──────┬───────┘      └──────────────┘       └──────┬──────┘
            │                                             │
            ▼                                             ▼
     ┌──────────────┐                              ┌──────────────┐
     │   Extractor  │                              │ Query Rewrite│
     └──────┬───────┘                              └──────┬───────┘
            │                                             ▼
            ▼                                      ┌──────────────┐
     ┌──────────────┐                              │   Embedding  │
     │   Cleaner    │                              │   Service    │
     └──────┬───────┘                              └──────┬───────┘
            │                                             ▼
            ▼                                      ┌──────────────┐
     ┌──────────────┐                              │ Vector Search│
     │   Chunker    │                              └──────┬───────┘
     └──────┬───────┘                                     ▼
            │                                      ┌──────────────┐
            ▼                                      │  Re-Ranking  │
     ┌──────────────┐                              └──────┬───────┘
     │  Embeddings  │                                     ▼
     └──────┬───────┘                              ┌──────────────┐
            │                                      │ Context Build│
            ▼                                      └──────┬───────┘
     ┌────────────────┐                                    ▼
     │  VECTOR DB     │                              ┌──────────────┐
     │ vectors        │                              │    LLM API   │
     │ chunks         │                              └──────┬───────┘
     │ metadata       │                                     ▼
     └────────────────┘                              ┌──────────────┐
                                                     │ Grounding +  │
                                                     │  Citations   │
                                                     └──────┬───────┘
                                                            ▼
                                                     Answer + Sources
```

## 3. Document Ingestion Sequence

``` text
User
 │
 │ Upload document
 ▼
API
 │
 ▼
Validation
 │
 ├── File type
 ├── MIME type
 ├── Size
 └── Duplicate hash
 │
 ▼
Document Loader
 │
 ▼
Text Extraction
 │
 ▼
Cleaning
 │
 ▼
Token-Aware Chunking
 │
 ▼
Chunk Metadata
 │
 ▼
Embedding API
 │
 ▼
Vector Database
 │
 ▼
Indexed
```

For large documents:

``` text
Upload
  ↓
API
  ↓
Job Queue
  ↓
Background Worker
  ├── Extract
  ├── Clean
  ├── Chunk
  ├── Embed
  └── Index
  ↓
Indexed
```

## 4. Query Execution Sequence

``` text
Student
   │
   ▼
Chat UI
   │
   ▼
POST /api/v1/query
   │
   ▼
Conversation Manager
   │
   ▼
Query Rewriter
   │
   ▼
Query Embedding
   │
   ▼
Vector Search
   │
   ├── Top-K
   ├── Metadata filters
   └── Similarity score
   │
   ▼
Re-Ranker
   │
   ▼
Context Builder
   │
   ▼
Grounding Check
   │
   ├── Insufficient → Refusal
   │
   └── Sufficient
          ↓
         LLM
          ↓
    Structured Output
          ↓
 Citation Validation
          ↓
 Answer + Citations
```

## 5. Query Sequence Diagram

``` text
Student        Frontend       Backend       Retriever       Vector DB       LLM
   │              │              │              │              │            │
   │──Question───►│              │              │              │            │
   │              │──POST───────►│              │              │            │
   │              │              │──Retrieve───►│              │            │
   │              │              │              │──Search──────►│            │
   │              │              │              │◄──Chunks─────│            │
   │              │              │◄──Results────│              │            │
   │              │              │              │              │            │
   │              │              │──────── Context + Prompt ──────────────►│
   │              │              │◄──────── Grounded Answer ──────────────│
   │              │◄──Response───│              │              │            │
   │◄──Answer─────│              │              │              │            │
```

## 6. Retrieval Design

The retrieval layer should support:

### Semantic Search

Convert query and chunks into vectors and calculate similarity.

### Metadata Filtering

Filter by:

-   Document type
-   Course
-   Author
-   Year
-   Subject
-   Uploaded collection

### Hybrid Search

Combine semantic similarity with keyword-based matching when useful.

### Re-ranking

Initial vector retrieval can return 10--20 candidates. A re-ranker then
selects the most relevant evidence for final context.

``` text
Query
 ↓
Top 20 candidates
 ↓
Re-ranker
 ↓
Top 5–8 evidence chunks
 ↓
Context Builder
```

## 7. Citation Architecture

Citation integrity is a core feature.

``` text
Document
   ↓
Chunk
   ↓
Embedding
   ↓
Retrieved Chunk
   ↓
Context
   ↓
LLM Answer
   ↓
Citation Validator
   ↓
Citation
```

Every citation should retain:

``` text
document_id
chunk_id
page_number
document_title
section
relevance_score
```

Example answer:

``` text
Federated learning trains models across decentralized
data sources without requiring raw training data to be
centrally collected.

[1] Federated Learning Survey, p. 8
```

## 8. Grounding and Hallucination Guardrails

The system should follow a strict rule:

> No sufficient evidence = no generated factual answer.

``` text
Retrieved Evidence
       ↓
Evidence Sufficiency
    /          \
Insufficient   Sufficient
     ↓             ↓
   Refuse         LLM
                   ↓
             Citation Check
                   ↓
              Final Answer
```

Example refusal:

``` text
I couldn't find enough supporting information in the
available academic sources to answer this reliably.
```

## 9. Conversational RAG

Follow-up questions should use conversation context.

Example:

``` text
User:
What is federated learning?

Assistant:
[Answer + citations]

User:
What are its main challenges?

        ↓

Conversation History
        +
Current Question
        ↓
Query Rewriting
        ↓
"What are the main challenges of federated learning?"
        ↓
Retrieval
```

This avoids treating every follow-up as an isolated query.

## 10. Streaming

For better UX:

``` text
User Query
    ↓
Retrieval
    ↓
LLM
    ↓
Token Stream
    ↓
Frontend
```

The frontend can progressively render:

``` text
Generating answer...

Federated learning is...
...
Sources:
[1] ...
[2] ...
```

Citations should only be displayed as verified references.

## 11. Caching

Useful cache targets:

-   Query embeddings
-   Frequently repeated retrieval results
-   Stable generated answers where appropriate

A safe cache key can include:

``` text
hash(
    normalized_query +
    corpus_version +
    retrieval_configuration
)
```

This prevents stale results after the academic corpus changes.

## 12. Monitoring

Track each request:

``` text
Request ID
API latency
Embedding latency
Retrieval latency
Re-ranking latency
LLM latency
Total latency
Input tokens
Output tokens
Estimated cost
Top retrieval score
Number of sources
Grounded / ungrounded
Citation validation result
```

Example:

``` text
Request: req_8392

Retrieval:      180ms
Re-ranking:      90ms
LLM:            1.8s
Total:          2.2s

Input tokens:   1240
Output tokens:   280

Sources: 4
Grounded: true
```

## 13. Evaluation System

Maintain a benchmark dataset containing:

``` json
{
  "question": "What is federated learning?",
  "expected_documents": [
    "federated_learning_survey.pdf"
  ],
  "expected_pages": [4, 5]
}
```

### Retrieval Metrics

``` text
Recall@K
Precision@K
Hit Rate
MRR
```

### Generation Metrics

``` text
Answer Relevance
Groundedness
Citation Accuracy
Refusal Accuracy
```

### System Metrics

``` text
Latency
Token Usage
Cost
Error Rate
```

## 14. Reliability and Failure Handling

### LLM Failure

``` text
LLM API
  ↓
Retry
  ↓
Still failing
  ↓
Graceful error
```

### Vector Database Failure

Return a temporary-unavailable response instead of answering from
unsupported general knowledge.

### Invalid LLM Output

``` text
LLM
 ↓
Schema Validation
 ↓
Invalid
 ↓
Retry / Fallback
```

### No Relevant Sources

``` text
Query
 ↓
Retrieval
 ↓
No reliable evidence
 ↓
Refusal
```

## 15. Scalability

### Initial Deployment

``` text
Frontend
    │
    ▼
Backend
    │
 ┌──┴───────────────┐
 ▼                  ▼
Vector DB          LLM API
```

### Scaled Deployment

``` text
                    Load Balancer
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         API Server 1          API Server 2
              │                     │
              └──────────┬──────────┘
                         │
                    RAG Service
                         │
              ┌──────────┼───────────┐
              ▼          ▼           ▼
          Cache       Vector DB    Database
              │
              ▼
           LLM API
```

## 16. Security Design

### Secrets

Never expose API keys in:

-   GitHub
-   Frontend
-   Source code
-   README

Use:

``` text
.env
.env.example
```

### File Security

Validate:

``` text
Extension
MIME type
File size
File content
```

### Prompt Injection

Uploaded documents are untrusted content.

``` text
System Instructions
       ↓
User Question
       ↓
Retrieved Documents
       ↓
LLM
```

Retrieved documents must never override system-level instructions.

## 17. Concept-to-System Mapping

    \# Concept                       Implementation
  ---- ----------------------------- ----------------------
     1 Environment Setup             Development
     2 GitHub Workflow               Git/CI
     3 LLM API                       LLM Service
     4 Prompt Construction & Roles   Prompt Layer
     5 Tokens & Cost                 LLM Monitoring
     6 Context Windows               Conversation Service
     7 Model Parameters              LLM Config
     8 Structured JSON               Response Parser
     9 Prompt Templates              Prompt Service
    10 Document Loading              Ingestion
    11 Text Cleaning                 Ingestion
    12 Document Chunking             Ingestion
    13 Chunk Metadata                Metadata Layer
    14 Token-Aware Chunking          Chunker
    15 Corpus Validation             Ingestion Validation
    16 Embeddings                    Embedding Service
    17 Embedding API                 Embedding Service
    18 Similarity Metrics            Retrieval
    19 Batch Embedding               Embedding Pipeline
    20 Embedding Quality             Evaluation
    21 Vector DB                     Storage
    22 Indexing                      Vector Storage
    23 Top-K Retrieval               Retrieval
    24 Hybrid Search                 Retrieval
    25 Retrieval Tuning              Retrieval
    26 Re-ranking                    Retrieval
    27 Recall Testing                Evaluation
    28 RAG Architecture              RAG Service
    29 Context Injection             RAG
    30 Grounded Answer               Grounding
    31 Source Citation               Citation Service
    32 Guardrails                    Safety/Grounding
    33 Conversational RAG            Conversation
    34 RAG Evaluation                Evaluation
    35 Backend API                   FastAPI
    36 Upload Endpoint               Document API
    37 Chat Interface                Frontend
    38 Streaming                     API/UI
    39 Caching & Monitoring          Observability
    40 Deployment                    DevOps

## 18. Viva Explanation

> ResearchMate is a modular RAG-based academic research assistant where
> documents go through extraction, cleaning, token-aware chunking and
> embedding into a vector database. User queries are embedded and
> retrieved using semantic and metadata-aware search, re-ranked,
> injected into a controlled LLM context, and returned as grounded
> answers with validated citations. The FastAPI backend coordinates
> document ingestion, querying, conversations, evaluation, and
> observability.
