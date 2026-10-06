# ResearchMate --- Low-Level Design (LLD)

## 1. Project Structure

``` text
researchmate/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/v1/
│   │   │   ├── query.py
│   │   │   ├── documents.py
│   │   │   ├── conversations.py
│   │   │   └── health.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logging.py
│   │   │   └── security.py
│   │   ├── models/
│   │   │   ├── document.py
│   │   │   ├── chunk.py
│   │   │   ├── citation.py
│   │   │   └── conversation.py
│   │   ├── schemas/
│   │   │   ├── query.py
│   │   │   ├── response.py
│   │   │   └── document.py
│   │   ├── services/
│   │   │   ├── llm/
│   │   │   │   ├── client.py
│   │   │   │   ├── prompts.py
│   │   │   │   └── parser.py
│   │   │   ├── embeddings/
│   │   │   │   ├── service.py
│   │   │   │   └── batcher.py
│   │   │   ├── documents/
│   │   │   │   ├── loader.py
│   │   │   │   ├── extractor.py
│   │   │   │   ├── cleaner.py
│   │   │   │   └── chunker.py
│   │   │   ├── retrieval/
│   │   │   │   ├── search.py
│   │   │   │   ├── filters.py
│   │   │   │   └── reranker.py
│   │   │   ├── rag/
│   │   │   │   ├── pipeline.py
│   │   │   │   ├── context.py
│   │   │   │   ├── grounding.py
│   │   │   │   └── citations.py
│   │   │   └── conversations/
│   │   │       ├── manager.py
│   │   │       └── query_rewriter.py
│   │   ├── repositories/
│   │   │   ├── documents.py
│   │   │   ├── chunks.py
│   │   │   └── conversations.py
│   │   └── evaluation/
│   │       ├── retrieval.py
│   │       └── rag.py
│   ├── tests/
│   │   ├── unit/
│   │   └── integration/
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       └── hooks/
│
├── docs/
│   ├── HLD.md
│   ├── LLD.md
│   └── SYSTEM_DESIGN.md
│
├── README.md
└── .gitignore
```

## 2. Data Models

### Document

``` text
Document
---------
id
title
filename
document_type
author
course
uploaded_at
file_hash
page_count
processing_status
```

### Chunk

``` text
Chunk
-----
id
document_id
chunk_index
text
page_number
section
token_count
embedding_id
metadata
created_at
```

The relationship between chunk, embedding, document, and citation must
always be preserved.

### Conversation

``` text
Conversation
------------
id
user_id
title
created_at
updated_at
```

### Message

``` text
Message
-------
id
conversation_id
role
content
created_at
```

Roles:

``` text
system
user
assistant
```

### Citation

``` text
Citation
--------
id
answer_id
document_id
chunk_id
page_number
document_title
relevance_score
citation_text
```

## 3. Vector Record

``` text
VectorRecord
------------
id
embedding[]
text
document_id
chunk_id
page_number
document_title
section
document_type
course
created_at
```

The vector database provides:

``` text
Embedding storage
       +
Similarity search
       +
Metadata filtering
```

## 4. Document Processing

``` text
Upload
  ↓
Validate
  ↓
Hash / Duplicate Check
  ↓
Extract
  ↓
Clean
  ↓
Chunk
  ↓
Embed
  ↓
Index
  ↓
Complete
```

Recommended document states:

``` text
UPLOADED
PROCESSING
EMBEDDING
INDEXING
INDEXED
FAILED
```

## 5. Embedding Service

``` text
chunk.text
    ↓
EmbeddingService
    ↓
Embedding API
    ↓
vector[]
    ↓
VectorRecord
    ↓
Vector Database
```

For queries:

``` text
User Query
    ↓
Embedding Service
    ↓
query_vector
    ↓
Vector Search
```

## 6. Retrieval

``` python
def retrieve(query, top_k=10, filters=None):
    query_vector = embedding_service.embed(query)

    candidates = vector_store.search(
        vector=query_vector,
        top_k=top_k,
        filters=filters
    )

    ranked = reranker.rank(
        query=query,
        documents=candidates
    )

    return ranked
```

## 7. RAG Pipeline

``` python
def answer_question(query, conversation_id=None):
    history = conversation_manager.get_history(conversation_id)

    contextual_query = query_rewriter.rewrite(
        query,
        history
    )

    results = retriever.retrieve(contextual_query)

    if not grounding.is_sufficient(results):
        return refusal_response()

    reranked_results = reranker.rank(
        contextual_query,
        results
    )

    context = context_builder.build(reranked_results)

    llm_response = llm.generate(
        query=query,
        context=context,
        history=history
    )

    parsed_response = response_parser.parse(llm_response)

    citations = citation_service.validate(
        parsed_response,
        reranked_results
    )

    return {
        "answer": parsed_response.answer,
        "citations": citations
    }
```

## 8. Context Builder

Do not concatenate random chunks. Preserve source metadata:

``` text
Retrieved Chunk
────────────────────────
Document: AI Research Paper
Page: 8
Section: Results
Chunk ID: c102

Text:
"..."
```

This enables reliable citation attribution.

## 9. Grounding

``` text
Retrieved Context
       ↓
 Evidence Check
    /       \
  YES        NO
   ↓          ↓
Generate    Refuse
 Answer
   ↓
Citation Validation
   ↓
Response
```

If retrieval evidence is below the configured threshold, the system
should refuse instead of generating an unsupported answer.

Example:

``` text
"I couldn't find enough supporting information
in the available academic sources."
```

## 10. API Contracts

### POST `/api/v1/query`

Request:

``` json
{
  "query": "What are the advantages of federated learning?",
  "conversation_id": "conv_123"
}
```

Response:

``` json
{
  "answer": "Federated learning allows models to be trained across decentralized data sources without directly centralizing raw data.",
  "citations": [
    {
      "document_id": "doc_123",
      "title": "Federated Learning Survey",
      "page": 8,
      "chunk_id": "chunk_45"
    }
  ],
  "grounded": true
}
```

### POST `/api/v1/documents`

``` text
multipart/form-data
```

Response:

``` json
{
  "document_id": "doc_123",
  "filename": "research-paper.pdf",
  "status": "indexed",
  "chunks": 142
}
```

## 11. Prompt Design

The prompt should separate:

``` text
System Instructions
        +
User Question
        +
Conversation Context
        +
Retrieved Evidence
```

Retrieved content must be explicitly marked as untrusted reference
material and must never override system instructions.

## 12. Error Handling

### LLM Failure

``` text
LLM API
  ↓
Retry
  ↓
Failure
  ↓
Graceful error response
```

### Vector DB Failure

Return a temporary-unavailable response rather than generating from
unsupported knowledge.

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
