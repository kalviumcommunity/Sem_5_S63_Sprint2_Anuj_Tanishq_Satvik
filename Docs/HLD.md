# ResearchMate --- High-Level Design (HLD)

> AI-powered academic research assistant for concise, citation-backed
> answers over research papers, theses, and course materials.

## 1. Overview

ResearchMate uses a Retrieval-Augmented Generation (RAG) architecture to
transform academic documents into searchable knowledge and answer
student questions using retrieved evidence.

``` text
Academic Documents
       ↓
   Extraction
       ↓
     Cleaning
       ↓
     Chunking
       ↓
    Embeddings
       ↓
   Vector Store
       ↓
   User Question
       ↓
 Query Understanding
       ↓
    Retrieval
       ↓
    Re-ranking
       ↓
 Context Construction
       ↓
       LLM
       ↓
Grounded Answer + Citations
```

## 2. High-Level Architecture

``` text
                         ┌───────────────────────┐
                         │       Student         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    ResearchMate UI    │
                         │ Chat / Upload /       │
                         │ Sources / Citations   │
                         └───────────┬───────────┘
                                     │ HTTPS
                                     ▼
                         ┌───────────────────────┐
                         │      Backend API      │
                         │       FastAPI         │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
             ┌────────────┐   ┌────────────┐   ┌──────────────┐
             │ Document   │   │    RAG     │   │ Conversation │
             │ Pipeline   │   │  Service   │   │   Service    │
             └─────┬──────┘   └─────┬──────┘   └──────────────┘
                   │                 │
                   ▼                 ▼
             ┌────────────┐   ┌──────────────┐
             │ Embedding  │   │   Retrieval  │
             │  Service   │   │    Engine    │
             └─────┬──────┘   └──────┬───────┘
                   │                 │
                   ▼                 ▼
             ┌──────────────────────────────┐
             │       Vector Database        │
             │   Vectors + Chunk Metadata   │
             └──────────────────────────────┘
                           │
                           │ Context
                           ▼
                    ┌──────────────┐
                    │   LLM API    │
                    └──────┬───────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Answer + Citations │
                 └────────────────────┘
```

## 3. Major Components

### Frontend

Responsible for:

-   Chat interface
-   Question input
-   Answer rendering
-   Citation rendering
-   Source preview
-   Document upload
-   Conversation history
-   Loading and streaming states
-   Error handling

### Backend API

Recommended endpoints:

``` text
/api/v1/query
/api/v1/documents
/api/v1/conversations
/api/v1/sources
/api/v1/health
/api/v1/evaluation
```

The backend coordinates the application but keeps retrieval, embeddings,
document processing, and LLM logic inside dedicated services.

### Document Processing

``` text
PDF / DOCX / TXT / MD
          ↓
    File Validation
          ↓
    Document Loader
          ↓
   Text Extraction
          ↓
    Text Cleaning
          ↓
   Token-Aware Chunking
          ↓
    Chunk Metadata
          ↓
    Embedding Service
          ↓
      Vector Store
```

### RAG Query System

``` text
Question
   ↓
Conversation Context
   ↓
Query Rewriting
   ↓
Query Embedding
   ↓
Vector Search
   ↓
Metadata Filtering
   ↓
Top-K Results
   ↓
Re-ranking
   ↓
Relevant Context
   ↓
LLM
   ↓
Grounded Answer
   ↓
Citation Validation
```

## 4. HLD Data Flows

### Document Ingestion

``` text
Student/Admin
     │
     ▼
Backend API
     │
     ▼
Document Processor
     ├── Extract
     ├── Clean
     └── Chunk
          │
          ▼
    Embedding Service
          │
          ▼
    Vector Database
```

### Question Answering

``` text
Student
   ↓
Chat UI
   ↓
Backend API
   ↓
RAG Service
   ├── Query Understanding
   ├── Query Embedding
   ├── Retrieval
   ├── Filtering
   ├── Re-ranking
   └── Context Building
           ↓
          LLM
           ↓
    Answer + Citations
```

## 5. Scalability

### MVP

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

### Production

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

Large document ingestion can be moved to asynchronous workers:

``` text
Upload → API → Job Queue → Document Worker → Vector DB
```

## 6. Security

-   Keep LLM and embedding API keys server-side.
-   Store secrets in environment variables.
-   Validate file extension, MIME type, size, and content.
-   Treat retrieved document text as data, not instructions.
-   Protect against prompt injection from uploaded documents.
-   Apply authentication and authorization before production deployment.

## 7. Observability

Track:

-   API latency
-   Retrieval latency
-   Embedding latency
-   Re-ranking latency
-   LLM latency
-   Total latency
-   Token usage
-   Estimated cost
-   Retrieval scores
-   Citation validation
-   Error rate

## 8. Recommended Technology Stack

``` text
Frontend        → React
Backend         → Python + FastAPI
Data Processing→ Python / Pandas
LLM             → LLM API
Embeddings      → Embedding API
Vector DB       → PostgreSQL + pgvector or selected vector DB
Cache           → Redis (optional)
Testing         → Pytest
API Testing     → Postman / Bruno
CI/CD           → GitHub Actions
Deployment      → Frontend + Python backend
```
