# 📚 ResearchMate

> **Ask one question. Get one clear answer. Verify it instantly.**

## 🚀 Overview

ResearchMate is an **AI-powered academic research assistant** designed to help university students find reliable information from research papers, theses, and course materials.

Instead of searching through dozens of documents, students can ask questions in natural language and receive **concise, citation-backed answers** with direct access to the relevant source.

---

## 🎯 Problem

University libraries contain huge amounts of valuable academic content, but students often struggle to find the exact information they need.

They have to:

* 🔍 Search through multiple documents
* 📄 Read lengthy PDFs
* ⏳ Spend significant time finding relevant information
* ❓ Verify whether an answer is actually supported by a source

**ResearchMate simplifies this entire process.**

---

## 💡 Solution

ResearchMate follows a simple workflow:

```text
Ask → Search → Understand → Cite → Verify
```

The system searches relevant academic documents, retrieves the most useful sections, generates a concise explanation, and provides citations pointing back to the original source.

---

## ✨ Key Features

### 🔎 Natural Language Search

Ask questions naturally instead of using complex keywords.

### 🤖 AI-Powered Answers

Get short, understandable explanations based on relevant academic sources.

### 📌 Citation-Backed Responses

Every answer includes supporting documents and page references.

### 📄 Source Preview

Jump directly from an answer to the relevant section of the original document.

### 💬 Follow-Up Questions

Continue asking questions while maintaining conversation context.

### 📚 Document Discovery

Find additional research papers, theses, and course materials related to your query.

---

## 🧠 How It Works

ResearchMate uses a **Retrieval-Augmented Generation (RAG)** approach.

```text
             User Question
                   ↓
          Query Understanding
                   ↓
             Semantic Search
                   ↓
        Relevant Document Chunks
                   ↓
              AI / LLM
                   ↓
        ┌──────────┴──────────┐
        ↓                     ↓
   Concise Answer          Citations
        ↓                     ↓
        └──────────┬──────────┘
                   ↓
             Source Preview
```

The AI uses retrieved academic content as context rather than relying only on general knowledge.

---

## 🛠️ MVP

The first version focuses on the core research experience:

* 📤 Upload academic PDFs
* 📑 Process and index documents
* 🔍 Search documents semantically
* 🤖 Generate AI answers
* 📌 Provide citations
* 📄 Preview supporting sources

---

## 👥 Target Users

### Primary

* University students
* Project teams
* Thesis students

### Secondary

* Faculty
* Academic researchers

---

## 🏗️ High-Level Architecture

```text
┌─────────────────────┐
│      Frontend       │
│  Search / Chat / UI │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│      Backend API    │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Document Processing │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Vector Database   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│       LLM / RAG     │
└──────────┬──────────┘
           ↓
    Answer + Citations
```
---

## 💻 Development & Setup

### Prerequisites
- Python 3.10+ (tested with Python 3.12)
- Git

### Installation
1. **Clone the repository and checkout the feature branch**:
   ```bash
   git clone https://github.com/kalviumcommunity/Sem_5_S63_Sprint2_Anuj_Tanishq_Satvik.git
   cd Sem_5_S63_Sprint2_Anuj_Tanishq_Satvik
   git checkout feature/01-environment-setup
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your LLM/Embedding API keys if available
   ```

### Running the Backend Server
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```
- **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health) or [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### Running Tests
Execute the full test suite with pytest:
```bash
pytest
```

---
## 📊 Success Metrics

The primary success metric is:

> **Percentage of questions resolved with a useful answer and a verifiable citation.**

Other metrics include:

* Average time to find an answer
* Citation accuracy
* Search success rate
* Citation click-through rate
* User satisfaction
* Number of questions successfully answered

---

## 🔮 Future Scope

* 📑 Automatic literature reviews
* 🔗 APA / MLA / IEEE citation generation
* 📊 Research paper comparison
* 🌐 Multilingual academic search
* 🎤 Voice-based queries
* 📚 University library integration
* 🧠 Personalized study assistance

---

## 🔐 Core Principle

> **No unsupported answer. No hidden source.**

ResearchMate is built around the idea that students should not only receive an answer — they should be able to **understand where that answer came from and verify it themselves.**

---

## 📜 License

This project is developed for educational and academic purposes.
