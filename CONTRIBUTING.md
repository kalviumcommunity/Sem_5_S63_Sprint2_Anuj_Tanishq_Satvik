# 🤝 ResearchMate Contributing & Team Workflow Guide

Welcome to the **ResearchMate** contributor guide. This document establishes the engineering practices, Git branching model, commit conventions, pull request workflows, and code review standards for our 3-member development team (**Anuj, Tanishq, Satvik**).

---

## 🏛️ Core Product Principles

Every pull request and contribution must honor our foundational rule:

> **"No unsupported answer. No hidden source."**

- **Never introduce invented/hallucinated behavior.**
- **Preserve citation traceability** (`document_id`, `page_number`, `chunk_id`) through every layer of the pipeline.
- **Never commit secrets, tokens, or API keys.**

---

## 👥 Team Structure & Ownership Matrix

| Member | Primary Focus Area | Secondary Areas |
| :--- | :--- | :--- |
| **Anuj** | LLM Services, Prompts, Guardrails & Evaluation | RAG Pipeline, Integration Tests |
| **Tanishq** | Document Ingestion, Extractors, Chunking & Embeddings | Vector Storage, Cleaning Utilities |
| **Satvik** | FastAPI Backend Architecture, Retrieval Engine & API Endpoints | Search Filters, Re-ranking, Client UI |

*Note: While domains have designated primary owners, any member may review and contribute across the stack following the PR guidelines below.*

---

## 🌿 Git Branching Strategy

ResearchMate follows an adapted **GitFlow** branching strategy designed for fast, conflict-free collaborative RAG development:

```text
  main (production-ready, tagged releases)
   ▲
   │ (Pull Request via Sprint Release)
   │
  develop (active sprint integration branch)
   ▲             ▲             ▲
   │             │             │
feature/01-*  feature/02-*  feature/03-* (individual concept branches)
```

### 1. `main` Branch
- Represents **stable, deployable production code**.
- Direct commits to `main` are strictly forbidden.
- Protected branch requiring passing test suites and at least 1 team peer review approval.

### 2. `develop` Branch
- Serves as the primary integration branch for ongoing sprint cycles.
- Features are merged into `develop` only through verified Pull Requests.
- CI tests must pass before merging.

### 3. Feature Branches (`feature/*`)
- Every concept or feature must be developed on an isolated branch.
- Base branch: Checkout from `develop` (or designated base branch for sprint concepts).
- Lifetime: Kept short-lived (1–2 days max) to prevent merge drift.

### 4. Bugfix & Hotfix Branches
- `bugfix/<issue-name>`: Branched from `develop` to address bugs discovered in integration.
- `hotfix/<patch-name>`: Branched from `main` to address critical production issues.

---

## 🏷️ Branch Naming Conventions

All branches must adhere strictly to the lowercase, kebab-case naming standard:

| Type | Pattern | Examples |
| :--- | :--- | :--- |
| **Sprint Concept** | `feature/<concept-number>-<short-description>` | `feature/01-environment-setup`<br>`feature/02-github-workflow`<br>`feature/03-llm-api` |
| **Feature** | `feature/<feature-name>` | `feature/pdf-table-extractor`<br>`feature/bm25-hybrid-search` |
| **Bugfix** | `bugfix/<issue-description>` | `bugfix/fix-citation-offset`<br>`bugfix/null-metadata-check` |
| **Hotfix** | `hotfix/<patch-description>` | `hotfix/api-key-header-parsing` |
| **Documentation** | `docs/<doc-subject>` | `docs/api-v1-swagger-update` |
| **Testing** | `test/<test-suite>` | `test/grounding-guardrail-eval` |

---

## 📝 Commit Message Conventions

ResearchMate adheres to the [Conventional Commits v1.0.0](https://www.conventionalcommits.org/) standard.

### Format
```text
<type>(<scope>): <imperative description>

[optional body explaining WHY, not WHAT]

[optional footer(s) / issue references]
```

### Allowed Types
- **`feat`**: A new user-facing or pipeline feature (e.g., `feat(embeddings): implement batch embedding generator`).
- **`fix`**: A bug fix (e.g., `fix(parser): handle missing bracket citations gracefully`).
- **`docs`**: Documentation changes only (e.g., `docs: establish github team workflow`).
- **`chore`**: Tooling, dependencies, environment, configuration (e.g., `chore: setup researchmate development environment`).
- **`test`**: Adding missing tests or correcting existing tests (e.g., `test(retrieval): add recall at k verification`).
- **`refactor`**: Code restructuring without behavior changes (e.g., `refactor(chunker): extract sliding window generator`).
- **`perf`**: Performance improvement (e.g., `perf(retrieval): optimize vector cosine distance calculations`).

### Quality Rules
- Write in the imperative mood (*"add feature"*, not *"added feature"* or *"adds feature"*).
- Keep the first line under 72 characters.
- **Never** use generic or lazy messages such as `"fix"`, `"update"`, `"wip"`, or `"changes"`.

---

## 🔀 Pull Request (PR) Workflow

### 1. Preparation
Before opening a PR:
1. Run local tests:
   ```bash
   pytest
   ```
2. Verify formatting and linting.
3. Confirm that no `.env`, keys, or test output files are staged:
   ```bash
   git status
   ```

### 2. Opening the PR
- Title: Follow the conventional commit format (e.g., `feat(rag): add context builder with chunk metadata`).
- Target Branch: `develop` (or specified sprint base branch).
- Description: Fill out the standard Pull Request template completely.
- Assignee: The author.
- Reviewers: At least one teammate (e.g., Anuj, Tanishq, or Satvik).

### 3. Review Process
- At least **1 approving review** from a team member is required before merging.
- Any requested changes must be resolved.
- CI / test suite must be green.

### 4. Merging Policy
- **Squash and Merge** (or Rebase Merge): Keeps a clean linear git history on the integration branch.
- Delete the feature branch after merging.
- Do not force push to shared branches (`main`, `develop`).

---

## 🔍 Code Review Expectations & Checklist

Every reviewer must verify the following before approving:

### 🔬 1. RAG Integrity & Grounding
- [ ] Does this change comply with *"No unsupported answer. No hidden source."*?
- [ ] Is source attribution metadata (`document_id`, `chunk_id`, `page_number`) preserved?
- [ ] Are hallucination guards and refusal triggers maintained?

### 🛡️ 2. Security & Credentials
- [ ] Are API keys, tokens, or credentials excluded from the commit?
- [ ] Are file uploads properly validated for extension and MIME type?
- [ ] Is user/document input sanitized to prevent prompt injection?

### 🧪 3. Quality & Testing
- [ ] Are unit tests provided for newly introduced logic?
- [ ] Do all existing unit and integration tests pass?
- [ ] Are edge cases (empty strings, missing metadata, API timeouts) handled?

### 🧩 4. Modularity & Architecture
- [ ] Does this adhere to the service-repository-controller separation?
- [ ] Are LLM, embeddings, document processing, and retrieval kept modular?
- [ ] Are dependencies declared in `requirements.txt`?

---

## 🚀 Quick Git Reference for Daily Work

```bash
# 1. Update develop branch
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/03-llm-api

# 3. Work and commit
git add <files>
git commit -m "feat(llm): add client interface and OpenAI provider"

# 4. Push to remote
git push -u origin feature/03-llm-api

# 5. Open Pull Request on GitHub and request review
```
