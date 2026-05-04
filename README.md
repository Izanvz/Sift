# Sift — On-premise Enterprise Knowledge Agent

> Query your company's documents with hybrid search, precise citations, and per-user permissions.  
> No data leaves your infrastructure.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-in%20development-orange)

---

## The problem

Enterprise teams sit on thousands of documents — internal wikis, PDFs, codebases, email threads — and can't query them effectively. SaaS RAG tools require sending sensitive data to third-party APIs. Building your own means stitching together retrieval, ranking, citations, access control, and evaluation from scratch.

**Sift does all of this on your own hardware.**

---

## What Sift does

Ask a question in natural language. Sift searches your indexed documents using hybrid retrieval, generates a grounded answer with inline citations (linked to the exact file, page, or line), and only returns it if a self-critique loop scores it above threshold.

```
User: "What is our refund policy for enterprise subscriptions?"

Sift: "Enterprise subscriptions can be refunded within 30 days of billing [1].
      Prorated refunds apply for annual plans cancelled mid-cycle [2]."

[1] policies/billing.pdf — page 4
[2] contracts/enterprise-template.md — section 'Cancellation'
```

---

## Architecture

```
Query
  │
  ▼
route_query ──(ambiguous)──► clarification_request ─┐
  │                                                   │
  ▼ (factual / analytical / comparative)             │
retrieve ◄─────────────────────────────────────────┘
  │
  │  HybridRetriever
  │  ├── VectorRetriever  (ChromaDB)
  │  ├── BM25Retriever    (rank_bm25, in-memory)
  │  ├── RRF fusion       (Reciprocal Rank Fusion, k=60)
  │  └── BGE Reranker     (BAAI/bge-reranker-base)
  │
  ▼
gather → evaluate_relevance
  │              │
  │         (low relevance, < 2 iterations)
  │              ▼
  │         rewrite_query ──► retrieve (cycle)
  │
  ▼ (sufficient relevance)
synthesize
  │   Answer with inline citations [1][2][3]
  ▼
self_critique
  │   Scores: faithfulness · completeness · citation quality
  │
  ├── (score < 8, < 2 rewrites) ──► rewrite_answer ──► synthesize (cycle)
  │
  └── (score ≥ 8 or max rewrites reached)
        ▼
    format_response → END
```

Graph built with **LangGraph** — cycles, conditional edges, and human-in-the-loop interrupts are first-class.

---

## Key features

| Feature | Details |
|---------|---------|
| **100% on-premise** | Ollama (qwen2.5:7b) + ChromaDB — no data sent to external APIs |
| **Multi-source ingestion** | PDF, Markdown, source code (Python/TS), email (mbox/Enron format) |
| **Hybrid search** | BM25 + vector similarity → RRF fusion → BGE cross-encoder reranker |
| **Precise citations** | Every claim linked to exact file, page number, or line range |
| **Self-critique loop** | LangGraph cycle scores faithfulness/completeness before returning |
| **Per-user permissions** | JWT auth + glob-based scope filters applied at retrieval time |
| **Full audit trail** | Every query, ingestion, and permission denial logged to SQLite |
| **RAGAS evaluation** | Reproducible benchmarks: faithfulness, answer relevancy, context precision/recall |

---

## Stack

```
Runtime      Python 3.12
Agent        LangGraph 0.2 (StateGraph, cycles, interrupt_before)
LLM          Ollama — qwen2.5:7b (local, no API key needed)
Embeddings   nomic-embed-text via Ollama
Vector DB    ChromaDB
Keyword      rank_bm25 (BM25Okapi, in-memory)
Reranker     BAAI/bge-reranker-base (HuggingFace sentence-transformers)
API          FastAPI + SSE streaming
Auth         JWT (python-jose) + passlib bcrypt
Eval         RAGAS
Deploy       Docker Compose (intentionally no cloud — on-premise by design)
```

---

## Quickstart

```bash
git clone https://github.com/Izanvz/Sift.git
cd Sift

# Start services (Ollama pulls models automatically on first run)
docker-compose up

# Seed demo users
python scripts/seed_users.py

# Index your documents
python scripts/ingest.py --source data/sources/personal --connector pdf
python scripts/ingest.py --source data/sources/enterprise --connector markdown

# Open http://localhost:8001
```

> **Note:** First startup takes a few minutes while Ollama pulls `qwen2.5:7b` (~4.7 GB) and `nomic-embed-text`.

---

## Demo users (after seeding)

| Email | Role | Access |
|-------|------|--------|
| admin@demo.com | admin | All documents |
| engineer@demo.com | employee | Code + Stripe docs |
| sales@demo.com | employee | Email corpus only |

---

## Project status

| Phase | Status | Description |
|-------|--------|-------------|
| 0. Foundation | ✅ Done | Repo structure, datasets, Docker |
| 1. Graph refactor | ✅ Done | SiftState, new nodes, centralized prompts |
| 2. Multi-source ingestion | 🔄 In progress | PDF, Markdown, Code, Email connectors |
| 3. Hybrid search + rerank | ⏳ Planned | BM25 + vector + RRF + BGE reranker |
| 4. Precise citations | ⏳ Planned | Inline [N] markers mapped to exact chunks |
| 5. Self-critique loop | ⏳ Planned | Faithfulness/completeness scoring |
| 6. RAGAS eval suite | ⏳ Planned | Reproducible benchmarks |
| 7. Per-user permissions | ⏳ Planned | JWT + scope filters at retrieval |
| 8. Audit logs | ⏳ Planned | Full query/ingest traceability |
| 9. UI | ⏳ Planned | HTMX chat with citations panel |
| 10. Benchmarks + docs | ⏳ Planned | RAGAS results, architecture docs |

---

## Why on-premise?

Most enterprise RAG demos send documents to OpenAI or Anthropic. That works for demos — not for companies with compliance requirements, sensitive IP, or legal constraints on data residency.

Sift runs entirely on your hardware. The LLM, embeddings, vector store, and reranker are all local. The only network calls are the ones you make.

---

## License

MIT
