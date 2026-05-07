# Sift — On-premise Enterprise Knowledge Agent

> Query your company's documents with hybrid search, precise inline citations, and per-user permissions.
> **Nothing leaves your infrastructure.**

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Tests](https://img.shields.io/badge/tests-145%20passing-brightgreen)

---

## The problem

Enterprise teams sit on thousands of documents — internal wikis, PDFs, codebases, email archives — and can't query them effectively. Hosted RAG tools mean shipping confidential data to third-party APIs. Building your own forces stitching together retrieval, ranking, citation extraction, access control, audit trails, and evaluation from scratch.

**Sift packages all of this and runs entirely on your hardware.**

---

## What Sift does

Ask a question in natural language. Sift searches indexed documents with hybrid retrieval (BM25 + dense vectors fused via RRF, then re-ranked with a cross-encoder), generates a grounded answer with **inline citations linked to the exact file, page, or line range**, and only returns it if a self-critique loop scores it above a quality threshold.

```
User:  "How does Vercel handle environment variables across deployments?"

Sift:  "Vercel scopes environment variables to Production, Preview, and
        Development environments [1]. They can be set via the dashboard,
        the CLI (`vercel env`), or in vercel.json [2]. Sensitive values
        are encrypted at rest [1]."

[1] vercel-docs/docs/concepts/projects/environment-variables.md
    section 'Encryption'
[2] vercel-docs/docs/cli.md
    section 'vercel env'
```

---

## Architecture

```
                                Query
                                  │
                                  ▼
                          ┌────────────┐
                          │ route_query│  classify: factual / analytical /
                          └─────┬──────┘  comparative / ambiguous
                                │
        (ambiguous) ─► clarification_request ─► (interrupt → human input)
                                │
                                ▼
                          ┌──────────┐
                          │ retrieve │  HybridRetriever
                          └─────┬────┘  ├── BM25 (rank_bm25)            ┐
                                │       ├── Vector (ChromaDB)           │ parallel
                                │       ├── RRF fusion (k=60)           │
                                │       └── Cross-encoder rerank        ┘
                                │           (BAAI/bge-reranker-base)
                                ▼
                          ┌────────┐  ┌─────────────────┐
                          │ gather │ →│ evaluate_relev. │
                          └────┬───┘  └────────┬────────┘
                               │               │
                               │   ┌───────────┘
                               │   │ (low score, < max_iter)
                               │   ▼
                               │ rewrite_query ──► retrieve (cycle)
                               │
                               ▼ (sufficient relevance)
                         ┌────────────┐
                         │ synthesize │  Answer with [1][2][3] markers
                         └──────┬─────┘  → build_citations()
                                │
                                ▼
                         ┌──────────────┐  faithfulness · completeness
                         │ self_critique│  · citation_quality (LLM as judge)
                         └──────┬───────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        (faithfulness < 6.0  OR        (passes gate)
         score < 8.0, < max_rewrites)         │
                │                              ▼
                ▼                      ┌────────────────┐
        rewrite_answer ─► synthesize   │ format_response│ → END
                                       └────────────────┘
```

Built with **LangGraph** — cycles, conditional edges, interrupts, and SQLite-backed checkpointing are first-class.

---

## Key features

| Feature | What it actually does |
|---------|----------------------|
| **100% on-premise** | Ollama (qwen2.5:7b) + ChromaDB + local BGE reranker. Zero outbound API calls. |
| **Multi-source ingestion** | PDF (pypdf), Markdown (with section paths), source code (tree-sitter Python/TS, by function), email (mbox/Enron format). All chunked with tiktoken at 512 tokens, 64 overlap. |
| **Hybrid retrieval** | BM25 + dense vectors run in parallel → fused with Reciprocal Rank Fusion (Cormack 2009, k=60) → top-N re-ranked with `BAAI/bge-reranker-base`. |
| **Inline citations** | Regex parses `[N]` markers in the answer, maps them to chunks, attaches `source_path`, `page_number`, `line_start/end`, and a 200-char snippet. |
| **Self-critique loop** | LLM-as-judge scores faithfulness, completeness, and citation_quality (0–10). Hard gate at faithfulness < 6.0 forces rewrite regardless of overall score. |
| **JWT auth + scope filters** | OAuth2 password flow, bcrypt-hashed passwords. ChromaDB `where` filter built per-user from `corpus` metadata (`{"corpus": {"$in": user.scopes}}`). Admin bypasses. |
| **Audit log** | Append-only SQLite table — every login, query, latency, error, IP. Endpoint `/audit/events` (admin) and `/audit/events/mine` (self). |
| **Human-in-the-loop** | Ambiguous queries trigger LangGraph `interrupt_before`. UI shows a checkpoint modal; resume injects feedback into state. |
| **RAGAS evaluation** | `scripts/eval.py --dataset golden_qa.jsonl` runs the full pipeline, scores with RAGAS (Ollama as judge), writes Markdown + JSON report. `--mock` mode for CI without external services. |
| **Tests** | 145 tests passing — unit + integration. Reranker, vector store, LLM clients are all dependency-injected for testability. |

---

## Stack

```
Runtime       Python 3.12+
Agent         LangGraph 1.1 (StateGraph, cycles, interrupt_before, SQLite checkpointer)
LLM           Ollama qwen2.5:7b (configurable via env)
Embeddings    ChromaDB default (all-MiniLM-L6-v2) — swappable
Vector DB     ChromaDB
Keyword       rank_bm25 (BM25Okapi, in-memory index)
Reranker      BAAI/bge-reranker-base (sentence-transformers, lazy-loaded)
API           FastAPI 0.115 + SSE streaming
Auth          python-jose JWT + bcrypt (passlib bypassed — broken on Py3.14 + bcrypt 5)
Eval          RAGAS 0.2 with Ollama judge wrapper
UI            Single-file HTML + vanilla JS + marked.js (no build step)
Storage       SQLite for users, sessions, audit; ChromaDB for vectors
Deploy        Docker Compose — intentionally no cloud
```

---

## Quickstart

```bash
git clone https://github.com/Izanvz/Sift.git
cd Sift

# 1. Boot infra (Ollama pulls qwen2.5:7b ~4.7GB on first run)
docker compose up -d

# 2. Install Python deps
pip install -r requirements.txt

# 3. Create an admin user
python scripts/bootstrap_admin.py -u admin -p <password> --scopes "*"

# 4. Index documents (any of these — pick what you have)
python scripts/ingest.py --source data/sources/enterprise/vercel-docs --connector markdown
python scripts/ingest.py --source data/sources/code/stripe-go        --connector code
python scripts/ingest.py --source data/sources/personal              --connector pdf

# 5. Run the API
uvicorn src.api.main:app --port 8001

# 6. Open the UI
open http://localhost:8001
```

---

## Per-user scopes

Documents are indexed with a `corpus` metadata field. Each user has a list of allowed corpora; admins (`scopes=["*"]` or `is_admin=True`) bypass the filter.

```bash
# User who can only see Vercel docs and Stripe Go SDK
python scripts/bootstrap_admin.py \
    -u alice -p hunter2 \
    --scopes vercel-docs stripe-go \
    --no-admin
```

The filter is applied at retrieval time inside ChromaDB's `where` clause — no post-filtering, so document scoping has no recall penalty.

---

## Evaluation

```bash
# Mock mode (no Ollama, no judge) — fast smoke test
python scripts/eval.py --dataset data/eval/golden_qa.jsonl --mock

# Real RAGAS run (requires Ollama up + indexed corpus)
python scripts/eval.py --dataset data/eval/golden_qa.jsonl

# Subset by tag or limit
python scripts/eval.py --dataset data/eval/golden_qa.jsonl --tag vercel --limit 5
```

Output: `data/eval/reports/report-<timestamp>.{md,json}` with aggregate metrics, per-question breakdown, source recall, and latency stats. See [docs/benchmarks.md](docs/benchmarks.md) for sample results.

The golden dataset (`data/eval/golden_qa.jsonl`) ships with **15 hand-crafted Q&A pairs** spanning Vercel docs, Stripe Go SDK, and Enron emails — covering factual, analytical, comparative, and ambiguous query types.

---

## Project status

| Phase | Status | What landed |
|-------|--------|-------------|
| 0 — Foundation | ✅ | Repo, Docker Compose, datasets |
| 1 — Graph refactor | ✅ | `SiftState`, modular nodes, centralized prompts |
| 2 — Multi-source ingestion | ✅ | PDF / Markdown / Code (tree-sitter) / Email connectors + tiktoken chunker |
| 3 — Hybrid retrieval | ✅ | BM25 + vector + RRF + BGE reranker, parallel search |
| 4 — Inline citations | ✅ | Regex `[N]` parser, chunk-mapped `Citation` dataclass with snippets |
| 5 — Self-critique loop | ✅ | Faithfulness hard gate + weighted scoring + answer rewrite |
| 6 — RAGAS eval | ✅ | Mock + real evaluators, 15-Q&A golden dataset, MD/JSON reports |
| 7 — JWT auth + scopes | ✅ | OAuth2 password flow, bcrypt, SQLite store, ChromaDB `$in` filter |
| 8 — Audit log | ✅ | Append-only SQLite, Starlette middleware, admin endpoints |
| 9 — UI | ✅ | Login overlay, DAG live status, citation cards, critique panel |
| 10 — README + benchmarks | ✅ | This file + `docs/architecture.md` + `docs/benchmarks.md` |
| 11 — Demo + write-up | ⏳ | Loom walkthrough + LinkedIn launch |

---

## Why on-premise?

Most enterprise RAG demos send documents to OpenAI or Anthropic. That works for demos — not for companies with compliance requirements (GDPR, HIPAA, SOC2 data residency clauses), sensitive IP, or contractual restrictions on third-party processors.

Sift runs entirely on your hardware. The LLM, embeddings, vector store, and reranker are all local. The only network calls are the ones the operator explicitly makes.

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — deeper dive into nodes, state, retrieval pipeline, scope model
- [docs/benchmarks.md](docs/benchmarks.md) — RAGAS results, latency, retrieval ablations
- [docs/sift-v2-roadmap.md](docs/sift-v2-roadmap.md) — design notes from the v2 rewrite

---

## License

MIT — Izan Villarejo
