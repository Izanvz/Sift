# Sift — Architecture Notes

In-depth view of the v2 rewrite. The README covers the surface; this file documents the design decisions and trade-offs.

---

## 1. Graph (LangGraph 1.1)

```
START
  │
  ▼
route_query        — classify (factual / analytical / comparative / ambiguous)
  │
  ├─ ambiguous ─►  clarification_request  (interrupt_before — waits for user)
  │                                        │
  │                                        ▼ (resume with clarification)
  ▼                                        │
retrieve  ◄─────────────────────────── ────┘
  │
  ▼
gather (dedupe + sort by relevance_score desc)
  │
  ▼
evaluate_relevance ─ if avg_score < threshold AND iter < max_search_iterations
  │                  ▼
  │           rewrite_query ─► retrieve (cycle)
  │
  ▼ (sufficient relevance)
synthesize ─► build_citations(answer, chunks)
  │
  ▼
self_critique ─ if faithfulness < 6.0 OR score < 8.0, < max_rewrite_iterations
  │             ▼
  │       rewrite_answer ─► self_critique (cycle, NOT synthesize, to keep latency bounded)
  │
  ▼ (passes gate)
format_response → END
```

**Why route first?** Ambiguous queries waste retrieval budget. The router is cheap (single LLM call with structured output via `instructor`) and short-circuits to a clarification interrupt.

**Why two separate cycles?** Query rewrite and answer rewrite address different failure modes:

- Low retrieval relevance → query is wrong, fix the query
- High retrieval relevance but low faithfulness → answer hallucinates, fix the answer

Mixing them in one cycle made the agent oscillate.

**Hard gate vs soft gate.** A weighted overall score (0.5·faithfulness + 0.3·completeness + 0.2·citation_quality) can hide a hallucination if completeness is high. The hard gate at faithfulness < 6.0 forces a rewrite even when the overall score looks healthy.

---

## 2. State (`SiftState`)

```python
class SiftState(TypedDict):
    # Input
    query: str
    user_id: str | None
    scopes: list[str]
    is_admin: bool

    # Routing
    query_type: str

    # Retrieval
    chunks: list[Chunk]
    relevance_scores: list[float]   # historical, one entry per cycle
    iterations: int                 # search iterations counter

    # Generation
    answer: str
    citations: list[Citation]

    # Critique
    critique: dict                  # CritiqueOutput serialized
    rewrite_iterations: int

    # Human-in-the-loop
    clarification: str | None

    # Bookkeeping
    metadata: dict
```

State is a `TypedDict` (not a Pydantic model) because LangGraph's reducers operate on dicts and TypedDict is closer to the wire format used by the SQLite checkpointer.

---

## 3. Hybrid retrieval

```
query
  │
  ├─► BM25 (rank_bm25 BM25Okapi, in-memory, lowercased tokenization)
  │   └─ returns top_n=30 docs with bm25_score
  │
  └─► Vector (ChromaDB query, where=scope_filter)
      └─ returns top_n=30 docs with relevance_score (1 - distance)
  │
  ▼
RRF fusion (Cormack et al. 2009, k=60)
  │   score(d) = Σ over rankings: 1 / (k + rank_d_in_ranking)
  │   keeps top n_candidates=20
  │
  ▼
Cross-encoder rerank (BAAI/bge-reranker-base, lazy-loaded)
  │   scores each (query, doc.content) pair
  │   keeps top_k=5
  │
  ▼
chunks with relevance_score = rerank_score (or rrf_score if rerank disabled)
```

### Design notes

- **BM25 vocabulary check.** rank_bm25's IDF goes negative when a term appears in ≥ 50% of a small corpus. The BM25 wrapper builds a vocabulary set on init and short-circuits to `[]` only when the query has zero vocabulary overlap — never based on score sign. This avoids dropping legitimate matches in small corpora.
- **Parallel I/O.** BM25 and vector search run in a 2-worker `ThreadPoolExecutor`. With cold ChromaDB at ~50ms and BM25 at <1ms, parallelization buys back the slow leg.
- **Reranker is opt-in.** `NoOpReranker` is the default in tests/CI (no GPU). Set `RERANKER_ENABLED=false` to skip in production.
- **Scope filter location.** Applied as a ChromaDB `where` clause on metadata `corpus`, **not** post-retrieval. Post-filtering would silently shrink the candidate pool and hurt recall.

---

## 4. Citations

```
synthesize → answer text "Sift uses BM25 [1] and vector search [2]."
                           │
                           ▼
              extract_citation_indices()  ← regex \[(\d+)\]
                           │
                           ▼  [1, 2]
              build_citations(answer, chunks)
                           │
                           ▼
[
  Citation(chunk_id=…, source_path="…/bm25.md", source_type="markdown",
           page_number=None, line_start=None, line_end=None,
           snippet="BM25 is a probabilistic retrieval…", relevance_score=0.87),
  Citation(…)
]
```

- `[N]` is **1-indexed** (matches what humans expect in a paper-style citation).
- Snippets are truncated at the first word boundary after 200 characters with an ellipsis.
- A citation is only emitted if `chunks[N-1]` exists — out-of-range indices are dropped silently and logged at debug.

---

## 5. Auth & scope model

### Token

```json
{
  "sub": "<user_id>",
  "username": "alice",
  "scopes": ["vercel-docs", "stripe-go"],
  "is_admin": false,
  "exp": 1717000000
}
```

Signed HS256 with `JWT_SECRET`. 8-hour default expiry.

### Scope filter rules

| User | `build_scope_filter()` returns |
|------|-------------------------------|
| `None` (no auth) | `None` (no filter — only used in tests) |
| Admin or `scopes=["*"]` | `None` |
| `scopes=["a","b"]` | `{"corpus": {"$in": ["a","b"]}}` |
| `scopes=[]` (valid but no scopes assigned) | `{"corpus": "__no_access__"}` (impossible value → no results) |

The empty-scopes case is critical: without it, a user with no scopes assigned would see everything.

### bcrypt direct (not passlib)

`passlib==1.7.4` is broken on Python 3.14 + bcrypt 5.x — the wraparound bug detector calls `verify()` with a 256-byte secret which bcrypt 5 rejects. Sift uses `bcrypt` directly with manual 72-byte truncation.

---

## 6. Audit

Single append-only SQLite table:

```sql
CREATE TABLE audit_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT UNIQUE NOT NULL,   -- UUID
    event_type     TEXT NOT NULL,          -- auth_login | research_start | research_query
    user_id        TEXT,
    username       TEXT,
    session_id     TEXT,
    query          TEXT,
    status         TEXT,                   -- ok | error
    error          TEXT,
    latency_ms     REAL,
    n_chunks       INTEGER,
    n_citations    INTEGER,
    critique_score REAL,
    scopes         TEXT,                   -- JSON array
    ip_address     TEXT,
    created_at     TEXT NOT NULL           -- ISO 8601 UTC
);
CREATE INDEX idx_audit_user ON audit_events(user_id);
CREATE INDEX idx_audit_ts   ON audit_events(created_at);
```

The middleware is **best-effort**: a failure inside `AuditStore.log()` never breaks the request. Audit must be passive; if the audit DB is full or locked, the API keeps serving.

---

## 7. Test architecture

```
tests/
├── unit/
│   ├── test_graph_structure.py    (state shape, node wiring, edge routing)
│   ├── test_chunker.py            (tiktoken token counting, sentence splits)
│   ├── test_retrieval.py          (BM25, RRF, reranker, hybrid integration with mocks)
│   ├── test_citations.py          (regex parsing, chunk mapping, snippet truncation)
│   ├── test_eval.py               (dataset I/O, mock evaluator, runner, report)
│   ├── test_auth.py               (JWT, bcrypt, store, scope, FastAPI deps)
│   └── test_audit.py              (store CRUD, middleware helpers)
└── integration/
    └── test_auth_api.py           (login flow, /me, admin-only endpoints)
```

**Dependency injection everywhere.** `HybridRetriever`, `Reranker`, `evaluate_fn`, `run_agent_fn`, `UserStore`, `AuditStore` all accept their collaborators in `__init__`. Tests never need Ollama, ChromaDB, or a GPU.

---

## 8. Deliberate non-goals

- **No multi-tenant SaaS layer.** This is a deployed-per-customer product. One organization per Sift instance.
- **No agent tool-use beyond retrieval.** Sift answers from indexed documents. It does not browse the web, call APIs, or write code.
- **No streaming token-by-token answer.** SSE streams node transitions, not LLM tokens. The critique loop means tokens may be discarded; streaming partial answers would be misleading.
- **No fine-tuning pipeline.** The bet is that hybrid retrieval + reranker + critique loop wins more than fine-tuning the synthesizer for a single deployment.
