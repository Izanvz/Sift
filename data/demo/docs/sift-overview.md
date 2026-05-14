# Sift — System Overview

Sift is an on-premise RAG (Retrieval-Augmented Generation) agent for querying enterprise documents. All processing happens locally — no data leaves your infrastructure.

## Architecture

Sift is built on a LangGraph state graph with the following nodes:

- **route_query** — classifies the query as factual, analytical, comparative, or ambiguous
- **retrieve** — runs hybrid BM25 + dense vector retrieval, fuses results with RRF, reranks with BGE
- **synthesize** — generates a grounded answer with inline citations
- **critique** — scores faithfulness, completeness, and citation quality
- **rewrite** — rewrites the answer if critique scores fall below thresholds

## Retrieval Pipeline

### BM25 (keyword search)
Uses `rank_bm25` with BM25Okapi. Tokenizes queries and documents with `tiktoken` (cl100k_base). Returns top-K chunks by BM25 score.

### Dense vector search
ChromaDB with `all-MiniLM-L6-v2` embeddings. Queries are embedded and cosine similarity is used to find relevant chunks.

### Reciprocal Rank Fusion (RRF)
BM25 and vector results are fused using RRF (Cormack 2009) with `k=60`. This reduces position bias and combines complementary signals.

### BGE Reranker
The fused candidates are reranked with `BAAI/bge-reranker-base` (cross-encoder). The model scores each (query, chunk) pair directly, yielding more accurate relevance ordering than bi-encoder embeddings.

## Authentication and Scopes

Sift uses JWT bearer tokens (OAuth2 password flow). Each user has a list of `scopes` that map to ChromaDB corpus names. The retrieval layer filters by `{"corpus": {"$in": user.scopes}}` so users can only see documents they have access to.

Admin users bypass scope filtering.

## Self-Critique Loop

After synthesis, a separate LLM-as-judge call scores the answer:
- **faithfulness** (0–10): is the answer grounded in the retrieved chunks?
- **completeness** (0–10): does it answer the full question?
- **citation_quality** (0–10): are citations accurate and present?

If `faithfulness < 6.0`, the answer is rewritten regardless of other scores. Up to 2 rewrite iterations are allowed.

## Audit Log

Every query, login, error, and system event is appended to a SQLite table. Records include timestamp, user ID, query, latency, and IP. The log is append-only — no update or delete operations are issued.
