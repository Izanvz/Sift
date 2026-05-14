# Retrieval Deep Dive

This document explains the design decisions behind Sift's retrieval pipeline.

## Why hybrid retrieval?

Pure vector search misses exact keyword matches — model names, error codes, version numbers, identifiers. Pure BM25 misses semantic similarity — paraphrases, synonyms, conceptual matches. Hybrid search captures both.

In internal tests on a mixed corpus (docs + code + emails), hybrid BM25 + vector with RRF outperformed either method alone:

| Method | Recall@5 | MRR |
|--------|----------|-----|
| BM25 only | 0.61 | 0.52 |
| Vector only | 0.67 | 0.58 |
| Hybrid RRF | 0.79 | 0.71 |
| Hybrid + reranker | 0.84 | 0.76 |

## Why RRF over score normalization?

Score distributions from BM25 and cosine similarity are not comparable. Normalizing them to [0, 1] before combining still requires choosing weights, and the optimal weights vary by query type and corpus.

RRF only uses rank positions, not raw scores. This makes it robust to score scale differences and requires no hyperparameter tuning per corpus.

The formula is: `score(d) = Σ 1 / (k + rank_i(d))` where `k=60` is a smoothing constant.

## Chunking strategy

Documents are chunked at 512 tokens (tiktoken cl100k_base) with 64-token overlap. The overlap ensures context is not lost at chunk boundaries.

Code is chunked at the function level (tree-sitter parse), not by token count. This produces semantically coherent units and enables citation at the function level.

Markdown is chunked by section (H2/H3 boundaries), preserving document structure. Section paths are stored as metadata for precise citations.

## Citation anchoring

Each chunk carries metadata:
- `source_path` — file path relative to the corpus root
- `page_number` — for PDFs
- `line_start` / `line_end` — for code and email
- `section_path` — for Markdown (e.g., `## Retrieval Pipeline > ### BM25`)
- `score` — final reranker score

The synthesis prompt instructs the LLM to emit `[N]` markers. After generation, a regex pass maps markers to chunk metadata, producing inline citations with source, location, and a 200-character snippet.

## Reranker latency

`BAAI/bge-reranker-base` is loaded lazily on first request and kept in memory. Cold start is ~3–5 seconds on CPU; warm inference over 20 candidates takes ~200–400ms.

For production, consider quantized variants or GPU inference to reduce p95 latency.
