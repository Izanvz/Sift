# Hybrid Retrieval in Sift

Sift uses a hybrid retrieval pipeline that combines BM25 keyword search with dense vector search, fused via Reciprocal Rank Fusion (RRF), and re-ranked with a cross-encoder.

## Why hybrid?

Neither BM25 nor dense vectors alone are sufficient:

- **BM25** excels at exact keyword matches and rare terms (product names, error codes, identifiers). It fails on semantic paraphrases.
- **Dense vectors** (embeddings) capture semantic meaning and synonyms. They fail on rare out-of-vocabulary terms.

Combining both gives higher recall than either alone.

## BM25

BM25 (Best Match 25) is a probabilistic ranking function. Sift uses `rank_bm25` (BM25Okapi variant) with an in-memory index built at startup from all indexed chunks.

Key parameters:
- `bm25_top_k: 30` — candidates retrieved per query
- Index is rebuilt on each ingestion run

## Dense vector search

Embeddings are stored in ChromaDB using the default `all-MiniLM-L6-v2` model. Queries are embedded at search time and matched via cosine similarity.

Key parameters:
- `vector_top_k: 30` — candidates retrieved per query
- `chromadb_collection: sift_documents` — single collection for all corpora

## Reciprocal Rank Fusion (RRF)

RRF merges the two ranked lists into a single ranking without requiring score normalization. For each document, its RRF score is:

```
score(d) = sum over lists L of: 1 / (k + rank_L(d))
```

Where `k = 60` (Cormack et al. 2009). Documents appearing in both lists get a combined boost; documents in only one list still appear.

Key parameters:
- `rrf_k: 60` — controls score compression. Higher k reduces the impact of top ranks.
- `retrieval_top_k: 20` — documents kept after RRF, before reranking

## Cross-encoder reranker

After RRF, the top-20 candidates are re-ranked by `BAAI/bge-reranker-base`, a cross-encoder that scores (query, document) pairs jointly. This is more accurate than bi-encoder similarity but too slow to run on the full index.

Key parameters:
- `reranker_model: BAAI/bge-reranker-base`
- `reranker_enabled: true` — can be disabled in CI
- `synthesis_top_k: 5` — final chunks passed to the synthesis LLM

## Scope filtering

All retrieval steps respect per-user corpus permissions. The ChromaDB `where` clause filters by `corpus` metadata before vector search. BM25 candidates are filtered post-retrieval. Only chunks the user is authorized to see reach the synthesis step.
