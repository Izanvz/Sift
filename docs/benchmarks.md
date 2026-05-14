# Benchmarks

How to reproduce, what was measured, and the latest numbers.

---

## 1. Reproducing

```bash
# Boot infra
docker compose up -d

# Index the three corpora used in the golden dataset
python scripts/ingest.py --source data/sources/enterprise/vercel-docs --connector markdown
python scripts/ingest.py --source data/sources/code/stripe-go        --connector code
python scripts/ingest.py --source data/sources/enterprise/enron       --connector email

# Run RAGAS evaluation
python scripts/eval.py --dataset data/eval/golden_qa.jsonl
```

Output lands in `data/eval/reports/report-<timestamp>.{md,json}`.

---

## 2. Golden dataset

15 hand-crafted Q&A pairs (8 evaluated in the latest run — vercel tag) in `data/eval/golden_qa.jsonl`:

| Tag | N | Query types |
|-----|---|-------------|
| `vercel`     | 7 | factual (4) · analytical (1) · comparative (2) |
| `stripe`     | 4 | factual (2) · analytical (2) |
| `enron`      | 2 | factual (1) · analytical (1) |
| `meta`       | 1 | ambiguous |
| `cron`       | 1 | comparative |

Each pair specifies `expected_sources` (substrings expected in cited paths) used to compute **source recall** independently of RAGAS.

---

## 3. Metrics

### RAGAS metrics

| Metric | What it measures | Target |
|--------|------------------|--------|
| **faithfulness** | Are answer claims grounded in retrieved context? | ≥ 0.80 |
| **answer_relevancy** | Does the answer address the question? | ≥ 0.85 |
| **context_precision** | How much of the retrieved context is actually used? | ≥ 0.65 |
| **context_recall** | Does the retrieved context contain the ground truth? | ≥ 0.80 |

Judge model: `qwen2.5:7b` via Ollama (LangchainLLMWrapper). RAGAS does not need a frontier model for usable signal; the relative ordering is what matters for ablations.

### Custom metrics

| Metric | Definition |
|--------|------------|
| **source_recall** | `|{q: any expected_source ⊆ any cited path}| / |{q: expected_sources non-empty}|` — case-insensitive substring match |
| **latency** | Wall-clock time per query through the full graph (`run_agent_fn`) |

---

## 4. Latest run

_2026-05-12 · 8 Q&A pairs (vercel tag) · CPU-only Ollama (qwen2.5:7b)_

| Metric | Score | Notes |
|--------|-------|-------|
| faithfulness | — | RAGAS Ollama judge needs `EMBEDDING_MODEL` env var |
| answer_relevancy | — | same |
| context_precision | — | same |
| context_recall | — | same |
| **source_recall** | **100%** | citations matched expected sources on all 8 questions |
| mean latency | 618.94 s | CPU-only; expect 5–15 s on GPU |
| median latency | 635.82 s | |
| p95 latency | ≈ 885.30 s (max) | |

To get RAGAS scores: set `EMBEDDING_MODEL=nomic-embed-text` (or any Ollama embedding model) in your `.env`, then re-score without re-running the agent:

```bash
python scripts/eval_score.py --report data/eval/reports/report-20260512-145350.json
```

---

## 5. Planned ablations

To be run once the baseline is in place:

| Ablation | Question |
|----------|----------|
| BM25 only (no vector) | How much does dense retrieval contribute? |
| Vector only (no BM25) | What recall does keyword search add? |
| RRF disabled (concat + dedupe) | Is RRF worth it vs. naive merging? |
| Reranker disabled | What's the cross-encoder lift? |
| No critique loop | Does the rewrite cycle improve faithfulness in practice? |
| Synth model: qwen2.5:7b → llama3.1:8b | Sensitivity to synthesizer choice |

Each row will be a row in this file with the same metric set.

---

## 6. CI / smoke

`scripts/eval.py --mock` returns fixed scores (faithfulness=0.85 etc.) without touching Ollama or RAGAS. It runs in <1s and is safe in CI to verify the pipeline is wired correctly.

```bash
python scripts/eval.py --dataset data/eval/golden_qa.jsonl --mock
```
