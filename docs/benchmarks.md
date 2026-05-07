# Benchmarks

How to reproduce, what was measured, and the latest numbers.

> **Status:** placeholders below until the full eval runs against the real corpus. Once the corpus is fully ingested and Ollama is warm, results will be filled in by re-running `scripts/eval.py` and committing the timestamped report.

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

15 hand-crafted Q&A pairs in `data/eval/golden_qa.jsonl`:

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

> Will be replaced after the next end-to-end run. Snapshot only.

| Metric | Score |
|--------|-------|
| faithfulness | _pending_ |
| answer_relevancy | _pending_ |
| context_precision | _pending_ |
| context_recall | _pending_ |
| source_recall | _pending_ |
| mean latency | _pending_ |
| p95 latency | _pending_ |

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
