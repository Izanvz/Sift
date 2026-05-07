"""CLI de evaluación — corre el agente sobre un dataset Q&A y genera reporte.

Uso:
    python scripts/eval.py --dataset data/eval/golden_qa.jsonl
    python scripts/eval.py --dataset data/eval/golden_qa.jsonl --mock
    python scripts/eval.py --dataset data/eval/golden_qa.jsonl --tag vercel
    python scripts/eval.py --dataset data/eval/golden_qa.jsonl --limit 5
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.dataset import filter_by_tag, load_jsonl
from src.eval.metrics import evaluate_with_mock, evaluate_with_ragas
from src.eval.report import save_report
from src.eval.runner import default_agent_run, run_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sift evaluation runner")
    parser.add_argument("--dataset", required=True, help="Path al JSONL")
    parser.add_argument("--tag", help="Filtra Q&A por tag")
    parser.add_argument("--limit", type=int, help="Limita número de preguntas")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Usa scores fijos (smoke test sin Ollama)",
    )
    parser.add_argument(
        "--output",
        default="data/eval/reports",
        help="Directorio del reporte",
    )
    args = parser.parse_args()

    dataset = load_jsonl(args.dataset)
    if args.tag:
        dataset = filter_by_tag(dataset, args.tag)
    if args.limit:
        dataset = dataset[:args.limit]

    logger.info("Loaded %d Q&A pairs from %s", len(dataset), args.dataset)
    if not dataset:
        logger.error("No Q&A pairs after filtering. Aborting.")
        raise SystemExit(1)

    evaluate_fn = evaluate_with_mock if args.mock else evaluate_with_ragas

    logger.info("Running agent (mock=%s)...", args.mock)
    result = run_dataset(
        dataset=dataset,
        run_agent_fn=default_agent_run,
        evaluate_fn=evaluate_fn,
    )

    paths = save_report(
        eval_result=result["eval_result"],
        agent_runs=result["agent_runs"],
        source_recall=result["source_recall"],
        n_total=result["n_total"],
        n_evaluated=result["n_evaluated"],
        errors=result["errors"],
        output_dir=args.output,
    )

    logger.info("Report written:\n  %s\n  %s", paths["markdown"], paths["json"])

    agg = result["eval_result"].aggregate
    if agg:
        logger.info("Aggregate scores:")
        for metric, score in agg.items():
            score_str = f"{score:.3f}" if score is not None else "—"
            logger.info("  %s: %s", metric, score_str)


if __name__ == "__main__":
    main()
