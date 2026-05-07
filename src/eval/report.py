"""Reporte markdown + JSON del eval."""
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.eval.metrics import EvalResult


def render_markdown(
    eval_result: EvalResult,
    agent_runs: list,
    source_recall: float | None,
    n_total: int,
    n_evaluated: int,
    errors: list[dict],
    title: str = "Sift — Evaluation Report",
) -> str:
    """Genera un markdown legible con métricas y per-question breakdown."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append(f"**Dataset size:** {n_total}  ")
    lines.append(f"**Evaluated:** {n_evaluated}  ")
    lines.append(f"**Errors:** {len(errors)}  ")
    if source_recall is not None:
        lines.append(f"**Source recall (citations match expected):** {source_recall:.1%}  ")
    if agent_runs:
        latencies = [r.latency_s for r in agent_runs if r.latency_s]
        if latencies:
            lines.append(
                f"**Latency:** mean={statistics.mean(latencies):.2f}s  "
                f"median={statistics.median(latencies):.2f}s  "
                f"max={max(latencies):.2f}s"
            )
    lines.append("")

    # Aggregate metrics
    lines.append("## Aggregate Metrics")
    lines.append("")
    if eval_result.aggregate:
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for metric, score in eval_result.aggregate.items():
            score_str = f"{score:.3f}" if score is not None else "—"
            lines.append(f"| {metric} | {score_str} |")
    else:
        lines.append("_No metrics computed._")
    lines.append("")

    # Per-question breakdown
    if eval_result.per_question:
        lines.append("## Per-Question Breakdown")
        lines.append("")
        metrics = [k for k in eval_result.per_question[0].keys() if k not in ("id", "query")]
        header = "| ID | Query | " + " | ".join(metrics) + " |"
        sep = "|----|-------|" + "|".join(["------"] * len(metrics)) + "|"
        lines.append(header)
        lines.append(sep)
        for q in eval_result.per_question:
            query_short = q["query"][:60].replace("|", "\\|")
            scores = [
                f"{q[m]:.2f}" if isinstance(q.get(m), (int, float)) else "—"
                for m in metrics
            ]
            lines.append(f"| {q['id']} | {query_short} | " + " | ".join(scores) + " |")
        lines.append("")

    # Errors
    if errors:
        lines.append("## Errors")
        lines.append("")
        for err in errors:
            lines.append(f"- **{err['id']}**: {err['error']}")
        lines.append("")

    return "\n".join(lines)


def save_report(
    eval_result: EvalResult,
    agent_runs: list,
    source_recall: float | None,
    n_total: int,
    n_evaluated: int,
    errors: list[dict],
    output_dir: str | Path = "data/eval/reports",
) -> dict:
    """Guarda markdown + JSON. Devuelve los paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    md_path = out / f"report-{timestamp}.md"
    json_path = out / f"report-{timestamp}.json"

    md_content = render_markdown(
        eval_result, agent_runs, source_recall, n_total, n_evaluated, errors
    )
    md_path.write_text(md_content, encoding="utf-8")

    json_content = {
        "timestamp": timestamp,
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "source_recall": source_recall,
        "aggregate": eval_result.aggregate,
        "per_question": eval_result.per_question,
        "errors": errors,
    }
    json_path.write_text(json.dumps(json_content, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}
