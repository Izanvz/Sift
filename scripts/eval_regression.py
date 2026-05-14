"""Compara dos reportes JSON de eval. Exit 0=PASS, 1=FAIL, 2=SKIP."""
import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path


FAITHFULNESS_THRESHOLD = 0.05   # drop > 5% → FAIL
LATENCY_THRESHOLD      = 0.20   # increase > 20% → FAIL


@dataclass
class RegressionResult:
    status: str            # "PASS" | "FAIL" | "SKIP"
    failures: list[str] = field(default_factory=list)
    faithfulness_delta: float | None = None
    latency_delta_pct: float | None = None


def _mean_latency(report: dict) -> float | None:
    runs = report.get("agent_runs") or []
    values = [r["latency_s"] for r in runs if r.get("latency_s") is not None]
    return statistics.mean(values) if values else None


def compare_reports(baseline: dict, current: dict) -> RegressionResult:
    failures: list[str] = []
    faith_delta = None
    lat_delta = None

    # --- faithfulness ---
    b_faith = (baseline.get("aggregate") or {}).get("faithfulness")
    c_faith = (current.get("aggregate") or {}).get("faithfulness")
    if b_faith is not None and c_faith is not None:
        faith_delta = c_faith - b_faith
        if faith_delta < -FAITHFULNESS_THRESHOLD:
            failures.append("faithfulness")

    # --- latency ---
    b_lat = _mean_latency(baseline)
    c_lat = _mean_latency(current)
    if b_lat is not None and c_lat is not None and b_lat > 0:
        lat_delta = (c_lat - b_lat) / b_lat
        if lat_delta > LATENCY_THRESHOLD:
            failures.append("latency")

    status = "FAIL" if failures else "PASS"
    return RegressionResult(
        status=status,
        failures=failures,
        faithfulness_delta=faith_delta,
        latency_delta_pct=lat_delta,
    )


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.1%}" if pct else f"{value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sift regression eval")
    parser.add_argument("--baseline", required=True, help="Path al JSON baseline")
    parser.add_argument("--current",  required=True, help="Path al JSON current")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    current_path  = Path(args.current)

    if not baseline_path.exists():
        print(f"ERROR: baseline no encontrado: {baseline_path}", file=sys.stderr)
        sys.exit(2)
    if not current_path.exists():
        print(f"ERROR: current no encontrado: {current_path}", file=sys.stderr)
        sys.exit(2)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current  = json.loads(current_path.read_text(encoding="utf-8"))

    result = compare_reports(baseline, current)

    print(f"\n{'='*50}")
    print(f"  Sift Regression Eval - {result.status}")
    print(f"{'='*50}")
    print(f"  Baseline : {baseline_path.name}")
    print(f"  Current  : {current_path.name}")
    print(f"{'-'*50}")
    print(f"  Faithfulness Delta : {_fmt(result.faithfulness_delta)}"
          f"  (threshold > -{FAITHFULNESS_THRESHOLD:.0%})")
    print(f"  Latency Delta      : {_fmt(result.latency_delta_pct, pct=True)}"
          f"  (threshold < +{LATENCY_THRESHOLD:.0%})")
    if result.failures:
        print(f"\n  FAILURES: {', '.join(result.failures)}")
    print(f"{'='*50}\n")

    sys.exit(0 if result.status == "PASS" else 1)


if __name__ == "__main__":
    main()
