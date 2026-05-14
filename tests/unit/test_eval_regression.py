"""Tests para eval_regression.py — lógica de comparación."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.eval_regression import compare_reports, RegressionResult


def _make_report(faithfulness=None, latency_s_list=None, source_recall=None):
    agent_runs = []
    if latency_s_list is not None:
        agent_runs = [{"latency_s": v, "answer": "", "contexts": [], "citations": [], "error": None}
                      for v in latency_s_list]
    return {
        "timestamp": "20260101-000000",
        "n_total": 8,
        "n_evaluated": 8,
        "source_recall": source_recall,
        "aggregate": {
            "faithfulness": faithfulness,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        },
        "per_question": [],
        "errors": [],
        "agent_runs": agent_runs,
    }


def test_pass_when_no_regression():
    baseline = _make_report(faithfulness=0.90, latency_s_list=[1.0, 2.0])
    current  = _make_report(faithfulness=0.88, latency_s_list=[1.1, 2.1])
    result = compare_reports(baseline, current)
    assert result.status == "PASS"
    assert result.faithfulness_delta == pytest.approx(-0.02)


def test_fail_faithfulness_drop():
    baseline = _make_report(faithfulness=0.90)
    current  = _make_report(faithfulness=0.84)  # -6% > threshold 5%
    result = compare_reports(baseline, current)
    assert result.status == "FAIL"
    assert "faithfulness" in result.failures


def test_pass_faithfulness_exactly_at_threshold():
    baseline = _make_report(faithfulness=0.90)
    current  = _make_report(faithfulness=0.855)  # -5% exacto → PASS (> not >=)
    result = compare_reports(baseline, current)
    assert result.status == "PASS"


def test_fail_latency_increase():
    baseline = _make_report(latency_s_list=[1.0, 1.0])
    current  = _make_report(latency_s_list=[1.3, 1.3])  # +30% > threshold 20%
    result = compare_reports(baseline, current)
    assert result.status == "FAIL"
    assert "latency" in result.failures


def test_pass_latency_exactly_at_threshold():
    baseline = _make_report(latency_s_list=[1.0, 1.0])
    current  = _make_report(latency_s_list=[1.2, 1.2])  # +20% exacto → PASS
    result = compare_reports(baseline, current)
    assert result.status == "PASS"


def test_skip_when_both_null_faithfulness():
    baseline = _make_report(faithfulness=None)
    current  = _make_report(faithfulness=None)
    result = compare_reports(baseline, current)
    # faithfulness no puede compararse — status puede ser PASS o SKIP
    assert "faithfulness" not in result.failures


def test_skip_when_no_agent_runs():
    baseline = _make_report(faithfulness=0.90, latency_s_list=None)
    current  = _make_report(faithfulness=0.855, latency_s_list=None)
    result = compare_reports(baseline, current)
    # latency no comparable — solo faithfulness importa
    assert result.status == "PASS"


def test_both_fail():
    baseline = _make_report(faithfulness=0.90, latency_s_list=[1.0])
    current  = _make_report(faithfulness=0.80, latency_s_list=[2.0])
    result = compare_reports(baseline, current)
    assert result.status == "FAIL"
    assert "faithfulness" in result.failures
    assert "latency" in result.failures
