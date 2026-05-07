"""Tests del módulo de evaluación — sin RAGAS, sin Ollama."""
import json
from pathlib import Path

import pytest

from src.eval.dataset import (
    QAPair,
    filter_by_tag,
    filter_by_type,
    load_jsonl,
    save_jsonl,
)
from src.eval.metrics import (
    EvalResult,
    EvalRow,
    evaluate_with_mock,
)
from src.eval.report import render_markdown, save_report
from src.eval.runner import (
    AgentRun,
    _check_source_recall,
    run_dataset,
)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def test_qapair_defaults():
    pair = QAPair(id="q1", query="What is X?", ground_truth="X is...")
    assert pair.expected_sources == []
    assert pair.tags == []
    assert pair.query_type == "factual"


def test_load_jsonl_roundtrip(tmp_path):
    path = tmp_path / "qa.jsonl"
    pairs = [
        QAPair(id="q1", query="A?", ground_truth="A is alpha", tags=["alpha"]),
        QAPair(id="q2", query="B?", ground_truth="B is beta", tags=["beta"]),
    ]
    save_jsonl(pairs, path)
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].id == "q1"
    assert loaded[1].tags == ["beta"]


def test_load_jsonl_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        "# comment\n"
        "\n"
        '{"id": "q1", "query": "A?", "ground_truth": "alpha"}\n'
        "# another comment\n"
        '{"id": "q2", "query": "B?", "ground_truth": "beta"}\n',
        encoding="utf-8",
    )
    loaded = load_jsonl(path)
    assert len(loaded) == 2


def test_load_jsonl_skips_malformed(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        '{"id": "q1", "query": "A?", "ground_truth": "alpha"}\n'
        "this is not json\n"
        '{"id": "q2", "query": "B?", "ground_truth": "beta"}\n',
        encoding="utf-8",
    )
    loaded = load_jsonl(path)
    assert len(loaded) == 2  # malformed line skipped


def test_load_jsonl_missing_file():
    with pytest.raises(FileNotFoundError):
        load_jsonl("/does/not/exist.jsonl")


def test_filter_by_tag():
    pairs = [
        QAPair(id="q1", query="A?", ground_truth="a", tags=["x"]),
        QAPair(id="q2", query="B?", ground_truth="b", tags=["y"]),
        QAPair(id="q3", query="C?", ground_truth="c", tags=["x", "y"]),
    ]
    assert len(filter_by_tag(pairs, "x")) == 2
    assert len(filter_by_tag(pairs, "y")) == 2
    assert len(filter_by_tag(pairs, "z")) == 0


def test_filter_by_type():
    pairs = [
        QAPair(id="q1", query="A?", ground_truth="a", query_type="factual"),
        QAPair(id="q2", query="B?", ground_truth="b", query_type="comparative"),
    ]
    assert len(filter_by_type(pairs, "factual")) == 1
    assert len(filter_by_type(pairs, "comparative")) == 1


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_evaluate_with_mock_returns_fixed_scores():
    rows = [
        EvalRow(id="q1", query="?", answer="a", contexts=["c"], ground_truth="g"),
        EvalRow(id="q2", query="?", answer="b", contexts=["c"], ground_truth="g"),
    ]
    result = evaluate_with_mock(rows)
    assert len(result.per_question) == 2
    assert result.aggregate["faithfulness"] == 0.85


def test_evaluate_with_mock_empty_rows():
    result = evaluate_with_mock([])
    assert result.per_question == []
    assert result.aggregate["faithfulness"] == 0.85


def test_evaluate_with_mock_custom_metrics():
    rows = [EvalRow(id="q1", query="?", answer="a", contexts=["c"], ground_truth="g")]
    result = evaluate_with_mock(rows, metrics=["faithfulness"])
    assert "faithfulness" in result.aggregate
    assert "answer_relevancy" not in result.aggregate


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def test_check_source_recall_substring_match():
    citations = [
        {"source_path": "/data/vercel-docs/edge.md"},
        {"source_path": "/data/other.md"},
    ]
    assert _check_source_recall(["vercel-docs"], citations) is True
    assert _check_source_recall(["nonexistent"], citations) is False


def test_check_source_recall_case_insensitive():
    citations = [{"source_path": "/data/Vercel-Docs/edge.md"}]
    assert _check_source_recall(["vercel-docs"], citations) is True


def test_check_source_recall_empty():
    assert _check_source_recall([], [{"source_path": "any"}]) is False
    assert _check_source_recall(["x"], []) is False


def test_run_dataset_with_mock_agent():
    dataset = [
        QAPair(id="q1", query="A?", ground_truth="alpha"),
        QAPair(id="q2", query="B?", ground_truth="beta"),
    ]

    def mock_agent(query: str) -> AgentRun:
        return AgentRun(
            answer=f"answer to {query}",
            contexts=["ctx1", "ctx2"],
            citations=[],
            chunks=[],
            latency_s=0.0,
        )

    result = run_dataset(dataset, mock_agent, evaluate_fn=evaluate_with_mock)
    assert result["n_total"] == 2
    assert result["n_evaluated"] == 2
    assert result["errors"] == []
    assert len(result["agent_runs"]) == 2


def test_run_dataset_handles_agent_errors():
    dataset = [
        QAPair(id="q1", query="A?", ground_truth="alpha"),
        QAPair(id="q2", query="B?", ground_truth="beta"),
    ]

    def broken_agent(query: str) -> AgentRun:
        if "A" in query:
            raise RuntimeError("Boom")
        return AgentRun(answer="ok", contexts=["c"], citations=[], chunks=[], latency_s=0.0)

    result = run_dataset(dataset, broken_agent, evaluate_fn=evaluate_with_mock)
    assert len(result["errors"]) == 1
    assert result["errors"][0]["id"] == "q1"
    assert result["n_evaluated"] == 1


def test_run_dataset_source_recall():
    dataset = [
        QAPair(
            id="q1", query="A?", ground_truth="a",
            expected_sources=["vercel-docs"],
        ),
        QAPair(
            id="q2", query="B?", ground_truth="b",
            expected_sources=["stripe"],
        ),
    ]

    def agent(query: str) -> AgentRun:
        return AgentRun(
            answer="ok",
            contexts=["c"],
            citations=[{"source_path": "/data/vercel-docs/edge.md"}],
            chunks=[],
            latency_s=0.0,
        )

    result = run_dataset(dataset, agent, evaluate_fn=evaluate_with_mock)
    # q1 matches (vercel-docs in path), q2 doesn't → 0.5
    assert result["source_recall"] == 0.5


def test_run_dataset_empty():
    result = run_dataset([], lambda q: AgentRun("", [], [], [], 0.0), evaluate_with_mock)
    assert result["n_total"] == 0
    assert result["n_evaluated"] == 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def test_render_markdown_basic():
    eval_result = EvalResult(
        per_question=[
            {"id": "q1", "query": "A?", "faithfulness": 0.9, "answer_relevancy": 0.85},
        ],
        aggregate={"faithfulness": 0.9, "answer_relevancy": 0.85},
    )
    runs = [AgentRun(answer="ok", contexts=[], citations=[], chunks=[], latency_s=1.5)]
    md = render_markdown(eval_result, runs, source_recall=0.8, n_total=1, n_evaluated=1, errors=[])
    assert "Aggregate Metrics" in md
    assert "faithfulness" in md
    assert "0.900" in md
    assert "Per-Question" in md


def test_render_markdown_handles_none_scores():
    eval_result = EvalResult(
        per_question=[{"id": "q1", "query": "A?", "faithfulness": None}],
        aggregate={"faithfulness": None},
    )
    md = render_markdown(eval_result, [], None, 1, 0, [])
    assert "—" in md  # placeholder para None


def test_render_markdown_with_errors():
    eval_result = EvalResult(per_question=[], aggregate={})
    md = render_markdown(
        eval_result, [], None, 1, 0,
        errors=[{"id": "q1", "error": "Connection refused"}],
    )
    assert "Errors" in md
    assert "Connection refused" in md


def test_save_report_creates_files(tmp_path):
    eval_result = EvalResult(
        per_question=[{"id": "q1", "query": "A?", "faithfulness": 0.9}],
        aggregate={"faithfulness": 0.9},
    )
    paths = save_report(
        eval_result=eval_result,
        agent_runs=[],
        source_recall=None,
        n_total=1,
        n_evaluated=1,
        errors=[],
        output_dir=tmp_path,
    )
    assert Path(paths["markdown"]).exists()
    assert Path(paths["json"]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["aggregate"]["faithfulness"] == 0.9
