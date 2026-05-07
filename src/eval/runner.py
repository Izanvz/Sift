"""Eval runner — corre el agente sobre un dataset de Q&A.

Pipeline:
    QAPair → run_agent_fn(query) → (answer, chunks)
           → EvalRow(query, answer, contexts, ground_truth)
           → evaluate_fn(rows) → EvalResult

Diseño:
    - run_agent_fn y evaluate_fn son inyectables → testeable sin Ollama/RAGAS
    - Manejo de errores por pregunta: si una falla, continúa con las demás
    - Devuelve además metadata útil (latencia, errores, recall sobre expected_sources)
"""
import logging
import time
from dataclasses import dataclass
from typing import Callable

from src.eval.dataset import QAPair
from src.eval.metrics import EvalResult, EvalRow, evaluate_with_ragas

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    """Resultado de correr el agente sobre una pregunta."""
    answer: str
    contexts: list[str]
    citations: list[dict]
    chunks: list[dict]
    latency_s: float
    error: str | None = None


# Tipo del callable que ejecuta el agente: query → AgentRun
AgentRunFn = Callable[[str], AgentRun]


def run_dataset(
    dataset: list[QAPair],
    run_agent_fn: AgentRunFn,
    evaluate_fn: Callable[[list[EvalRow]], EvalResult] = evaluate_with_ragas,
) -> dict:
    """Corre el agente sobre cada QAPair y evalúa con RAGAS.

    Returns:
        dict con keys: 'eval_result', 'agent_runs', 'source_recall', 'errors'
    """
    rows: list[EvalRow] = []
    runs: list[AgentRun] = []
    source_hits = 0
    source_total = 0
    errors: list[dict] = []

    for pair in dataset:
        run = _safe_run(pair, run_agent_fn)
        runs.append(run)

        if run.error:
            errors.append({"id": pair.id, "error": run.error})
            continue

        # Recall sobre expected_sources (¿alguna citation matchea?)
        if pair.expected_sources:
            source_total += 1
            if _check_source_recall(pair.expected_sources, run.citations):
                source_hits += 1

        rows.append(EvalRow(
            id=pair.id,
            query=pair.query,
            answer=run.answer,
            contexts=run.contexts,
            ground_truth=pair.ground_truth,
        ))

    eval_result = evaluate_fn(rows) if rows else EvalResult(per_question=[], aggregate={})
    source_recall = source_hits / source_total if source_total else None

    return {
        "eval_result": eval_result,
        "agent_runs": runs,
        "source_recall": source_recall,
        "errors": errors,
        "n_total": len(dataset),
        "n_evaluated": len(rows),
    }


def _safe_run(pair: QAPair, run_agent_fn: AgentRunFn) -> AgentRun:
    """Ejecuta el agente capturando excepciones."""
    start = time.perf_counter()
    try:
        run = run_agent_fn(pair.query)
        run.latency_s = time.perf_counter() - start
        return run
    except Exception as exc:
        logger.warning("Agent failed on %s: %s", pair.id, exc)
        return AgentRun(
            answer="",
            contexts=[],
            citations=[],
            chunks=[],
            latency_s=time.perf_counter() - start,
            error=str(exc),
        )


def _check_source_recall(expected: list[str], citations: list[dict]) -> bool:
    """Devuelve True si alguna citation matchea algún expected source.

    Match: substring case-insensitive sobre source_path.
    """
    cited_paths = [
        (c.get("source_path") or "").lower()
        for c in citations
    ]
    for exp in expected:
        exp_low = exp.lower()
        if any(exp_low in path for path in cited_paths):
            return True
    return False


# ---------------------------------------------------------------------------
# Default: corre el agente real (Sift) sobre una query
# ---------------------------------------------------------------------------

def default_agent_run(query: str) -> AgentRun:
    """Default agent runner — invoca el grafo Sift compilado.

    Lazy import para evitar cargar el grafo en tests.
    """
    from src.agent.graph import build_graph

    graph = build_graph()
    final_state = graph.invoke({"query": query, "user_id": None})

    chunks = final_state.get("chunks", [])
    citations = final_state.get("citations", [])

    return AgentRun(
        answer=final_state.get("answer", ""),
        contexts=[c.get("content", "") for c in chunks],
        citations=[
            dict(c) if not isinstance(c, dict) else c
            for c in citations
        ],
        chunks=[dict(c) if not isinstance(c, dict) else c for c in chunks],
        latency_s=0.0,  # se rellena en _safe_run
    )
