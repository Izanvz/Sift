"""RAGAS metrics wrapper — lazy import, Ollama como LLM judge.

Métricas:
- faithfulness: el answer está soportado por los contexts (no alucinaciones)
- answer_relevancy: el answer responde a la query
- context_precision: los contexts top-k son relevantes
- context_recall: los contexts cubren el ground_truth

RAGAS usa un LLM como juez. Por defecto OpenAI; aquí lo apuntamos a Ollama
local vía OpenAI-compatible API (mismo patrón que en nodes.py).

Para tests/CI sin Ollama: usar evaluate_with_mock() o pasar mock_metrics=True.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvalRow:
    """Una fila lista para evaluar — tras correr el agente sobre una QAPair."""
    id: str
    query: str
    answer: str
    contexts: list[str]
    ground_truth: str


@dataclass
class EvalResult:
    """Resultado agregado de una evaluación."""
    per_question: list[dict]   # uno por EvalRow, con sus métricas
    aggregate: dict            # promedio de cada métrica


# ---------------------------------------------------------------------------
# RAGAS evaluation (lazy import)
# ---------------------------------------------------------------------------

def evaluate_with_ragas(
    rows: list[EvalRow],
    metrics: list[str] | None = None,
    llm_model: str | None = None,
) -> EvalResult:
    """Corre RAGAS sobre los rows. Importa ragas/datasets lazy.

    Args:
        rows: filas con (query, answer, contexts, ground_truth).
        metrics: lista de métricas a calcular. Default: las 4 estándar.
        llm_model: modelo Ollama a usar como juez (default: settings.model_routing).

    Returns:
        EvalResult con per_question + aggregate.
    """
    if not rows:
        return EvalResult(per_question=[], aggregate={})

    metric_names = metrics or [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        logger.error("RAGAS or datasets not installed: %s", exc)
        return _empty_result(rows, metric_names)

    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    selected = [metric_map[m] for m in metric_names if m in metric_map]

    ds = Dataset.from_list([
        {
            "question": r.query,
            "answer": r.answer,
            "contexts": r.contexts,
            "ground_truth": r.ground_truth,
        }
        for r in rows
    ])

    try:
        llm = _build_ollama_judge(llm_model)
        result = evaluate(ds, metrics=selected, llm=llm) if llm else evaluate(ds, metrics=selected)
    except Exception as exc:
        logger.error("RAGAS evaluate failed: %s", exc)
        return _empty_result(rows, metric_names)

    return _format_result(rows, result, metric_names)


def evaluate_with_mock(rows: list[EvalRow], metrics: list[str] | None = None) -> EvalResult:
    """Mock: devuelve scores fijos. Útil para tests y smoke tests sin Ollama."""
    metric_names = metrics or [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    ]
    per_q = []
    for r in rows:
        scores = {m: 0.85 for m in metric_names}
        per_q.append({"id": r.id, "query": r.query, **scores})

    aggregate = {m: 0.85 for m in metric_names}
    return EvalResult(per_question=per_q, aggregate=aggregate)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ollama_judge(model_name: str | None):
    """Construye un LLM Ollama para usar como juez RAGAS.

    Ragas espera un LangChain ChatModel. Usamos langchain-ollama si está
    disponible. Si no, devolvemos None y RAGAS intentará OpenAI (fallará si
    no hay API key — esperado en setup local).
    """
    from config.settings import settings
    name = model_name or settings.model_routing
    try:
        from langchain_ollama import ChatOllama
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(ChatOllama(model=name, temperature=0))
    except Exception as exc:
        logger.warning("Could not build Ollama judge: %s", exc)
        return None


def _empty_result(rows: list[EvalRow], metric_names: list[str]) -> EvalResult:
    per_q = [
        {"id": r.id, "query": r.query, **{m: None for m in metric_names}}
        for r in rows
    ]
    aggregate = {m: None for m in metric_names}
    return EvalResult(per_question=per_q, aggregate=aggregate)


def _format_result(rows: list[EvalRow], ragas_result, metric_names: list[str]) -> EvalResult:
    """Convierte el resultado de RAGAS a EvalResult."""
    # ragas_result soporta .to_pandas() y dict-like access en versiones recientes
    try:
        df = ragas_result.to_pandas()
    except Exception:
        # fallback: intentar serializar directamente
        return _empty_result(rows, metric_names)

    per_q: list[dict] = []
    for i, row in enumerate(rows):
        entry = {"id": row.id, "query": row.query}
        for metric in metric_names:
            try:
                value = float(df.iloc[i][metric])
            except (KeyError, ValueError, IndexError):
                value = None
            entry[metric] = value
        per_q.append(entry)

    aggregate = {}
    for metric in metric_names:
        try:
            values = [q[metric] for q in per_q if q[metric] is not None]
            aggregate[metric] = sum(values) / len(values) if values else None
        except Exception:
            aggregate[metric] = None

    return EvalResult(per_question=per_q, aggregate=aggregate)
