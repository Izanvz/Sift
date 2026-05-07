from src.agent.state import SiftState
from config.settings import settings


def route_after_classification(state: SiftState) -> str:
    """Si la query es ambigua, pedir clarificación. Si no, recuperar."""
    if state.get("query_type") == "ambiguous":
        return "clarification_request"
    return "retrieve"


def route_after_relevance(state: SiftState) -> str:
    """Decide si refinar la query o sintetizar."""
    scores = state.get("relevance_scores", [])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    iterations = state.get("iterations", 0)

    if avg_score < settings.relevance_threshold and iterations < settings.max_search_iterations:
        return "rewrite_query"
    return "synthesize"


def route_after_critique(state: SiftState) -> str:
    """Decide si reescribir la respuesta o dar por buena.

    Reescribe si:
    - score general < quality_gate_score  (defecto 8.0/10), O
    - faithfulness < faithfulness_hard_gate (defecto 6.0/10) — siempre corregir hallucinations

    Límite: max_rewrite_iterations para evitar bucle infinito.
    """
    critique = state.get("critique", {})
    score = critique.get("score", 10.0)
    faithfulness = critique.get("faithfulness", 10.0)
    rewrite_iterations = state.get("rewrite_iterations", 0)

    needs_rewrite = (
        score < settings.quality_gate_score
        or faithfulness < settings.faithfulness_hard_gate
    )

    if needs_rewrite and rewrite_iterations < settings.max_rewrite_iterations:
        return "rewrite_answer"
    return "format_response"
