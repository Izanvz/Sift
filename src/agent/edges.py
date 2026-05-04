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
    """Decide si reescribir la respuesta o dar por buena."""
    score = state.get("critique", {}).get("score", 10.0)
    rewrite_iterations = state.get("rewrite_iterations", 0)

    if score < settings.quality_gate_score and rewrite_iterations < settings.max_rewrite_iterations:
        return "rewrite_answer"
    return "format_response"
