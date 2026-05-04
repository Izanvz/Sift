from src.agent.state import ResearchState
from config.settings import settings


def route_after_quality_eval(state: ResearchState) -> str:
    """Decide si refinar la búsqueda o sintetizar."""
    avg_quality = (
        sum(state["quality_scores"]) / len(state["quality_scores"])
        if state["quality_scores"]
        else 0.0
    )
    iterations = state.get("iterations", 0)
    if avg_quality < settings.quality_threshold and iterations < settings.max_search_iterations:
        return "refine_query"
    return "synthesize"


def route_after_critique(state: ResearchState) -> str:
    """Decide si reescribir o pasar a human checkpoint."""
    critique = state.get("critique", {})
    score = critique.get("score", 10.0)
    rewrite_iterations = state.get("rewrite_iterations", 0)
    if score < settings.quality_gate_score and rewrite_iterations < settings.max_rewrite_iterations:
        return "rewrite"
    return "human_checkpoint"


def route_to_searches(state: ResearchState):
    """Fan-out paralelo via Send() API."""
    from langgraph.types import Send

    sends = []
    for subtopic in state["research_plan"]:
        sends.append(Send("search_web", {"query": subtopic, "source": "web"}))
        sends.append(Send("search_chromadb", {"query": subtopic, "source": "chromadb"}))
    return sends
