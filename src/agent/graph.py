from langgraph.graph import StateGraph, START, END

from src.agent.state import ResearchState
from src.agent.edges import route_after_quality_eval, route_after_critique, route_to_searches
from src.agent.nodes import (
    plan_research,
    search_web,
    search_chromadb,
    gather_results,
    evaluate_quality,
    refine_query,
    synthesize,
    self_critique,
    rewrite,
    human_checkpoint_node,
    generate_report,
)
from config.settings import settings


def _try_import_arxiv_node():
    """Importa el nodo search_arxiv solo si está disponible."""
    try:
        from src.agent.nodes import search_arxiv
        return search_arxiv
    except ImportError:
        return None


def build_graph(checkpointer=None):
    """Construye y compila el grafo LangGraph del Research Agent.

    Args:
        checkpointer: Instancia de checkpointer compatible con LangGraph
                      (e.g., SqliteSaver). Si es None se compila sin persistencia.

    Returns:
        CompiledStateGraph listo para invocar.
    """
    builder = StateGraph(ResearchState)

    # --- Nodos ---
    builder.add_node("plan_research", plan_research)
    builder.add_node("search_web", search_web)
    builder.add_node("search_chromadb", search_chromadb)

    search_arxiv = _try_import_arxiv_node()
    if search_arxiv is not None:
        builder.add_node("search_arxiv", search_arxiv)

    builder.add_node("gather_results", gather_results)
    builder.add_node("evaluate_quality", evaluate_quality)
    builder.add_node("refine_query", refine_query)
    builder.add_node("synthesize", synthesize)
    builder.add_node("self_critique", self_critique)
    builder.add_node("rewrite", rewrite)
    builder.add_node("human_checkpoint", human_checkpoint_node)
    builder.add_node("generate_report", generate_report)

    # --- Edges: inicio → planificación ---
    builder.add_edge(START, "plan_research")

    # --- Fan-out: plan_research → search_* via Send() ---
    builder.add_conditional_edges(
        "plan_research",
        route_to_searches,
        ["search_web", "search_chromadb"],
    )

    # --- Fan-in: search_* → gather_results ---
    builder.add_edge("search_web", "gather_results")
    builder.add_edge("search_chromadb", "gather_results")

    if search_arxiv is not None:
        builder.add_edge("search_arxiv", "gather_results")

    # --- gather_results → evaluate_quality ---
    builder.add_edge("gather_results", "evaluate_quality")

    # --- Ciclo 1: quality loop (evaluate → refine → fan-out de nuevo) ---
    builder.add_conditional_edges(
        "evaluate_quality",
        route_after_quality_eval,
        {
            "refine_query": "refine_query",
            "synthesize": "synthesize",
        },
    )

    # refine_query vuelve al fan-out via Send()
    builder.add_conditional_edges(
        "refine_query",
        route_to_searches,
        ["search_web", "search_chromadb"],
    )

    # --- Ciclo 2: rewrite loop (synthesize → critique → rewrite → synthesize) ---
    builder.add_edge("synthesize", "self_critique")

    builder.add_conditional_edges(
        "self_critique",
        route_after_critique,
        {
            "rewrite": "rewrite",
            "human_checkpoint": "human_checkpoint",
        },
    )

    builder.add_edge("rewrite", "synthesize")

    # --- human_checkpoint → generate_report → END ---
    builder.add_edge("human_checkpoint", "generate_report")
    builder.add_edge("generate_report", END)

    # --- Compilación ---
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    graph = builder.compile(
        interrupt_before=["human_checkpoint"],
        **compile_kwargs,
    )

    return graph
