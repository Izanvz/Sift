from langgraph.graph import StateGraph, START, END

from src.agent.state import SiftState
from src.agent.edges import (
    route_after_classification,
    route_after_critique,
    route_after_relevance,
)
from src.agent.nodes import (
    clarification_request,
    evaluate_relevance,
    format_response,
    gather,
    retrieve,
    rewrite_answer,
    rewrite_query,
    route_query,
    self_critique,
    synthesize,
)


def build_graph(checkpointer=None):
    """Construye y compila el grafo LangGraph de Sift v2.

    Flujo principal:
        START → route_query → [clarification_request | retrieve]
                             → gather → evaluate_relevance
                             → [rewrite_query → retrieve (ciclo) | synthesize]
                             → self_critique
                             → [rewrite_answer → synthesize (ciclo) | format_response]
                             → END

    Args:
        checkpointer: SqliteSaver u otro checkpointer de LangGraph.
                      None compila sin persistencia.

    Returns:
        CompiledStateGraph listo para invocar.
    """
    builder = StateGraph(SiftState)

    # --- Nodos ---
    builder.add_node("route_query", route_query)
    builder.add_node("clarification_request", clarification_request)
    builder.add_node("retrieve", retrieve)
    builder.add_node("gather", gather)
    builder.add_node("evaluate_relevance", evaluate_relevance)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("synthesize", synthesize)
    builder.add_node("self_critique", self_critique)
    builder.add_node("rewrite_answer", rewrite_answer)
    builder.add_node("format_response", format_response)

    # --- Edges ---
    builder.add_edge(START, "route_query")

    # Clasificación → clarificación (si ambigua) o retrieval
    builder.add_conditional_edges(
        "route_query",
        route_after_classification,
        {"clarification_request": "clarification_request", "retrieve": "retrieve"},
    )

    # Tras clarificación (human-in-the-loop), continuar con retrieval
    builder.add_edge("clarification_request", "retrieve")

    # Retrieval → gather → evaluate
    builder.add_edge("retrieve", "gather")
    builder.add_edge("gather", "evaluate_relevance")

    # Ciclo 1: relevancia insuficiente → reescribir query → retrieve de nuevo
    builder.add_conditional_edges(
        "evaluate_relevance",
        route_after_relevance,
        {"rewrite_query": "rewrite_query", "synthesize": "synthesize"},
    )
    builder.add_edge("rewrite_query", "retrieve")

    # Generación → critique
    builder.add_edge("synthesize", "self_critique")

    # Ciclo 2: score bajo → reescribir respuesta → synthesize de nuevo
    builder.add_conditional_edges(
        "self_critique",
        route_after_critique,
        {"rewrite_answer": "rewrite_answer", "format_response": "format_response"},
    )
    builder.add_edge("rewrite_answer", "synthesize")

    # Fin
    builder.add_edge("format_response", END)

    # --- Compilación ---
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return builder.compile(
        interrupt_before=["clarification_request"],
        **compile_kwargs,
    )
