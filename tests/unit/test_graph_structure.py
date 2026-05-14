"""Tests de estructura del grafo Sift v2 — no requieren Ollama ni ChromaDB."""
import pytest
from src.agent.graph import build_graph
from src.agent.state import SiftState, CritiqueOutput, QueryClassification


def test_graph_builds_without_checkpointer():
    """El grafo debe compilar sin errores sin checkpointer."""
    graph = build_graph(checkpointer=None)
    assert graph is not None


def test_graph_has_expected_nodes():
    """El grafo debe tener exactamente los nodos definidos en el roadmap."""
    graph = build_graph(checkpointer=None)
    expected = {
        "route_query",
        "clarification_request",
        "retrieve",
        "gather",
        "evaluate_relevance",
        "rewrite_query",
        "synthesize",
        "self_critique",
        "rewrite_answer",
        "format_response",
    }
    assert expected.issubset(set(graph.nodes))


def test_sift_state_keys():
    """SiftState debe tener todos los campos definidos en el roadmap."""
    required_keys = {
        "query", "user_id", "scopes", "is_admin", "query_type",
        "chunks", "relevance_scores", "iterations", "retrieval_debug",
        "answer", "citations",
        "critique", "rewrite_iterations",
        "clarification", "metadata",
    }
    assert required_keys == set(SiftState.__annotations__.keys())


def test_critique_output_fields():
    """CritiqueOutput debe tener los campos de scoring multi-dimensión."""
    fields = set(CritiqueOutput.model_fields.keys())
    assert "score" in fields
    assert "faithfulness" in fields
    assert "completeness" in fields
    assert "citation_quality" in fields
    assert "gaps" in fields
    assert "recommendation" in fields
    # Campo del v1 que se eliminó
    assert "strengths" not in fields


def test_query_classification_types():
    """QueryClassification solo acepta los 4 tipos definidos."""
    valid = QueryClassification(query_type="factual")
    assert valid.query_type == "factual"

    with pytest.raises(Exception):
        QueryClassification(query_type="invalid_type")
