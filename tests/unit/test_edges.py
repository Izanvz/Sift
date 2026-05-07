"""Tests del routing del grafo — sin dependencias externas."""
import pytest
from src.agent.edges import route_after_classification, route_after_critique, route_after_relevance
from config.settings import settings


def _state(**kwargs) -> dict:
    """Construye un estado mínimo de SiftState para tests."""
    defaults = {
        "query": "test query",
        "user_id": None,
        "query_type": "factual",
        "chunks": [],
        "relevance_scores": [],
        "iterations": 0,
        "answer": "",
        "citations": [],
        "critique": {},
        "rewrite_iterations": 0,
        "clarification": None,
        "metadata": {},
    }
    return {**defaults, **kwargs}


# ---------------------------------------------------------------------------
# route_after_classification
# ---------------------------------------------------------------------------

def test_route_ambiguous_goes_to_clarification():
    state = _state(query_type="ambiguous")
    assert route_after_classification(state) == "clarification_request"


@pytest.mark.parametrize("qtype", ["factual", "analytical", "comparative"])
def test_route_non_ambiguous_goes_to_retrieve(qtype):
    state = _state(query_type=qtype)
    assert route_after_classification(state) == "retrieve"


# ---------------------------------------------------------------------------
# route_after_relevance
# ---------------------------------------------------------------------------

def test_route_low_relevance_rewrites_query():
    state = _state(relevance_scores=[0.1, 0.2], iterations=1)
    assert route_after_relevance(state) == "rewrite_query"


def test_route_high_relevance_synthesizes():
    state = _state(relevance_scores=[0.9, 0.8], iterations=1)
    assert route_after_relevance(state) == "synthesize"


def test_route_max_iterations_forces_synthesize():
    """Aunque la relevancia sea baja, al llegar al límite debe sintetizar."""
    state = _state(
        relevance_scores=[0.1],
        iterations=settings.max_search_iterations,
    )
    assert route_after_relevance(state) == "synthesize"


# ---------------------------------------------------------------------------
# route_after_critique
# ---------------------------------------------------------------------------

def test_route_low_score_rewrites_answer():
    state = _state(critique={"score": 4.0}, rewrite_iterations=0)
    assert route_after_critique(state) == "rewrite_answer"


def test_route_high_score_formats_response():
    state = _state(critique={"score": 9.0}, rewrite_iterations=0)
    assert route_after_critique(state) == "format_response"


def test_route_max_rewrites_forces_format():
    """Al llegar al límite de rewrites, debe formatear aunque el score sea bajo."""
    state = _state(
        critique={"score": 2.0},
        rewrite_iterations=settings.max_rewrite_iterations,
    )
    assert route_after_critique(state) == "format_response"
