"""Tests del self-critique loop — sin LLM."""
import pytest

from src.agent.edges import route_after_critique
from src.agent.state import SiftState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**critique_fields) -> SiftState:
    critique = {
        "score": 9.0,
        "faithfulness": 9.0,
        "completeness": 9.0,
        "citation_quality": 9.0,
        "gaps": [],
        "recommendation": "Good.",
        **critique_fields,
    }
    return SiftState(
        query="test query",
        user_id=None,
        query_type="factual",
        chunks=[],
        relevance_scores=[],
        iterations=0,
        answer="test answer",
        citations=[],
        critique=critique,
        rewrite_iterations=0,
        clarification=None,
        metadata={},
    )


# ---------------------------------------------------------------------------
# route_after_critique
# ---------------------------------------------------------------------------

def test_route_good_answer_goes_to_format():
    state = _state(score=9.0, faithfulness=9.0)
    assert route_after_critique(state) == "format_response"


def test_route_low_score_triggers_rewrite():
    state = _state(score=5.0, faithfulness=8.0)
    assert route_after_critique(state) == "rewrite_answer"


def test_route_low_faithfulness_triggers_rewrite():
    """Faithfulness baja siempre reescribe aunque score general sea OK."""
    state = _state(score=8.5, faithfulness=4.0)
    assert route_after_critique(state) == "rewrite_answer"


def test_route_max_iterations_stops_loop():
    """Aunque el score sea bajo, no reescribir si se alcanzó el límite."""
    from config.settings import settings
    state = _state(score=3.0, faithfulness=3.0)
    state["rewrite_iterations"] = settings.max_rewrite_iterations
    assert route_after_critique(state) == "format_response"


def test_route_exact_quality_gate_passes():
    """Score == quality_gate_score debe pasar (no reescribir)."""
    from config.settings import settings
    state = _state(score=settings.quality_gate_score, faithfulness=settings.faithfulness_hard_gate)
    assert route_after_critique(state) == "format_response"


def test_route_just_below_quality_gate_rewrites():
    from config.settings import settings
    state = _state(score=settings.quality_gate_score - 0.1, faithfulness=9.0)
    assert route_after_critique(state) == "rewrite_answer"


def test_route_just_below_faithfulness_gate_rewrites():
    from config.settings import settings
    state = _state(score=9.0, faithfulness=settings.faithfulness_hard_gate - 0.1)
    assert route_after_critique(state) == "rewrite_answer"


def test_route_missing_critique_defaults_to_format():
    """Sin critique en state → score=10 por defecto → format_response."""
    state = _state()
    state["critique"] = {}
    assert route_after_critique(state) == "format_response"


def test_route_first_iteration_can_rewrite():
    state = _state(score=4.0, faithfulness=4.0)
    state["rewrite_iterations"] = 0
    assert route_after_critique(state) == "rewrite_answer"


def test_route_second_iteration_still_rewrites_if_under_limit():
    from config.settings import settings
    state = _state(score=4.0, faithfulness=4.0)
    state["rewrite_iterations"] = settings.max_rewrite_iterations - 1
    assert route_after_critique(state) == "rewrite_answer"
