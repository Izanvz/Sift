# tests/unit/test_explain_endpoint.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


def _make_debug(query="test"):
    return {
        "query": query,
        "query_original": query,
        "bm25_top": [{"id": "d1", "content_preview": "hello", "source_path": "a.md",
                      "source_type": "markdown", "bm25_score": 1.5}],
        "vector_top": [{"id": "d1", "content_preview": "hello", "source_path": "a.md",
                        "source_type": "markdown", "relevance_score": 0.9}],
        "rrf_top": [{"id": "d1", "content_preview": "hello", "source_path": "a.md",
                     "source_type": "markdown", "rrf_score": 0.016, "rrf_rank": 1}],
        "final_top": [{"id": "d1", "content_preview": "hello", "source_path": "a.md",
                       "source_type": "markdown", "relevance_score": 0.9,
                       "rerank_score": None, "rrf_score": 0.016}],
    }


def _make_state(user_id="user1", debug=None):
    return {
        "query": "test",
        "user_id": user_id,
        "query_type": "factual",
        "metadata": {"query": "test"},
        "retrieval_debug": debug if debug is not None else _make_debug(),
    }


@pytest.fixture()
def app_with_checkpoint():
    from fastapi import FastAPI
    from src.api.explain_routes import router
    from src.auth.dependencies import get_current_user
    from src.auth.models import TokenData

    app = FastAPI()
    app.include_router(router, prefix="/research")

    checkpointer = MagicMock()
    app.state.checkpointer = checkpointer

    def override_user():
        return TokenData(sub="user1", username="user1", scopes=[], is_admin=False)

    app.dependency_overrides[get_current_user] = override_user
    return app, checkpointer


def test_explain_returns_debug(app_with_checkpoint):
    app, checkpointer = app_with_checkpoint
    checkpointer.get.return_value = {"channel_values": _make_state()}

    resp = TestClient(app).get("/research/sess123/explain")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("query", "bm25_top", "vector_top", "rrf_top", "final_top"):
        assert key in data


def test_explain_404_if_session_not_found(app_with_checkpoint):
    app, checkpointer = app_with_checkpoint
    checkpointer.get.return_value = None

    resp = TestClient(app).get("/research/no-such/explain")
    assert resp.status_code == 404


def test_explain_403_if_not_owner(app_with_checkpoint):
    app, checkpointer = app_with_checkpoint
    checkpointer.get.return_value = {"channel_values": _make_state(user_id="other_user")}

    resp = TestClient(app).get("/research/sess123/explain")
    assert resp.status_code == 403


def test_explain_404_if_no_debug(app_with_checkpoint):
    app, checkpointer = app_with_checkpoint
    state = _make_state()
    state["retrieval_debug"] = None
    checkpointer.get.return_value = {"channel_values": state}

    resp = TestClient(app).get("/research/sess123/explain")
    assert resp.status_code == 404
