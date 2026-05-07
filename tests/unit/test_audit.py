"""Tests del módulo audit — store SQLite y middleware."""
import pytest

from src.audit.middleware import _event_type, _extract_bearer
from src.audit.store import AuditStore


# ---------------------------------------------------------------------------
# AuditStore
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return AuditStore(tmp_path / "audit.db")


def test_store_log_and_count(store):
    assert store.count() == 0
    store.log(event_type="research_start", user_id="u1", username="alice")
    store.log(event_type="research_start", user_id="u2", username="bob")
    assert store.count() == 2
    assert store.count(user_id="u1") == 1


def test_store_log_returns_event_id(store):
    eid = store.log(event_type="auth_login")
    assert isinstance(eid, str)
    assert len(eid) == 36  # UUID


def test_store_query_filter_user(store):
    store.log(event_type="auth_login", user_id="u1")
    store.log(event_type="auth_login", user_id="u2")
    rows = store.query_events(user_id="u1")
    assert len(rows) == 1
    assert rows[0]["user_id"] == "u1"


def test_store_query_filter_event_type(store):
    store.log(event_type="auth_login", user_id="u1")
    store.log(event_type="research_start", user_id="u1")
    rows = store.query_events(event_type="research_start")
    assert len(rows) == 1
    assert rows[0]["event_type"] == "research_start"


def test_store_all_fields(store):
    store.log(
        event_type="research_end",
        user_id="u1",
        username="alice",
        session_id="sess-1",
        query="What is X?",
        status="ok",
        latency_ms=123.4,
        n_chunks=5,
        n_citations=3,
        critique_score=8.5,
        scopes=["vercel-docs"],
        ip_address="127.0.0.1",
    )
    rows = store.query_events()
    assert len(rows) == 1
    r = rows[0]
    assert r["username"] == "alice"
    assert r["latency_ms"] == pytest.approx(123.4)
    assert r["n_chunks"] == 5
    assert r["critique_score"] == pytest.approx(8.5)
    assert r["ip_address"] == "127.0.0.1"


def test_store_limit_offset(store):
    for i in range(10):
        store.log(event_type="e", user_id=f"u{i}")
    assert len(store.query_events(limit=3)) == 3
    assert len(store.query_events(limit=3, offset=8)) == 2


def test_store_log_never_raises_on_bad_data(store):
    # No debe explotar aunque falten campos
    eid = store.log(event_type="x")
    assert isinstance(eid, str)


# ---------------------------------------------------------------------------
# Middleware helpers
# ---------------------------------------------------------------------------

def test_event_type_auth_login():
    assert _event_type("/auth/login", "POST") == "auth_login"


def test_event_type_research_start():
    assert _event_type("/research", "POST") == "research_start"
    assert _event_type("/research/abc-123/resume", "POST") == "research_start"


def test_event_type_research_query():
    assert _event_type("/research/abc-123", "GET") == "research_query"


def test_extract_bearer():
    class FakeRequest:
        headers = {"authorization": "Bearer tok123"}
    assert _extract_bearer(FakeRequest()) == "tok123"


def test_extract_bearer_missing():
    class FakeRequest:
        headers = {}
    assert _extract_bearer(FakeRequest()) is None
