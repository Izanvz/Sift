"""
Sprint 0 assumption-validation tests.

These tests verify that the external dependencies and LangGraph primitives
behave as expected BEFORE production code is written.  Each test is designed
to pass once the required packages are installed and a local Ollama instance
with qwen2.5:14b is running.

Run with:
    pytest tests/test_sprint0.py -v
"""

import operator
import sys
import os

# Make src/ importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio
import sqlite3
import tempfile
from typing import Annotated, TypedDict

import httpx
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ollama_instructor_client():
    """Return an instructor-patched Ollama client targeting the local server."""
    import instructor
    from openai import OpenAI  # ollama exposes an OpenAI-compatible API

    raw_client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required by the client but ignored by Ollama
    )
    # Mode.JSON is required for Ollama — it does not support tool_call mode
    return instructor.from_openai(raw_client, mode=instructor.Mode.JSON)


# ===========================================================================
# Test 1 – Instructor + Ollama JSON reliability
# ===========================================================================

@pytest.mark.parametrize("_run", range(10))
def test_instructor_ollama_json_reliability(_run):
    """
    Call the configured model via instructor 10 times (parametrized) and assert that
    each response is a valid CritiqueOutput with all required fields and a
    score in [0, 10].
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.state import CritiqueOutput
    from config.settings import settings

    client = _make_ollama_instructor_client()
    model = settings.model_planning

    response: CritiqueOutput = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Evaluate the following research snippet and return a "
                    "structured critique.\n\n"
                    "Snippet: 'Quantum entanglement enables faster-than-light "
                    "communication between particles.'"
                ),
            }
        ],
        response_model=CritiqueOutput,
    )

    assert isinstance(response, CritiqueOutput), "Response must be a CritiqueOutput instance"
    assert 0 <= response.score <= 10, f"score {response.score} out of range [0, 10]"
    assert isinstance(response.strengths, list), "strengths must be a list"
    assert isinstance(response.gaps, list), "gaps must be a list"
    assert isinstance(response.recommendation, str) and response.recommendation, (
        "recommendation must be a non-empty string"
    )


# ===========================================================================
# Test 2 – SqliteSaver with a cyclic LangGraph
# ===========================================================================

class _CycleState(TypedDict):
    count: int


def _build_cycle_graph(checkpointer):
    """
    Build a minimal graph with a cycle: A → B → (continue? → A | stop → END).
    The cycle runs at most 3 times.
    """
    from langgraph.graph import END, START, StateGraph

    def node_a(state: _CycleState) -> _CycleState:
        return {"count": state["count"] + 1}

    def node_b(state: _CycleState) -> _CycleState:
        # B does nothing except act as the interrupt / decision point.
        # LangGraph 0.2.x requires at least one field to be returned.
        return {"count": state["count"]}

    def should_continue(state: _CycleState) -> str:
        return "node_a" if state["count"] < 3 else END

    builder = StateGraph(_CycleState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_conditional_edges("node_b", should_continue, {"node_a": "node_a", END: END})

    return builder.compile(checkpointer=checkpointer, interrupt_before=["node_b"])


def test_sqlite_saver_cycle(tmp_path):
    """
    Verify that SqliteSaver persists state across separate graph invocations
    and that interrupt_before="node_b" pauses execution after node_a.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = str(tmp_path / "checkpoints.db")
    thread_id = "test-cycle-thread"
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = _build_cycle_graph(checkpointer)

        # First invocation – starts at count=0, node_a increments to 1,
        # then execution pauses before node_b.
        result = graph.invoke({"count": 0}, config=config)
        assert result["count"] == 1, "After first run count must be 1"

        # Resume – node_b runs, conditional sends us back to node_a (count → 2),
        # pauses again before node_b.
        result = graph.invoke(None, config=config)
        assert result["count"] == 2, "After first resume count must be 2"

        # Resume again – count → 3, conditional routes to END.
        result = graph.invoke(None, config=config)
        assert result["count"] == 3, "After second resume count must be 3"

        # Verify persistence: re-open the DB and confirm the saved state.
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        saved = checkpointer.get(config)
        assert saved is not None, "Checkpoint must survive closing and reopening the DB"
        # In langgraph-checkpoint-sqlite 2.x, get() returns a plain dict
        cv = saved["channel_values"] if isinstance(saved, dict) else saved.channel_values
        saved_count = cv.get("count")
        assert saved_count == 3, (
            f"Persisted count must be 3, got {saved_count}"
        )


# ===========================================================================
# Test 3 – interrupt_before in FastAPI (via httpx.AsyncClient)
# ===========================================================================

@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def fastapi_app(tmp_path):
    """Build a minimal FastAPI app that runs a LangGraph with interrupt_before."""
    import sqlite3
    import uuid

    from fastapi import FastAPI
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph

    class _FAPIState(TypedDict):
        step: int

    db_path = str(tmp_path / "fapi_checkpoints.db")

    app = FastAPI()

    # Use a raw sqlite3 connection so we control lifecycle without context manager quirks.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    def step_one(state: _FAPIState) -> _FAPIState:
        return {"step": 1}

    def checkpoint_node(state: _FAPIState) -> _FAPIState:
        # LangGraph 0.2.x requires at least one field returned from each node.
        return {"step": state["step"]}

    def step_two(state: _FAPIState) -> _FAPIState:
        return {"step": 2}

    builder = StateGraph(_FAPIState)
    builder.add_node("step_one", step_one)
    builder.add_node("checkpoint", checkpoint_node)
    builder.add_node("step_two", step_two)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "checkpoint")
    builder.add_edge("checkpoint", "step_two")
    builder.add_edge("step_two", END)

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["checkpoint"],
    )

    @app.post("/run", status_code=202)
    async def run_graph():
        session_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}
        # Runs until the interrupt; does NOT complete the graph.
        graph.invoke({"step": 0}, config=config)
        return {"session_id": session_id}

    @app.post("/resume/{session_id}")
    async def resume_graph(session_id: str):
        config = {"configurable": {"thread_id": session_id}}
        result = graph.invoke(None, config=config)
        return {"step": result["step"]}

    yield app

    # Cleanup
    conn.close()


@pytest.mark.anyio
async def test_interrupt_before_fastapi(fastapi_app):
    """
    POST /run  → 202, graph paused (step still 1, not 2).
    POST /resume/{session_id} → graph completes, step == 2.
    """
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fastapi_app), base_url="http://test"
    ) as client:
        # Start the graph – should return 202 with a session_id.
        response = await client.post("/run")
        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body
        session_id = body["session_id"]

        # The graph must NOT have completed yet: step_two has not run.
        # (We verify this indirectly by checking that /resume advances it.)

        # Resume – graph completes and step_two sets step=2.
        resume_response = await client.post(f"/resume/{session_id}")
        assert resume_response.status_code == 200
        resume_body = resume_response.json()
        assert resume_body["step"] == 2, (
            f"Expected step=2 after resume, got {resume_body['step']}"
        )


# ===========================================================================
# Test 4 – Send() + Annotated reducer (fan-out / gather)
# ===========================================================================

class _FanOutState(TypedDict):
    search_results: Annotated[list[str], operator.add]


def test_send_annotated_reducer():
    """
    Fan-out via Send() to 3 parallel branches; each branch appends one item.
    The gather node must see exactly 3 accumulated items.
    """
    from langgraph.constants import Send
    from langgraph.graph import END, START, StateGraph

    gathered_items: list[list[str]] = []

    def fan_out(state: _FanOutState):
        """Return Send objects targeting the 'branch' node for each item."""
        return [
            Send("branch", {"item": "result_a"}),
            Send("branch", {"item": "result_b"}),
            Send("branch", {"item": "result_c"}),
        ]

    def branch(state: dict) -> _FanOutState:
        """Each branch adds its item to search_results."""
        return {"search_results": [state["item"]]}

    def gather(state: _FanOutState) -> _FanOutState:
        """Record what was accumulated, then pass through."""
        gathered_items.append(list(state["search_results"]))
        return {"search_results": []}  # empty list is no-op for operator.add

    builder = StateGraph(_FanOutState)
    builder.add_node("branch", branch)
    builder.add_node("gather", gather)

    # fan_out is implemented as a conditional edge from START.
    builder.add_conditional_edges(START, fan_out, ["branch"])
    builder.add_edge("branch", "gather")
    builder.add_edge("gather", END)

    graph = builder.compile()
    final_state = graph.invoke({"search_results": []})

    assert len(final_state["search_results"]) == 3, (
        f"Expected 3 accumulated results, got {len(final_state['search_results'])}"
    )
    assert set(final_state["search_results"]) == {"result_a", "result_b", "result_c"}, (
        f"Unexpected items: {final_state['search_results']}"
    )
