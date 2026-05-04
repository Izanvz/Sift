import asyncio
import json
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agent.state import HumanFeedback

logger = logging.getLogger(__name__)
router = APIRouter()


class ResearchRequest(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# GET /info  — metadata de la instancia
# ---------------------------------------------------------------------------

@router.get("/info")
async def get_info():
    from config.settings import settings
    return {
        "model_planning": settings.model_planning,
        "model_synthesis": settings.model_synthesis,
        "quality_threshold": settings.quality_threshold,
        "max_search_iterations": settings.max_search_iterations,
    }


# ---------------------------------------------------------------------------
# POST /research  — lanza nueva sesión con SSE streaming
# ---------------------------------------------------------------------------

@router.post("")
async def start_research(request: Request, body: ResearchRequest):
    graph = request.app.state.graph
    session_id = str(uuid4())
    config = {"configurable": {"thread_id": session_id}}

    async def event_stream():
        try:
            yield f"data: {json.dumps({'session_id': session_id, 'status': 'started'})}\n\n"

            async for event in graph.astream_events(
                {"query": body.query}, config, version="v2"
            ):
                event_type = event.get("event", "")
                node_name = event.get("name", "")

                if event_type == "on_chain_start" and node_name not in ("LangGraph", ""):
                    yield f"data: {json.dumps({'node': node_name, 'status': 'running', 'session_id': session_id})}\n\n"

                elif event_type == "on_chain_end" and node_name not in ("LangGraph", ""):
                    output = event.get("data", {}).get("output", {})

                    # Detectar pausa en human_checkpoint
                    if node_name == "human_checkpoint" or (
                        isinstance(output, dict) and "__interrupt__" in str(output)
                    ):
                        yield f"data: {json.dumps({'node': 'human_checkpoint', 'status': 'waiting_human', 'session_id': session_id})}\n\n"
                        return

                    yield f"data: {json.dumps({'node': node_name, 'status': 'done', 'session_id': session_id})}\n\n"

                    # Reporte final
                    if node_name == "generate_report" and isinstance(output, dict):
                        report = output.get("report", "")
                        yield f"data: {json.dumps({'status': 'completed', 'session_id': session_id, 'report': report})}\n\n"

        except Exception as exc:
            logger.exception("Streaming error: %s", exc)
            yield f"data: {json.dumps({'status': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# GET /research/{session_id}  — estado actual del grafo
# ---------------------------------------------------------------------------

@router.get("/{session_id}")
async def get_session(request: Request, session_id: str):
    checkpointer = request.app.state.checkpointer
    config = {"configurable": {"thread_id": session_id}}

    checkpoint = await asyncio.to_thread(checkpointer.get, config)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    state = checkpoint.get("channel_values", {})
    return {
        "session_id": session_id,
        "state": state,
        "status": "waiting_human" if state.get("human_feedback") is None and state.get("synthesis") else "active",
    }


# ---------------------------------------------------------------------------
# POST /research/{session_id}/resume  — continuar tras human checkpoint
# ---------------------------------------------------------------------------

@router.post("/{session_id}/resume")
async def resume_research(request: Request, session_id: str, feedback: HumanFeedback):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": session_id}}

    await asyncio.to_thread(
        graph.update_state, config, {"human_feedback": feedback.content}
    )
    result = await asyncio.to_thread(graph.invoke, None, config)
    return {"session_id": session_id, "status": "completed", "report": result.get("report", "")}


# ---------------------------------------------------------------------------
# GET /research  — lista de sesiones
# ---------------------------------------------------------------------------

@router.get("")
async def list_sessions(request: Request):
    checkpointer = request.app.state.checkpointer
    try:
        sessions = await asyncio.to_thread(list, checkpointer.list({}))
        return [
            {
                "session_id": s.config.get("configurable", {}).get("thread_id", ""),
                "timestamp": s.metadata.get("created_at", "") if s.metadata else "",
            }
            for s in sessions
        ]
    except Exception as exc:
        logger.warning("Could not list sessions: %s", exc)
        return []
