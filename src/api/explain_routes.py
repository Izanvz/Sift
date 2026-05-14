import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.auth.dependencies import get_current_user
from src.auth.models import TokenData

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{session_id}/explain")
async def get_retrieval_explain(
    request: Request,
    session_id: str,
    user: TokenData = Depends(get_current_user),
):
    """Devuelve el debug del pipeline de retrieval para una sesion."""
    checkpointer = request.app.state.checkpointer
    config = {"configurable": {"thread_id": session_id}}

    checkpoint = await asyncio.to_thread(checkpointer.get, config)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    state = checkpoint.get("channel_values", {})
    owner = state.get("user_id")
    if owner and owner != user.sub and not user.is_admin:
        raise HTTPException(status_code=403, detail="Not your session")

    debug = state.get("retrieval_debug")
    if debug is None:
        raise HTTPException(status_code=404, detail="No retrieval debug available for this session")

    return {
        **debug,
        "session_id": session_id,
        "query_current": state.get("query"),
        "query_type": state.get("query_type"),
    }
