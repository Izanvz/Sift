"""Endpoints de audit (solo admin): /audit/events, /audit/stats."""
from fastapi import APIRouter, Depends, Query

from src.audit.store import get_audit_store
from src.auth.dependencies import get_current_user, require_admin
from src.auth.models import TokenData

router = APIRouter()


@router.get("/events")
async def list_events(
    user_id: str | None = Query(None, description="Filtrar por user_id"),
    event_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _admin: TokenData = Depends(require_admin),
):
    """Lista eventos de audit (solo admins)."""
    store = get_audit_store()
    return store.query_events(user_id=user_id, event_type=event_type, limit=limit, offset=offset)


@router.get("/events/mine")
async def my_events(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: TokenData = Depends(get_current_user),
):
    """Eventos propios del usuario autenticado."""
    store = get_audit_store()
    return store.query_events(user_id=user.sub, limit=limit, offset=offset)


@router.get("/stats")
async def audit_stats(_admin: TokenData = Depends(require_admin)):
    """Estadísticas globales (solo admins)."""
    store = get_audit_store()
    total = store.count()
    return {"total_events": total}
