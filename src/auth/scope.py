"""Construcción de filtros de scope para ChromaDB.

Modelo de scopes:
    - Cada doc ingestado lleva metadata 'corpus' (ej. "vercel-docs", "stripe-go").
    - User.scopes = lista de corpora permitidos. ["*"] = todos (admin).
    - El where filter se aplica en ChromaDB sobre la metadata.

Returns:
    where dict válido para ChromaDB, o None si el usuario es admin.
"""
from src.auth.models import TokenData


def build_scope_filter(user: TokenData | None) -> dict | None:
    """Devuelve un where dict para ChromaDB según los scopes del usuario.

    - Sin user → no filter (modo abierto, p. ej. tests)
    - Admin o scopes=["*"] → no filter
    - Scopes específicos → {"corpus": {"$in": [...]}}
    - Sin scopes pero usuario válido → filtro imposible (None de ningún corpus)
    """
    if user is None:
        return None
    if user.is_admin or "*" in user.scopes:
        return None
    if not user.scopes:
        # Usuario sin scopes asignados → no debe ver nada.
        # ChromaDB no acepta {"$in": []} directamente: usamos un valor que no existe.
        return {"corpus": "__no_access__"}
    return {"corpus": {"$in": user.scopes}}


def merge_where(*filters: dict | None) -> dict | None:
    """Combina varios where dicts con AND lógico para ChromaDB.

    ChromaDB usa $and a nivel raíz cuando hay múltiples claves; aquí
    serializamos a {"$and": [...]} si hay 2+ filtros no vacíos.
    """
    valid = [f for f in filters if f]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return {"$and": valid}
