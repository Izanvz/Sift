"""Starlette middleware que loguea cada request autenticado a AuditStore.

Solo registra requests a /research (POST) y /auth/login.
El middleware es best-effort: si falla no rompe el request.
"""
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.audit.store import get_audit_store
from src.auth.jwt_utils import decode_token

logger = logging.getLogger(__name__)

# Rutas que se auditan (prefix match)
_AUDIT_PREFIXES = ("/research", "/auth/login")


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        method = request.method

        # Solo auditar las rutas configuradas
        if not any(path.startswith(p) for p in _AUDIT_PREFIXES):
            return await call_next(request)

        # Extraer user del token (best-effort — puede no existir)
        user_id = None
        username = None
        scopes = None
        token = _extract_bearer(request)
        if token:
            try:
                td = decode_token(token)
                user_id = td.sub
                username = td.username
                scopes = td.scopes
            except Exception:
                pass

        ip_address = _get_ip(request)
        start = time.perf_counter()
        status = "ok"
        error = None
        response: Response | None = None

        try:
            response = await call_next(request)
            if response.status_code >= 400:
                status = "error"
                error = f"HTTP {response.status_code}"
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            event_type = _event_type(path, method)
            try:
                get_audit_store().log(
                    event_type=event_type,
                    user_id=user_id,
                    username=username,
                    status=status,
                    error=error,
                    latency_ms=round(latency_ms, 2),
                    scopes=scopes,
                    ip_address=ip_address,
                )
            except Exception as exc:
                logger.warning("AuditMiddleware log failed: %s", exc)

        return response


def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:]
    return None


def _get_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return ""


def _event_type(path: str, method: str) -> str:
    if path.startswith("/auth/login"):
        return "auth_login"
    if path.startswith("/research") and method == "POST":
        return "research_start"
    if path.startswith("/research") and method == "GET":
        return "research_query"
    return f"{method.lower()}:{path}"
