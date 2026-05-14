"""Audit log — escribe eventos de uso a SQLite.

Schema (append-only, nunca se borra):
    audit_events(
        id          INTEGER PK AUTOINCREMENT,
        event_id    TEXT UNIQUE,   -- UUID del evento
        event_type  TEXT,          -- "research_start" | "research_end" | "auth_login" | ...
        user_id     TEXT,
        username    TEXT,
        session_id  TEXT,
        query       TEXT,
        status      TEXT,          -- "ok" | "error"
        error       TEXT,
        latency_ms  REAL,
        n_chunks    INTEGER,
        n_citations INTEGER,
        critique_score REAL,
        scopes      TEXT,          -- JSON
        ip_address  TEXT,
        created_at  TEXT NOT NULL
    )
"""
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.audit.redact import redact_query

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT UNIQUE NOT NULL,
    event_type     TEXT NOT NULL,
    user_id        TEXT,
    username       TEXT,
    session_id     TEXT,
    query          TEXT,
    status         TEXT,
    error          TEXT,
    latency_ms     REAL,
    n_chunks       INTEGER,
    n_citations    INTEGER,
    critique_score REAL,
    scopes         TEXT,
    ip_address     TEXT,
    created_at     TEXT NOT NULL
)
"""
_CREATE_IDX_USER = "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)"
_CREATE_IDX_TS   = "CREATE INDEX IF NOT EXISTS idx_audit_ts   ON audit_events(created_at)"


class AuditStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_IDX_USER)
            conn.execute(_CREATE_IDX_TS)

    # ------------------------------------------------------------------

    def log(
        self,
        *,
        event_type: str,
        user_id: str | None = None,
        username: str | None = None,
        session_id: str | None = None,
        query: str | None = None,
        status: str = "ok",
        error: str | None = None,
        latency_ms: float | None = None,
        n_chunks: int | None = None,
        n_citations: int | None = None,
        critique_score: float | None = None,
        scopes: list[str] | None = None,
        ip_address: str | None = None,
    ) -> str:
        """Inserta un evento. Devuelve el event_id generado."""
        event_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        query = redact_query(query)
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, event_type, user_id, username, session_id,
                        query, status, error, latency_ms, n_chunks, n_citations,
                        critique_score, scopes, ip_address, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        event_id, event_type, user_id, username, session_id,
                        query, status, error, latency_ms, n_chunks, n_citations,
                        critique_score,
                        json.dumps(scopes) if scopes is not None else None,
                        ip_address, created_at,
                    ),
                )
        except Exception as exc:
            logger.warning("AuditStore.log failed: %s", exc)
        return event_id

    def query_events(
        self,
        user_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        clauses = []
        params: list = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params += [limit, offset]
        sql = f"SELECT * FROM audit_events {where} ORDER BY id DESC LIMIT ? OFFSET ?"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def count(self, user_id: str | None = None) -> int:
        if user_id:
            return self._conn().execute(
                "SELECT COUNT(*) AS n FROM audit_events WHERE user_id = ?", (user_id,)
            ).fetchone()["n"]
        return self._conn().execute(
            "SELECT COUNT(*) AS n FROM audit_events"
        ).fetchone()["n"]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_singleton: AuditStore | None = None


def get_audit_store() -> AuditStore:
    from config.settings import settings

    global _singleton
    if _singleton is None:
        _singleton = AuditStore(settings.audit_db_path)
    return _singleton


def reset_audit_store() -> None:
    global _singleton
    _singleton = None
