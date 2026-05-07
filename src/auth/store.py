"""User store en SQLite — simple, sin ORM.

Schema:
    users(user_id TEXT PK, username TEXT UNIQUE, password_hash TEXT,
          scopes TEXT JSON, is_admin INT, created_at TEXT)
"""
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.auth.jwt_utils import hash_password, verify_password
from src.auth.models import UserCreate, UserInDB

logger = logging.getLogger(__name__)


class UserStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id       TEXT PRIMARY KEY,
                    username      TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    scopes        TEXT NOT NULL DEFAULT '[]',
                    is_admin      INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, payload: UserCreate) -> UserInDB:
        user_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        password_hash = hash_password(payload.password)
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO users(user_id, username, password_hash, scopes, is_admin, created_at) "
                    "VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        payload.username,
                        password_hash,
                        json.dumps(payload.scopes),
                        int(payload.is_admin),
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Username already exists: {payload.username}") from exc

        return UserInDB(
            user_id=user_id,
            username=payload.username,
            scopes=payload.scopes,
            is_admin=payload.is_admin,
            created_at=datetime.fromisoformat(created_at),
            password_hash=password_hash,
        )

    def get_by_username(self, username: str) -> UserInDB | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_by_id(self, user_id: str) -> UserInDB | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def authenticate(self, username: str, password: str) -> UserInDB | None:
        user = self.get_by_username(username)
        if not user or not verify_password(password, user.password_hash):
            return None
        return user

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> UserInDB:
        return UserInDB(
            user_id=row["user_id"],
            username=row["username"],
            password_hash=row["password_hash"],
            scopes=json.loads(row["scopes"] or "[]"),
            is_admin=bool(row["is_admin"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_singleton: UserStore | None = None


def get_user_store() -> UserStore:
    """Default store en data/sift.db (mismo SQLite del proyecto)."""
    from config.settings import settings

    global _singleton
    if _singleton is None:
        _singleton = UserStore(settings.sqlite_db_path)
    return _singleton


def reset_user_store() -> None:
    global _singleton
    _singleton = None
