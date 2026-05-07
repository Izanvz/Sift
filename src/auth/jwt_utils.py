"""JWT encoding/decoding + password hashing.

Usa python-jose (JWT) y passlib (bcrypt).
"""
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from config.settings import settings
from src.auth.models import TokenData

# bcrypt máx 72 bytes — truncamos si supera. passlib roto en Py3.14 + bcrypt 5.x.
_BCRYPT_MAX_BYTES = 72


def _to_bytes(password: str) -> bytes:
    encoded = password.encode("utf-8")
    return encoded[:_BCRYPT_MAX_BYTES]


# ---------------------------------------------------------------------------
# Password hashing (bcrypt directo)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(_to_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_to_bytes(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# JWT encode/decode
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: str,
    username: str,
    scopes: list[str],
    is_admin: bool = False,
    expires_minutes: int | None = None,
) -> tuple[str, int]:
    """Genera un JWT firmado. Devuelve (token, expires_in_seconds)."""
    minutes = expires_minutes or settings.jwt_expire_minutes
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    payload = {
        "sub": user_id,
        "username": username,
        "scopes": scopes,
        "is_admin": is_admin,
        "exp": int(expire.timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, minutes * 60


def decode_token(token: str) -> TokenData:
    """Decodifica un JWT. Lanza JWTError si inválido o expirado."""
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
    return TokenData(
        sub=payload["sub"],
        username=payload.get("username", ""),
        scopes=payload.get("scopes", []),
        is_admin=payload.get("is_admin", False),
        exp=payload.get("exp"),
    )


__all__ = [
    "JWTError",
    "create_access_token",
    "decode_token",
    "hash_password",
    "verify_password",
]
