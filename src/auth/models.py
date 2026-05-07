"""Modelos Pydantic de autenticación y autorización."""
from datetime import datetime

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    """Payload de registro / creación administrativa."""
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    scopes: list[str] = Field(default_factory=list)
    is_admin: bool = False


class UserOut(BaseModel):
    """Vista pública del usuario — sin password_hash."""
    user_id: str
    username: str
    scopes: list[str]
    is_admin: bool
    created_at: datetime


class UserInDB(UserOut):
    """Vista interna con hash — nunca expuesta por la API."""
    password_hash: str


class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int   # segundos


class TokenData(BaseModel):
    """Payload del JWT (claims)."""
    sub: str                # user_id
    username: str
    scopes: list[str] = Field(default_factory=list)
    is_admin: bool = False
    exp: int | None = None
