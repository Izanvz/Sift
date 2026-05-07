"""Módulo de auth — JWT + bcrypt + scopes en ChromaDB."""
from src.auth.dependencies import (
    get_current_user,
    get_optional_user,
    require_admin,
)
from src.auth.jwt_utils import (
    JWTError,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.auth.models import (
    LoginRequest,
    Token,
    TokenData,
    UserCreate,
    UserInDB,
    UserOut,
)
from src.auth.scope import build_scope_filter, merge_where
from src.auth.store import UserStore, get_user_store, reset_user_store

__all__ = [
    "JWTError",
    "LoginRequest",
    "Token",
    "TokenData",
    "UserCreate",
    "UserInDB",
    "UserOut",
    "UserStore",
    "build_scope_filter",
    "create_access_token",
    "decode_token",
    "get_current_user",
    "get_optional_user",
    "get_user_store",
    "hash_password",
    "merge_where",
    "require_admin",
    "reset_user_store",
    "verify_password",
]
