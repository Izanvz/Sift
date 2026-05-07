"""Tests del módulo auth — JWT, store SQLite, scope filter, dependencies."""
import time

import pytest
from fastapi import HTTPException

from src.auth.dependencies import get_current_user, get_optional_user, require_admin
from src.auth.jwt_utils import (
    JWTError,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.auth.models import TokenData, UserCreate
from src.auth.scope import build_scope_filter, merge_where
from src.auth.store import UserStore


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def test_hash_and_verify_password():
    h = hash_password("s3cret!")
    assert h != "s3cret!"
    assert verify_password("s3cret!", h) is True
    assert verify_password("wrong", h) is False


def test_verify_password_handles_invalid_hash():
    assert verify_password("anything", "not-a-real-hash") is False


# ---------------------------------------------------------------------------
# JWT encode/decode
# ---------------------------------------------------------------------------

def test_create_and_decode_access_token():
    token, expires_in = create_access_token(
        user_id="u1", username="alice", scopes=["vercel-docs"], is_admin=False,
    )
    assert isinstance(token, str)
    assert expires_in > 0
    data = decode_token(token)
    assert data.sub == "u1"
    assert data.username == "alice"
    assert data.scopes == ["vercel-docs"]
    assert data.is_admin is False


def test_decode_invalid_token_raises():
    with pytest.raises(JWTError):
        decode_token("not.a.jwt")


def test_decode_expired_token_raises():
    token, _ = create_access_token("u1", "alice", [], expires_minutes=-1)
    # negativo → ya expirado al crearse
    time.sleep(0.01)
    with pytest.raises(JWTError):
        decode_token(token)


# ---------------------------------------------------------------------------
# UserStore (SQLite)
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return UserStore(tmp_path / "users.db")


def test_store_create_and_get(store):
    user = store.create(UserCreate(
        username="alice", password="hunter2", scopes=["a", "b"], is_admin=False,
    ))
    assert user.username == "alice"
    assert user.scopes == ["a", "b"]
    assert user.password_hash != "hunter2"

    fetched = store.get_by_username("alice")
    assert fetched is not None
    assert fetched.user_id == user.user_id


def test_store_unique_username(store):
    store.create(UserCreate(username="alice", password="pw1234"))
    with pytest.raises(ValueError):
        store.create(UserCreate(username="alice", password="other7"))


def test_store_authenticate(store):
    store.create(UserCreate(username="bob", password="goodpass"))
    assert store.authenticate("bob", "goodpass") is not None
    assert store.authenticate("bob", "wrongpass") is None
    assert store.authenticate("nobody", "x") is None


def test_store_count(store):
    assert store.count() == 0
    store.create(UserCreate(username="alice", password="123456"))
    store.create(UserCreate(username="bob", password="123456"))
    assert store.count() == 2


# ---------------------------------------------------------------------------
# Scope filter
# ---------------------------------------------------------------------------

def test_scope_filter_none_for_no_user():
    assert build_scope_filter(None) is None


def test_scope_filter_none_for_admin():
    user = TokenData(sub="u1", username="admin", scopes=[], is_admin=True)
    assert build_scope_filter(user) is None


def test_scope_filter_none_for_wildcard():
    user = TokenData(sub="u1", username="x", scopes=["*"], is_admin=False)
    assert build_scope_filter(user) is None


def test_scope_filter_specific_scopes():
    user = TokenData(sub="u1", username="x", scopes=["vercel-docs", "stripe-go"])
    assert build_scope_filter(user) == {"corpus": {"$in": ["vercel-docs", "stripe-go"]}}


def test_scope_filter_empty_blocks_all():
    user = TokenData(sub="u1", username="x", scopes=[], is_admin=False)
    assert build_scope_filter(user) == {"corpus": "__no_access__"}


def test_merge_where_combines():
    assert merge_where(None, None) is None
    assert merge_where({"a": 1}, None) == {"a": 1}
    assert merge_where({"a": 1}, {"b": 2}) == {"$and": [{"a": 1}, {"b": 2}]}


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def test_get_current_user_missing_token():
    with pytest.raises(HTTPException) as exc:
        get_current_user(token=None)
    assert exc.value.status_code == 401


def test_get_current_user_invalid_token():
    with pytest.raises(HTTPException) as exc:
        get_current_user(token="garbage")
    assert exc.value.status_code == 401


def test_get_current_user_valid_token():
    token, _ = create_access_token("u1", "alice", ["x"])
    user = get_current_user(token=token)
    assert user.sub == "u1"
    assert user.scopes == ["x"]


def test_get_optional_user_no_token():
    assert get_optional_user(token=None) is None


def test_get_optional_user_invalid_token():
    assert get_optional_user(token="bad") is None


def test_require_admin_blocks_non_admin():
    user = TokenData(sub="u1", username="x", scopes=[], is_admin=False)
    with pytest.raises(HTTPException) as exc:
        require_admin(user=user)
    assert exc.value.status_code == 403


def test_require_admin_allows_admin():
    user = TokenData(sub="u1", username="x", scopes=[], is_admin=True)
    assert require_admin(user=user) is user
