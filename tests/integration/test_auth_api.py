"""Smoke test del flow auth: login → /auth/me → /research protegido."""
import pytest
from fastapi.testclient import TestClient

from src.api.auth_routes import router as auth_router
from src.auth.models import UserCreate
from src.auth.store import UserStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Monta un FastAPI mínimo con un user store en tmp."""
    from fastapi import FastAPI

    from src.auth import store as store_module

    store = UserStore(tmp_path / "users.db")
    monkeypatch.setattr(store_module, "_singleton", store)

    app = FastAPI()
    app.include_router(auth_router, prefix="/auth")
    return TestClient(app), store


def test_login_invalid_credentials(client):
    test_client, _ = client
    res = test_client.post(
        "/auth/login",
        data={"username": "nobody", "password": "x"},
    )
    assert res.status_code == 401


def test_login_and_me(client):
    test_client, store = client
    store.create(UserCreate(username="alice", password="hunter2", scopes=["vercel-docs"]))

    res = test_client.post(
        "/auth/login",
        data={"username": "alice", "password": "hunter2"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["token_type"] == "bearer"
    token = body["access_token"]

    me = test_client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "alice"
    assert me.json()["scopes"] == ["vercel-docs"]


def test_me_requires_token(client):
    test_client, _ = client
    res = test_client.get("/auth/me")
    assert res.status_code == 401


def test_create_user_requires_admin(client):
    test_client, store = client
    store.create(UserCreate(username="reg", password="passwd1", is_admin=False))

    res = test_client.post(
        "/auth/login",
        data={"username": "reg", "password": "passwd1"},
    )
    token = res.json()["access_token"]

    create = test_client.post(
        "/auth/users",
        json={"username": "newby", "password": "secret1", "scopes": []},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert create.status_code == 403


def test_admin_can_create_user(client):
    test_client, store = client
    store.create(UserCreate(username="root", password="rootpw1", is_admin=True))

    res = test_client.post(
        "/auth/login",
        data={"username": "root", "password": "rootpw1"},
    )
    token = res.json()["access_token"]

    create = test_client.post(
        "/auth/users",
        json={"username": "alice", "password": "hunter2", "scopes": ["vercel-docs"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert create.status_code == 201
    assert create.json()["username"] == "alice"
