"""Endpoints de autenticación: /auth/login, /auth/me, /auth/users (admin)."""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from src.auth.dependencies import get_current_user, require_admin
from src.auth.jwt_utils import create_access_token
from src.auth.models import Token, TokenData, UserCreate, UserOut
from src.auth.store import get_user_store

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /auth/login — OAuth2 password flow (form-encoded)
# ---------------------------------------------------------------------------

@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()) -> Token:
    store = get_user_store()
    user = store.authenticate(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(
        user_id=user.user_id,
        username=user.username,
        scopes=user.scopes,
        is_admin=user.is_admin,
    )
    logger.info("Login success: user=%s admin=%s", user.username, user.is_admin)
    return Token(access_token=token, expires_in=expires_in)


# ---------------------------------------------------------------------------
# GET /auth/me — info del usuario autenticado
# ---------------------------------------------------------------------------

@router.get("/me", response_model=UserOut)
async def me(user: TokenData = Depends(get_current_user)) -> UserOut:
    db_user = get_user_store().get_by_id(user.sub)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists",
        )
    return UserOut(**db_user.model_dump(exclude={"password_hash"}))


# ---------------------------------------------------------------------------
# POST /auth/users — alta de usuarios (solo admin)
# ---------------------------------------------------------------------------

@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    _admin: TokenData = Depends(require_admin),
) -> UserOut:
    store = get_user_store()
    try:
        user = store.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return UserOut(**user.model_dump(exclude={"password_hash"}))
