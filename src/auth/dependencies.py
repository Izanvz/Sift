"""FastAPI dependencies para autenticación."""
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.auth.jwt_utils import JWTError, decode_token
from src.auth.models import TokenData

logger = logging.getLogger(__name__)

# tokenUrl es solo para Swagger UI; el endpoint real puede tener otro prefijo
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_current_user(token: str | None = Depends(oauth2_scheme)) -> TokenData:
    """Dependencia que valida el JWT y devuelve el TokenData."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(token)
    except JWTError as exc:
        logger.info("Invalid JWT: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def require_admin(user: TokenData = Depends(get_current_user)) -> TokenData:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def get_optional_user(token: str | None = Depends(oauth2_scheme)) -> TokenData | None:
    """Versión que no exige token — útil para endpoints públicos opcionales."""
    if not token:
        return None
    try:
        return decode_token(token)
    except JWTError:
        return None
