import logging
from typing import Optional

logger = logging.getLogger(__name__)

_handler = None
_initialized = False


def init_langfuse():
    """Llama una vez en startup. Configura el callback handler de Langfuse si está habilitado."""
    global _handler, _initialized
    if _initialized:
        return
    _initialized = True

    from config.settings import settings
    if not settings.langfuse_enabled:
        logger.info("Langfuse disabled")
        return

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "Langfuse enabled but LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY missing — tracing skipped"
        )
        return

    try:
        from langfuse.callback import CallbackHandler
        _handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled — host: %s", settings.langfuse_host)
    except Exception as e:
        logger.warning("Langfuse init failed: %s — tracing skipped", e)


def get_callbacks() -> list:
    """Devuelve la lista de callbacks para runs de LangGraph. Lista vacía si Langfuse no está activo."""
    if _handler is None:
        return []
    return [_handler]
