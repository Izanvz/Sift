"""Cross-encoder reranker — refina top-N candidatos antes de la síntesis.

Usa BGE reranker (BAAI/bge-reranker-base por defecto). Carga lazy: el modelo
no se descarga ni instancia hasta la primera llamada a rerank().

En CI/tests sin GPU, settings.reranker_enabled=False evita cargar el modelo
y rerank() devuelve los docs sin modificar (passthrough).
"""
import logging
from typing import Protocol

from config.settings import settings

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Interfaz para inyección de dependencias en HybridRetriever."""

    def rerank(self, query: str, docs: list[dict], top_k: int | None = None) -> list[dict]:
        ...


class CrossEncoderReranker:
    """Reranker basado en CrossEncoder de sentence-transformers.

    Modelo recomendado: BAAI/bge-reranker-base (~280MB, ~50ms/doc en CPU).
    Para mayor precisión: BAAI/bge-reranker-v2-m3 (multilingüe, más pesado).
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None  # lazy

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # import local pesado
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int | None = None,
        content_key: str = "content",
    ) -> list[dict]:
        """Reordena docs por score cross-encoder (query, doc).

        Devuelve los docs originales (mismos dicts) con un campo añadido
        'rerank_score', ordenados desc por ese score.
        """
        if not docs:
            return []

        model = self._load()
        pairs = [(query, d.get(content_key, "")) for d in docs]
        scores = model.predict(pairs)

        scored = [
            {**doc, "rerank_score": float(score)}
            for doc, score in zip(docs, scores)
        ]
        scored.sort(key=lambda d: d["rerank_score"], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]
        return scored


class NoOpReranker:
    """Reranker passthrough — para tests/CI sin sentence-transformers."""

    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int | None = None,
        content_key: str = "content",
    ) -> list[dict]:
        out = list(docs)
        if top_k is not None:
            out = out[:top_k]
        return out


def get_reranker() -> Reranker:
    """Factory: devuelve cross-encoder si está habilitado, no-op si no."""
    if settings.reranker_enabled:
        return CrossEncoderReranker()
    return NoOpReranker()
