"""BM25 retriever — léxico, complementa al vectorial.

Construye un índice BM25 sobre un corpus de chunks. El corpus se carga lazy
desde ChromaDB la primera vez que se invoca, y se cachea en memoria.
"""
import logging
import re
from typing import Iterable

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenización simple: word-chars Unicode, lowercase."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    """Retriever léxico BM25 sobre una lista de documentos.

    Args:
        corpus: lista de dicts con keys 'id' y 'content' (mínimo).
                El resto de campos se preservan para devolverlos en query().
    """

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self._tokenized: list[list[str]] = [tokenize(d["content"]) for d in corpus]
        self._vocab: set[str] = {t for doc in self._tokenized for t in doc}
        # Si el corpus está vacío, BM25Okapi falla — protegemos
        self._bm25: BM25Okapi | None = (
            BM25Okapi(self._tokenized) if self._tokenized else None
        )

    def __len__(self) -> int:
        return len(self.corpus)

    def query(self, query: str, top_k: int = 30) -> list[dict]:
        """Devuelve top_k documentos ordenados por score BM25 desc.

        Cada dict incluye los campos originales del corpus + 'bm25_score'.
        Filtra docs solo cuando ningún término de la query aparece en el
        vocabulario (no usa `score > 0`: BM25Okapi puede producir scores
        negativos con corpus pequeños por IDF log).
        """
        if not self._bm25 or not self.corpus:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        # Si ningún término de la query existe en el corpus → no hay match real
        if not any(t in self._vocab for t in tokens):
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(float(s) for s in scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            {**self.corpus[i], "bm25_score": score}
            for i, score in ranked
        ]


# ---------------------------------------------------------------------------
# Lazy singleton: se construye desde ChromaDB la primera vez
# ---------------------------------------------------------------------------

_singleton: BM25Retriever | None = None


def get_bm25_retriever(corpus_loader=None) -> BM25Retriever:
    """Singleton lazy.

    Args:
        corpus_loader: callable opcional que devuelve list[dict]. Si None,
                       carga desde ChromaDB.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    if corpus_loader is None:
        corpus_loader = _load_corpus_from_chromadb

    logger.info("Building BM25 index...")
    corpus = corpus_loader()
    _singleton = BM25Retriever(corpus)
    logger.info("BM25 index ready: %d documents", len(_singleton))
    return _singleton


def reset_bm25_retriever() -> None:
    """Fuerza reconstruir el índice (útil tras ingestion o en tests)."""
    global _singleton
    _singleton = None


def _load_corpus_from_chromadb() -> list[dict]:
    """Carga todos los chunks de la colección principal."""
    from src.db.vector_store import _get_collection  # import local

    try:
        collection = _get_collection()
        # ChromaDB get() devuelve todo si no se pasa where ni ids
        result = collection.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        ids = result.get("ids") or []
        metas = result.get("metadatas") or []

        return [
            {
                "id": ids[i],
                "content": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
            }
            for i in range(len(docs))
        ]
    except Exception as exc:
        logger.warning("Could not load BM25 corpus from ChromaDB: %s", exc)
        return []
