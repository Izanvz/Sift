"""HybridRetriever — orquesta BM25 + vectorial → RRF → reranker.

Pipeline:
    1. Query lanza en paralelo BM25 (top_n) y vectorial (top_n)
    2. RRF fusiona ambos rankings → top_n candidatos
    3. Cross-encoder reranker → top_k finales
    4. Devuelve list[dict] con keys homogéneas para nodes.retrieve()

Diseño:
    - Inyección de dependencias en __init__: testeable sin red ni modelos
    - Defaults razonables: factory get_hybrid_retriever() conecta a producción
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

from config.settings import settings
from src.retrieval.bm25 import BM25Retriever, get_bm25_retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import NoOpReranker, Reranker, get_reranker

logger = logging.getLogger(__name__)

# Tipo del retriever vectorial: query → list[dict] con al menos 'id' y 'content'
VectorRetrieverFn = Callable[[str, int], list[dict]]


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        vector_fn: VectorRetrieverFn | None = None,
        reranker: Reranker | None = None,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
        rrf_k: int | None = None,
    ):
        self.bm25 = bm25
        self.vector_fn = vector_fn or _default_vector_fn
        self.reranker = reranker or NoOpReranker()
        self.bm25_top_k = bm25_top_k or settings.bm25_top_k
        self.vector_top_k = vector_top_k or settings.vector_top_k
        self.rrf_k = rrf_k or settings.rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidates: int | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Pipeline híbrido completo.

        Args:
            query: texto de búsqueda.
            top_k: chunks finales tras reranker.
            candidates: nº candidatos tras RRF (antes del reranker).
                        Default = settings.retrieval_top_k.
            where: filtros para el retriever vectorial (ej. user_id).

        Returns:
            Lista de docs ordenados por relevance_score desc.
        """
        n_candidates = candidates or settings.retrieval_top_k

        # 1. Búsqueda paralela
        bm25_results, vector_results = self._search_parallel(query, where)
        logger.info(
            "Hybrid: bm25=%d, vector=%d, query=%r",
            len(bm25_results), len(vector_results), query[:80],
        )

        # 2. RRF fusion
        if not bm25_results and not vector_results:
            return []

        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=self.rrf_k,
            id_key="id",
            top_k=n_candidates,
        )

        # 3. Reranker
        reranked = self.reranker.rerank(query, fused, top_k=top_k)

        # 4. Normalizar score final
        for doc in reranked:
            doc["relevance_score"] = doc.get(
                "rerank_score",
                doc.get("rrf_score", 0.0),
            )

        return reranked

    def _search_parallel(
        self, query: str, where: dict | None
    ) -> tuple[list[dict], list[dict]]:
        """Lanza BM25 y vectorial en paralelo (I/O-bound)."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            bm25_future = pool.submit(self._bm25_search, query, where)
            vector_future = pool.submit(self._vector_search, query, where)
            return bm25_future.result(), vector_future.result()

    def _bm25_search(self, query: str, where: dict | None = None) -> list[dict]:
        bm25 = self.bm25 or get_bm25_retriever()
        try:
            results = bm25.query(query, top_k=self.bm25_top_k)
            return _apply_corpus_filter(results, where)
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []

    def _vector_search(self, query: str, where: dict | None) -> list[dict]:
        try:
            return self.vector_fn(query, self.vector_top_k, where)  # type: ignore
        except TypeError:
            # vector_fn no acepta 'where' — llamada sin filtro
            try:
                return self.vector_fn(query, self.vector_top_k)
            except Exception as exc:
                logger.warning("Vector search failed: %s", exc)
                return []
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Defaults de producción
# ---------------------------------------------------------------------------

def _default_vector_fn(
    query: str, top_k: int, where: dict | None = None
) -> list[dict]:
    """Wrapper sobre query_chromadb que normaliza al formato esperado."""
    from src.db.vector_store import query_chromadb  # import local
    raw = query_chromadb(query, n_results=top_k, where=where)
    # query_chromadb ya devuelve dicts con 'id', 'content', 'relevance_score', etc.
    return raw


_singleton: HybridRetriever | None = None


def _apply_corpus_filter(docs: list[dict], where: dict | None) -> list[dict]:
    """Post-filters BM25 results by corpus scope to match ChromaDB where semantics.

    where=None            → no filter (admin or open mode)
    where={"corpus": X}   → keep only docs whose metadata.corpus matches X
    """
    if where is None:
        return docs

    corpus_filter = where.get("corpus")
    if corpus_filter is None:
        return docs

    # {"corpus": "__no_access__"} → user has no scopes, block everything
    if corpus_filter == "__no_access__":
        return []

    # {"corpus": {"$in": [...]}}
    if isinstance(corpus_filter, dict):
        allowed = set(corpus_filter.get("$in", []))
        return [d for d in docs if d.get("metadata", {}).get("corpus") in allowed]

    # {"corpus": "exact-value"}
    return [d for d in docs if d.get("metadata", {}).get("corpus") == corpus_filter]


def get_hybrid_retriever() -> HybridRetriever:
    """Singleton — reutiliza BM25 index y reranker entre llamadas."""
    global _singleton
    if _singleton is None:
        _singleton = HybridRetriever(reranker=get_reranker())
    return _singleton


def reset_hybrid_retriever() -> None:
    global _singleton
    _singleton = None
