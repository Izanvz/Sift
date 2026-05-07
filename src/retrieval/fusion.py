"""Reciprocal Rank Fusion (RRF) — combina rankings heterogéneos.

Cormack, Clarke & Buettcher (2009): "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods".

Score fusionado para un documento d:
    RRF(d) = sum( 1 / (k + rank_i(d)) )  para cada ranking i donde aparece d

Ventaja: no necesita normalizar scores entre rankings (BM25 vs cosine
similarity tienen escalas distintas). Solo usa el rango.
"""
from typing import Iterable


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = 60,
    id_key: str = "id",
    top_k: int | None = None,
) -> list[dict]:
    """Fusiona varios rankings de documentos por RRF.

    Args:
        rankings: lista de rankings; cada ranking es list[dict] ya ordenada
                  (mejor primero). Cada dict debe tener al menos `id_key`.
        k: constante RRF (60 = Cormack default; suaviza colas largas).
        id_key: clave que identifica unívocamente al documento entre rankings.
        top_k: si se pasa, recorta el resultado a top_k.

    Returns:
        Lista de dicts (los originales, primer ranking que los contenga gana
        los campos), añadiendo `rrf_score` y `rrf_rank`. Ordenado por score
        descendente.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            doc_id = doc.get(id_key)
            if doc_id is None:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            # Preservar primer dict completo que vimos para este id
            if doc_id not in docs:
                docs[doc_id] = dict(doc)

    fused = sorted(
        (
            {**docs[doc_id], "rrf_score": score}
            for doc_id, score in scores.items()
        ),
        key=lambda d: d["rrf_score"],
        reverse=True,
    )

    for i, doc in enumerate(fused, start=1):
        doc["rrf_rank"] = i

    if top_k is not None:
        fused = fused[:top_k]

    return fused
