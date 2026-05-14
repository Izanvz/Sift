"""Reciprocal Rank Fusion implementation used in Sift's hybrid retrieval."""
from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def reciprocal_rank_fusion(
    *rankings: list[T],
    k: int = 60,
) -> list[tuple[T, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        *rankings: Ranked lists of items. Earlier items have higher rank.
        k: Smoothing constant. Higher k reduces the impact of top-ranked items.

    Returns:
        List of (item, score) tuples sorted by fused score descending.

    Reference:
        Cormack, Clarke & Buettcher (2009). Reciprocal Rank Fusion outperforms
        Condorcet and individual Rank Learning Methods. SIGIR 2009.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_retrieval_results(
    bm25_results: list[str],
    vector_results: list[str],
    k: int = 60,
    top_n: int = 20,
) -> list[str]:
    """Fuse BM25 and vector retrieval results for Sift's hybrid pipeline."""
    fused = reciprocal_rank_fusion(bm25_results, vector_results, k=k)
    return [item for item, _ in fused[:top_n]]
