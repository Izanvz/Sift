"""Tests de retrieval — sin red, sin modelos pesados."""
import pytest

from src.retrieval.bm25 import BM25Retriever, tokenize
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import NoOpReranker


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def test_tokenize_lowercase_unicode():
    assert tokenize("Hola Mundo") == ["hola", "mundo"]
    assert tokenize("Año 2026, café") == ["año", "2026", "café"]


def test_tokenize_drops_punctuation():
    assert tokenize("¿Qué tal? ¡Bien!") == ["qué", "tal", "bien"]


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

def _corpus():
    # Términos discriminantes: cada uno aparece en 1/N docs para que BM25 tenga IDF > 0
    return [
        {"id": "a", "content": "LangGraph orchestrates agentic workflows."},
        {"id": "b", "content": "Python is a great programming language for AI."},
        {"id": "c", "content": "Cycles allow self-correction loops in graphs."},
        {"id": "d", "content": "Vector databases store embeddings for retrieval."},
        {"id": "e", "content": "Filler document about cooking pasta recipes."},
        {"id": "f", "content": "Another filler about gardening and plants."},
    ]


def test_bm25_returns_relevant_docs_first():
    retr = BM25Retriever(_corpus())
    # 'LangGraph' solo aparece en 'a' → debe ser primero
    results = retr.query("LangGraph workflows", top_k=6)
    assert results[0]["id"] == "a"


def test_bm25_empty_corpus():
    retr = BM25Retriever([])
    assert retr.query("anything") == []
    assert len(retr) == 0


def test_bm25_empty_query():
    retr = BM25Retriever(_corpus())
    assert retr.query("") == []


def test_bm25_no_matches_returns_empty():
    retr = BM25Retriever(_corpus())
    # término que no aparece en ningún doc
    results = retr.query("zzzzzz", top_k=4)
    assert results == []


def test_bm25_preserves_corpus_fields():
    corpus = [{"id": "x", "content": "hello world", "extra": 42}]
    retr = BM25Retriever(corpus)
    results = retr.query("hello")
    assert results[0]["extra"] == 42
    assert "bm25_score" in results[0]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def test_rrf_basic_fusion():
    r1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    r2 = [{"id": "b"}, {"id": "c"}, {"id": "a"}]
    fused = reciprocal_rank_fusion([r1, r2], k=60)
    # 'b' aparece en pos 2 y 1 → 1/61 + 1/62 (las mejor combinadas)
    # 'a' aparece en pos 1 y 3 → 1/61 + 1/63
    # 'c' aparece en pos 3 y 2 → 1/62 + 1/63
    # b > c > a (b: 0.0325, c: 0.0320, a: 0.0317)
    ids = [d["id"] for d in fused]
    assert ids[0] == "b"


def test_rrf_single_ranking():
    r1 = [{"id": "a"}, {"id": "b"}]
    fused = reciprocal_rank_fusion([r1])
    assert [d["id"] for d in fused] == ["a", "b"]
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_rrf_empty_rankings():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_top_k_truncates():
    r1 = [{"id": str(i)} for i in range(10)]
    fused = reciprocal_rank_fusion([r1], top_k=3)
    assert len(fused) == 3


def test_rrf_preserves_doc_fields():
    r1 = [{"id": "a", "content": "hello", "score": 0.9}]
    fused = reciprocal_rank_fusion([r1])
    assert fused[0]["content"] == "hello"
    assert fused[0]["score"] == 0.9
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_rank"] == 1


def test_rrf_skips_docs_without_id():
    r1 = [{"id": "a"}, {"content": "no id"}]
    fused = reciprocal_rank_fusion([r1])
    assert len(fused) == 1
    assert fused[0]["id"] == "a"


# ---------------------------------------------------------------------------
# NoOpReranker
# ---------------------------------------------------------------------------

def test_noop_reranker_passthrough():
    docs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    out = NoOpReranker().rerank("query", docs)
    assert out == docs


def test_noop_reranker_top_k():
    docs = [{"id": str(i)} for i in range(5)]
    out = NoOpReranker().rerank("q", docs, top_k=2)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# HybridRetriever (con mocks)
# ---------------------------------------------------------------------------

def _mock_vector_fn(query: str, top_k: int, where: dict | None = None) -> list[dict]:
    """Simula ChromaDB devolviendo docs con id + content."""
    docs = [
        {"id": "v1", "content": "vector result 1", "relevance_score": 0.9},
        {"id": "v2", "content": "vector result 2", "relevance_score": 0.8},
        {"id": "shared", "content": "appears in both", "relevance_score": 0.7},
    ]
    return docs[:top_k]


def test_hybrid_combines_bm25_and_vector():
    bm25 = BM25Retriever([
        {"id": "b1", "content": "bm25 result one"},
        {"id": "b2", "content": "bm25 result two"},
        {"id": "shared", "content": "appears in both"},
    ])
    retr = HybridRetriever(
        bm25=bm25,
        vector_fn=_mock_vector_fn,
        reranker=NoOpReranker(),
    )
    results = retr.retrieve("result", top_k=10)
    ids = {d["id"] for d in results}
    # debe contener resultados de ambas ramas
    assert "b1" in ids or "b2" in ids
    assert "v1" in ids or "v2" in ids


def test_hybrid_empty_when_both_empty():
    bm25 = BM25Retriever([])

    def empty_vec(q, k, where=None):
        return []

    retr = HybridRetriever(bm25=bm25, vector_fn=empty_vec, reranker=NoOpReranker())
    assert retr.retrieve("anything") == []


def test_hybrid_dedupes_via_rrf():
    """Si un doc aparece en ambas ramas, debe aparecer una sola vez."""
    bm25 = BM25Retriever([{"id": "shared", "content": "appears in both"}])
    retr = HybridRetriever(
        bm25=bm25,
        vector_fn=_mock_vector_fn,
        reranker=NoOpReranker(),
    )
    results = retr.retrieve("appears", top_k=10)
    ids = [d["id"] for d in results]
    assert ids.count("shared") == 1


def test_hybrid_assigns_relevance_score():
    bm25 = BM25Retriever([{"id": "b1", "content": "result alpha"}])
    retr = HybridRetriever(
        bm25=bm25,
        vector_fn=_mock_vector_fn,
        reranker=NoOpReranker(),
    )
    results = retr.retrieve("result", top_k=5)
    for r in results:
        assert "relevance_score" in r
        assert isinstance(r["relevance_score"], float)


def test_hybrid_respects_top_k():
    bm25 = BM25Retriever([
        {"id": f"b{i}", "content": f"result {i}"} for i in range(10)
    ])
    retr = HybridRetriever(
        bm25=bm25,
        vector_fn=_mock_vector_fn,
        reranker=NoOpReranker(),
    )
    results = retr.retrieve("result", top_k=3)
    assert len(results) <= 3


def test_hybrid_handles_vector_fn_exception():
    bm25 = BM25Retriever([{"id": "b1", "content": "fallback result"}])

    def broken_vec(q, k, where=None):
        raise RuntimeError("ChromaDB down")

    retr = HybridRetriever(bm25=bm25, vector_fn=broken_vec, reranker=NoOpReranker())
    results = retr.retrieve("fallback", top_k=5)
    # debe seguir devolviendo resultados de BM25
    assert any(r["id"] == "b1" for r in results)
