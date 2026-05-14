# tests/unit/test_retrieval_explain.py
import pytest
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import NoOpReranker

CORPUS = [
    {"id": f"d{i}", "content": f"document about topic {i}", "source_path": f"doc{i}.md",
     "source_type": "markdown", "relevance_score": 0.5}
    for i in range(5)
]


def _mock_vector_fn(query: str, top_k: int, where=None) -> list[dict]:
    return [
        {**doc, "relevance_score": 0.9 - i * 0.1}
        for i, doc in enumerate(CORPUS[:top_k])
    ]


def _make_retriever() -> HybridRetriever:
    from src.retrieval.bm25 import BM25Retriever
    bm25 = BM25Retriever(CORPUS)
    return HybridRetriever(
        bm25=bm25,
        vector_fn=_mock_vector_fn,
        reranker=NoOpReranker(),
        bm25_top_k=5,
        vector_top_k=5,
        rrf_k=60,
    )


def test_retrieve_with_explain_returns_tuple():
    r = _make_retriever()
    result = r.retrieve_with_explain("topic 0", top_k=3)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_retrieve_with_explain_docs_same_as_retrieve():
    r = _make_retriever()
    docs, _ = r.retrieve_with_explain("topic 0", top_k=3)
    direct = r.retrieve("topic 0", top_k=3)
    assert [d["id"] for d in docs] == [d["id"] for d in direct]


def test_explain_has_required_keys():
    r = _make_retriever()
    _, debug = r.retrieve_with_explain("topic 0", top_k=3)
    for key in ("query", "bm25_top", "vector_top", "rrf_top", "final_top"):
        assert key in debug


def test_explain_query_matches():
    r = _make_retriever()
    _, debug = r.retrieve_with_explain("topic 0", top_k=3)
    assert debug["query"] == "topic 0"


def test_explain_bm25_top_has_preview_and_score():
    r = _make_retriever()
    _, debug = r.retrieve_with_explain("topic 0", top_k=3)
    for item in debug["bm25_top"]:
        assert "id" in item
        assert "content_preview" in item
        assert "bm25_score" in item
        assert "source_path" in item


def test_explain_vector_top_has_preview_and_score():
    r = _make_retriever()
    _, debug = r.retrieve_with_explain("topic 0", top_k=3)
    for item in debug["vector_top"]:
        assert "id" in item
        assert "content_preview" in item
        assert "relevance_score" in item
        assert "source_path" in item


def test_explain_rrf_top_has_rank():
    r = _make_retriever()
    _, debug = r.retrieve_with_explain("topic 0", top_k=3)
    for item in debug["rrf_top"]:
        assert "id" in item
        assert "rrf_score" in item
        assert "rrf_rank" in item


def test_explain_final_top_limited_to_top_k():
    r = _make_retriever()
    _, debug = r.retrieve_with_explain("topic 0", top_k=2)
    assert len(debug["final_top"]) <= 2


def test_explain_empty_corpus():
    from src.retrieval.bm25 import BM25Retriever
    r = HybridRetriever(
        bm25=BM25Retriever([]),
        vector_fn=lambda q, k, w=None: [],
        reranker=NoOpReranker(),
    )
    docs, debug = r.retrieve_with_explain("anything", top_k=3)
    assert docs == []
    assert debug["bm25_top"] == []
    assert debug["vector_top"] == []
    assert debug["rrf_top"] == []
    assert debug["final_top"] == []
