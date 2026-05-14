"""Tests de permisos y scope enforcement.

Group 1: build_scope_filter (lógica pura, sin mocks)
Group 2: _apply_corpus_filter (lógica pura, sin mocks)
Group 3: HybridRetriever scope integration (inyección de deps, sin ChromaDB)
"""
from src.auth.scope import build_scope_filter
from src.auth.models import TokenData
from src.retrieval.hybrid import HybridRetriever, _apply_corpus_filter
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import NoOpReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(scopes, is_admin=False):
    return TokenData(sub="u1", username="u1", scopes=scopes, is_admin=is_admin)


def _doc(corpus):
    return {"id": "x", "content": "y", "metadata": {"corpus": corpus}}


def _make_corpus():
    return [
        {"id": "d1", "content": "hybrid retrieval BM25 vector fusion", "metadata": {"corpus": "docs"}},
        {"id": "d2", "content": "authentication scopes JWT permissions", "metadata": {"corpus": "docs"}},
        {"id": "d3", "content": "sales pipeline revenue forecast", "metadata": {"corpus": "sales"}},
        {"id": "d4", "content": "engineering code review pull request", "metadata": {"corpus": "code"}},
    ]


# ---------------------------------------------------------------------------
# Group 1: build_scope_filter
# ---------------------------------------------------------------------------

def test_admin_gets_no_filter():
    assert build_scope_filter(_user(scopes=[], is_admin=True)) is None


def test_wildcard_scope_gets_no_filter():
    assert build_scope_filter(_user(scopes=["*"])) is None


def test_empty_scopes_gets_no_access():
    assert build_scope_filter(_user(scopes=[])) == {"corpus": "__no_access__"}


def test_single_scope_filter():
    assert build_scope_filter(_user(scopes=["docs"])) == {"corpus": {"$in": ["docs"]}}


def test_multi_scope_filter():
    result = build_scope_filter(_user(scopes=["docs", "code"]))
    assert result == {"corpus": {"$in": ["docs", "code"]}}


def test_none_user_gets_no_filter():
    assert build_scope_filter(None) is None


# ---------------------------------------------------------------------------
# Group 2: _apply_corpus_filter
# ---------------------------------------------------------------------------

def test_filter_none_passes_all():
    docs = [_doc("docs"), _doc("sales"), _doc("code")]
    assert _apply_corpus_filter(docs, None) == docs


def test_filter_no_access_blocks_all():
    docs = [_doc("docs"), _doc("sales")]
    assert _apply_corpus_filter(docs, {"corpus": "__no_access__"}) == []


def test_filter_in_keeps_matching():
    docs_doc = _doc("docs")
    code_doc = _doc("code")
    result = _apply_corpus_filter([docs_doc, code_doc], {"corpus": {"$in": ["docs"]}})
    assert result == [docs_doc]


def test_filter_in_multi_scope():
    docs_doc = _doc("docs")
    code_doc = _doc("code")
    sales_doc = _doc("sales")
    result = _apply_corpus_filter(
        [docs_doc, code_doc, sales_doc],
        {"corpus": {"$in": ["docs", "code"]}},
    )
    assert docs_doc in result
    assert code_doc in result
    assert sales_doc not in result


def test_filter_no_corpus_metadata():
    doc_no_meta = {"id": "x", "content": "y", "metadata": {}}
    result = _apply_corpus_filter([doc_no_meta], {"corpus": {"$in": ["docs"]}})
    assert result == []


def test_filter_admin_passes_all():
    docs = [_doc("docs"), _doc("sales"), _doc("code")]
    # Admin gets where=None from build_scope_filter
    assert _apply_corpus_filter(docs, None) == docs


# ---------------------------------------------------------------------------
# Group 3: HybridRetriever scope integration
# ---------------------------------------------------------------------------

def _make_retriever():
    corpus = _make_corpus()
    bm25 = BM25Retriever(corpus)
    return HybridRetriever(
        bm25=bm25,
        vector_fn=lambda q, n, w=None: [],
        reranker=NoOpReranker(),
    )


def test_no_scope_user_gets_empty_bm25():
    retriever = _make_retriever()
    results = retriever.retrieve("hybrid retrieval", where={"corpus": "__no_access__"})
    assert results == []


def test_scoped_user_only_sees_own_corpus():
    retriever = _make_retriever()
    results = retriever.retrieve(
        "hybrid retrieval",
        where={"corpus": {"$in": ["docs"]}},
    )
    assert len(results) > 0
    corpora = {r["metadata"]["corpus"] for r in results}
    assert corpora == {"docs"}
    assert "sales" not in corpora
    assert "code" not in corpora


def test_admin_sees_all_corpora():
    retriever = _make_retriever()
    results = retriever.retrieve("retrieval sales engineering", where=None)
    corpora = {r["metadata"]["corpus"] for r in results}
    assert len(corpora) > 1


def test_engineer_cannot_see_sales():
    retriever = _make_retriever()
    results = retriever.retrieve(
        "sales pipeline",
        where={"corpus": {"$in": ["docs", "code"]}},
    )
    corpora = {r["metadata"]["corpus"] for r in results}
    assert "sales" not in corpora
