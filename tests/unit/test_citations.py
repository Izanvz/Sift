"""Tests del parser de citas."""
import pytest

from src.agent.citations import (
    build_citations,
    extract_citation_indices,
    _extract_snippet,
    _int_or_none,
)
from src.agent.state import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(n: int, source_type: str = "markdown", **meta) -> Chunk:
    return Chunk(
        chunk_id=f"chunk-{n}",
        document_id=f"doc-{n}",
        source_path=f"/tmp/doc{n}.md",
        source_type=source_type,
        content=f"Content of chunk number {n}. " * 10,
        relevance_score=round(1.0 - n * 0.1, 2),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# extract_citation_indices
# ---------------------------------------------------------------------------

def test_extract_indices_basic():
    answer = "Respuesta [1] y también [3][2]."
    assert extract_citation_indices(answer) == [1, 2, 3]


def test_extract_indices_deduplicates():
    answer = "Ver [1] y [1] de nuevo y [2]."
    assert extract_citation_indices(answer) == [1, 2]


def test_extract_indices_empty_answer():
    assert extract_citation_indices("") == []
    assert extract_citation_indices("Sin marcadores.") == []


def test_extract_indices_preserves_order():
    answer = "Primero [3] luego [1] luego [2]."
    # debe retornar ordenado numéricamente
    assert extract_citation_indices(answer) == [1, 2, 3]


# ---------------------------------------------------------------------------
# _extract_snippet
# ---------------------------------------------------------------------------

def test_snippet_short_content():
    text = "Short content."
    assert _extract_snippet(text) == "Short content."


def test_snippet_truncates_at_word_boundary():
    text = "word " * 50  # 250 chars
    snippet = _extract_snippet(text, max_chars=20)
    assert len(snippet) <= 21  # max + ellipsis
    assert snippet.endswith("…")
    # no debe cortar a mitad de palabra
    assert not snippet[:-1].endswith(" ")


def test_snippet_strips_whitespace():
    assert _extract_snippet("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# _int_or_none
# ---------------------------------------------------------------------------

def test_int_or_none_valid():
    assert _int_or_none(5) == 5
    assert _int_or_none("10") == 10


def test_int_or_none_zero_returns_none():
    assert _int_or_none(0) is None


def test_int_or_none_none():
    assert _int_or_none(None) is None


def test_int_or_none_invalid():
    assert _int_or_none("abc") is None


# ---------------------------------------------------------------------------
# build_citations
# ---------------------------------------------------------------------------

def test_build_citations_basic():
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    answer = "Respuesta basada en [1] y [3]."
    citations = build_citations(answer, chunks)
    assert len(citations) == 2
    assert citations[0]["chunk_id"] == "chunk-1"
    assert citations[1]["chunk_id"] == "chunk-3"


def test_build_citations_no_markers():
    chunks = [_chunk(1), _chunk(2)]
    citations = build_citations("Sin citas.", chunks)
    assert citations == []


def test_build_citations_out_of_range_ignored():
    chunks = [_chunk(1)]  # solo 1 chunk
    answer = "Mira [1] y [99]."  # [99] fuera de rango
    citations = build_citations(answer, chunks)
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "chunk-1"


def test_build_citations_zero_index_ignored():
    chunks = [_chunk(1)]
    answer = "Mira [0]."  # 0 no es válido
    citations = build_citations(answer, chunks)
    assert citations == []


def test_build_citations_empty_chunks():
    citations = build_citations("Respuesta [1].", [])
    assert citations == []


def test_build_citations_pdf_metadata():
    chunk = Chunk(
        chunk_id="pdf-1",
        document_id="d1",
        source_path="/docs/report.pdf",
        source_type="pdf",
        content="Contenido del PDF.",
        relevance_score=0.9,
        metadata={"page_number": 7, "title": "Mi Informe"},
    )
    citations = build_citations("Ver [1].", [chunk])
    assert citations[0]["page_number"] == 7
    assert citations[0]["title"] == "Mi Informe"
    assert citations[0]["source_type"] == "pdf"


def test_build_citations_code_metadata():
    chunk = Chunk(
        chunk_id="code-1",
        document_id="d1",
        source_path="/src/app.py",
        source_type="code",
        content="def main(): pass",
        relevance_score=0.8,
        metadata={"line_start": 10, "line_end": 20},
    )
    citations = build_citations("Función en [1].", [chunk])
    assert citations[0]["line_start"] == 10
    assert citations[0]["line_end"] == 20


def test_build_citations_snippet_truncated():
    long_content = "word " * 100  # 500 chars
    chunk = _chunk(1)
    chunk["content"] = long_content
    citations = build_citations("[1]", [chunk])
    assert len(citations[0]["snippet"]) <= 210  # 200 chars + posible "…"
    assert citations[0]["snippet"].endswith("…")


def test_build_citations_deduplicates():
    chunks = [_chunk(1), _chunk(2)]
    answer = "Ver [1] y [1] otra vez."
    citations = build_citations(answer, chunks)
    assert len(citations) == 1


def test_build_citations_preserves_relevance_score():
    chunk = _chunk(1)
    chunk["relevance_score"] = 0.95
    citations = build_citations("[1]", [chunk])
    assert citations[0]["relevance_score"] == 0.95
