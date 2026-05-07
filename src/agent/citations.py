"""Citation parser — extrae marcadores [N] del answer y los mapea a Chunks.

El LLM genera respuestas con referencias inline tipo:
    "La arquitectura LangGraph [1] soporta ciclos [2][3]..."

Este módulo:
1. Extrae qué índices [N] aparecen en el answer
2. Los mapea a los Chunks correspondientes (basado en orden de la lista)
3. Construye objetos Citation con snippet + localización exacta

Notas de diseño:
- Los índices en el answer son 1-based (el LLM ve [1], [2]...)
- chunks[0] corresponde a [1], chunks[1] a [2], etc.
- Si el LLM cita [N] fuera de rango → se ignora silenciosamente
"""
import re
from src.agent.state import Chunk, Citation

_MARKER_RE = re.compile(r"\[(\d+)\]")


def extract_citation_indices(answer: str) -> list[int]:
    """Extrae índices únicos de los marcadores [N], ordenados."""
    found = {int(m) for m in _MARKER_RE.findall(answer)}
    return sorted(found)


def _extract_snippet(content: str, max_chars: int = 200) -> str:
    """Toma las primeras max_chars del chunk como snippet."""
    snippet = content.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "…"
    return snippet


def build_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    """Construye la lista de Citation a partir del answer y los chunks.

    Args:
        answer: Texto del LLM con marcadores [1], [2]…
        chunks: Lista de Chunk en el mismo orden presentado al LLM.

    Returns:
        Lista de Citation, una por índice citado único, en orden de aparición.
    """
    indices = extract_citation_indices(answer)
    citations: list[Citation] = []

    for idx in indices:
        # índices son 1-based en el answer
        chunk_pos = idx - 1
        if chunk_pos < 0 or chunk_pos >= len(chunks):
            continue  # referencia fuera de rango → ignorar

        chunk = chunks[chunk_pos]
        meta = chunk.get("metadata", {}) or {}

        citations.append(Citation(
            chunk_id=chunk["chunk_id"],
            source_path=chunk["source_path"],
            source_type=chunk["source_type"],
            title=meta.get("title") or None,
            page_number=_int_or_none(meta.get("page_number")),
            line_start=_int_or_none(meta.get("line_start")),
            line_end=_int_or_none(meta.get("line_end")),
            snippet=_extract_snippet(chunk["content"]),
            relevance_score=chunk["relevance_score"],
        ))

    return citations


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
        return result if result > 0 else None
    except (ValueError, TypeError):
        return None
