"""Chunker semántico basado en tokens (tiktoken).

Estrategia:
- Divide por párrafos primero (respeta fronteras semánticas)
- Agrupa párrafos hasta llegar a chunk_size tokens
- Overlap de chunk_overlap tokens del chunk anterior
- Nunca corta palabras a mitad
"""
import re
from functools import lru_cache

import tiktoken

from config.settings import settings
from src.ingestion.base import Document, IngestChunk


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    """cl100k_base es compatible con la mayoría de modelos modernos."""
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _split_paragraphs(text: str) -> list[str]:
    """Divide por líneas en blanco; elimina párrafos vacíos."""
    paras = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in paras if p.strip()]


def split(
    doc: Document,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[IngestChunk]:
    """Divide un Document en IngestChunks.

    Args:
        doc: Documento a dividir.
        chunk_size: Tokens por chunk (default: settings.chunk_size_tokens).
        chunk_overlap: Tokens de overlap (default: settings.chunk_overlap_tokens).

    Returns:
        Lista de IngestChunk en orden.
    """
    size = chunk_size or settings.chunk_size_tokens
    overlap = chunk_overlap or settings.chunk_overlap_tokens

    paragraphs = _split_paragraphs(doc.content)
    if not paragraphs:
        return []

    chunks: list[IngestChunk] = []
    current_paras: list[str] = []
    current_tokens = 0
    overlap_text = ""

    def _flush(paras: list[str], index: int) -> IngestChunk:
        content = overlap_text + "\n\n".join(paras) if overlap_text else "\n\n".join(paras)
        return IngestChunk(
            document_id=doc.id,
            content=content.strip(),
            chunk_index=index,
            source_path=doc.source_path,
            source_type=doc.source_type,
            metadata={**doc.metadata, "title": doc.title},
        )

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # Párrafo solo ya excede el límite → chunkearlo por frases
        if para_tokens > size:
            if current_paras:
                chunks.append(_flush(current_paras, len(chunks)))
                overlap_text = _build_overlap(current_paras, overlap)
                current_paras, current_tokens = [], 0

            for sub in _split_by_sentences(para, size, overlap):
                chunks.append(IngestChunk(
                    document_id=doc.id,
                    content=sub,
                    chunk_index=len(chunks),
                    source_path=doc.source_path,
                    source_type=doc.source_type,
                    metadata={**doc.metadata, "title": doc.title},
                ))
            overlap_text = ""
            continue

        if current_tokens + para_tokens > size and current_paras:
            chunks.append(_flush(current_paras, len(chunks)))
            overlap_text = _build_overlap(current_paras, overlap)
            current_paras, current_tokens = [], 0

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunks.append(_flush(current_paras, len(chunks)))

    return chunks


def _build_overlap(paras: list[str], overlap_tokens: int) -> str:
    """Toma suficientes párrafos del final para alcanzar overlap_tokens."""
    enc = _get_encoder()
    result: list[str] = []
    tokens = 0
    for para in reversed(paras):
        t = len(enc.encode(para))
        if tokens + t > overlap_tokens:
            break
        result.insert(0, para)
        tokens += t
    return "\n\n".join(result) + "\n\n" if result else ""


def _split_by_sentences(text: str, size: int, overlap: int) -> list[str]:
    """Fallback: divide por frases cuando un párrafo supera size tokens."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        st = _count_tokens(sent)

        # Frase individual demasiado larga → dividir por palabras
        if st > size:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.extend(_split_by_words(sent, size))
            continue

        if current_tokens + st > size and current:
            chunks.append(" ".join(current))
            # overlap: mantener última frase
            current = current[-1:] if overlap > 0 else []
            current_tokens = _count_tokens(" ".join(current))
        current.append(sent)
        current_tokens += st

    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_by_words(text: str, size: int) -> list[str]:
    """Último recurso: divide por palabras cuando no hay puntuación."""
    enc = _get_encoder()
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for word in words:
        wt = len(enc.encode(word))
        if current_tokens + wt > size and current:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += wt

    if current:
        chunks.append(" ".join(current))
    return chunks
