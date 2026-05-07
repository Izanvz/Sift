import logging
import uuid

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from config.settings import settings
from src.ingestion.base import IngestChunk

logger = logging.getLogger(__name__)


def get_chromadb_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=settings.chromadb_host, port=settings.chromadb_port)


def _get_embedding_fn() -> OllamaEmbeddingFunction:
    return OllamaEmbeddingFunction(
        url=f"http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )


def _get_collection(client: chromadb.HttpClient | None = None):
    c = client or get_chromadb_client()
    return c.get_or_create_collection(
        name=settings.chromadb_collection,
        embedding_function=_get_embedding_fn(),
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: list[IngestChunk]) -> None:
    """Persiste chunks en ChromaDB. Idempotente: usa chunk_id como id."""
    if not chunks:
        return
    try:
        collection = _get_collection()
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[{
                "document_id": c.document_id,
                "source_path": c.source_path,
                "source_type": c.source_type,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number or 0,
                "line_start": c.line_start or 0,
                "line_end": c.line_end or 0,
                **{k: str(v) for k, v in c.metadata.items()
                   if isinstance(v, (str, int, float, bool))},
            } for c in chunks],
        )
    except Exception as exc:
        logger.error("ChromaDB upsert failed (%d chunks): %s", len(chunks), exc)
        raise


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def query_chromadb(
    query: str,
    n_results: int = 20,
    where: dict | None = None,
) -> list[dict]:
    """Búsqueda vectorial en la colección principal.

    Returns lista de dicts con keys: id, content, source_path, source_type,
    relevance_score (1 - distance), metadata.
    """
    try:
        collection = _get_collection()
        kwargs = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        docs = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "id": ids[i],
                "content": docs[i],
                "source_path": metas[i].get("source_path", ""),
                "source_type": metas[i].get("source_type", "unknown"),
                "document_id": metas[i].get("document_id", ""),
                "relevance_score": max(0.0, 1.0 - distances[i]),
                "metadata": metas[i],
            }
            for i in range(len(docs))
        ]
    except Exception as exc:
        logger.warning("ChromaDB query failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Legacy (mantener compatibilidad con nodes.py)
# ---------------------------------------------------------------------------

def save_report(report_content: str, metadata: dict) -> bool:
    """Deprecated: guarda un informe en la colección principal."""
    try:
        collection = _get_collection()
        collection.add(
            documents=[report_content],
            metadatas=[{k: str(v) for k, v in metadata.items()}],
            ids=[str(uuid.uuid4())],
        )
        return True
    except Exception as exc:
        logger.warning("ChromaDB save_report failed: %s", exc)
        return False
