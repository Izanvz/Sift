"""Pipeline de ingestion — orquesta connector → chunker → vector store."""
import logging
from typing import Callable

from src.ingestion.base import BaseConnector, IngestChunk
from src.ingestion import chunker as chunker_module

logger = logging.getLogger(__name__)

_MAX_DOC_BYTES = 10 * 1024 * 1024  # 10 MB


def ingest(
    source_path: str,
    connector: BaseConnector,
    upsert_fn: Callable[[list[IngestChunk]], None],
    on_progress: Callable[[str, int], None] | None = None,
) -> dict:
    """Ingesta todos los archivos de source_path usando connector.

    Args:
        source_path: Directorio raíz a procesar.
        connector: Instancia de BaseConnector (PDF, Markdown, Code, Email).
        upsert_fn: Función que recibe list[IngestChunk] y los persiste.
        on_progress: Callback opcional (file_path, n_chunks).

    Returns:
        Resumen: {"processed": N, "chunks": N, "errors": N, "skipped": N}
    """
    stats = {"processed": 0, "chunks": 0, "errors": 0, "skipped": 0}

    for file_path in connector.discover(source_path):
        try:
            doc = connector.parse(file_path)

            if not doc.content.strip():
                logger.info("Skipped (empty content): %s", file_path)
                stats["skipped"] += 1
                continue

            if len(doc.content.encode("utf-8")) > _MAX_DOC_BYTES:
                logger.warning("Skipped (exceeds %d MB): %s", _MAX_DOC_BYTES // 1024 // 1024, file_path)
                stats["skipped"] += 1
                continue

            chunks = chunker_module.split(doc)

            if not chunks:
                logger.info("Skipped (no chunks): %s", file_path)
                stats["skipped"] += 1
                continue

            upsert_fn(chunks)

            stats["processed"] += 1
            stats["chunks"] += len(chunks)

            if on_progress:
                on_progress(file_path, len(chunks))

            logger.info("Ingested %s → %d chunks", file_path, len(chunks))

        except Exception as exc:
            logger.error("Error ingesting %s: %s", file_path, exc)
            stats["errors"] += 1

    return stats
