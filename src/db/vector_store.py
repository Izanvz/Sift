import logging
import uuid

import chromadb

from config.settings import settings

logger = logging.getLogger(__name__)


def get_chromadb_client() -> chromadb.HttpClient:
    """Return a ChromaDB HTTP client configured from settings."""
    return chromadb.HttpClient(host=settings.chromadb_host, port=settings.chromadb_port)


def save_report(report_content: str, metadata: dict) -> bool:
    """Persist a research report to the ChromaDB research_reports collection.

    Returns True on success, False if the write fails.
    """
    try:
        client = get_chromadb_client()
        collection = client.get_or_create_collection(
            settings.chromadb_collection_research
        )
        doc_id = str(uuid.uuid4())
        collection.add(
            documents=[report_content],
            metadatas=[metadata],
            ids=[doc_id],
        )
        return True
    except Exception as exc:
        logger.warning("ChromaDB save failed: %s", exc)
        return False


def search_reports(query: str, n_results: int = 5) -> list[dict]:
    """Query the ChromaDB research_reports collection for similar reports.

    Returns the raw ChromaDB query result dict, or an empty list on failure.
    """
    try:
        client = get_chromadb_client()
        collection = client.get_or_create_collection(
            settings.chromadb_collection_research
        )
        results = collection.query(query_texts=[query], n_results=n_results)
        return results
    except Exception as exc:
        logger.warning("ChromaDB search failed: %s", exc)
        return []
