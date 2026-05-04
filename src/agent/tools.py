import logging

from src.agent.state import SearchResult
from config.settings import settings

logger = logging.getLogger(__name__)


def search_web_tool(query: str) -> list[SearchResult]:
    """Search the web using Tavily (primary) or DuckDuckGo (fallback).

    Returns up to 5 SearchResult entries with source="web".
    Returns an empty list if both providers fail.
    """
    # --- Tavily (primary) ---
    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.tavily_api_key)
            response = client.search(query, max_results=5)
            results: list[SearchResult] = []
            for item in response.get("results", []):
                results.append(
                    SearchResult(
                        source="web",
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        relevance=item.get("score", 0.7),
                    )
                )
            return results
        except Exception as exc:
            logger.warning("Tavily search failed, falling back to DuckDuckGo: %s", exc)

    # --- DuckDuckGo (fallback) ---
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=5))

        results = []
        for item in raw:
            results.append(
                SearchResult(
                    source="web",
                    url=item.get("href", ""),
                    content=item.get("body", ""),
                    relevance=0.6,
                )
            )
        return results
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []


def search_chromadb_tool(query: str) -> list[SearchResult]:
    """Search the ChromaDB meeting-transcripts collection.

    Returns up to 5 SearchResult entries with source="chromadb".
    Returns an empty list if ChromaDB is unavailable.
    """
    try:
        import chromadb

        client = chromadb.HttpClient(
            host=settings.chromadb_host, port=settings.chromadb_port
        )
        collection = client.get_or_create_collection(
            settings.chromadb_collection_meeting
        )
        raw = collection.query(query_texts=[query], n_results=5)

        results: list[SearchResult] = []
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert L2 distance to a 0-1 relevance score (smaller distance = higher relevance).
            relevance = max(0.0, min(1.0, 1.0 / (1.0 + dist)))
            url = meta.get("url", meta.get("source", "")) if meta else ""
            results.append(
                SearchResult(
                    source="chromadb",
                    url=url,
                    content=doc,
                    relevance=relevance,
                )
            )
        return results
    except (ConnectionError, Exception) as exc:
        logger.warning("ChromaDB search failed: %s", exc)
        return []


def search_arxiv_tool(query: str) -> list[SearchResult]:
    """Search arXiv for academic papers matching the query.

    Returns up to 3 SearchResult entries with source="arxiv".
    Raises an exception on failure (caller is responsible for handling it).
    """
    import arxiv

    search = arxiv.Search(query=query, max_results=3)
    results: list[SearchResult] = []
    for entry in search.results():
        results.append(
            SearchResult(
                source="arxiv",
                url=entry.pdf_url,
                content=entry.summary,
                relevance=0.8,
            )
        )
    return results
