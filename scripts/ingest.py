"""CLI de ingestion — indexa documentos en ChromaDB.

Uso:
    python scripts/ingest.py --source data/sources/personal --connector pdf
    python scripts/ingest.py --source data/sources/code --connector code
    python scripts/ingest.py --source data/sources/enterprise/enron/sampled --connector email
    python scripts/ingest.py --source data/sources/enterprise/vercel-docs --connector markdown
    python scripts/ingest.py --source data/sources --connector all
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.code import CodeConnector
from src.ingestion.email import EmailConnector
from src.ingestion.markdown import MarkdownConnector
from src.ingestion.pdf import PDFConnector
from src.ingestion.pipeline import ingest
from src.db.vector_store import upsert_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONNECTORS = {
    "pdf": PDFConnector,
    "markdown": MarkdownConnector,
    "code": CodeConnector,
    "email": EmailConnector,
}


def _progress(file_path: str, n_chunks: int) -> None:
    logger.info("  ✓ %s → %d chunks", os.path.basename(file_path), n_chunks)


def run(source: str, connector_name: str) -> None:
    if connector_name == "all":
        names = list(CONNECTORS.keys())
    else:
        names = [connector_name]

    total_stats = {"processed": 0, "chunks": 0, "errors": 0, "skipped": 0}

    for name in names:
        connector = CONNECTORS[name]()
        logger.info("=== Ingesting [%s] from %s ===", name, source)
        stats = ingest(
            source_path=source,
            connector=connector,
            upsert_fn=upsert_chunks,
            on_progress=_progress,
        )
        for k in total_stats:
            total_stats[k] += stats[k]
        logger.info("  → processed=%d chunks=%d errors=%d skipped=%d",
                    stats["processed"], stats["chunks"], stats["errors"], stats["skipped"])

    logger.info("=== TOTAL: processed=%d chunks=%d errors=%d skipped=%d ===",
                total_stats["processed"], total_stats["chunks"],
                total_stats["errors"], total_stats["skipped"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sift ingestion CLI")
    parser.add_argument("--source", required=True, help="Directorio raíz a indexar")
    parser.add_argument(
        "--connector",
        required=True,
        choices=[*CONNECTORS.keys(), "all"],
        help="Tipo de conector a usar",
    )
    args = parser.parse_args()

    if not os.path.exists(args.source):
        logger.error("Source path not found: %s", args.source)
        raise SystemExit(1)

    run(args.source, args.connector)


if __name__ == "__main__":
    main()
