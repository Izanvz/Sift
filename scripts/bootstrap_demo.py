"""Crea usuarios demo y corpus de ejemplo para Sift.

Tras `docker compose up -d` y `pip install -r requirements.txt`:

    python scripts/bootstrap_demo.py

Resultado:
  - 4 usuarios demo con distintos scopes
  - Corpus demo indexado en ChromaDB
  - Queries sugeridas impresas al final
"""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.auth.models import UserCreate
from src.auth.store import get_user_store
from src.db.vector_store import upsert_chunks
from src.ingestion.markdown import MarkdownConnector
from src.ingestion.code import CodeConnector
from src.ingestion.pipeline import ingest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEMO_USERS = [
    {
        "username": "admin",
        "password": "admin123",
        "scopes": ["*"],
        "is_admin": True,
    },
    {
        "username": "engineer",
        "password": "engineer123",
        "scopes": ["docs", "code"],
        "is_admin": False,
    },
    {
        "username": "sales",
        "password": "sales123",
        "scopes": ["docs"],
        "is_admin": False,
    },
    {
        "username": "no_scope",
        "password": "noscope123",
        "scopes": [],
        "is_admin": False,
    },
]

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demo")


def _create_users() -> None:
    store = get_user_store()
    for u in DEMO_USERS:
        payload = UserCreate(**u)
        try:
            user = store.create(payload)
            logger.info("Created  %-12s  admin=%-5s  scopes=%s", user.username, user.is_admin, user.scopes)
        except ValueError:
            logger.info("Skipped  %-12s  (already exists)", u["username"])


def _index_demo_corpus() -> None:
    docs_dir = os.path.join(DEMO_DIR, "docs")
    code_dir = os.path.join(DEMO_DIR, "code")

    if os.path.isdir(docs_dir):
        logger.info("Indexing demo docs: %s", docs_dir)
        stats = ingest(source_path=docs_dir, connector=MarkdownConnector(), upsert_fn=upsert_chunks)
        logger.info("Indexed %d doc chunks (%d errors)", stats["chunks"], stats["errors"])
    else:
        logger.warning("Demo docs dir not found: %s", docs_dir)

    if os.path.isdir(code_dir):
        logger.info("Indexing demo code: %s", code_dir)
        stats = ingest(source_path=code_dir, connector=CodeConnector(), upsert_fn=upsert_chunks)
        logger.info("Indexed %d code chunks (%d errors)", stats["chunks"], stats["errors"])
    else:
        logger.warning("Demo code dir not found: %s", code_dir)


def _print_summary() -> None:
    print("\n" + "=" * 60)
    print("  Sift demo ready — open http://localhost:8001")
    print("=" * 60)
    print("\nUsers:")
    for u in DEMO_USERS:
        scope_label = ", ".join(u["scopes"]) if u["scopes"] else "(none)"
        print(f"  {u['username']:<12}  pw: {u['password']:<14}  scopes: {scope_label}")
    print("\nSuggested queries:")
    print("  • What does the hybrid retrieval pipeline do?")
    print("  • How is BM25 fused with vector search?")
    print("  • Show me the authentication flow.")
    print("  • What is Reciprocal Rank Fusion?")
    print()


def main() -> None:
    logger.info("Bootstrapping Sift demo...")
    _create_users()
    _index_demo_corpus()
    _print_summary()


if __name__ == "__main__":
    main()
