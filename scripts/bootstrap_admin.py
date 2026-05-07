"""Crea un usuario admin inicial en la SQLite de Sift.

Uso:
    python scripts/bootstrap_admin.py --username admin --password <pw>
    # con scopes específicos:
    python scripts/bootstrap_admin.py -u alice -p hunter2 --scopes vercel-docs stripe-go
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.auth.models import UserCreate
from src.auth.store import get_user_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap user en Sift")
    p.add_argument("-u", "--username", required=True)
    p.add_argument("-p", "--password", required=True)
    p.add_argument("--scopes", nargs="*", default=["*"], help='Default ["*"] = todos')
    p.add_argument("--no-admin", action="store_true", help="Crea usuario regular")
    args = p.parse_args()

    store = get_user_store()
    payload = UserCreate(
        username=args.username,
        password=args.password,
        scopes=args.scopes,
        is_admin=not args.no_admin,
    )
    try:
        user = store.create(payload)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(2)

    logger.info(
        "Created user_id=%s username=%s admin=%s scopes=%s",
        user.user_id, user.username, user.is_admin, user.scopes,
    )


if __name__ == "__main__":
    main()
