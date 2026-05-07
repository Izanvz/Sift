"""Muestrea N emails del CSV Enron (1.3 GB) y los guarda como archivos .txt individuales.

Uso:
    python scripts/sample_enron.py --csv data/sources/enterprise/enron/emails.csv --n 1000
    python scripts/sample_enron.py --csv data/sources/enterprise/enron/emails.csv --n 1000 --seed 42
"""
import argparse
import csv
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def sample_enron(csv_path: str, output_dir: str, n: int, seed: int = 42) -> None:
    """Lee el CSV en streaming, muestrea n filas con reservoir sampling."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    total = 0

    logger.info("Reading %s (streaming)...", csv_path)
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            if len(reservoir) < n:
                reservoir.append(row)
            else:
                j = rng.randint(0, total - 1)
                if j < n:
                    reservoir[j] = row

    logger.info("Total rows read: %d — sampled: %d", total, len(reservoir))

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for i, row in enumerate(reservoir):
        message = row.get("message", row.get("Message", "")).strip()
        if not message:
            continue
        out_path = os.path.join(output_dir, f"email_{i:05d}.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(message)
        saved += 1

    logger.info("Saved %d email files to %s", saved, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Enron emails from CSV")
    parser.add_argument("--csv", required=True, help="Path to emails.csv")
    parser.add_argument("--n", type=int, default=1000, help="Number of emails to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/sources/enterprise/enron/sampled",
        help="Output directory for .txt files",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        logger.error("CSV not found: %s", args.csv)
        raise SystemExit(1)

    sample_enron(args.csv, args.output, args.n, args.seed)


if __name__ == "__main__":
    main()
