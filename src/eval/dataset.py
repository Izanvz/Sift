"""Dataset de evaluación — Q&A pairs con ground truth.

Formato JSONL:
    {"id", "query", "ground_truth", "expected_sources", "query_type", "tags"}

- id: identificador único (ej. "q001")
- query: pregunta del usuario
- ground_truth: respuesta correcta esperada (texto)
- expected_sources: lista de paths/patterns que deberían aparecer en citations
- query_type: factual | analytical | comparative | ambiguous
- tags: opcionales, para filtrar/segmentar (ej. ["vercel", "infra"])
"""
import json
import logging
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    id: str
    query: str
    ground_truth: str
    expected_sources: list[str] = Field(default_factory=list)
    query_type: str = "factual"
    tags: list[str] = Field(default_factory=list)


def load_jsonl(path: str | Path) -> list[QAPair]:
    """Carga Q&A pairs desde JSONL."""
    pairs: list[QAPair] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                pairs.append(QAPair(**data))
            except Exception as exc:
                logger.warning("Skipping malformed line %d: %s", i, exc)
    return pairs


def save_jsonl(pairs: Iterable[QAPair], path: str | Path) -> None:
    """Guarda Q&A pairs a JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.model_dump(), ensure_ascii=False) + "\n")


def filter_by_tag(pairs: list[QAPair], tag: str) -> list[QAPair]:
    return [p for p in pairs if tag in p.tags]


def filter_by_type(pairs: list[QAPair], query_type: str) -> list[QAPair]:
    return [p for p in pairs if p.query_type == query_type]
