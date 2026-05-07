"""EmailConnector — parsea emails del dataset Enron.

Soporta dos formatos:
- CSV (columnas 'File' y 'Message') — formato Kaggle enron-email-dataset
- Archivos .txt / sin extensión con headers RFC 2822
"""
import email as stdlib_email
import logging
import os
from typing import Iterator

from src.ingestion.base import BaseConnector, Document

logger = logging.getLogger(__name__)

_EMAIL_EXTENSIONS = {".txt", ".email", ""}  # Enron usa archivos sin extensión


class EmailConnector(BaseConnector):
    """Parsea un directorio de archivos de email individuales (formato Enron mbox-like).

    Uso con CSV Enron (post-sampling):
        Primero ejecutar scripts/sample_enron.py para generar emails individuales en
        data/sources/enterprise/enron/ y luego usar este connector.
    """
    source_type = "email"

    def discover(self, root_path: str) -> Iterator[str]:
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _EMAIL_EXTENSIONS and not fname.startswith("."):
                    yield os.path.join(dirpath, fname)

    def parse(self, file_path: str) -> Document:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        return _parse_email_text(raw, file_path)


class EnronCSVConnector(BaseConnector):
    """Lee directamente el CSV de Kaggle (1.3 GB) sin sampling previo.

    Uso: EnronCSVConnector().stream_csv(csv_path, max_rows=1000)
    No implementa discover/parse — usa stream_csv() directamente.
    """
    source_type = "email"

    def discover(self, root_path: str) -> Iterator[str]:
        for fname in os.listdir(root_path):
            if fname.endswith(".csv"):
                yield os.path.join(root_path, fname)

    def parse(self, file_path: str) -> Document:
        # No usar directamente — preferir stream_csv()
        raise NotImplementedError("Use stream_csv() for CSV files")

    def stream_csv(self, csv_path: str, max_rows: int = 1000) -> Iterator[Document]:
        """Lee el CSV de Enron fila a fila (streaming, no carga en RAM)."""
        import csv
        count = 0
        with open(csv_path, encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if count >= max_rows:
                    break
                message_text = row.get("message", row.get("Message", ""))
                file_name = row.get("file", row.get("File", f"email_{count}"))
                if message_text.strip():
                    yield _parse_email_text(message_text, file_name)
                    count += 1
        logger.info("Streamed %d emails from %s", count, csv_path)


# ---------------------------------------------------------------------------
# Parser común
# ---------------------------------------------------------------------------

def _parse_email_text(raw: str, source_path: str) -> Document:
    """Parsea texto RFC 2822 a Document."""
    msg = stdlib_email.message_from_string(raw)

    subject = _decode_header(msg.get("Subject", ""))
    from_ = _decode_header(msg.get("From", ""))
    to = _decode_header(msg.get("To", ""))
    cc = _decode_header(msg.get("Cc", ""))
    date = _decode_header(msg.get("Date", ""))
    message_id = _decode_header(msg.get("Message-ID", ""))

    body = _extract_body(msg)

    # Contenido indexable: subject + body
    content = f"Subject: {subject}\n\n{body}".strip()

    return Document(
        content=content,
        source_path=source_path,
        source_type="email",
        title=subject or "(sin asunto)",
        author=from_,
        created_at=date,
        metadata={
            "from": from_,
            "to": to,
            "cc": cc,
            "subject": subject,
            "date": date,
            "message_id": message_id,
            "thread_id": _extract_thread_id(message_id),
        },
    )


def _extract_body(msg) -> str:
    """Extrae el body de texto plano del email."""
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    parts.append(payload.decode("utf-8", errors="replace"))
        return "\n".join(parts)
    payload = msg.get_payload(decode=True)
    if payload:
        return payload.decode("utf-8", errors="replace")
    return msg.get_payload() or ""


def _decode_header(value: str) -> str:
    if not value:
        return ""
    return str(value).strip()


def _extract_thread_id(message_id: str) -> str:
    """Usa message_id como thread_id aproximado (Enron no tiene In-Reply-To consistente)."""
    return message_id.strip("<>").split("@")[0] if message_id else ""
