import logging
import os
from typing import Iterator

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.ingestion.base import BaseConnector, Document

logger = logging.getLogger(__name__)

_PDF_EXTENSIONS = {".pdf"}


class PDFConnector(BaseConnector):
    source_type = "pdf"

    def discover(self, root_path: str) -> Iterator[str]:
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in _PDF_EXTENSIONS:
                    yield os.path.join(dirpath, fname)

    def parse(self, file_path: str) -> Document:
        try:
            reader = PdfReader(file_path)
        except PdfReadError as exc:
            raise ValueError(f"Cannot read PDF {file_path}: {exc}") from exc

        # Metadatos del PDF
        info = reader.metadata or {}
        title = _clean(info.get("/Title")) or os.path.basename(file_path)
        author = _clean(info.get("/Author"))
        created_at = _clean(info.get("/CreationDate"))

        # Extraer texto página a página
        pages_text: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
            else:
                logger.debug("PDF %s page %d has no extractable text — skipped", file_path, i)

        if not pages_text:
            logger.warning("PDF %s has no extractable text (scanned?)", file_path)

        content = "\n\n".join(pages_text)

        return Document(
            content=content,
            source_path=file_path,
            source_type=self.source_type,
            title=title,
            author=author,
            created_at=created_at,
            metadata={
                "page_count": len(reader.pages),
                "pages_with_text": len(pages_text),
            },
        )


def _clean(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None
