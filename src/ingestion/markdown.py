import os
import re
from typing import Iterator

import yaml

from src.ingestion.base import BaseConnector, Document

_MD_EXTENSIONS = {".md", ".mdx", ".markdown"}
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class MarkdownConnector(BaseConnector):
    source_type = "markdown"

    def discover(self, root_path: str) -> Iterator[str]:
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in _MD_EXTENSIONS:
                    yield os.path.join(dirpath, fname)

    def parse(self, file_path: str) -> Document:
        with open(file_path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read()

        frontmatter, content = _parse_frontmatter(raw)

        title = frontmatter.get("title") or os.path.splitext(os.path.basename(file_path))[0]
        author = frontmatter.get("author") or frontmatter.get("authors")
        created_at = str(frontmatter.get("date", "")) or None
        tags = frontmatter.get("tags", [])

        # Construir section_path desde el árbol de headers
        section_path = _extract_first_header(content) or title

        return Document(
            content=content,
            source_path=file_path,
            source_type=self.source_type,
            title=title,
            author=str(author) if author else None,
            created_at=created_at,
            metadata={
                "section_path": section_path,
                "tags": tags if isinstance(tags, list) else [tags],
                "frontmatter_keys": list(frontmatter.keys()),
            },
        )


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Extrae YAML frontmatter si existe. Devuelve (meta_dict, content_sin_frontmatter)."""
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    content = raw[match.end():]
    return meta, content


def _extract_first_header(content: str) -> str | None:
    """Devuelve el texto del primer header Markdown (#, ##, etc.)."""
    for line in content.splitlines():
        stripped = line.lstrip("#").strip()
        if line.startswith("#") and stripped:
            return stripped
    return None
