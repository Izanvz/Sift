"""CodeConnector — extrae funciones y clases como chunks atómicos.

Python: usa `ast` stdlib (preciso, cero dependencias extra).
TypeScript/JS y otros: fallback regex por funciones/clases.
"""
import ast
import logging
import os
import re
from typing import Iterator

from src.ingestion.base import BaseConnector, Document

logger = logging.getLogger(__name__)

_PY_EXTENSIONS = {".py"}
_TS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}
_GO_EXTENSIONS = {".go"}
_ALL_CODE_EXTENSIONS = _PY_EXTENSIONS | _TS_EXTENSIONS | _GO_EXTENSIONS

# Directorios a ignorar (artefactos de build, deps)
_IGNORE_DIRS = {
    "__pycache__", ".git", "node_modules", "dist", "build",
    ".venv", "venv", "env", ".mypy_cache", ".pytest_cache",
}


class CodeConnector(BaseConnector):
    source_type = "code"

    def __init__(self, extensions: set[str] | None = None):
        self.extensions = extensions or _ALL_CODE_EXTENSIONS

    def discover(self, root_path: str) -> Iterator[str]:
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Poda in-place para que os.walk no baje a dirs ignorados
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in self.extensions:
                    yield os.path.join(dirpath, fname)

    def parse(self, file_path: str) -> Document:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                source = fh.read()
        except OSError as exc:
            raise ValueError(f"Cannot read {file_path}: {exc}") from exc

        if ext in _PY_EXTENSIONS:
            blocks = _extract_python_blocks(source, file_path)
        elif ext in _TS_EXTENSIONS:
            blocks = _extract_ts_blocks(source)
        elif ext in _GO_EXTENSIONS:
            blocks = _extract_go_blocks(source)
        else:
            blocks = [source]

        # Unimos los bloques con separadores para el chunker
        content = "\n\n".join(blocks) if blocks else source

        return Document(
            content=content,
            source_path=file_path,
            source_type=self.source_type,
            title=os.path.basename(file_path),
            metadata={
                "language": _lang(ext),
                "block_count": len(blocks),
                "file_size_bytes": len(source.encode()),
            },
        )


# ---------------------------------------------------------------------------
# Python: ast
# ---------------------------------------------------------------------------

def _extract_python_blocks(source: str, file_path: str) -> list[str]:
    """Extrae funciones y clases de nivel superior como strings."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.warning("SyntaxError in %s: %s", file_path, exc)
        return [source]

    lines = source.splitlines(keepends=True)
    blocks: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno or len(lines)
            block = "".join(lines[start:end])
            blocks.append(block)

    # Si no hay bloques (archivo solo con imports/constantes), devolver todo
    return blocks if blocks else [source]


# ---------------------------------------------------------------------------
# TypeScript/JS: regex
# ---------------------------------------------------------------------------

_TS_FUNC_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+\w+\s*\([^)]*\)[^{]*\{",
    re.MULTILINE,
)
_TS_ARROW_RE = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\S+\s*)?=>",
    re.MULTILINE,
)
_TS_CLASS_RE = re.compile(
    r"(?:export\s+)?(?:abstract\s+)?class\s+\w+",
    re.MULTILINE,
)


def _extract_ts_blocks(source: str) -> list[str]:
    lines = source.splitlines()
    starts = sorted({
        m.start() for m in [*_TS_FUNC_RE.finditer(source),
                             *_TS_ARROW_RE.finditer(source),
                             *_TS_CLASS_RE.finditer(source)]
    })
    if not starts:
        return [source]

    blocks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(source)
        blocks.append(source[start:end].strip())
    return blocks


# ---------------------------------------------------------------------------
# Go: regex
# ---------------------------------------------------------------------------

_GO_FUNC_RE = re.compile(r"^func\s+", re.MULTILINE)


def _extract_go_blocks(source: str) -> list[str]:
    starts = [m.start() for m in _GO_FUNC_RE.finditer(source)]
    if not starts:
        return [source]
    blocks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(source)
        blocks.append(source[start:end].strip())
    return blocks


def _lang(ext: str) -> str:
    mapping = {".py": "python", ".ts": "typescript", ".tsx": "typescript",
               ".js": "javascript", ".jsx": "javascript", ".go": "go"}
    return mapping.get(ext, "unknown")
