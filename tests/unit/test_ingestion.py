"""Tests de ingestion — sin ChromaDB ni Ollama."""
import os
import tempfile
import textwrap

import pytest

from src.ingestion.base import Document, IngestChunk
from src.ingestion.chunker import split
from src.ingestion.code import CodeConnector, _extract_python_blocks
from src.ingestion.email import _parse_email_text
from src.ingestion.markdown import MarkdownConnector, _parse_frontmatter
from src.ingestion.pdf import PDFConnector
from src.ingestion.pipeline import ingest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(content: str, source_type: str = "markdown") -> Document:
    return Document(
        content=content,
        source_path="/tmp/test.md",
        source_type=source_type,
    )


def _write_temp(content: str, suffix: str, dir: str) -> str:
    path = os.path.join(dir, f"test{suffix}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# IngestChunk
# ---------------------------------------------------------------------------

def test_ingest_chunk_defaults():
    chunk = IngestChunk(
        document_id="doc1",
        content="hello",
        chunk_index=0,
        source_path="/tmp/f.md",
        source_type="markdown",
    )
    assert chunk.chunk_id  # auto-generado
    assert chunk.page_number is None
    assert chunk.line_start is None


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def test_chunker_splits_long_document():
    # ~600 palabras → debe producir al menos 2 chunks con size=512
    words = " ".join(["word"] * 600)
    doc = _doc(words)
    chunks = split(doc, chunk_size=512, chunk_overlap=64)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.content.strip()
        assert c.document_id == doc.id


def test_chunker_short_document_single_chunk():
    doc = _doc("This is a short document.")
    chunks = split(doc, chunk_size=512, chunk_overlap=64)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_chunker_preserves_metadata():
    doc = Document(
        content="Some content here.",
        source_path="/tmp/x.pdf",
        source_type="pdf",
        title="My Doc",
        metadata={"page_count": 3},
    )
    chunks = split(doc)
    assert all(c.source_path == "/tmp/x.pdf" for c in chunks)
    assert all(c.source_type == "pdf" for c in chunks)


def test_chunker_empty_document():
    doc = _doc("")
    chunks = split(doc)
    assert chunks == []


def test_chunker_chunk_indices_sequential():
    content = "\n\n".join([f"Paragraph {i}. " + "word " * 80 for i in range(10)])
    doc = _doc(content)
    chunks = split(doc, chunk_size=200, chunk_overlap=20)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Markdown connector
# ---------------------------------------------------------------------------

def test_markdown_parses_frontmatter():
    raw = textwrap.dedent("""\
        ---
        title: Test Doc
        author: Izan
        tags: [rag, sift]
        ---
        # Introduction

        Some content here.
    """)
    meta, content = _parse_frontmatter(raw)
    assert meta["title"] == "Test Doc"
    assert meta["author"] == "Izan"
    assert "Introduction" in content


def test_markdown_connector_discover(tmp_path):
    (tmp_path / "doc.md").write_text("# Hello", encoding="utf-8")
    (tmp_path / "other.txt").write_text("ignore me", encoding="utf-8")
    connector = MarkdownConnector()
    found = list(connector.discover(str(tmp_path)))
    assert len(found) == 1
    assert found[0].endswith(".md")


def test_markdown_connector_parse(tmp_path):
    content = textwrap.dedent("""\
        ---
        title: My Guide
        ---
        # My Guide

        This is body text.
    """)
    path = str(tmp_path / "guide.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    connector = MarkdownConnector()
    doc = connector.parse(path)
    assert doc.title == "My Guide"
    assert doc.source_type == "markdown"
    assert "body text" in doc.content


# ---------------------------------------------------------------------------
# Code connector
# ---------------------------------------------------------------------------

def test_code_extract_python_functions():
    source = textwrap.dedent("""\
        import os

        def hello():
            return "hello"

        def world(x: int) -> int:
            return x * 2

        CONSTANT = 42
    """)
    blocks = _extract_python_blocks(source, "test.py")
    assert len(blocks) == 2
    assert "def hello" in blocks[0]
    assert "def world" in blocks[1]


def test_code_connector_discover_python(tmp_path):
    (tmp_path / "main.py").write_text("def foo(): pass", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# readme", encoding="utf-8")
    connector = CodeConnector(extensions={".py"})
    found = list(connector.discover(str(tmp_path)))
    assert len(found) == 1
    assert found[0].endswith(".py")


def test_code_connector_ignores_pycache(tmp_path):
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "module.pyc").write_bytes(b"")
    (tmp_path / "real.py").write_text("x = 1", encoding="utf-8")
    connector = CodeConnector(extensions={".py"})
    found = list(connector.discover(str(tmp_path)))
    assert all("__pycache__" not in f for f in found)


def test_code_connector_parse_python(tmp_path):
    src = textwrap.dedent("""\
        def greet(name: str) -> str:
            return f"Hello {name}"
    """)
    path = str(tmp_path / "greet.py")
    with open(path, "w") as fh:
        fh.write(src)
    connector = CodeConnector()
    doc = connector.parse(path)
    assert doc.source_type == "code"
    assert "greet" in doc.content


# ---------------------------------------------------------------------------
# Email connector
# ---------------------------------------------------------------------------

def test_email_parse_basic():
    raw = textwrap.dedent("""\
        Message-ID: <123@enron.com>
        Date: Mon, 14 May 2001 09:00:00
        From: sender@enron.com
        To: receiver@enron.com
        Subject: Test subject

        This is the email body.
    """)
    doc = _parse_email_text(raw, "email_001.txt")
    assert doc.source_type == "email"
    assert doc.title == "Test subject"
    assert doc.author == "sender@enron.com"
    assert "email body" in doc.content


def test_email_parse_missing_headers():
    raw = "Just a plain text with no headers."
    doc = _parse_email_text(raw, "bare.txt")
    assert doc.source_type == "email"
    assert doc.content  # no debe explotar


# ---------------------------------------------------------------------------
# Pipeline (sin ChromaDB — mock upsert)
# ---------------------------------------------------------------------------

def test_pipeline_processes_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("# Doc A\n\nSome content here.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Doc B\n\nMore content here.", encoding="utf-8")

    collected: list[IngestChunk] = []

    def mock_upsert(chunks):
        collected.extend(chunks)

    connector = MarkdownConnector()
    stats = ingest(str(tmp_path), connector, mock_upsert)

    assert stats["processed"] == 2
    assert stats["errors"] == 0
    assert len(collected) >= 2


def test_pipeline_skips_empty_files(tmp_path):
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    called = []
    connector = MarkdownConnector()
    stats = ingest(str(tmp_path), connector, lambda c: called.extend(c))

    assert stats["skipped"] == 1
    assert stats["processed"] == 0
    assert called == []


def test_pipeline_handles_connector_error(tmp_path):
    """Si un archivo falla al parsear, el pipeline continúa."""
    (tmp_path / "good.md").write_text("# Good\n\nContent.", encoding="utf-8")
    (tmp_path / "bad.md").write_text("\x00\x01\x02", encoding="utf-8")  # bytes raros

    collected: list[IngestChunk] = []

    class ErrorConnector(MarkdownConnector):
        def parse(self, file_path):
            if "bad" in file_path:
                raise ValueError("Intentional error")
            return super().parse(file_path)

    stats = ingest(str(tmp_path), ErrorConnector(), lambda c: collected.extend(c))
    assert stats["errors"] == 1
    assert stats["processed"] == 1
