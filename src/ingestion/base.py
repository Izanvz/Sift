import uuid
from abc import ABC, abstractmethod
from typing import Iterator

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Documento completo parseado de un archivo fuente."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source_path: str
    source_type: str        # "pdf" | "markdown" | "code" | "email"
    title: str | None = None
    author: str | None = None
    created_at: str | None = None
    metadata: dict = Field(default_factory=dict)


class IngestChunk(BaseModel):
    """Fragmento de documento listo para indexar en ChromaDB."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    chunk_index: int
    source_path: str
    source_type: str
    page_number: int | None = None      # PDFs
    line_start: int | None = None       # Código
    line_end: int | None = None
    metadata: dict = Field(default_factory=dict)


class BaseConnector(ABC):
    """Interfaz común para todos los conectores de ingestion."""
    source_type: str

    @abstractmethod
    def discover(self, root_path: str) -> Iterator[str]:
        """Encuentra archivos a procesar bajo root_path."""

    @abstractmethod
    def parse(self, file_path: str) -> Document:
        """Parsea un archivo a Document."""
