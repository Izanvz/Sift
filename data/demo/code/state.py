from typing import Annotated, Literal, TypedDict
import operator

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tipos de datos del dominio
# ---------------------------------------------------------------------------

class HumanFeedback(BaseModel):
    """Respuesta del usuario a una petición de clarificación."""
    clarification: str


class Chunk(TypedDict):
    """Fragmento de documento recuperado del vector store."""
    chunk_id: str
    document_id: str
    source_path: str
    source_type: str        # "pdf" | "markdown" | "code" | "email"
    content: str
    relevance_score: float
    metadata: dict


class Citation(TypedDict):
    """Referencia a un chunk concreto usada en la respuesta final."""
    chunk_id: str
    source_path: str
    source_type: str
    title: str | None
    page_number: int | None     # Para PDFs
    line_start: int | None      # Para código
    line_end: int | None
    snippet: str                # Texto exacto citado
    relevance_score: float


# ---------------------------------------------------------------------------
# Modelos de salida estructurada (instructor)
# ---------------------------------------------------------------------------

class QueryClassification(BaseModel):
    query_type: Literal["factual", "analytical", "comparative", "ambiguous"]
    reasoning: str = Field(default="")


class RelevanceEval(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class CritiqueOutput(BaseModel):
    score: float = Field(..., ge=0, le=10)
    faithfulness: float = Field(..., ge=0, le=10)
    completeness: float = Field(..., ge=0, le=10)
    citation_quality: float = Field(..., ge=0, le=10)
    gaps: list[str]
    recommendation: str


# ---------------------------------------------------------------------------
# Estado principal del grafo
# ---------------------------------------------------------------------------

class SiftState(TypedDict):
    # Input
    query: str
    user_id: str | None             # Para permisos (Fase 7)
    scopes: list[str]               # Corpora permitidos para el usuario (Fase 7)
    is_admin: bool                  # Bypass scope filter (Fase 7)

    # Clasificación
    query_type: str                 # "factual" | "analytical" | "comparative" | "ambiguous"

    # Retrieval
    chunks: list[Chunk]             # Chunks recuperados (acumulativos entre iteraciones)
    relevance_scores: list[float]   # Historial de scores de relevancia
    iterations: int                 # Contador de ciclos de búsqueda

    # Generación
    answer: str                     # Respuesta con marcadores [N]
    citations: list[Citation]       # Mapeadas desde los [N] del answer

    # Critique loop
    critique: dict                  # CritiqueOutput serializado
    rewrite_iterations: int

    # Human in the loop
    clarification: str | None       # Pregunta de clarificación generada

    # Metadata
    metadata: dict
