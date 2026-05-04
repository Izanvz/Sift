# Sift v2 — Roadmap completo

**Versión:** 1.0
**Estado:** planificación
**Duración estimada:** 4 semanas part-time (~60-80h totales)
**Pivote desde:** ResearchAgent v1 (web research agent)
**Posicionamiento:** On-premise enterprise knowledge agent

---

## Tabla de contenidos

- [0. Pre-requisitos y decisiones](#0-pre-requisitos-y-decisiones)
- [1. Refactor base del proyecto](#1-refactor-base-del-proyecto)
- [2. Ingestion multi-source](#2-ingestion-multi-source)
- [3. Hybrid search + re-ranking](#3-hybrid-search--re-ranking)
- [4. Citaciones precisas](#4-citaciones-precisas)
- [5. Self-critique loop adaptado](#5-self-critique-loop-adaptado)
- [6. Eval suite con RAGAS](#6-eval-suite-con-ragas)
- [7. Permisos por usuario](#7-permisos-por-usuario)
- [8. Logs de auditoría](#8-logs-de-auditoría)
- [9. UI](#9-ui)
- [10. README + benchmarks](#10-readme--benchmarks)
- [11. Demo + deploy](#11-demo--deploy)
- [12. Checklist final](#12-checklist-final)

---

## 0. Pre-requisitos y decisiones

### 0.1 Decisiones de producto (cerrar antes de empezar)

| Decisión | Recomendado | Alternativa |
|---|---|---|
| Nombre | **Sift** | KnowledgeAgent / Vault |
| Posicionamiento | "On-premise enterprise knowledge agent" | "Self-hosted RAG for teams" |
| Casos de uso primarios | Personal docs + Enterprise multi-source | Solo enterprise |
| LLM por defecto | Ollama (`qwen2.5:7b` o `mistral:7b`) | OpenAI fallback |
| Embeddings | `nomic-embed-text` (Ollama, local) | OpenAI ada-002 |
| Vector store | ChromaDB (ya en uso) | Qdrant |
| BM25 | `rank_bm25` (in-memory) | Elasticsearch |
| Re-ranker | `BAAI/bge-reranker-base` (HuggingFace) | Cohere Rerank |

### 0.2 Stack técnico final

```
Backend:    Python 3.12 + FastAPI + LangGraph + Pydantic v2
Storage:    SQLite (metadata + audit) + ChromaDB (vectores)
LLM:        Ollama (local) con fallback OpenAI/Anthropic via env
Embeddings: nomic-embed-text via Ollama
Search:     Hybrid (BM25 + vectorial) + RRF fusion + BGE-reranker
Eval:       RAGAS + dataset propio
UI:         Mínima (FastAPI + HTMX o Next.js si hay tiempo)
Deploy:     Docker Compose (no nube)
```

### 0.3 Dataset de pruebas (preparar primero)

Crear directorio `data/sources/` con tres orígenes:

```
data/sources/
├── personal/         # Tus docs reales: CV, certificados, cartas
│   ├── cv/
│   ├── certificates/
│   └── letters/
├── code/             # Tu carpeta /Claude (proyectos)
│   └── (symlink o copia de proyectos seleccionados)
└── enterprise/       # Dataset simulado empresarial
    ├── enron/        # Subset de Enron emails (~1000 emails)
    ├── stripe-docs/  # Clone de github.com/stripe/stripe-docs
    └── vercel-docs/  # Clone de github.com/vercel/docs
```

**Acciones concretas:**
- [ ] Descargar Enron subset desde Kaggle: `enron-email-dataset` (filtrar a 1000 emails representativos)
- [ ] `git clone` de stripe-docs y vercel-docs (ambos públicos, MIT/Apache)
- [ ] Copiar tus docs personales sin información sensible (revisar antes de indexar)

### 0.4 Branching strategy

```bash
git checkout -b feature/sift-v2
# Desarrollo en esta rama durante 4 semanas
# Merge a master cuando esté completo + tag v2.0.0
```

### 0.5 Estructura final del proyecto (después del refactor)

```
sift/
├── config/
│   ├── settings.py              # Pydantic settings (extender el actual)
│   └── prompts.py               # Prompts centralizados
├── src/
│   ├── agent/
│   │   ├── graph.py             # Grafo LangGraph (refactorizar)
│   │   ├── nodes.py             # Nodos del grafo
│   │   ├── edges.py             # Routing
│   │   ├── state.py             # State con citaciones
│   │   └── prompts.py           # Prompts del agente
│   ├── ingestion/
│   │   ├── base.py              # BaseConnector ABC
│   │   ├── pdf.py               # PDFConnector
│   │   ├── markdown.py          # MarkdownConnector
│   │   ├── code.py              # CodeConnector
│   │   ├── email.py             # EmailConnector (Enron-like)
│   │   ├── chunker.py           # Chunking semántico
│   │   └── pipeline.py          # Orquestación
│   ├── retrieval/
│   │   ├── vector.py            # Búsqueda vectorial (Chroma)
│   │   ├── bm25.py              # BM25 in-memory
│   │   ├── hybrid.py            # RRF fusion
│   │   └── reranker.py          # BGE reranker
│   ├── auth/
│   │   ├── models.py            # User, Role, Permission
│   │   ├── scope.py             # Filtros de scope por usuario
│   │   └── middleware.py        # Auth middleware FastAPI
│   ├── audit/
│   │   ├── logger.py            # Logger estructurado
│   │   └── models.py            # AuditEvent
│   ├── api/
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── static/              # UI mínima
│   ├── db/
│   │   ├── sqlite.py            # Migrations + queries
│   │   ├── vector_store.py      # Wrapper Chroma
│   │   └── checkpointer.py      # LangGraph checkpointer
│   └── eval/
│       ├── ragas_runner.py      # Suite RAGAS
│       ├── dataset.py           # Q&A dataset propio
│       └── benchmark.py         # Comparativa v1 vs v2
├── data/
│   ├── sources/                 # Datasets de prueba
│   ├── chroma/                  # Vector store (gitignored)
│   ├── bm25_index/              # Índice BM25 serializado
│   └── audit.db                 # SQLite audit log
├── docs/
│   ├── architecture.md          # Diagrama + decisiones
│   ├── benchmarks.md            # Resultados RAGAS
│   ├── deployment.md            # Cómo correrlo
│   └── sift-v2-roadmap.md       # Este archivo
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/
├── docker-compose.yml           # NUEVO — añadir al proyecto
├── Dockerfile                   # NUEVO
├── requirements.txt
└── README.md                    # Reescribir completamente
```

---

## 1. Refactor base del proyecto

**Duración:** 1 día (8h)
**Objetivo:** dejar el proyecto preparado para añadir features sin caos

### 1.1 Renombrar proyecto a Sift

- [ ] Crear directorio nuevo `~/Documents/Claude/Sift/` (o renombrar in-place)
- [ ] Actualizar `README.md` cabecera con nombre Sift
- [ ] Buscar y reemplazar referencias a "ResearchAgent" en código y docs
- [ ] Actualizar `pyproject.toml` (si existe) o crear uno con `name = "sift"`

### 1.2 Reorganizar estructura

- [ ] Crear directorios nuevos: `src/ingestion/`, `src/retrieval/`, `src/auth/`, `src/audit/`, `src/eval/`
- [ ] Mover `src/agent/tools.py` → eliminar (su lógica se reparte en `ingestion/` y `retrieval/`)
- [ ] Crear `src/agent/prompts.py` y mover todos los prompts inline ahí

### 1.3 Limpiar grafo actual

El grafo actual de ResearchAgent tiene nodos que ya no aplican (`search_web`, `search_arxiv`).
Adaptar:

| Nodo viejo | Nodo nuevo | Cambio |
|---|---|---|
| `plan_research` | `route_query` | Clasificar: factual / analytical / comparative |
| `search_web` | ❌ eliminar | Sustituido por hybrid search |
| `search_chromadb` | `retrieve` | Usar HybridRetriever nuevo |
| `search_arxiv` | ❌ eliminar | |
| `gather_results` | `gather` | Mantener, simplificar |
| `evaluate_quality` | `evaluate_relevance` | Renombrar |
| `refine_query` | `rewrite_query` | Mantener |
| `synthesize` | `synthesize` | Mantener pero con citaciones |
| `self_critique` | `self_critique` | Mantener |
| `rewrite` | `rewrite_answer` | Renombrar |
| `human_checkpoint` | `clarification_request` | Solo si query ambigua |
| `generate_report` | `format_response` | Simplificar — no es informe largo |

### 1.4 Setup de dependencias

Añadir a `requirements.txt`:

```
# Existentes a mantener:
langgraph==0.2.55
langgraph-checkpoint-sqlite==2.0.11
langchain-community==0.3.14
langchain-ollama==0.2.3
langchain-core>=0.3.33,<0.4.0
chromadb>=1.0.0
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.5
pydantic-settings==2.7.1
instructor==1.7.2

# Nuevas:
rank-bm25==0.2.2              # BM25 in-memory
sentence-transformers==3.3.1  # Para BGE reranker
pypdf==5.1.0                  # PDF parsing
python-magic==0.4.27          # Detección de tipos de archivo
markdown==3.7                 # Markdown parsing
tree-sitter==0.23.2           # Code parsing
tree-sitter-python==0.23.6
tree-sitter-typescript==0.23.2
ragas==0.2.10                 # Evaluación RAG
datasets==3.2.0               # Para RAGAS
passlib[bcrypt]==1.7.4        # Auth (hash passwords)
python-jose[cryptography]==3.3.0  # JWT
python-multipart==0.0.20      # Form data
```

### 1.5 Setup de Docker

Crear `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libmagic1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

Crear `docker-compose.yml`:

```yaml
services:
  api:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./data:/app/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - CHROMADB_HOST=chromadb
      - CHROMADB_PORT=8000
    depends_on:
      ollama:
        condition: service_healthy
      chromadb:
        condition: service_healthy

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    entrypoint: /bin/sh -c "ollama serve & sleep 5 && ollama pull qwen2.5:7b && ollama pull nomic-embed-text & wait"
    healthcheck:
      test: ["CMD", "bash", "-c", "exec 3<>/dev/tcp/localhost/11434"]
      interval: 5s
      timeout: 5s
      retries: 12

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8002:8000"
    volumes:
      - chroma_data:/chroma/chroma
    healthcheck:
      test: ["CMD", "bash", "-c", "exec 3<>/dev/tcp/localhost/8000"]
      interval: 5s
      timeout: 5s
      retries: 12

volumes:
  ollama_data:
  chroma_data:
```

### 1.6 Validación del refactor

- [ ] `docker-compose up` arranca los 3 servicios sin error
- [ ] `pytest tests/` pasa los tests existentes (los que sigan aplicando)
- [ ] Eliminar tests obsoletos (los que prueben funcionalidad removida)

---

## 2. Ingestion multi-source

**Duración:** 3 días (24h)
**Objetivo:** poder indexar PDFs + Markdown + código + emails con metadatos ricos

### 2.1 Diseñar `BaseConnector` (ABC)

`src/ingestion/base.py`:

```python
from abc import ABC, abstractmethod
from typing import Iterator
from pydantic import BaseModel

class Document(BaseModel):
    content: str
    source_path: str           # Path absoluto al archivo
    source_type: str           # "pdf" | "markdown" | "code" | "email"
    title: str | None = None
    author: str | None = None
    created_at: str | None = None
    metadata: dict             # Específico por tipo

class Chunk(BaseModel):
    document_id: str           # FK al Document padre
    chunk_id: str              # UUID único
    content: str
    chunk_index: int           # Posición en el documento
    page_number: int | None    # Para PDFs
    line_start: int | None     # Para código
    line_end: int | None
    metadata: dict             # Hereda + añade

class BaseConnector(ABC):
    source_type: str

    @abstractmethod
    def discover(self, root_path: str) -> Iterator[str]:
        """Encuentra archivos a procesar."""

    @abstractmethod
    def parse(self, file_path: str) -> Document:
        """Parsea un archivo a Document."""
```

### 2.2 Implementar `PDFConnector`

`src/ingestion/pdf.py`:
- [ ] Usar `pypdf` para extraer texto página por página
- [ ] Preservar `page_number` en metadatos por chunk
- [ ] Extraer título y autor del PDF metadata si existe
- [ ] Manejar PDFs escaneados (skip si no hay texto extraíble, log warning)

### 2.3 Implementar `MarkdownConnector`

`src/ingestion/markdown.py`:
- [ ] Parsear frontmatter YAML si existe (título, tags, autor)
- [ ] Chunking respetando headers (no partir en medio de una sección)
- [ ] Preservar jerarquía de headers como `metadata.section_path`

### 2.4 Implementar `CodeConnector`

`src/ingestion/code.py`:
- [ ] Usar `tree-sitter` para parsear funciones/clases como chunks atómicos
- [ ] Soportar Python y TypeScript (suficiente para tu caso)
- [ ] `chunk.metadata`: `{"function_name", "class_name", "imports": [...]}`
- [ ] Preservar `line_start` y `line_end` para citas precisas

### 2.5 Implementar `EmailConnector`

`src/ingestion/email.py`:
- [ ] Parsear formato Enron (mbox / archivos .txt con headers)
- [ ] Extraer: From, To, Cc, Subject, Date, Body
- [ ] Un chunk por email (o por párrafo si email muy largo)
- [ ] `metadata`: `{"from", "to", "subject", "date", "thread_id"}`

### 2.6 Chunker inteligente

`src/ingestion/chunker.py`:
- [ ] Estrategia: 512 tokens por chunk, overlap de 64 tokens
- [ ] Usar `tiktoken` para contar tokens (compatible con la mayoría de modelos)
- [ ] Respetar fronteras semánticas: párrafos > frases > palabras (nunca cortar palabra)
- [ ] Para código: chunks por función completa (no contar tokens)

### 2.7 Pipeline de ingestion

`src/ingestion/pipeline.py`:

```python
def ingest(source_path: str, connector: BaseConnector, store: VectorStore):
    for file_path in connector.discover(source_path):
        try:
            doc = connector.parse(file_path)
            chunks = chunker.split(doc)
            embeddings = embedder.embed([c.content for c in chunks])
            store.upsert(chunks, embeddings)
            audit_log.log("ingest_success", {"path": file_path, "chunks": len(chunks)})
        except Exception as e:
            audit_log.log("ingest_error", {"path": file_path, "error": str(e)})
```

### 2.8 CLI de ingestion

Crear `scripts/ingest.py`:

```bash
python scripts/ingest.py --source data/sources/personal --connector pdf
python scripts/ingest.py --source data/sources/code --connector code
python scripts/ingest.py --source data/sources/enterprise/enron --connector email
python scripts/ingest.py --source data/sources/enterprise/stripe-docs --connector markdown
```

### 2.9 Tests de ingestion

`tests/unit/test_ingestion.py`:
- [ ] Test cada connector con un fixture de ejemplo
- [ ] Test que metadatos se preservan correctamente
- [ ] Test que chunking respeta fronteras semánticas
- [ ] Test idempotencia: re-ingestar no duplica chunks

---

## 3. Hybrid search + re-ranking

**Duración:** 2 días (16h)
**Objetivo:** búsqueda significativamente mejor que vectorial puro

### 3.1 Implementar `VectorRetriever`

`src/retrieval/vector.py`:
- [ ] Wrapper sobre ChromaDB que devuelve `list[ScoredChunk]`
- [ ] Soporta filtros por metadata (autor, source_type, fecha, etc.)
- [ ] Top-k configurable (default: 20)

### 3.2 Implementar `BM25Retriever`

`src/retrieval/bm25.py`:
- [ ] Usar `rank_bm25` (BM25Okapi)
- [ ] Construir índice al arranque desde ChromaDB (cargar todos los chunks en memoria)
- [ ] Persistir índice serializado en `data/bm25_index/` para arranque rápido
- [ ] Re-construir cuando se ingestan nuevos docs (hook al final del pipeline)
- [ ] Soporta los mismos filtros que `VectorRetriever`

### 3.3 Implementar `HybridRetriever` con RRF

`src/retrieval/hybrid.py`:

Reciprocal Rank Fusion (RRF) — algoritmo simple y efectivo:

```python
def rrf_fuse(vector_results: list[ScoredChunk],
             bm25_results: list[ScoredChunk],
             k: int = 60) -> list[ScoredChunk]:
    """
    RRF score = sum(1 / (k + rank_i)) para cada lista donde aparece el chunk.
    k=60 es el valor estándar de la literatura.
    """
    scores = defaultdict(float)
    for rank, chunk in enumerate(vector_results, start=1):
        scores[chunk.chunk_id] += 1 / (k + rank)
    for rank, chunk in enumerate(bm25_results, start=1):
        scores[chunk.chunk_id] += 1 / (k + rank)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [chunk_by_id(cid) for cid in sorted_ids]
```

### 3.4 Re-ranking con BGE

`src/retrieval/reranker.py`:

```python
from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        pairs = [[query, c.content] for c in chunks]
        scores = self.model.predict(pairs)
        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
```

- [ ] Cargar modelo una vez al arranque (singleton)
- [ ] Llamar después de RRF: top 20 candidatos → re-rank → top 5 finales

### 3.5 Pipeline de retrieval completo

```python
def retrieve(query: str, user: User, top_k: int = 5) -> list[Chunk]:
    scope_filter = build_scope_filter(user)
    vec_results = vector_retriever.search(query, k=20, filter=scope_filter)
    bm25_results = bm25_retriever.search(query, k=20, filter=scope_filter)
    fused = rrf_fuse(vec_results, bm25_results)
    reranked = reranker.rerank(query, fused[:20], top_k=top_k)
    return reranked
```

### 3.6 Tests de retrieval

- [ ] Test que hybrid encuentra chunks que vector solo no encuentra (y viceversa)
- [ ] Test que re-ranking mejora el top-1 vs solo RRF
- [ ] Test que filtros de scope aplican correctamente

---

## 4. Citaciones precisas

**Duración:** 2 días (16h)
**Objetivo:** cada respuesta tiene fuentes clickables y verificables

### 4.1 Modelo de citación

`src/agent/state.py`:

```python
class Citation(BaseModel):
    chunk_id: str
    document_id: str
    source_path: str
    source_type: str
    title: str | None
    page_number: int | None
    line_start: int | None
    line_end: int | None
    snippet: str           # Texto exacto citado
    relevance_score: float

class AnswerWithCitations(BaseModel):
    answer: str            # Respuesta con marcadores [1] [2] inline
    citations: list[Citation]
```

### 4.2 Modificar prompt de síntesis

```python
SYNTHESIS_PROMPT = """
Eres un asistente de conocimiento empresarial. Responde la pregunta usando
EXCLUSIVAMENTE la información de los siguientes fragmentos.

Reglas estrictas:
1. Cada afirmación debe llevar una cita inline [N] al chunk que la respalda
2. Si la información no está en los fragmentos, di "No tengo información sobre esto"
3. NO inventes información ni cites conocimiento general
4. Sé conciso

Pregunta: {query}

Fragmentos:
{chunks_with_ids}

Respuesta (con citas inline [N]):
"""
```

Donde `{chunks_with_ids}` es:
```
[1] (PDF, página 4) "El proceso de onboarding consta de 5 fases..."
[2] (Markdown, sección 'Setup') "Para configurar el entorno..."
[3] (Code, file.py:42-58) "def setup_environment():..."
```

### 4.3 Parser de citas inline

Después de la generación, parsear `[1]`, `[2]` del texto y mapear a la lista de chunks.
Validar que todas las citas mencionadas existen (sino, alucinación).

### 4.4 Renderizado de citas en UI

- Texto: `"El onboarding tiene 5 fases [1]. La primera es setup [2]."`
- Bajo el texto: lista numerada con cada citación clickable
- Click → abre modal/panel con el snippet completo + link al archivo original

### 4.5 Tests de citaciones

- [ ] Test que toda cita en el texto existe en `citations`
- [ ] Test que no hay citas huérfanas (en `citations` pero no en texto)
- [ ] Test que `snippet` corresponde realmente al `chunk_id` referenciado

---

## 5. Self-critique loop adaptado

**Duración:** 1 día (8h)
**Objetivo:** reciclar el ciclo de crítica del v1, adaptado a Q&A

### 5.1 Adaptar nodo `self_critique`

Cambiar el prompt de "evalúa esta síntesis" a "evalúa esta respuesta a la pregunta":

```python
CRITIQUE_PROMPT = """
Evalúa la siguiente respuesta a una pregunta del usuario.

Pregunta: {query}
Respuesta: {answer}
Fuentes consultadas: {citations}

Evalúa en una escala 0-10:
- Faithfulness: ¿la respuesta se apoya solo en las fuentes? (alucinaciones penalizan)
- Completeness: ¿responde completamente la pregunta?
- Citation quality: ¿las citas son precisas y suficientes?

Devuelve: score, gaps, recommendation.
"""
```

### 5.2 Routing condicional

`src/agent/edges.py`:

```python
def route_after_critique(state) -> str:
    score = state["critique"]["score"]
    iters = state.get("rewrite_iterations", 0)
    if score >= 8.0 or iters >= 2:
        return "format_response"
    return "rewrite_answer"
```

### 5.3 Nodo `rewrite_answer`

Reescribe la respuesta abordando los `gaps` del critique. Mantiene las mismas citas (no añadir nuevas en rewrite).

---

## 6. Eval suite con RAGAS

**Duración:** 2 días (16h)
**Objetivo:** demostrar rigor científico con benchmarks reproducibles

### 6.1 Crear dataset de evaluación

`src/eval/dataset.py`:

20-30 pares Q&A creados a mano sobre los datasets indexados:

```python
EVAL_DATASET = [
    {
        "question": "¿Cuál es la política de reembolsos de Stripe para suscripciones?",
        "ground_truth": "Stripe permite reembolsos parciales o totales...",
        "expected_sources": ["stripe-docs/refunds.md"],
        "category": "factual"
    },
    {
        "question": "¿Quién aprobó el contrato con Lay Cooperation en 2001?",
        "ground_truth": "Kenneth Lay autorizó...",
        "expected_sources": ["enron/email_12345.txt"],
        "category": "investigative"
    },
    # ... 20-30 ejemplos
]
```

Categorías: `factual`, `analytical`, `comparative`, `investigative`, `code_lookup`.

### 6.2 Runner de RAGAS

`src/eval/ragas_runner.py`:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

def run_evaluation(agent, dataset):
    samples = []
    for item in dataset:
        result = agent.invoke({"query": item["question"]})
        samples.append({
            "question": item["question"],
            "answer": result["answer"],
            "contexts": [c.content for c in result["citations"]],
            "ground_truth": item["ground_truth"],
        })

    return evaluate(
        dataset=samples,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
```

### 6.3 Benchmark v1 vs v2

`src/eval/benchmark.py`:

Correr la misma eval con 3 configuraciones:
- **Baseline:** vectorial puro, sin re-rank, sin self-critique
- **Hybrid:** + BM25 + RRF
- **Hybrid+rerank:** + BGE reranker
- **Full:** + self-critique loop

Generar tabla en `docs/benchmarks.md`:

```
                    Baseline   Hybrid   Hybrid+Rerank   Full
Faithfulness        0.72       0.78     0.85            0.91
Answer Relevancy    0.78       0.80     0.82            0.88
Context Precision   0.61       0.74     0.79            0.79
Context Recall      0.65       0.82     0.85            0.85
```

### 6.4 CI: eval on PR

- [ ] GitHub Action que corre eval en cada push a main
- [ ] Falla si métricas regresan más de 5% vs baseline guardado

---

## 7. Permisos por usuario

**Duración:** 2 días (16h)
**Objetivo:** cada usuario solo accede a su scope autorizado

### 7.1 Modelo de datos

`src/auth/models.py`:

```python
class User(BaseModel):
    id: str
    email: str
    role: str                    # "admin" | "employee" | "external"
    teams: list[str]             # ["engineering", "sales"]

class Permission(BaseModel):
    user_id: str
    source_paths: list[str]      # Patrones glob: "data/sources/engineering/**"
    source_types: list[str]      # ["code", "markdown"]
    deny: list[str] = []         # Excepciones
```

### 7.2 Tablas SQLite

Migration en `src/db/sqlite.py`:

```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    teams TEXT NOT NULL,         -- JSON array
    created_at TIMESTAMP
);

CREATE TABLE permissions (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    source_paths TEXT NOT NULL,  -- JSON array de patrones
    source_types TEXT NOT NULL,  -- JSON array
    deny TEXT,                   -- JSON array
    created_at TIMESTAMP
);
```

### 7.3 Auth con JWT

`src/auth/middleware.py`:
- [ ] Endpoint `POST /auth/login` → JWT
- [ ] Middleware que valida JWT en cada request a `/research/*`
- [ ] `Depends(get_current_user)` en todos los endpoints protegidos

### 7.4 Filtros de scope en retrieval

`src/auth/scope.py`:

```python
def build_scope_filter(user: User) -> dict:
    """Convierte permisos del usuario en filtros ChromaDB."""
    perms = get_user_permissions(user.id)
    allowed_paths = expand_globs(perms.source_paths)
    return {
        "source_path": {"$in": allowed_paths},
        "source_type": {"$in": perms.source_types},
    }
```

Aplicar este filtro en VectorRetriever Y BM25Retriever (BM25 filtra in-memory).

### 7.5 Seed de usuarios de demo

`scripts/seed_users.py`:

```python
USERS = [
    {"email": "admin@demo.com", "role": "admin", "scope": ["**"]},
    {"email": "engineer@demo.com", "role": "employee", "teams": ["engineering"],
     "scope": ["data/sources/code/**", "data/sources/enterprise/stripe-docs/**"]},
    {"email": "sales@demo.com", "role": "employee", "teams": ["sales"],
     "scope": ["data/sources/enterprise/enron/**"]},
]
```

### 7.6 Tests de permisos

- [ ] User engineer no puede recuperar chunks de scope sales (test crítico)
- [ ] User admin puede ver todo
- [ ] Filtros aplican consistentemente en vector + BM25

---

## 8. Logs de auditoría

**Duración:** 1 día (8h)
**Objetivo:** trazabilidad completa de quién consultó qué

### 8.1 Modelo de evento

`src/audit/models.py`:

```python
class AuditEvent(BaseModel):
    id: str
    timestamp: datetime
    user_id: str
    event_type: str          # "query" | "ingest" | "permission_change" | "auth_login"
    payload: dict            # Específico por tipo
    outcome: str             # "success" | "denied" | "error"
```

### 8.2 Logger estructurado

`src/audit/logger.py`:
- [ ] Persistir en `data/audit.db` (SQLite separado del principal)
- [ ] WAL mode habilitado para no bloquear
- [ ] Hook en cada query: log con `query`, `chunks_retrieved`, `answer_score`
- [ ] Hook en cada ingestion: log con `source_path`, `chunks_created`
- [ ] Hook en denegaciones de permiso: log explícito

### 8.3 Endpoint admin

`GET /admin/audit?user_id=X&from=...&to=...`

Solo accesible por role `admin`. Devuelve eventos paginados.

### 8.4 Tests

- [ ] Toda query genera al menos 1 evento
- [ ] Denegaciones de permiso quedan registradas con `outcome=denied`

---

## 9. UI

**Duración:** 3 días (24h)
**Objetivo:** UI mínima viable que demuestre todas las features visualmente

### 9.1 Decisión: HTMX o Next.js

| HTMX (recomendado) | Next.js |
|---|---|
| Servido desde FastAPI | Proyecto separado |
| 1 día de desarrollo | 3 días |
| Suficiente para demo | Más profesional |

**Recomendación:** HTMX para entrega rápida. Si queda tiempo, Next.js.

### 9.2 Pantallas

**Login** (`/login`):
- Email + password
- Botones "Login as admin/engineer/sales" para demo

**Chat** (`/`):
- Lista de conversaciones a la izquierda (sidebar)
- Chat principal: input + histórico
- Cada respuesta del bot muestra: respuesta + citas numeradas clickables

**Citas modal:**
- Click en `[1]` → modal con el snippet completo + path al archivo + botón "abrir archivo"

**Sources** (`/sources`):
- Tabla de documentos indexados (filtrable por tipo)
- Botón "Re-ingest" para admin

**Audit** (`/admin/audit`, solo admin):
- Tabla de eventos con filtros
- Export CSV

### 9.3 Streaming SSE

- [ ] Reusar el SSE del `routes.py` actual, adaptado para mostrar:
  - Estado: "Buscando..." → "Recuperados N chunks" → "Generando respuesta..." → respuesta final
- [ ] Mostrar visualmente los nodos del grafo activos (similar al PipelineSimulator de MeetingAgentWeb)

---

## 10. README + benchmarks

**Duración:** 1.5 días (12h)
**Objetivo:** que cualquier hiring manager entienda el valor en 30 segundos

### 10.1 README estructura

```markdown
# Sift — On-premise enterprise knowledge agent

[Hero image / GIF demo]

## ¿Qué es?
1 párrafo. Problema → solución → diferenciador.

## Demo
[Video Loom 90s]
[Link a deploy si existe]

## Arquitectura
[Diagrama Mermaid del grafo LangGraph]
[Diagrama de componentes: ingestion → retrieval → agent → API → UI]

## Características diferenciadoras
- ✅ 100% on-premise (Ollama + ChromaDB)
- ✅ Multi-source: PDF, Markdown, código, emails
- ✅ Hybrid search (BM25 + vectorial + RRF + BGE re-ranker)
- ✅ Citaciones precisas con archivo + línea
- ✅ Self-critique loop (LangGraph cycles)
- ✅ Permisos por usuario (multi-tenant)
- ✅ Logs de auditoría completos
- ✅ Eval suite con RAGAS

## Benchmarks
[Tabla RAGAS: Baseline vs Hybrid vs Hybrid+Rerank vs Full]

## Stack
- Python 3.12, FastAPI, LangGraph
- Ollama (qwen2.5:7b, nomic-embed-text)
- ChromaDB + BM25 (rank_bm25)
- BGE-reranker-base (HuggingFace)
- SQLite (metadata + audit)
- HTMX (UI)

## Quickstart (5 min)
```
git clone ...
cp .env.example .env
docker-compose up
python scripts/seed_users.py
python scripts/ingest.py --source data/sources --connector all
# Open http://localhost:8001
```

## Decisiones técnicas
[Link a docs/architecture.md]

## Roadmap
[Próximos pasos]
```

### 10.2 docs/architecture.md

- Diagrama del grafo LangGraph completo
- Justificación de cada decisión: ¿por qué BM25+vectorial? ¿por qué BGE-reranker? ¿por qué Ollama?
- Trade-offs explícitos

### 10.3 docs/benchmarks.md

- Metodología (dataset, métricas, hardware)
- Tabla con resultados
- Análisis: dónde mejora más, dónde aún flojea

### 10.4 Diagramas

- Mermaid para el grafo LangGraph (commit en el repo)
- Excalidraw para arquitectura de componentes (export PNG)

---

## 11. Demo + deploy

**Duración:** 1 día (8h)

### 11.1 Loom demo

Script de 3 minutos:
- 0:00-0:30: problema (qué es enterprise RAG, por qué on-premise importa)
- 0:30-1:30: demo personal (login engineer, pregunta sobre código)
- 1:30-2:30: demo enterprise (login admin, pregunta sobre Enron, mostrar citaciones)
- 2:30-3:00: arquitectura + benchmarks + cierre

### 11.2 Deploy: NO desplegar en cloud

Decisión consciente: este proyecto es **on-premise** por diseño. Desplegarlo en cloud rompe el pitch.

Alternativa para "demo viva":
- Cloudflare Tunnel desde tu PC cuando alguien lo pida (entrevistas, etc.)
- README brutal con screenshots y video Loom
- Repo público con `docker-compose up` que arranca todo

### 11.3 LinkedIn post de lanzamiento

Estructura:
- Hook: problema real (datos sensibles, compliance, coste de OpenAI)
- Solución: Sift, lo que hace
- Stack técnico (orgullo de las decisiones)
- Benchmarks como prueba
- Link al repo
- CTA: feedback bienvenido

### 11.4 Blog post (opcional pero recomendado)

Título: "Building an on-premise enterprise RAG agent with LangGraph: lessons from 4 weeks of work"

Secciones:
1. ¿Por qué on-premise?
2. Arquitectura: por qué LangGraph y no alternativas
3. Hybrid search: la diferencia que hace BM25 + RRF + re-ranker
4. Citaciones precisas: el detalle que más cuesta hacer bien
5. Eval con RAGAS: cómo medir RAG sin mentirse
6. Permisos: el aspecto enterprise que casi nadie implementa en demos
7. Resultados y aprendizajes

---

## 12. Checklist final

Antes de publicar y considerar v2.0 hecho:

### Funcional
- [ ] Indexa PDF, Markdown, código y emails sin errores
- [ ] Hybrid search (BM25 + vectorial + RRF + re-ranker) funciona end-to-end
- [ ] Cada respuesta tiene citaciones validadas
- [ ] Self-critique loop activa rewrite cuando score < 8
- [ ] Permisos: engineer NO puede ver datos de sales (verificado con test)
- [ ] Audit logs guardan todas las queries
- [ ] UI permite hacer todo el flujo en menos de 5 clicks

### Calidad
- [ ] RAGAS faithfulness ≥ 0.85 en dataset propio
- [ ] Tests pasan (>80% coverage en módulos críticos)
- [ ] `docker-compose up` arranca sin errores en máquina nueva
- [ ] No hay TODOs ni `print()` en código

### Documentación
- [ ] README con badges, hero image, quickstart, benchmarks
- [ ] docs/architecture.md con diagramas y trade-offs
- [ ] docs/benchmarks.md con tabla RAGAS
- [ ] docs/deployment.md con todos los pasos
- [ ] docstrings en todas las funciones públicas

### Marketing
- [ ] Loom demo de 3 min en YouTube/Loom público
- [ ] LinkedIn post publicado
- [ ] Repo público en GitHub con tag v2.0.0
- [ ] (Opcional) Blog post en Medium/Dev.to

---

## Resumen ejecutivo

**Total:** 4 semanas part-time (~80h efectivas).

**Distribución por área:**
- Refactor base: 1 día
- Ingestion: 3 días
- Retrieval (hybrid + rerank): 2 días
- Citaciones: 2 días
- Self-critique: 1 día
- Eval suite: 2 días
- Permisos: 2 días
- Auditoría: 1 día
- UI: 3 días
- Docs + benchmarks: 1.5 días
- Demo + lanzamiento: 1 día

**Total:** ~19.5 días × 4h diarias ≈ 80h.

**Por qué este orden:**
1. Refactor primero para no acumular deuda
2. Ingestion antes que retrieval (no se puede buscar sin datos)
3. Retrieval antes que self-critique (el loop necesita buenos resultados base)
4. Citaciones en paralelo con retrieval
5. Eval después de retrieval para medir mejoras
6. Permisos al final del backend (filtros que aplican sobre todo lo anterior)
7. UI al final (cuando el backend está estable)
8. Docs y demo lo último (cuando hay algo real que mostrar)

**Hito intermedio recomendado:** al terminar semana 2 (retrieval + citas + RAGAS),
hacer un commit + tag `v2.0.0-alpha` y validar contigo si la dirección sigue siendo buena
antes de invertir las semanas 3-4 en permisos/UI.
