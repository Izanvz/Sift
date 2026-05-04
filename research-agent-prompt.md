# ResearchAgent — Prompt Maestro de Implementación

## Contexto del proyecto

Estoy construyendo un Autonomous Research Agent en Python usando LangGraph.
Es mi proyecto de aprendizaje después de MeetingAgent (pipeline lineal).
El objetivo es dominar LangGraph avanzado: ciclos, fan-out paralelo con Send(),
checkpointing, human-in-the-loop e interrupt_before.

## Stack técnico

- Python 3.11+
- LangGraph 0.2.x (pinear versión exacta)
- LangChain Community + langchain-ollama
- Ollama local con modelo qwen2.5:14b (hardware: 32GB RAM, 6GB VRAM)
- ChromaDB (ya tengo instancia de MeetingAgent, reutilizar con nuevo namespace)
- Tavily para web search (con fallback a duckduckgo-search gratis)
- FastAPI + SSE para streaming de progreso
- SQLite via SqliteSaver para checkpointing
- instructor + pydantic para structured output FIABLE de Ollama (CRÍTICO)
- anyio para async compatibility con SqliteSaver en FastAPI

## Arquitectura del grafo (12 nodos, 3 ciclos, 2 ramas paralelas)

```
START
  ↓
[plan_research]
  ↓ Send() API — fan-out paralelo
  ├─ [search_web]       (Tavily)
  ├─ [search_chromadb]  (RAG local)
  └─ [search_arxiv]     (papers, opcional)
  ↓ fan-in
[gather_results]
  ↓
[evaluate_quality]
  ↓ conditional edge
¿needs_more_research? (quality < 0.7 AND iter < 3)
  ├─ YES → [refine_query] → loop a fan-out
  └─ NO ↓
[synthesize]
  ↓
[self_critique]
  ↓ conditional edge
¿quality_gate? (score < 7)
  ├─ FAIL → [rewrite] → loop a synthesize
  └─ PASS ↓
[human_checkpoint]  (interrupt_before)
  ↓ APPROVE/EDIT/REJECT
[generate_report]
  ↓
END
```

## State Schema completo (con correcciones críticas)

```python
from typing import Annotated, TypedDict
import operator
from pydantic import BaseModel

class SearchResult(TypedDict):
    source: str        # "web" | "chromadb" | "arxiv"
    url: str
    content: str
    relevance: float

class CritiqueOutput(BaseModel):  # Pydantic para instructor
    score: float       # 0-10
    strengths: list[str]
    gaps: list[str]
    recommendation: str

class ResearchState(TypedDict):
    query: str
    research_plan: list[str]
    # CRÍTICO: Annotated con operator.add para que fan-in acumule resultados
    search_results: Annotated[list[SearchResult], operator.add]
    quality_scores: list[float]
    synthesis: str
    critique: dict
    iterations: int           # contador loop de búsqueda
    rewrite_iterations: int   # contador loop de synthesis (separado)
    human_feedback: str | None
    report: str
    metadata: dict
```

## Estructura de archivos

```
ResearchAgent/
├── src/
│   ├── agent/
│   │   ├── graph.py     # Compilación del grafo + checkpointer
│   │   ├── nodes.py     # Implementación de cada nodo
│   │   ├── state.py     # ResearchState + modelos Pydantic
│   │   ├── edges.py     # Funciones conditional_edge
│   │   └── tools.py     # search_web, search_chromadb, search_arxiv
│   ├── api/
│   │   ├── main.py      # FastAPI app + lifespan
│   │   ├── routes.py    # /research endpoints
│   │   └── static/      # SPA frontend (HTML/CSS/JS)
│   └── db/
│       ├── checkpointer.py   # SqliteSaver setup
│       └── vector_store.py   # ChromaDB (reutilizado)
├── tests/
│   ├── test_sprint0.py   # 4 tests de validación de asunciones
│   ├── test_nodes.py
│   └── test_graph.py
├── config/
│   └── settings.py       # Pydantic Settings (env vars)
└── requirements.txt
```

## Requisitos de implementación — IMPORTANTE

### 1. Structured output con instructor (OBLIGATORIO)

NO usar `json.loads(llm.invoke())` directamente.
SIEMPRE usar instructor + Pydantic para nodos que devuelven JSON:

```python
import instructor
from ollama import Client

ollama_client = Client()
instructor_client = instructor.from_ollama(ollama_client)

# Ejemplo en self_critique:
critique = instructor_client.chat.completions.create(
    model="qwen2.5:14b",
    response_model=CritiqueOutput,
    messages=[{"role": "user", "content": critique_prompt}]
)
```

### 2. Send() API para fan-out paralelo

```python
from langgraph.types import Send

def route_to_searches(state: ResearchState) -> list[Send]:
    return [
        Send("search_web", {"query": sub, "source": "web"})
        for sub in state["research_plan"]
    ]

# En el grafo:
graph.add_conditional_edges(
    "plan_research",
    route_to_searches,
    ["search_web", "search_chromadb"]
)
```

### 3. Checkpointing + interrupt_before en FastAPI

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import asyncio

# En lifespan de FastAPI:
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
compiled_graph = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_checkpoint"]
)

# Para reanudar después de human checkpoint:
@router.post("/research/{session_id}/resume")
async def resume(session_id: str, feedback: HumanFeedback):
    config = {"configurable": {"thread_id": session_id}}
    compiled_graph.update_state(config, {"human_feedback": feedback.content})
    result = await asyncio.to_thread(compiled_graph.invoke, None, config)
    return result
```

### 4. SSE streaming de progreso por nodo

```python
from fastapi.responses import StreamingResponse
from uuid import uuid4
import json

@router.post("/research")
async def start_research(request: ResearchRequest):
    async def event_stream():
        config = {"configurable": {"thread_id": str(uuid4())}}
        async for event in compiled_graph.astream_events(
            {"query": request.query}, config, version="v2"
        ):
            if event["event"] == "on_chain_start":
                node_name = event.get("name", "")
                yield f"data: {json.dumps({'node': node_name, 'status': 'running'})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

## Sprint 0 — Tests de validación ANTES de código de producción

Implementar `tests/test_sprint0.py` con estos 4 tests:

### Test 1: Instructor + Ollama JSON reliability

- Llamar 10 veces a qwen2.5:14b via instructor con CritiqueOutput
- Assertion: 10/10 éxitos, todos los campos presentes, score entre 0-10

### Test 2: SqliteSaver con ciclo

- Crear grafo mínimo con ciclo: A → B → ¿continuar? → A (max 3 veces)
- Usar SqliteSaver, pausar en nodo B (interrupt_before), reanudar
- Assertion: el estado se persiste correctamente entre sesiones

### Test 3: interrupt_before en FastAPI

- Endpoint que lanza grafo con `interrupt_before=["checkpoint"]`
- Llamada POST → debe retornar 202 con session_id
- Segunda llamada POST `/resume/{session_id}` → completa el grafo
- Assertion: el grafo no completa hasta el resume

### Test 4: Send() + Annotated reducer

- Grafo: START → fan_out (Send 3 branches) → gather → END
- Cada branch añade 1 item a search_results
- Assertion: gather recibe exactamente 3 items acumulados

## Comportamiento de los nodos principales

### plan_research
- Input: query (str)
- Output: research_plan (list de 3-5 strings, subtemas específicos)
- Modelo: qwen2.5:14b via instructor → `PlanOutput(subtopics: list[str])`

### evaluate_quality
- Input: search_results acumulados
- Output: quality_scores + decision de continuar o no
- Threshold: si avg_score < 0.7 Y iterations < 3 → refine
- Modelo: qwen2.5:14b o lógica determinista (score por keywords)

### synthesize
- Input: search_results (todos) + research_plan
- Output: synthesis (string estructurado con secciones)
- Prompt: forzar estructura markdown: `## Introducción`, `## Hallazgos`, `## Síntesis`

### self_critique
- Input: synthesis
- Output: CritiqueOutput via instructor (score 0-10 + gaps + strengths)
- Threshold quality_gate: score < 7 → rewrite

### generate_report
- Input: synthesis + human_feedback (si existe) + metadata
- Output: informe final en Markdown con: título, resumen ejecutivo, hallazgos por sección, síntesis, fuentes citadas
- También guardar en ChromaDB namespace `"research_reports"`

## API endpoints requeridos

- `POST /research` — lanza nueva sesión, retorna session_id + SSE stream
- `GET /research/{session_id}` — estado actual del grafo
- `POST /research/{session_id}/resume` — continuar tras human checkpoint
- `GET /research` — lista de sesiones anteriores (de SqliteSaver)

## Frontend SPA (similar a MeetingAgent)

- Input textarea para la query
- Barra de progreso con nodos activos (SSE)
- Panel lateral: research_plan en tiempo real
- Vista del informe final renderizado en Markdown
- Botón de aprobación/edición en el human checkpoint

## Lo que NO hacer

- NO usar `json.loads()` directamente sobre outputs de LLM
- NO olvidar `Annotated[list, operator.add]` en search_results del State
- NO compilar el grafo sin checkpointer si se usa interrupt_before
- NO hardcodear API keys (usar Pydantic Settings + .env)
- NO hacer `await` directamente en SqliteSaver (usar `asyncio.to_thread`)
- NO usar un solo modelo para todos los nodos (usar 7b para routing, 14b para síntesis)

## Orden de implementación

1. **Sprint 0:** `test_sprint0.py` — los 4 tests deben pasar antes de continuar
2. `state.py` + `graph.py` skeleton (stubs)
3. `nodes.py` (uno a uno, empezando por plan_research)
4. `edges.py` (conditional edges)
5. `tools.py` (search_chromadb primero, luego Tavily)
6. Integración del grafo completo (terminal, sin API)
7. `checkpointer.py` + interrupt
8. `api/main.py` + `routes.py` + SSE
9. Frontend SPA
