# LinkedIn — Post lanzamiento Sift v2

> Regla: link en primer comentario, no en el cuerpo.
> Hook debe funcionar solo antes del "ver más".

---

## Post principal

---

Construí un agente RAG on-premise que no envía ni un token a ninguna API externa.

Se llama Sift. Aquí está lo que hay dentro:

**Búsqueda híbrida real**
BM25 + búsqueda vectorial corren en paralelo. RRF fusiona los rankings. Un cross-encoder BGE reordena los top-20 finales. No es "RAG naive" — es el stack que usan los papers cuando les importa el recall.

**Citas inline ancladas a chunks**
La respuesta sale con marcadores [1][2][3]. Un parser de regex los mapea al fichero exacto, página o rango de líneas, y un snippet del texto recuperado. Si el chunk no existe, la cita no se emite.

**Loop de autocrítica antes de responder**
Un nodo LangGraph evalúa faithfulness, completeness y citation_quality. Puerta dura en faithfulness < 6.0: si el modelo alucina, se reescribe la respuesta y vuelve a evaluarse. El ciclo no es un if — es un edge condicional que puede iterar indefinidamente.

**Control de acceso por usuario**
Cada usuario tiene una lista de corpora permitidos. El filtro se aplica como cláusula where en ChromaDB, no en postprocesado — así el recall no se degrada en colecciones grandes.

**Human-in-the-loop nativo**
Queries ambiguas disparan un interrupt_before de LangGraph antes de retrieve. El agente para, pide clarificación, y el estado persiste en SQLite — si el servidor cae en medio, el thread se reanuda desde el mismo punto.

**Trazabilidad completa**
Cada query queda en una tabla append-only: user, latencia, chunks recuperados, score de critique, IP. Un fallo en el audit nunca rompe la petición.

**145 tests, cero dependencias externas en CI**
HybridRetriever, reranker, LLM client, stores — todo inyectado. Los tests no tocan Ollama ni ChromaDB.

---

Stack: LangGraph · FastAPI · ChromaDB · Ollama (qwen2.5:7b local) · rank_bm25 · BAAI/bge-reranker-base · RAGAS · SQLite · Docker Compose

Demo en Loom + código en GitHub 👇 (primer comentario)

---

## Primer comentario

```
🎬 Demo Loom (5 min): [URL]
💻 GitHub: https://github.com/Izanvz/Sift
📄 Docs arquitectura: https://github.com/Izanvz/Sift/blob/main/docs/architecture.md
```

---

## Variante más corta (si el largo no engancha)

---

Construí un agente RAG que corre 100% en tu hardware y no envía datos a ninguna API.

Lo que lo diferencia de un tutorial RAG:

→ BM25 + vector en paralelo → RRF → reranker cross-encoder
→ Self-critique loop con puerta dura en faithfulness antes de responder
→ Control de acceso por usuario aplicado dentro de ChromaDB (no postprocesado)
→ Human-in-the-loop nativo: queries ambiguas pausan el grafo y piden clarificación
→ Audit log append-only en SQLite — compliance sin Elasticsearch

145 tests. Nada toca Ollama en CI.

Stack: LangGraph · FastAPI · ChromaDB · Ollama · RAGAS

Demo y código en comentarios 👇

---

## Hashtags sugeridos

#LangGraph #RAG #Python #FastAPI #AIEngineering #MachineLearning #OpenSource
