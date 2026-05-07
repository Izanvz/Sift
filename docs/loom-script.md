# Sift v2 — Demo Script (Loom, ~5 min)

> Screen: terminal + browser side by side. Ollama pre-warm, corpus indexed.
> Pace: slow and deliberate. Let the UI settle before speaking.

---

## 0. Intro (0:00 – 0:20)

**[pantalla: repo GitHub Sift]**

> "Este es Sift — un agente RAG on-premise que construí para demostrar
> búsqueda híbrida, control de acceso por usuario y trazabilidad completa
> sobre documentos empresariales propios. Nada sale de tu infraestructura.
> Voy a mostrarte cómo funciona en cinco minutos."

---

## 1. Arranque (0:20 – 0:50)

**[terminal]**

```bash
docker compose up -d
uvicorn src.api.main:app --port 8001
```

> "La infraestructura es Ollama local con qwen2.5:7b, ChromaDB para vectores
> y SQLite para usuarios, sesiones y audit. Un solo docker compose up."

**[browser → localhost:8001]**
> "La UI aparece con un overlay de login. Sin token JWT válido no hay acceso."

---

## 2. Login y escopos (0:50 – 1:30)

**[login con user `alice`, password visible en pantalla]**

> "Alice tiene acceso solo a vercel-docs y stripe-go. Eso viene codificado
> en su JWT. Voy a loguearme."

**[login exitoso, overlay cierra]**

> "El token se guarda en sessionStorage. Cada llamada a la API lleva
> Authorization Bearer — no hay cookies de sesión."

**[abrir devtools → Network → petición /research → header Authorization visible]**

> "Aquí se ve el header. El scope filter se aplica dentro de ChromaDB como
> cláusula where, no en postprocesado — así el recall no se degrada."

---

## 3. Query factual — Vercel env vars (1:30 – 2:30)

**[escribir en el input:]**
```
How does Vercel scope environment variables across deployments?
```

**[click Buscar, DAG empieza a animarse]**

> "El grafo LangGraph tiene ciclos reales: route_query clasifica la query
> como factual, retrieve lanza BM25 y búsqueda vectorial en paralelo,
> RRF fusiona los rankings, y el reranker BGE ordena los top-5 finales."

**[DAG muestra nodos encendiéndose: route → retrieve → gather → synthesize]**

> "Synthesize genera la respuesta con marcadores [1][2] que son indices
> a los chunks recuperados. build_citations los mapea a rutas reales."

**[respuesta aparece en tab Respuesta]**

> "La respuesta cita tres variables de entorno, cada una con su fichero
> fuente. Nada inventado — si el chunk no existe, la cita no se emite."

**[click tab Citas]**

> "Cada cita: ruta del fichero, sección o rango de líneas, y un snippet
> de 200 caracteres del fragmento exacto recuperado."

---

## 4. Self-critique loop (2:30 – 3:10)

**[click tab Critique]**

> "El agente se evalúa solo antes de responder. Tres métricas: faithfulness,
> completeness, citation_quality — cada una de 0 a 10."

**[progress bars visibles]**

> "La puerta dura está en faithfulness < 6.0: si el modelo alucina,
> rewrite_answer se dispara y vuelve a critique. Aquí pasó a la primera —
> faithfulness 8.2, score total 8.7."

> "El ciclo es el nodo clave. No es un if — es un edge condicional en
> LangGraph que puede regresar al nodo anterior indefinidamente hasta
> que el score pasa el umbral o se agota el límite de reintentos."

---

## 5. Query comparativa — Ambigua (3:10 – 3:40)

**[nueva query:]**
```
Compare caching strategies
```

**[DAG muestra route → clarification_request, UI abre modal]**

> "Esta query es ambigua — ¿caching de qué? El router lo detecta y
> LangGraph dispara un interrupt_before antes de retrieve. El agente
> para y pide clarificación."

**[escribir en modal:]**
```
Vercel runtime cache vs CDN edge cache
```

**[resumir, retrieve continúa]**

> "El estado se persiste en SQLite via el checkpointer de LangGraph.
> Si el servidor cae aquí, el thread se puede reanudar exactamente
> desde este punto."

---

## 6. Admin: audit log (3:40 – 4:30)

**[abrir nueva pestaña, login con `admin`]**

> "Ahora con el admin. Misma UI, pero con acceso a todos los corpora
> y a los endpoints de administración."

**[abrir DevTools o curl directo en terminal:]**

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
     http://localhost:8001/audit/events | python -m json.tool | head -40
```

> "Cada query queda registrada: user_id, query, latencia en ms, número
> de chunks recuperados, score de critique, scopes activos, IP.
> La tabla es append-only. Un fallo en el audit nunca rompe la petición."

**[mostrar también /audit/stats]**

> "Y stats agregadas por usuario — queries totales, latencia media,
> errores. Útil para compliance sin montar Elasticsearch."

---

## 7. Eval (4:30 – 4:50)

**[terminal:]**

```bash
python scripts/eval.py --dataset data/eval/golden_qa.jsonl --mock
```

> "El pipeline de evaluación tiene 15 pares Q&A hand-crafted — factual,
> analíticas, comparativas, ambiguas. RAGAS con Ollama como juez.
> El flag --mock corre sin Ollama para CI. El real genera un report
> Markdown con los scores y las ablaciones planificadas."

---

## 8. Cierre (4:50 – 5:00)

**[volver al repo GitHub]**

> "145 tests, todo inyectado — ningún test toca Ollama ni ChromaDB.
> Stack completo: LangGraph, FastAPI, ChromaDB, Ollama, SQLite, RAGAS.
> Link en la descripción."

---

## Notas de grabación

- Resolución: 1920×1080, zoom browser al 110%
- Terminal: fuente 16px, fondo oscuro
- Ollama caliente antes de grabar (primera query tarda ~4s más)
- Hacer una pasada en seco antes de grabar — el DAG animado necesita
  ~800ms para pintar todos los nodos; no avanzar la pantalla antes
- No mostrar contraseñas reales en la grabación final — usar `hunter2`
  o similar que sea obviamente demo
