import logging
import uuid
from datetime import datetime

import instructor
from langchain_ollama import OllamaLLM
from openai import OpenAI

from config.prompts import (
    CLARIFICATION_PROMPT,
    CRITIQUE_PROMPT,
    EVALUATE_RELEVANCE_PROMPT,
    REWRITE_ANSWER_PROMPT,
    REWRITE_QUERY_PROMPT,
    ROUTE_QUERY_PROMPT,
    SYNTHESIS_PROMPT,
)
from config.settings import settings
from src.agent.state import (
    Chunk,
    CritiqueOutput,
    QueryClassification,
    RelevanceEval,
    SiftState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clientes LLM (singleton por proceso)
# ---------------------------------------------------------------------------

_instructor_client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)
_llm = OllamaLLM(model=settings.model_synthesis)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_chunks_for_prompt(chunks: list[Chunk]) -> str:
    """Formatea los chunks con IDs para incluir en prompts de síntesis."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk.get("source_type", "?")
        path = chunk.get("source_path", "")
        meta = chunk.get("metadata", {})

        # Localización específica por tipo
        loc = ""
        if src == "pdf" and meta.get("page_number"):
            loc = f", página {meta['page_number']}"
        elif src == "code" and meta.get("line_start"):
            loc = f", líneas {meta['line_start']}-{meta.get('line_end', '?')}"
        elif src == "markdown" and meta.get("section_path"):
            loc = f", sección '{meta['section_path']}'"

        lines.append(f"[{i}] ({src}{loc}) {path!r}\n{chunk['content'][:400]}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Nodo 1: route_query — clasifica la query antes de buscar
# ---------------------------------------------------------------------------

def route_query(state: SiftState) -> dict:
    query = state["query"]
    classification: QueryClassification = _instructor_client.chat.completions.create(
        model=settings.model_routing,
        response_model=QueryClassification,
        messages=[{"role": "user", "content": ROUTE_QUERY_PROMPT.format(query=query)}],
    )
    return {
        "query_type": classification.query_type,
        "iterations": 0,
        "rewrite_iterations": 0,
        "chunks": [],
        "relevance_scores": [],
        "citations": [],
        "metadata": {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# Nodo 2: retrieve — HybridRetriever (BM25 + vectorial + RRF + reranker)
# ---------------------------------------------------------------------------

def retrieve(state: SiftState) -> dict:
    """Recupera chunks relevantes vía HybridRetriever.

    Pipeline: BM25 (top_n) || vectorial (top_n) → RRF → cross-encoder rerank
    → top_k. El user_id se traduce a filtro vectorial (Fase 7).
    """
    from src.auth.models import TokenData
    from src.auth.scope import build_scope_filter
    from src.retrieval.hybrid import get_hybrid_retriever  # import local

    query = state["query"]
    user_id = state.get("user_id")
    scopes = state.get("scopes") or []
    is_admin = bool(state.get("is_admin", False))

    # Filtros: scope (corpora permitidos) y user_id (docs privados del usuario).
    # Admin / scopes=["*"] → no filter de scope.
    user_token = TokenData(
        sub=user_id or "anon",
        username=user_id or "anon",
        scopes=scopes,
        is_admin=is_admin,
    ) if user_id else None
    where = build_scope_filter(user_token)

    try:
        results = get_hybrid_retriever().retrieve(
            query,
            top_k=settings.synthesis_top_k,
            candidates=settings.retrieval_top_k,
            where=where,
        )
    except Exception as exc:
        logger.warning("Hybrid retrieval failed: %s", exc)
        results = []

    chunks: list[Chunk] = []
    for r in results:
        meta = r.get("metadata", {}) or {}
        chunks.append(Chunk(
            chunk_id=r.get("id", str(uuid.uuid4())),
            document_id=r.get("document_id") or meta.get("document_id", ""),
            source_path=r.get("source_path") or meta.get("source_path", ""),
            source_type=r.get("source_type") or meta.get("source_type", "unknown"),
            content=r.get("content", ""),
            relevance_score=float(r.get("relevance_score", 0.0)),
            metadata=meta,
        ))

    return {"chunks": chunks}


# ---------------------------------------------------------------------------
# Nodo 3: gather — deduplica y ordena chunks
# ---------------------------------------------------------------------------

def gather(state: SiftState) -> dict:
    chunks = state.get("chunks", [])
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for chunk in chunks:
        if chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            deduped.append(chunk)

    # Ordenar por relevancia descendente
    deduped.sort(key=lambda c: c["relevance_score"], reverse=True)

    meta = dict(state.get("metadata", {}))
    meta["chunks_retrieved"] = len(deduped)
    return {"chunks": deduped, "metadata": meta}


# ---------------------------------------------------------------------------
# Nodo 4: evaluate_relevance — decide si los chunks son suficientes
# ---------------------------------------------------------------------------

def evaluate_relevance(state: SiftState) -> dict:
    chunks = state.get("chunks", [])
    query = state["query"]

    if not chunks:
        score = 0.0
    else:
        previews = "\n".join(
            f"[{i+1}] {c['content'][:150]}..." for i, c in enumerate(chunks[:5])
        )
        prompt = EVALUATE_RELEVANCE_PROMPT.format(
            query=query,
            n_chunks=len(chunks),
            chunk_previews=previews,
        )
        try:
            eval_result: RelevanceEval = _instructor_client.chat.completions.create(
                model=settings.model_routing,
                response_model=RelevanceEval,
                messages=[{"role": "user", "content": prompt}],
            )
            score = eval_result.score
        except Exception as exc:
            logger.warning("Relevance eval failed: %s", exc)
            score = 0.5  # asumir relevancia media si falla

    scores = list(state.get("relevance_scores", []))
    scores.append(score)
    iterations = state.get("iterations", 0) + 1
    return {"relevance_scores": scores, "iterations": iterations}


# ---------------------------------------------------------------------------
# Nodo 5: rewrite_query — mejora la query si los resultados son pobres
# ---------------------------------------------------------------------------

def rewrite_query(state: SiftState) -> dict:
    scores = state.get("relevance_scores", [])
    avg_score = sum(scores) / len(scores) if scores else 0.0
    prompt = REWRITE_QUERY_PROMPT.format(
        query=state["query"],
        score=avg_score,
    )
    rewritten = _llm.invoke(prompt).strip()
    logger.info("Query rewritten: %r → %r", state["query"], rewritten)
    return {"query": rewritten}


# ---------------------------------------------------------------------------
# Nodo 6: synthesize — genera respuesta con citas inline [N]
# ---------------------------------------------------------------------------

def synthesize(state: SiftState) -> dict:
    from src.agent.citations import build_citations  # import local

    chunks = state.get("chunks", [])[:settings.synthesis_top_k]
    query = state["query"]

    chunks_text = _format_chunks_for_prompt(chunks)
    prompt = SYNTHESIS_PROMPT.format(query=query, chunks_with_ids=chunks_text)
    answer = _llm.invoke(prompt)

    citations = build_citations(answer, chunks)
    return {"answer": answer, "citations": citations}


# ---------------------------------------------------------------------------
# Nodo 7: self_critique — evalúa faithfulness, completeness, citation_quality
# ---------------------------------------------------------------------------

def self_critique(state: SiftState) -> dict:
    chunks = state.get("chunks", [])[:settings.synthesis_top_k]
    sources = _format_chunks_for_prompt(chunks) or "(no sources retrieved)"

    prompt = CRITIQUE_PROMPT.format(
        query=state["query"],
        sources=sources,
        answer=state.get("answer", ""),
        n_citations=len(state.get("citations", [])),
    )
    try:
        critique: CritiqueOutput = _instructor_client.chat.completions.create(
            model=settings.model_synthesis,
            response_model=CritiqueOutput,
            messages=[{"role": "user", "content": prompt}],
        )
        critique_dict = critique.model_dump()
    except Exception as exc:
        logger.warning("self_critique LLM failed: %s — using neutral score", exc)
        critique_dict = {
            "score": 5.0,
            "faithfulness": 5.0,
            "completeness": 5.0,
            "citation_quality": 5.0,
            "gaps": [],
            "recommendation": "Could not evaluate automatically.",
        }

    rewrite_iterations = state.get("rewrite_iterations", 0) + 1
    return {
        "critique": critique_dict,
        "rewrite_iterations": rewrite_iterations,
    }


# ---------------------------------------------------------------------------
# Nodo 8: rewrite_answer — mejora el answer según gaps del critique
# ---------------------------------------------------------------------------

def rewrite_answer(state: SiftState) -> dict:
    from src.agent.citations import build_citations  # import local

    chunks = state.get("chunks", [])[:settings.synthesis_top_k]
    sources = _format_chunks_for_prompt(chunks) or "(no sources)"
    critique = state.get("critique", {})
    gaps_text = "\n".join(f"- {g}" for g in critique.get("gaps", [])) or "None identified."

    prompt = REWRITE_ANSWER_PROMPT.format(
        query=state["query"],
        sources=sources,
        score=critique.get("score", 0),
        gaps=gaps_text,
        recommendation=critique.get("recommendation", ""),
        answer=state.get("answer", ""),
    )
    improved = _llm.invoke(prompt)

    # Reconstruir citations desde el answer reescrito
    citations = build_citations(improved, chunks)
    return {"answer": improved, "citations": citations}


# ---------------------------------------------------------------------------
# Nodo 9: clarification_request — solo si query_type == "ambiguous"
# ---------------------------------------------------------------------------

def clarification_request(state: SiftState) -> dict:
    """LangGraph pausa ANTES de este nodo vía interrupt_before.
    Al reanudarse, la clarificación ya está en state["clarification"]."""
    return {"clarification": state.get("clarification")}


# ---------------------------------------------------------------------------
# Nodo 10: format_response — formatea la respuesta final
# ---------------------------------------------------------------------------

def format_response(state: SiftState) -> dict:
    meta = dict(state.get("metadata", {}))
    meta.update({
        "chunks_used": len(state.get("chunks", [])),
        "citations_count": len(state.get("citations", [])),
        "rewrite_iterations": state.get("rewrite_iterations", 0),
        "final_score": state.get("critique", {}).get("score"),
    })
    return {"metadata": meta}
