import logging
from datetime import datetime

import instructor
from langchain_ollama import OllamaLLM
from openai import OpenAI

from config.settings import settings
from src.agent.state import CritiqueOutput, PlanOutput, ResearchState, SearchResult
from src.agent.tools import search_arxiv_tool, search_chromadb_tool, search_web_tool
from src.db.vector_store import save_report

logger = logging.getLogger(__name__)

# Instructor via OpenAI-compatible endpoint de Ollama
_instructor_client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)
_llm_synthesis = OllamaLLM(model=settings.model_synthesis)
_llm_routing = OllamaLLM(model=settings.model_routing)


# ---------------------------------------------------------------------------
# Nodo 1: plan_research
# ---------------------------------------------------------------------------

def plan_research(state: ResearchState) -> dict:
    query = state["query"]
    prompt = (
        f"You are a research planner. Break the following research query into 3-5 "
        f"specific, non-overlapping subtopics that together cover the topic comprehensively.\n\n"
        f"Query: {query}\n\n"
        f"Return a list of subtopics as strings."
    )
    plan: PlanOutput = _instructor_client.chat.completions.create(
        model=settings.model_planning,
        response_model=PlanOutput,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "research_plan": plan.subtopics,
        "iterations": 0,
        "rewrite_iterations": 0,
        "metadata": {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }


# ---------------------------------------------------------------------------
# Nodo 2: search_web
# ---------------------------------------------------------------------------

def search_web(state: dict) -> dict:
    query = state.get("query", "")
    results = search_web_tool(query)
    return {"search_results": results}


# ---------------------------------------------------------------------------
# Nodo 3: search_chromadb
# ---------------------------------------------------------------------------

def search_chromadb(state: dict) -> dict:
    query = state.get("query", "")
    results = search_chromadb_tool(query)
    return {"search_results": results}


# ---------------------------------------------------------------------------
# Nodo 4: search_arxiv (opcional)
# ---------------------------------------------------------------------------

def search_arxiv(state: dict) -> dict:
    query = state.get("query", "")
    try:
        results = search_arxiv_tool(query)
    except Exception as exc:
        logger.warning("ArXiv search failed: %s", exc)
        results = []
    return {"search_results": results}


# ---------------------------------------------------------------------------
# Nodo 5: gather_results
# ---------------------------------------------------------------------------

def gather_results(state: ResearchState) -> dict:
    raw = state.get("search_results", [])
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for r in raw:
        key = f"{r.get('url', '')}::{r.get('content', '')[:100]}"
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    meta = dict(state.get("metadata", {}))
    meta["result_count"] = len(deduped)
    meta["sources"] = list({r.get("source", "") for r in deduped})
    return {"search_results": deduped, "metadata": meta}


# ---------------------------------------------------------------------------
# Nodo 6: evaluate_quality
# ---------------------------------------------------------------------------

def evaluate_quality(state: ResearchState) -> dict:
    results = state.get("search_results", [])
    if not results:
        score = 0.0
    else:
        avg_relevance = sum(r.get("relevance", 0.0) for r in results) / len(results)
        sources = {r.get("source", "") for r in results}
        diversity_bonus = min(0.15 * (len(sources) - 1), 0.3)
        score = min(1.0, avg_relevance + diversity_bonus)

    existing = list(state.get("quality_scores", []))
    existing.append(score)
    iterations = state.get("iterations", 0) + 1
    return {"quality_scores": existing, "iterations": iterations}


# ---------------------------------------------------------------------------
# Nodo 7: refine_query
# ---------------------------------------------------------------------------

def refine_query(state: ResearchState) -> dict:
    query = state["query"]
    plan = state.get("research_plan", [])
    scores = state.get("quality_scores", [])
    avg_score = sum(scores) / len(scores) if scores else 0.0

    prompt = (
        f"The initial research on '{query}' returned low-quality results "
        f"(average relevance: {avg_score:.2f}).\n\n"
        f"Current subtopics:\n" + "\n".join(f"- {s}" for s in plan) + "\n\n"
        f"Generate 3-5 improved, more specific subtopics that are likely to yield "
        f"better search results. Be more precise and use different angles."
    )
    refined: PlanOutput = _instructor_client.chat.completions.create(
        model=settings.model_planning,
        response_model=PlanOutput,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"research_plan": refined.subtopics}


# ---------------------------------------------------------------------------
# Nodo 8: synthesize
# ---------------------------------------------------------------------------

def synthesize(state: ResearchState) -> dict:
    results = state.get("search_results", [])
    plan = state.get("research_plan", [])
    query = state["query"]

    sources_text = "\n\n".join(
        f"[{r.get('source','?')}] {r.get('url','')}\n{r.get('content','')[:500]}"
        for r in results[:20]
    )
    plan_text = "\n".join(f"- {s}" for s in plan)

    prompt = (
        f"You are a research synthesizer. Based on the following research materials, "
        f"write a comprehensive synthesis for the query: '{query}'\n\n"
        f"Research plan subtopics:\n{plan_text}\n\n"
        f"Sources:\n{sources_text}\n\n"
        f"Write the synthesis using EXACTLY this structure:\n\n"
        f"## Introducción\n[2-3 sentences framing the topic]\n\n"
        f"## Hallazgos\n[Key findings organized by subtopic]\n\n"
        f"## Síntesis\n[Integrated conclusions and insights]"
    )
    synthesis = _llm_synthesis.invoke(prompt)
    return {"synthesis": synthesis}


# ---------------------------------------------------------------------------
# Nodo 9: self_critique
# ---------------------------------------------------------------------------

def self_critique(state: ResearchState) -> dict:
    synthesis = state.get("synthesis", "")
    query = state["query"]

    prompt = (
        f"You are a critical reviewer. Evaluate the following research synthesis "
        f"for the query: '{query}'\n\n"
        f"Synthesis:\n{synthesis}\n\n"
        f"Provide a detailed critique with a score from 0 to 10."
    )
    critique: CritiqueOutput = _instructor_client.chat.completions.create(
        model=settings.model_synthesis,
        response_model=CritiqueOutput,
        messages=[{"role": "user", "content": prompt}],
    )
    rewrite_iterations = state.get("rewrite_iterations", 0) + 1
    return {"critique": critique.model_dump(), "rewrite_iterations": rewrite_iterations}


# ---------------------------------------------------------------------------
# Nodo 10: rewrite
# ---------------------------------------------------------------------------

def rewrite(state: ResearchState) -> dict:
    synthesis = state.get("synthesis", "")
    critique = state.get("critique", {})
    gaps = critique.get("gaps", [])
    recommendation = critique.get("recommendation", "")
    score = critique.get("score", 0)

    gaps_text = "\n".join(f"- {g}" for g in gaps)
    prompt = (
        f"You are a research writer. Improve the following synthesis based on critic feedback.\n\n"
        f"Current score: {score}/10\n"
        f"Gaps to address:\n{gaps_text}\n"
        f"Recommendation: {recommendation}\n\n"
        f"Current synthesis:\n{synthesis}\n\n"
        f"Write an improved synthesis using the same structure:\n"
        f"## Introducción\n## Hallazgos\n## Síntesis"
    )
    improved = _llm_synthesis.invoke(prompt)
    return {"synthesis": improved}


# ---------------------------------------------------------------------------
# Nodo 11: human_checkpoint_node
# ---------------------------------------------------------------------------

def human_checkpoint_node(state: ResearchState) -> dict:
    # LangGraph pausa ANTES de este nodo via interrupt_before.
    # Si se llega aquí es porque se reanudó tras el human review.
    # LangGraph 0.2.x requiere al menos un campo en el return.
    return {"human_feedback": state.get("human_feedback")}


# ---------------------------------------------------------------------------
# Nodo 12: generate_report
# ---------------------------------------------------------------------------

def generate_report(state: ResearchState) -> dict:
    synthesis = state.get("synthesis", "")
    human_feedback = state.get("human_feedback")
    meta = state.get("metadata", {})
    query = state["query"]
    plan = state.get("research_plan", [])
    results = state.get("search_results", [])

    feedback_section = ""
    if human_feedback:
        feedback_section = f"\n\n## Revisión Humana\n{human_feedback}"

    sources_section = "\n".join(
        f"- [{r.get('source','?')}] {r.get('url','')}"
        for r in results
        if r.get("url")
    )

    timestamp = meta.get("timestamp", datetime.utcnow().isoformat())
    plan_text = "\n".join(f"- {s}" for s in plan)

    report = (
        f"# Informe de Investigación: {query}\n\n"
        f"**Fecha:** {timestamp}  \n"
        f"**Fuentes consultadas:** {len(results)}  \n"
        f"**Subtemas investigados:** {len(plan)}\n\n"
        f"## Resumen Ejecutivo\n"
        f"Este informe presenta los hallazgos de una investigación automatizada sobre: *{query}*\n\n"
        f"**Plan de investigación:**\n{plan_text}\n\n"
        f"{synthesis}"
        f"{feedback_section}\n\n"
        f"## Fuentes Citadas\n{sources_section}\n"
    )

    report_meta = {**meta, "report_saved": False}
    saved = save_report(report, {"query": query, "timestamp": timestamp})
    report_meta["report_saved"] = saved

    return {"report": report, "metadata": report_meta}
