from typing import Annotated, Literal, TypedDict
import operator

from pydantic import BaseModel, Field


class SearchResult(TypedDict):
    source: str        # "web" | "chromadb" | "arxiv"
    url: str
    content: str
    relevance: float


class CritiqueOutput(BaseModel):
    score: float = Field(..., ge=0, le=10)
    strengths: list[str]
    gaps: list[str]
    recommendation: str


class PlanOutput(BaseModel):
    subtopics: list[str] = Field(..., min_length=3, max_length=5)


class HumanFeedback(BaseModel):
    content: str
    action: Literal["approve", "edit", "reject"]


class ResearchState(TypedDict):
    query: str
    research_plan: list[str]
    search_results: Annotated[list[SearchResult], operator.add]
    quality_scores: list[float]
    synthesis: str
    critique: dict
    iterations: int
    rewrite_iterations: int
    human_feedback: str | None
    report: str
    metadata: dict
