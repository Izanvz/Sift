# Prompts centralizados de Sift v2
# Todos los prompts del agente viven aquí para facilitar tuning sin tocar lógica.

# ---------------------------------------------------------------------------
# route_query
# ---------------------------------------------------------------------------

ROUTE_QUERY_PROMPT = """Classify the following user query into one of these types:
- "factual": asks for a specific fact, date, name, or piece of information
- "analytical": asks to explain, compare, or reason about something
- "comparative": explicitly compares two or more things
- "ambiguous": too vague to answer without clarification

Query: {query}

Respond with the query type only."""


# ---------------------------------------------------------------------------
# evaluate_relevance
# ---------------------------------------------------------------------------

EVALUATE_RELEVANCE_PROMPT = """You are evaluating whether retrieved document chunks are \
sufficient to answer a user query.

Query: {query}
Number of chunks retrieved: {n_chunks}
Chunk previews:
{chunk_previews}

Rate the overall relevance from 0.0 to 1.0:
- 1.0 = chunks directly and fully answer the query
- 0.5 = chunks partially relevant, answer may be incomplete
- 0.0 = chunks are irrelevant or empty

Return only a float between 0.0 and 1.0."""


# ---------------------------------------------------------------------------
# rewrite_query
# ---------------------------------------------------------------------------

REWRITE_QUERY_PROMPT = """The query "{query}" returned low-relevance results \
(relevance score: {score:.2f}).

Rewrite the query to be more specific and likely to retrieve better results \
from an enterprise document corpus. Keep it concise (under 20 words).

Return only the rewritten query."""


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are an enterprise knowledge assistant. Answer the question \
using EXCLUSIVELY the information from the fragments below.

Strict rules:
1. Each claim must include an inline citation [N] referencing the fragment that supports it
2. If the information is not in the fragments, say "I don't have information on this"
3. Do NOT invent information or cite general knowledge
4. Be concise and direct

Question: {query}

Fragments:
{chunks_with_ids}

Answer (with inline citations [N]):"""


# ---------------------------------------------------------------------------
# self_critique
# ---------------------------------------------------------------------------

CRITIQUE_PROMPT = """You are a strict quality evaluator for an enterprise knowledge assistant.

Question: {query}

Source fragments available (ground truth):
{sources}

Generated answer:
{answer}

Citations used: {n_citations}

Evaluate on a scale 0–10:
- faithfulness (10 = every claim traceable to a source fragment; 0 = hallucinations present)
- completeness (10 = question fully answered; 0 = key aspects missing)
- citation_quality (10 = citations precise and correctly placed; 0 = missing or wrong)

Overall score = weighted average (faithfulness ×0.5, completeness ×0.3, citation_quality ×0.2).

List specific gaps (claims without source, missing aspects, wrong citations).
Provide a concrete recommendation for improvement.

Return structured output with: score, faithfulness, completeness, citation_quality, gaps, recommendation."""


# ---------------------------------------------------------------------------
# rewrite_answer
# ---------------------------------------------------------------------------

REWRITE_ANSWER_PROMPT = """Improve the following answer using the critic feedback below.

Rules:
- Use ONLY the source fragments provided — do NOT add information from memory
- Keep existing citation markers [N] where they are correct; fix or remove wrong ones
- Address each gap explicitly

Question: {query}

Source fragments (only these can be cited):
{sources}

Critic feedback (score {score}/10):
Gaps: {gaps}
Recommendation: {recommendation}

Current answer:
{answer}

Improved answer (with corrected citations [N]):"""


# ---------------------------------------------------------------------------
# clarification_request
# ---------------------------------------------------------------------------

CLARIFICATION_PROMPT = """The query "{query}" is ambiguous. \
Ask the user one clarifying question to understand what they are looking for. \
Be concise (one sentence)."""
