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

CRITIQUE_PROMPT = """Evaluate the following answer to a user question.

Question: {query}
Answer: {answer}
Number of citations used: {n_citations}

Score from 0 to 10 on each dimension:
- faithfulness: does the answer rely only on the provided fragments? (hallucinations penalize)
- completeness: does the answer fully address the question?
- citation_quality: are citations precise and sufficient?

Overall score = average of the three dimensions.

Return: overall score (float 0-10), list of gaps (strings), recommendation (string)."""


# ---------------------------------------------------------------------------
# rewrite_answer
# ---------------------------------------------------------------------------

REWRITE_ANSWER_PROMPT = """Improve the following answer based on critic feedback. \
Use the same citation markers [N] — do NOT add new ones.

Current score: {score}/10
Gaps to address:
{gaps}
Recommendation: {recommendation}

Current answer:
{answer}

Improved answer:"""


# ---------------------------------------------------------------------------
# clarification_request
# ---------------------------------------------------------------------------

CLARIFICATION_PROMPT = """The query "{query}" is ambiguous. \
Ask the user one clarifying question to understand what they are looking for. \
Be concise (one sentence)."""
