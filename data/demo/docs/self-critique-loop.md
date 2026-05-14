# Self-Critique Loop

After synthesis, Sift runs a self-critique step where the LLM evaluates its own answer before returning it to the user.

## What the critique scores

The critique LLM (same model as synthesis) scores the answer on three dimensions (0–10 each):

| Dimension | What it measures |
|-----------|-----------------|
| `faithfulness` | Are all claims grounded in the retrieved context? No hallucinations. |
| `completeness` | Does the answer address all parts of the query? |
| `citation_quality` | Are `[N]` markers present, accurate, and well-placed? |

The overall score is a weighted average. The exact weights are in `config/prompts.py`.

## Hard gate

Regardless of the overall score, a `faithfulness` score below `6.0` always triggers a rewrite. This is the faithfulness hard gate — it prevents hallucinated answers from passing even if completeness and citation_quality are high.

## What happens on failure

If the answer fails the quality gate (overall score < `8.0` or faithfulness < `6.0`), the graph enters `rewrite_answer`, which instructs the LLM to fix the specific issues flagged by the critique. The rewritten answer goes through synthesis again, then critique again.

The maximum number of rewrite cycles is controlled by `max_rewrite_iterations` (default: 2). After that, the best answer so far is returned.

## Query rewriting

Separate from answer critique, Sift also rewrites the *query* if retrieved context scores low on relevance. The `evaluate_relevance` node checks whether retrieved chunks are relevant to the query. If not, `rewrite_query` rephrases the query and retrieves again.

Maximum query rewrite cycles: `max_search_iterations` (default: 2).

## Human-in-the-loop

Ambiguous queries (classified by the router as `ambiguous`) trigger a LangGraph `interrupt_before` at the `clarification_request` node. The UI displays a modal asking the user for clarification. The user's response is injected back into the graph state before retrieval begins.
