# Inline Citations in Sift

Sift attaches precise citations to every answer — linked to the exact file, page number, or line range where the information was found.

## How citations work

During synthesis, the LLM is prompted to insert `[N]` markers inline in its answer text, where N corresponds to a numbered chunk in the context window. After generation, Sift parses these markers with a regex and maps each to its source chunk.

Each citation contains:

| Field | Example |
|-------|---------|
| `chunk_id` | `abc123` |
| `source_path` | `vercel-docs/concepts/environment-variables.md` |
| `source_type` | `markdown`, `pdf`, `code`, `email` |
| `page_number` | `4` (PDFs only) |
| `line_start` / `line_end` | `42` / `67` (code only) |
| `section_path` | `Concepts > Environment Variables` (Markdown) |
| `snippet` | First 200 characters of the chunk |
| `score` | Reranker score (0–1) |

## Citation types by source

**Markdown**: cited to section path (e.g., `## Encryption > Key rotation`).

**PDF**: cited to page number extracted by pypdf during ingestion.

**Code**: cited to function name and line range extracted by tree-sitter.

**Email**: cited to sender, date, and subject extracted from mbox headers.

## What happens when markers are missing

If the LLM omits `[N]` markers, Sift still returns an answer but with an empty citations list. The critique loop penalizes low `citation_quality` scores, which can trigger an answer rewrite.

## UI display

The UI renders citations as expandable cards below the answer tab:
- Source path and type badge
- Snippet preview
- Page / line / section locator
- Reranker score

Citations are the primary trust signal for users evaluating answer quality.
