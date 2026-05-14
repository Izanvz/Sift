# Security

This document describes Sift's security model, known limits, and deployment recommendations.

Sift is a **portfolio-grade / MVP project**. It is designed to demonstrate security-aware engineering, not to replace a certified enterprise product. Read the known limits section before deploying in a sensitive environment.

---

## JWT secret management

Sift uses HS256 JWT tokens. The signing key is `JWT_SECRET` in `.env`.

**Default:** `dev-secret-change-in-production` — intentionally weak to signal it must be replaced.

**In any non-demo deployment:**

```bash
# Generate a cryptographically random secret
openssl rand -hex 32
```

Set the result as `JWT_SECRET` in your environment before starting the API. The application logs a warning at startup if the default value is detected.

Token expiry defaults to 8 hours (`JWT_EXPIRE_MINUTES=480`). Reduce for higher-security contexts.

---

## Token rotation

There is no built-in token revocation list. Tokens are valid until expiry. To invalidate a token immediately:

1. Rotate `JWT_SECRET` — this invalidates all existing tokens.
2. Restart the API.

For long-lived deployments, use short token expiry (`JWT_EXPIRE_MINUTES=60`) and rely on expiry-based rotation.

---

## Per-user corpus scopes

Each user has a `scopes` list of allowed corpus names (e.g., `["vercel-docs", "stripe-go"]`). Admins have `scopes=["*"]` and bypass all filters.

Scope enforcement happens at retrieval time:

- **Vector search:** ChromaDB `where={"corpus": {"$in": user.scopes}}` applied before query execution.
- **BM25 search:** Results post-filtered by `metadata.corpus` before RRF fusion.
- **No post-hoc filtering:** Unauthorized documents never enter the context window.

Users with empty scopes (`scopes=[]`) receive a filter that matches no corpus (`__no_access__`), returning empty results on every query.

This is verified by 16 automated tests in `tests/unit/test_permissions.py`.

---

## Sensitive data handling

### Passwords

Passwords are hashed with bcrypt before storage. Plaintext passwords are never persisted.

Note: `passlib` has a known incompatibility with Python 3.14+ and `bcrypt>=5`. Sift works around this by calling `bcrypt` directly. See `src/auth/store.py`.

### Audit log

The audit log records user activity in an append-only SQLite table. Logged fields include:

- `user_id`, `session_id`, `ip_address`
- `status`, `latency_ms`, `critique_score`
- `query` — truncated and redacted before storage (see below)
- `scopes` — stored as JSON array

**Query redaction:** Before persisting, queries are scanned for patterns that suggest sensitive data:

- Email addresses
- Phone numbers
- Bearer tokens and API keys
- High-entropy strings (potential secrets)

Matched patterns are replaced with `[REDACTED]`. The redacted query is what gets stored.

### JWT tokens

JWTs are transmitted in `Authorization: Bearer` headers. They are not logged. The audit middleware extracts only `user_id` and `scopes` from the decoded payload.

---

## Network exposure

Sift is designed to run inside a private network or on a local machine. By default:

- The API binds to `0.0.0.0:8001` — reachable on all interfaces.
- For local-only operation, set `API_HOST=127.0.0.1`.
- ChromaDB binds to `localhost:8000` inside Docker Compose.
- Ollama binds to `localhost:11434`.

**Zero outbound API calls** when Langfuse is disabled (default). If Langfuse is enabled, traces go to the configured `LANGFUSE_HOST`.

---

## Dependencies: local vs external

| Component | Runs locally | External dependency |
|-----------|:-----------:|:-------------------:|
| LLM (qwen2.5:7b) | ✓ Ollama | — |
| Embeddings (nomic-embed-text) | ✓ Ollama | — |
| Vector store | ✓ ChromaDB | — |
| BM25 index | ✓ in-memory | — |
| Reranker (bge-reranker-base) | ✓ sentence-transformers | downloads on first use |
| UI assets (marked.js) | ✓ vendored | — |
| Observability | optional | Langfuse Cloud (if enabled) |

The reranker model (`BAAI/bge-reranker-base`, ~270 MB) is downloaded from Hugging Face on first use and cached locally. After that, no external calls are made.

---

## Known limits

- **No token revocation.** Tokens live until expiry. Rotate `JWT_SECRET` to invalidate all tokens.
- **No rate limiting.** The API does not limit request frequency. Add a reverse proxy (nginx, Caddy) with rate limiting before exposing to untrusted clients.
- **BM25 index is global.** The BM25 index is built from all indexed documents at startup. Corpus filtering is applied post-query. A user with scopes cannot extract unauthorized content through query phrasing, but the BM25 model itself is aware of all documents.
- **Single-tenant SQLite.** The audit and user databases are SQLite files. Not suitable for multi-node or high-concurrency deployments without migration to PostgreSQL.
- **No MFA.** Authentication is username + password only.
- **Self-signed / no TLS.** The API does not terminate TLS. Put it behind a TLS-terminating proxy for any non-localhost deployment.

---

## Reporting issues

This is a portfolio project. Open a GitHub issue if you find a security concern.
