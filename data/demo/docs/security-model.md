# Security Model

## Authentication

Sift uses OAuth2 password flow with JWT bearer tokens.

- Passwords are hashed with bcrypt (via passlib).
- Tokens are signed with `JWT_SECRET` (HS256).
- Token expiry is configurable via `ACCESS_TOKEN_EXPIRE_MINUTES` (default: 30).
- All endpoints except `/auth/token` and `/health` require a valid token.

## Authorization

Each user has a `scopes` list. Scopes map to ChromaDB corpus names.

Examples:
- `["*"]` — admin: access to all corpora
- `["docs", "code"]` — engineer: access to docs and code corpora
- `["docs"]` — sales: access to docs only
- `[]` — no_scope: no access to any corpus; retrieval returns empty

The ChromaDB `where` filter is built per-request: `{"corpus": {"$in": user.scopes}}`. Admin users bypass this filter.

## Audit log

Every event is recorded in an append-only SQLite table:
- Login attempts (success and failure)
- Every query with user ID, query text, latency, and result count
- System errors with stack traces (redacted before storage)
- IP address and timestamp

The `/audit/events` endpoint is admin-only. Users can query their own history at `/audit/events/mine`.

## Known limitations (portfolio-grade MVP)

- `JWT_SECRET` is read from env; no automatic rotation.
- No rate limiting on auth endpoints (brute-force risk in open deployments).
- Audit log IP is taken from `X-Forwarded-For` header — can be spoofed without a trusted proxy.
- No TLS termination built in — use a reverse proxy (nginx, Caddy) in front of uvicorn.
- Not a certified security product. Do not use for regulated data (HIPAA, PCI-DSS, etc.) without additional hardening.

## On-premise guarantee

Sift makes zero outbound API calls when configured correctly:
- LLM: Ollama running locally
- Embeddings: ChromaDB built-in (all-MiniLM-L6-v2, downloaded once at startup)
- Reranker: BAAI/bge-reranker-base (downloaded once at startup)
- Observability: Langfuse is optional; disabled by default

The only exception is if `LANGFUSE_ENABLED=true` points to an external host.
