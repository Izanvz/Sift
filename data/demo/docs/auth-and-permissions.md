# Authentication and Per-User Permissions

Sift implements JWT-based authentication with per-user corpus scopes enforced at retrieval time.

## Authentication flow

1. User posts `username` + `password` to `/auth/token` (OAuth2 password flow).
2. Server verifies the bcrypt-hashed password from the SQLite user store.
3. Server issues a signed JWT containing `sub` (username), `scopes`, and `is_admin`.
4. All subsequent requests include `Authorization: Bearer <token>`.
5. FastAPI dependency `get_current_user` decodes and validates the token on every protected endpoint.

JWT parameters:
- Algorithm: `HS256`
- Expiry: 8 hours (configurable via `JWT_EXPIRE_MINUTES`)
- Secret: `JWT_SECRET` env var — change from default in any non-demo deployment

## Scopes

Each user has a list of corpus names they are authorized to access (e.g., `["vercel-docs", "stripe-go"]`). Admins have `scopes=["*"]` and bypass all filters.

Scopes are stored in the SQLite users table and embedded in the JWT at login time.

## Retrieval-time enforcement

Scope filtering happens inside the retrieval step — not post-hoc:

- **Vector search**: ChromaDB `where={"corpus": {"$in": user.scopes}}` applied at query time.
- **BM25**: candidates filtered by corpus after retrieval from the in-memory index.

This means unauthorized documents are never loaded into the context window, regardless of query phrasing.

## Demo users

`scripts/bootstrap_demo.py` creates four users to demonstrate scope isolation:

| User | Scopes | Sees |
|------|--------|------|
| `admin` | `*` | Everything |
| `engineer` | `docs`, `code` | Docs and code corpora |
| `sales` | `docs` | Docs corpus only |
| `no_scope` | (none) | Nothing — returns empty results |

## Audit log

Every login attempt, query, and error is recorded in an append-only SQLite audit table with timestamp, user ID, IP address, session ID, and latency. Admins can query `/audit/events`; users can query `/audit/events/mine`.
