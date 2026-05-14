from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Ollama local) ---
    model_planning: str = "qwen2.5:7b"
    model_synthesis: str = "qwen2.5:7b"
    model_routing: str = "qwen2.5:7b"

    # --- ChromaDB ---
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    chromadb_collection: str = "sift_documents"     # Colección única para todos los docs

    # --- SQLite ---
    sqlite_db_path: str = "data/sift.db"
    audit_db_path: str = "data/audit.db"

    # --- Retrieval ---
    retrieval_top_k: int = 20           # Chunks candidatos antes del reranker
    synthesis_top_k: int = 5            # Chunks finales que van al prompt de síntesis
    relevance_threshold: float = 0.5    # Mínimo para no reescribir la query
    bm25_top_k: int = 30                # Candidatos por rama BM25
    vector_top_k: int = 30              # Candidatos por rama vectorial
    rrf_k: int = 60                     # Constante de RRF (Cormack et al. 2009)
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_enabled: bool = True       # Set False en tests/CI sin GPU

    # --- Ciclos del grafo ---
    max_search_iterations: int = 2      # Máximo de rewrites de query
    max_rewrite_iterations: int = 2     # Máximo de rewrites de respuesta
    quality_gate_score: float = 8.0          # Score mínimo del critique para no reescribir
    faithfulness_hard_gate: float = 6.0      # Faithfulness mínima — siempre reescribir si falla

    # --- Chunking ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    # --- Auth (Fase 7) ---
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 8   # 8 horas

    # --- Observability (Langfuse) ---
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


settings = Settings()
