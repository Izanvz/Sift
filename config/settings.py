from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama models — cambiar a qwen2.5:14b cuando haya ≥8GB RAM libres
    model_planning: str = "qwen2.5:7b"
    model_synthesis: str = "qwen2.5:7b"
    model_routing: str = "qwen2.5:7b"

    # APIs
    tavily_api_key: str = ""

    # ChromaDB
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    chromadb_collection_research: str = "research_reports"
    chromadb_collection_meeting: str = "meeting_transcripts"

    # SQLite checkpointing
    sqlite_db_path: str = "checkpoints.db"

    # Thresholds
    quality_threshold: float = 0.7
    quality_gate_score: float = 7.0
    max_search_iterations: int = 3
    max_rewrite_iterations: int = 3

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001


settings = Settings()
