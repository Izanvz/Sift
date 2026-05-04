from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import settings


def get_checkpointer() -> SqliteSaver:
    """Return a SqliteSaver instance for the configured database path."""
    return SqliteSaver.from_conn_string(settings.sqlite_db_path)
