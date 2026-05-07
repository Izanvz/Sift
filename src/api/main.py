from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.agent.graph import build_graph
    from src.auth.store import get_user_store
    from src.db.checkpointer import get_checkpointer

    checkpointer = get_checkpointer()
    app.state.graph = build_graph(checkpointer=checkpointer)
    app.state.checkpointer = checkpointer
    # Inicializa el user store (crea schema si no existe)
    get_user_store()
    yield


app = FastAPI(title="ResearchAgent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.auth_routes import router as auth_router  # noqa: E402
from src.api.routes import router  # noqa: E402

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(router, prefix="/research", tags=["research"])
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")
