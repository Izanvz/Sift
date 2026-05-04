from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.agent.graph import build_graph
    from src.db.checkpointer import get_checkpointer

    checkpointer = get_checkpointer()
    app.state.graph = build_graph(checkpointer=checkpointer)
    app.state.checkpointer = checkpointer
    yield


app = FastAPI(title="ResearchAgent API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.routes import router  # noqa: E402

app.include_router(router, prefix="/research")
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")
