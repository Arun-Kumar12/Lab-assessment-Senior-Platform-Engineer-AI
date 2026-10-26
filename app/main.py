from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.agent.llm import LLMClient, build_llm
from app.agent.orchestrator import Orchestrator
from app.api.routes import router
from app.config import Settings, get_settings
from app.data.repository import Repository
from app.logging_config import configure_logging
from app.state.store import RunStore

UI_PATH = Path(__file__).resolve().parent / "ui" / "index.html"


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version="1.0.0")
    repo = Repository(settings.fixtures_dir)
    store = RunStore()
    llm = llm or build_llm(settings)

    app.state.settings = settings
    app.state.repo = repo
    app.state.store = store
    app.state.llm = llm
    app.state.orchestrator = Orchestrator(llm, repo, store, settings)
    app.state.api_keys = settings.parsed_api_keys()
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def ui() -> HTMLResponse:
        return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))

    return app


app = create_app()
