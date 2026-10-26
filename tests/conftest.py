from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent.llm import FakeLLM
from app.agent.orchestrator import Orchestrator
from app.config import Settings
from app.data.repository import Repository
from app.main import create_app
from app.state.store import RunStore

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"
OPERATOR = {"X-API-Key": "operator-key"}
VIEWER = {"X-API-Key": "viewer-key"}


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="fake", fixtures_dir=FIXTURES, tool_timeout_seconds=0.2, log_level="WARNING")


@pytest.fixture
def repo(settings) -> Repository:
    return Repository(settings.fixtures_dir)


@pytest.fixture
def store() -> RunStore:
    return RunStore()


@pytest.fixture
def make_orchestrator(settings, repo, store):
    def factory(mode: str = "good") -> Orchestrator:
        return Orchestrator(FakeLLM(mode), repo, store, settings)

    return factory


@pytest.fixture
def make_client(settings):
    def factory(mode: str = "good") -> TestClient:
        return TestClient(create_app(settings, llm=FakeLLM(mode)))

    return factory


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
