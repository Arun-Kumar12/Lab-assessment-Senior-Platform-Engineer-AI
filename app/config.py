from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Refund Triage Agent"
    log_level: str = "INFO"

    llm_provider: str = Field(default="ollama", pattern="^(ollama|fake)$")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_timeout_seconds: float = 180.0
    max_agent_steps: int = 6
    max_decision_retries: int = 2

    tool_timeout_seconds: float = 1.0
    stale_after_hours: int = 24
    # The fixture data is a snapshot; policy windows are evaluated against the snapshot date.
    reference_date: datetime = datetime(2026, 10, 20, tzinfo=timezone.utc)

    fixtures_dir: Path = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    api_keys: str = "operator-key:operator,viewer-key:viewer"

    def parsed_api_keys(self) -> dict[str, str]:
        pairs = (item.split(":", 1) for item in self.api_keys.split(",") if ":" in item)
        return {key.strip(): role.strip() for key, role in pairs}


@lru_cache
def get_settings() -> Settings:
    return Settings()
