from typing import Literal

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=500)


class LLMHealth(BaseModel):
    provider: str
    model: str
    reachable: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm: LLMHealth


class ErrorResponse(BaseModel):
    detail: str
