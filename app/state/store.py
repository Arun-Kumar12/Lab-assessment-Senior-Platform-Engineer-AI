import threading
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.REJECTED, RunStatus.FAILED}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Step(BaseModel):
    index: int
    kind: str
    name: str
    input: Any = None
    output: Any = None
    error: dict[str, str] | None = None
    duration_ms: int = 0
    at: datetime = Field(default_factory=utcnow)


class Proposal(BaseModel):
    recommendation: str
    reason: str
    customer_reply: str
    model_recommendation: str | None
    policy_outcome: str | None
    overridden: bool = False
    override_reason: str | None = None
    reply_source: str = "model"
    safety_flags: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)


class OperatorDecision(BaseModel):
    decision: str
    note: str | None = None
    decided_by: str
    decided_at: datetime = Field(default_factory=utcnow)


class RunError(BaseModel):
    code: str
    message: str


class Run(BaseModel):
    run_id: str
    ticket_id: str
    status: RunStatus
    provider: str
    model: str
    steps: list[Step] = Field(default_factory=list)
    proposal: Proposal | None = None
    decision: OperatorDecision | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)
    error: RunError | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def add_step(self, kind: str, name: str, **fields: Any) -> Step:
        step = Step(index=len(self.steps), kind=kind, name=name, **fields)
        self.steps.append(step)
        return step


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def create(self, ticket_id: str, provider: str, model: str) -> Run:
        run = Run(run_id=uuid.uuid4().hex[:12], ticket_id=ticket_id, status=RunStatus.QUEUED,
                  provider=provider, model=model)
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            run = self._runs.get(run_id)
            return run.model_copy(deep=True) if run else None

    def save(self, run: Run) -> Run:
        run.updated_at = utcnow()
        with self._lock:
            self._runs[run.run_id] = run.model_copy(deep=True)
        return run

    def list(self, ticket_id: str | None = None) -> list[Run]:
        with self._lock:
            runs = [r.model_copy(deep=True) for r in self._runs.values() if ticket_id is None or r.ticket_id == ticket_id]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
