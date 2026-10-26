from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from app.agent.orchestrator import InvalidTransition
from app.api.auth import Principal, authenticate, require_role
from app.api.schemas import DecisionRequest, ErrorResponse, HealthResponse, LLMHealth
from app.data.models import TicketView
from app.state.store import Run

router = APIRouter()
NOT_FOUND = {404: {"model": ErrorResponse}}
AUTH_ERRORS = {401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}}


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    llm = request.app.state.llm
    reachable = llm.healthy()
    return HealthResponse(status="ok" if reachable else "degraded",
                          llm=LLMHealth(provider=llm.provider, model=llm.model, reachable=reachable))


@router.get("/api/tickets", response_model=list[TicketView], responses=AUTH_ERRORS)
def list_tickets(request: Request, _: Principal = Depends(authenticate)) -> list[TicketView]:
    repo = request.app.state.repo
    return [repo.ticket_view(t) for t in repo.list_tickets()]


@router.get("/api/tickets/{ticket_id}", response_model=TicketView, responses={**AUTH_ERRORS, **NOT_FOUND})
def get_ticket(ticket_id: str, request: Request, _: Principal = Depends(authenticate)) -> TicketView:
    repo = request.app.state.repo
    ticket = repo.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"ticket {ticket_id} not found")
    return repo.ticket_view(ticket)


@router.post("/api/tickets/{ticket_id}/runs", response_model=Run, status_code=202, responses={**AUTH_ERRORS, **NOT_FOUND})
def start_run(ticket_id: str, request: Request, background: BackgroundTasks,
              wait: bool = Query(default=False), _: Principal = Depends(require_role("operator"))) -> Run:
    orchestrator = request.app.state.orchestrator
    try:
        run = orchestrator.start(ticket_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"ticket {ticket_id} not found")
    if wait:
        return orchestrator.execute(run.run_id)
    background.add_task(orchestrator.execute, run.run_id)
    return run


@router.get("/api/runs", response_model=list[Run], responses=AUTH_ERRORS)
def list_runs(request: Request, ticket_id: str | None = None, _: Principal = Depends(authenticate)) -> list[Run]:
    return request.app.state.store.list(ticket_id)


@router.get("/api/runs/{run_id}", response_model=Run, responses={**AUTH_ERRORS, **NOT_FOUND})
def get_run(run_id: str, request: Request, _: Principal = Depends(authenticate)) -> Run:
    run = request.app.state.store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return run


@router.get("/api/runs/{run_id}/history", responses={**AUTH_ERRORS, **NOT_FOUND})
def run_history(run_id: str, request: Request, _: Principal = Depends(authenticate)) -> dict:
    run = request.app.state.store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return {
        "run_id": run.run_id,
        "ticket_id": run.ticket_id,
        "status": run.status,
        "provider": run.provider,
        "model": run.model,
        "steps": [s.model_dump(mode="json") for s in run.steps],
        "tool_calls": [s.name for s in run.steps if s.kind == "tool"],
        "errors": [s.error for s in run.steps if s.error],
        "total_duration_ms": sum(s.duration_ms for s in run.steps),
    }


@router.post("/api/runs/{run_id}/decision", response_model=Run,
             responses={**AUTH_ERRORS, **NOT_FOUND, 409: {"model": ErrorResponse}})
def decide(run_id: str, body: DecisionRequest, request: Request,
           principal: Principal = Depends(require_role("operator"))) -> Run:
    try:
        return request.app.state.orchestrator.decide(run_id, body.decision, body.note, principal.role)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    except InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
