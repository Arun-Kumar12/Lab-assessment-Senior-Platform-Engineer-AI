# Refund Triage Agent

An AI agent that triages customer refund tickets. It looks up the order with tools, checks the refund policy, and proposes a decision with a draft reply. A human operator approves or rejects in the web UI before any refund or reply is executed.

Runs fully offline with Docker Compose. The model is `qwen2.5:1.5b` served by Ollama inside Compose. No API keys needed.

## Requirements

- Docker Desktop or Docker Engine with Compose v2
- About 3 GB of free disk for images and the model

## Start

```bash
docker compose up --build
```

The first start downloads the model (about 1 GB). Open <http://localhost:8000> once the `app` container is healthy.

Optional, with an NVIDIA GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Use

1. Select a ticket and click **Run agent**.
2. Review the proposal, reason, draft reply, and execution trace.
3. Click **Approve & execute** or **Reject**.

API keys: `operator-key` can run and approve, `viewer-key` can only read. The UI pre-fills the operator key. API docs: <http://localhost:8000/docs>.

| Ticket | Scenario |
|---|---|
| TCK-001 | Eligible refund |
| TCK-002, TCK-004 | Refund denied by policy |
| TCK-003 | Above auto-refund limit, escalated |
| TCK-005 | Stale order data (injected failure) |
| TCK-006 | Customer email mismatch |
| TCK-007 | Tool timeout (injected failure) |
| TCK-008 | Unsafe request blocked by guardrail |
| TCK-009 | Malformed order reference (injected failure) |

## Tests and evaluation

```bash
docker compose --profile test run --rm tests    # unit, integration, API, data, browser tests (fake LLM, ~10 s)
docker compose --profile eval run --rm evals    # AI evaluation against the real model, 3 repeats (~5 min on CPU)
```

Outputs: `reports/junit.xml`, `reports/screenshots/`, `evals/reports/eval_report.md`.

Without Docker:

```bash
pip install -r requirements-dev.txt && python -m playwright install chromium
LLM_PROVIDER=fake uvicorn app.main:app
pytest
python -m evals.runner --provider ollama --repeats 3
```

## Architecture

```
app/api          HTTP routes, API-key auth with roles, schemas
app/agent        orchestrator (state machine + approval gate), tools, LLM clients, prompts, guardrails
app/policy.py    refund policy rules
app/data         pydantic models, fixture repository
app/state        in-memory run store with execution history
app/ui           single-page operator console
data/fixtures    orders, tickets, policy, injected faults
evals            evaluation dataset, scoring, runner, reports
tests            unit, integration, api, data, browser, evals
```

Run states: `queued -> running -> awaiting_approval -> completed | rejected`, or `failed`.

## Design decisions

- Business rules live in code and are exposed to the model as tools. The model gathers evidence and writes the reply.
- A policy layer checks the model's recommendation against the tool evidence and overrides it with a template reply if it is ungrounded.
- Refund and reply actions are never exposed to the model. They run only after operator approval.
- Tool errors (timeout, stale data, invalid input, invalid response) stop the run and force escalation. Model outages end the run as `failed`.
- The final decision uses JSON-schema output with validation and bounded retries.
- A deterministic fake LLM makes all tests reproducible. The real model is used only by the evaluation harness.

## Evaluation

Nine cases scored on task completion, tool accuracy, groundedness, relevance, safety, and failure handling, with repeats to expose variance. Committed run: 27 runs, 100% pass, all criteria at 1.00. Thresholds are in `evals/dataset.json` and the runner exits non-zero when missed.

## CI

`.github/workflows/ci.yml`: tests and a Docker smoke test on every pull request, the live evaluation on a weekly schedule or manual trigger.

## Assumptions and limitations

- Fixture data is a snapshot dated 2026-10-20; policy windows are evaluated against that date.
- State is in memory. Refund and reply actions are mocked and return receipts.
- API keys are static demo credentials.
- The guardrail is pattern based; scoring uses deterministic checks, not an LLM judge.

## With more time

Database persistence, free-text order extraction, operator edits to replies, LLM-as-judge scoring with a larger adversarial dataset, OpenTelemetry traces, canary evaluation before prompt or model changes.
