import pytest

from tests.conftest import OPERATOR, VIEWER


def start_and_wait(client, ticket_id, headers=OPERATOR):
    response = client.post(f"/api/tickets/{ticket_id}/runs?wait=true", headers=headers)
    assert response.status_code == 202, response.text
    return response.json()


def test_health_reports_provider(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm"] == {"provider": "fake", "model": "scripted-v1", "reachable": True}


def test_health_degraded_when_llm_unreachable(make_client):
    body = make_client("unavailable").get("/health").json()
    assert body["status"] == "degraded" and body["llm"]["reachable"] is False


def test_list_tickets_joins_order_data(client):
    tickets = client.get("/api/tickets", headers=VIEWER).json()
    assert [t["ticket_id"] for t in tickets][:3] == ["TCK-001", "TCK-002", "TCK-003"]
    first = tickets[0]
    assert first["order_found"] is True
    assert first["amount_display"] == "49.99 USD"
    assert first["order_summary"] == "Wireless Headphones (delivered)"


def test_ticket_with_unknown_order_is_marked(client):
    ticket = client.get("/api/tickets/TCK-009", headers=VIEWER).json()
    assert ticket["order_found"] is False and ticket["amount_display"] is None


def test_unknown_ticket_returns_404(client):
    assert client.get("/api/tickets/TCK-999", headers=VIEWER).status_code == 404
    assert client.post("/api/tickets/TCK-999/runs", headers=OPERATOR).status_code == 404


@pytest.mark.parametrize("path,method", [
    ("/api/tickets", "get"),
    ("/api/runs", "get"),
    ("/api/tickets/TCK-001/runs", "post"),
])
def test_missing_api_key_is_unauthorized(client, path, method):
    assert getattr(client, method)(path).status_code == 401


def test_invalid_api_key_is_unauthorized(client):
    assert client.get("/api/tickets", headers={"X-API-Key": "nope"}).status_code == 401


def test_viewer_cannot_start_runs_or_decide(client):
    assert client.post("/api/tickets/TCK-001/runs", headers=VIEWER).status_code == 403
    run = start_and_wait(client, "TCK-001")
    response = client.post(f"/api/runs/{run['run_id']}/decision", headers=VIEWER, json={"decision": "approve"})
    assert response.status_code == 403
    assert client.get(f"/api/runs/{run['run_id']}", headers=VIEWER).status_code == 200


def test_background_run_reaches_terminal_state(client):
    response = client.post("/api/tickets/TCK-001/runs", headers=OPERATOR)
    assert response.status_code == 202
    run = client.get(f"/api/runs/{response.json()['run_id']}", headers=OPERATOR).json()
    assert run["status"] == "awaiting_approval"
    assert run["proposal"]["recommendation"] == "approve_refund"


def test_full_approval_flow(client):
    run = start_and_wait(client, "TCK-001")
    response = client.post(f"/api/runs/{run['run_id']}/decision", headers=OPERATOR,
                           json={"decision": "approve", "note": "verified with customer"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert [a["action"] for a in body["actions"]] == ["issue_refund", "send_reply"]
    assert body["decision"]["note"] == "verified with customer"


@pytest.mark.parametrize("payload", [
    {},
    {"decision": "maybe"},
    {"decision": "approve", "note": "x" * 501},
    {"decision": 1},
])
def test_decision_validation_errors(client, payload):
    run = start_and_wait(client, "TCK-001")
    assert client.post(f"/api/runs/{run['run_id']}/decision", headers=OPERATOR, json=payload).status_code == 422


def test_decision_on_completed_run_conflicts(client):
    run = start_and_wait(client, "TCK-001")
    client.post(f"/api/runs/{run['run_id']}/decision", headers=OPERATOR, json={"decision": "reject"})
    response = client.post(f"/api/runs/{run['run_id']}/decision", headers=OPERATOR, json={"decision": "approve"})
    assert response.status_code == 409
    assert "rejected" in response.json()["detail"]


def test_decision_on_unknown_run_is_404(client):
    assert client.post("/api/runs/nope/decision", headers=OPERATOR, json={"decision": "approve"}).status_code == 404


def test_llm_outage_is_visible_as_failed_run(make_client):
    client = make_client("unavailable")
    run = start_and_wait(client, "TCK-001")
    assert run["status"] == "failed"
    assert run["error"]["code"] == "llm_unavailable"
    response = client.post(f"/api/runs/{run['run_id']}/decision", headers=OPERATOR, json={"decision": "approve"})
    assert response.status_code == 409


def test_tool_timeout_run_escalates(client):
    run = start_and_wait(client, "TCK-007")
    assert run["status"] == "awaiting_approval"
    assert run["proposal"]["recommendation"] == "escalate"
    assert run["proposal"]["tool_errors"] == ["timeout"]


def test_history_endpoint_exposes_trace(client):
    run = start_and_wait(client, "TCK-005")
    history = client.get(f"/api/runs/{run['run_id']}/history", headers=VIEWER).json()
    assert history["tool_calls"] == ["get_order"]
    assert history["errors"][0]["error"] == "stale_data"
    assert {s["kind"] for s in history["steps"]} >= {"guardrail", "llm", "tool", "policy"}


def test_runs_can_be_filtered_by_ticket(client):
    start_and_wait(client, "TCK-001")
    start_and_wait(client, "TCK-002")
    runs = client.get("/api/runs?ticket_id=TCK-002", headers=VIEWER).json()
    assert len(runs) == 1 and runs[0]["ticket_id"] == "TCK-002"


def test_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Refund Triage Agent" in response.text
