import pytest

from app.agent.orchestrator import InvalidTransition
from app.state.store import RunStatus


def run_ticket(orchestrator, ticket_id):
    run = orchestrator.start(ticket_id)
    assert run.status == RunStatus.QUEUED
    return orchestrator.execute(run.run_id)


def tool_names(run):
    return [s.name for s in run.steps if s.kind == "tool"]


def test_happy_path_proposes_refund_and_waits_for_operator(make_orchestrator):
    orchestrator = make_orchestrator()
    run = run_ticket(orchestrator, "TCK-001")
    assert run.status == RunStatus.AWAITING_APPROVAL
    assert tool_names(run) == ["get_order", "check_refund_eligibility"]
    assert run.proposal.recommendation == "approve_refund"
    assert run.proposal.policy_outcome == "refund_allowed"
    assert not run.proposal.overridden
    assert "ORD-1001" in run.proposal.customer_reply
    assert run.actions == []


def test_approval_executes_refund_and_reply(make_orchestrator):
    orchestrator = make_orchestrator()
    run = run_ticket(orchestrator, "TCK-001")
    final = orchestrator.decide(run.run_id, "approve", "looks fine", "operator")
    assert final.status == RunStatus.COMPLETED
    assert [a["action"] for a in final.actions] == ["issue_refund", "send_reply"]
    assert final.actions[0]["amount"] == 49.99
    assert final.decision.decided_by == "operator"


def test_rejection_executes_nothing(make_orchestrator):
    orchestrator = make_orchestrator()
    run = run_ticket(orchestrator, "TCK-001")
    final = orchestrator.decide(run.run_id, "reject", "customer already refunded", "operator")
    assert final.status == RunStatus.REJECTED
    assert final.actions == []


def test_decision_requires_awaiting_state(make_orchestrator):
    orchestrator = make_orchestrator()
    run = run_ticket(orchestrator, "TCK-001")
    orchestrator.decide(run.run_id, "approve", None, "operator")
    with pytest.raises(InvalidTransition):
        orchestrator.decide(run.run_id, "approve", None, "operator")


def test_denied_refund_sends_reply_only(make_orchestrator):
    orchestrator = make_orchestrator()
    run = run_ticket(orchestrator, "TCK-002")
    assert run.proposal.recommendation == "deny_refund"
    final = orchestrator.decide(run.run_id, "approve", None, "operator")
    assert [a["action"] for a in final.actions] == ["send_reply"]


def test_manual_review_hands_off_to_specialist(make_orchestrator):
    orchestrator = make_orchestrator()
    run = run_ticket(orchestrator, "TCK-003")
    assert run.proposal.recommendation == "escalate"
    assert run.proposal.policy_outcome == "manual_review_required"
    final = orchestrator.decide(run.run_id, "approve", None, "operator")
    assert [a["action"] for a in final.actions] == ["handoff_to_specialist", "send_reply"]


@pytest.mark.parametrize("ticket_id,error_code", [
    ("TCK-005", "stale_data"),
    ("TCK-007", "timeout"),
    ("TCK-009", "invalid_argument"),
])
def test_tool_failures_escalate_visibly(make_orchestrator, ticket_id, error_code):
    run = run_ticket(make_orchestrator(), ticket_id)
    assert run.status == RunStatus.AWAITING_APPROVAL
    assert run.proposal.recommendation == "escalate"
    assert run.proposal.tool_errors == [error_code]
    assert tool_names(run) == ["get_order"]
    assert any(s.error and s.error["error"] == error_code for s in run.steps)


def test_unsafe_request_is_blocked_before_any_tool_call(make_orchestrator):
    run = run_ticket(make_orchestrator(), "TCK-008")
    assert run.status == RunStatus.AWAITING_APPROVAL
    assert tool_names(run) == []
    assert run.proposal.recommendation == "escalate"
    assert set(run.proposal.safety_flags) >= {"prompt_injection", "bulk_action"}
    assert "refund" not in run.proposal.customer_reply.lower() or "cannot" in run.proposal.customer_reply.lower()


def test_ungrounded_model_recommendation_is_overridden(make_orchestrator):
    run = run_ticket(make_orchestrator("skip_tools"), "TCK-001")
    assert run.status == RunStatus.AWAITING_APPROVAL
    assert run.proposal.model_recommendation == "approve_refund"
    assert run.proposal.recommendation == "escalate"
    assert run.proposal.overridden
    assert "never checked" in run.proposal.override_reason
    assert run.proposal.reply_source == "template"
    assert "refunded" not in run.proposal.customer_reply
    assert "specialist" in run.proposal.customer_reply


def test_reply_without_order_id_is_normalized(make_orchestrator):
    run = run_ticket(make_orchestrator("terse_reply"), "TCK-001")
    assert run.proposal.customer_reply.startswith("Regarding order ORD-1001: ")
    assert run.proposal.reply_source == "model"


def test_malformed_model_output_is_retried(make_orchestrator):
    run = run_ticket(make_orchestrator("malformed_once"), "TCK-001")
    assert run.status == RunStatus.AWAITING_APPROVAL
    assert run.proposal.recommendation == "approve_refund"
    assert any(s.name == "decision_parse" for s in run.steps)


def test_unknown_tool_call_is_recorded_and_recovered(make_orchestrator):
    run = run_ticket(make_orchestrator("unknown_tool"), "TCK-001")
    assert run.status == RunStatus.AWAITING_APPROVAL
    assert tool_names(run)[0] == "delete_order"
    assert run.proposal.recommendation == "escalate"
    assert run.proposal.tool_errors == ["unknown_tool"]


def test_model_that_answers_without_tools_is_nudged(make_orchestrator):
    run = run_ticket(make_orchestrator("lazy_once"), "TCK-001")
    assert run.status == RunStatus.AWAITING_APPROVAL
    assert tool_names(run) == ["get_order", "check_refund_eligibility"]
    assert run.proposal.recommendation == "approve_refund"
    assert sum(1 for s in run.steps if s.kind == "llm" and s.input["tools"]) == 3


def test_llm_outage_fails_run_with_clear_error(make_orchestrator):
    run = run_ticket(make_orchestrator("unavailable"), "TCK-001")
    assert run.status == RunStatus.FAILED
    assert run.error.code == "llm_unavailable"
    assert run.proposal is None


def test_unknown_ticket_cannot_start(make_orchestrator):
    with pytest.raises(KeyError):
        make_orchestrator().start("TCK-999")


def test_store_returns_copies_not_references(make_orchestrator, store):
    orchestrator = make_orchestrator()
    run = run_ticket(orchestrator, "TCK-001")
    snapshot = store.get(run.run_id)
    snapshot.status = RunStatus.FAILED
    assert store.get(run.run_id).status == RunStatus.AWAITING_APPROVAL
