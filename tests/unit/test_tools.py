import pytest

from app.agent.tools import TOOLS, ToolContext, ToolError, execute_tool


@pytest.fixture
def ctx(repo, settings):
    return ToolContext(ticket=repo.get_ticket("TCK-001"), repo=repo, settings=settings)


def test_tool_schemas_are_well_formed():
    for spec in TOOLS.values():
        schema = spec.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == spec.name
        assert "properties" in schema["function"]["parameters"]


def test_get_order_returns_validated_payload(ctx):
    output = execute_tool("get_order", {"order_id": "ORD-1001"}, ctx)
    assert output["item"] == "Wireless Headphones"
    assert output["status"] == "delivered"
    assert "last_synced_at" not in output


def test_eligibility_uses_ticket_customer(ctx, repo):
    ctx.ticket = repo.get_ticket("TCK-006")
    output = execute_tool("check_refund_eligibility", {"order_id": "ORD-1006"}, ctx)
    assert output["reason_code"] == "customer_mismatch"


def test_unknown_tool_is_rejected(ctx):
    with pytest.raises(ToolError) as exc:
        execute_tool("delete_order", {"order_id": "ORD-1001"}, ctx)
    assert exc.value.code == "unknown_tool"


@pytest.mark.parametrize("args", [{}, {"order_id": "12345-X"}, {"order_id": 42}])
def test_invalid_arguments_are_rejected(ctx, args):
    with pytest.raises(ToolError) as exc:
        execute_tool("get_order", args, ctx)
    assert exc.value.code == "invalid_argument"


def test_missing_order_is_not_found(ctx):
    with pytest.raises(ToolError) as exc:
        execute_tool("get_order", {"order_id": "ORD-9999"}, ctx)
    assert exc.value.code == "not_found"


def test_stale_order_is_flagged(ctx):
    with pytest.raises(ToolError) as exc:
        execute_tool("get_order", {"order_id": "ORD-1005"}, ctx)
    assert exc.value.code == "stale_data"


def test_slow_dependency_times_out(ctx):
    with pytest.raises(ToolError) as exc:
        execute_tool("get_order", {"order_id": "ORD-1007"}, ctx)
    assert exc.value.code == "timeout"


def test_invalid_tool_response_is_detected(ctx, repo):
    repo.faults["ORD-1001"] = "invalid_response"
    try:
        with pytest.raises(ToolError) as exc:
            execute_tool("get_order", {"order_id": "ORD-1001"}, ctx)
        assert exc.value.code == "invalid_tool_response"
    finally:
        repo.faults.pop("ORD-1001")


def test_unavailable_dependency_is_reported(ctx, repo):
    repo.faults["ORD-1001"] = "unavailable"
    try:
        with pytest.raises(ToolError) as exc:
            execute_tool("check_refund_eligibility", {"order_id": "ORD-1001"}, ctx)
        assert exc.value.code == "unavailable"
    finally:
        repo.faults.pop("ORD-1001")
