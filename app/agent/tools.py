import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.data.models import FaultType, OrderId, Ticket
from app.data.repository import Repository
from app.policy import EligibilityResult, evaluate_refund

DEPENDENCY_ERROR_CODES = {"timeout", "unavailable", "stale_data", "invalid_tool_response"}


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"error": self.code, "message": self.message}


@dataclass
class ToolContext:
    ticket: Ticket
    repo: Repository
    settings: Settings


class OrderArgs(BaseModel):
    order_id: OrderId = Field(description="Order identifier in the form ORD-1234")


class NoArgs(BaseModel):
    pass


class OrderResult(BaseModel):
    order_id: str
    customer_email: str
    item: str
    amount: float
    currency: str
    status: str
    ordered_at: str
    delivered_at: str | None


class PolicyResult(BaseModel):
    window_days: int
    max_auto_refund_amount: float
    eligible_statuses: list[str]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[ToolContext, Any], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


def _load_order(ctx: ToolContext, order_id: str):
    fault = ctx.repo.fault_for(order_id)
    if fault == FaultType.TIMEOUT:
        time.sleep(ctx.settings.tool_timeout_seconds + 1)
    if fault == FaultType.UNAVAILABLE:
        raise ToolError("unavailable", "Order service is unavailable")
    order = ctx.repo.get_order(order_id)
    if order is None:
        raise ToolError("not_found", f"Order {order_id} does not exist")
    age_hours = (ctx.settings.reference_date - order.last_synced_at).total_seconds() / 3600
    if age_hours > ctx.settings.stale_after_hours:
        raise ToolError("stale_data", f"Order {order_id} was last synced {age_hours:.0f} hours ago")
    return order


def get_order(ctx: ToolContext, args: OrderArgs) -> dict[str, Any]:
    if ctx.repo.fault_for(args.order_id) == FaultType.INVALID_RESPONSE:
        return {"order_id": args.order_id, "unexpected": "payload"}
    order = _load_order(ctx, args.order_id)
    return order.model_dump(mode="json", exclude={"last_synced_at"})


def check_refund_eligibility(ctx: ToolContext, args: OrderArgs) -> dict[str, Any]:
    order = _load_order(ctx, args.order_id)
    result = evaluate_refund(order, ctx.ticket.customer_email, ctx.repo.policy, ctx.settings.reference_date)
    return result.model_dump(mode="json")


def get_refund_policy(ctx: ToolContext, args: NoArgs) -> dict[str, Any]:
    return ctx.repo.policy.model_dump(mode="json")


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "get_order",
            "Fetch an order by order_id to see the item, amount, status and dates.",
            OrderArgs, OrderResult, get_order,
        ),
        ToolSpec(
            "check_refund_eligibility",
            "Evaluate the refund policy for an order on behalf of the customer who opened the ticket.",
            OrderArgs, EligibilityResult, check_refund_eligibility,
        ),
        ToolSpec(
            "get_refund_policy",
            "Return the refund policy rules.",
            NoArgs, PolicyResult, get_refund_policy,
        ),
    ]
}

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


def execute_tool(name: str, raw_args: dict[str, Any] | None, ctx: ToolContext) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        raise ToolError("unknown_tool", f"Tool {name} does not exist")
    try:
        args = spec.args_model.model_validate(raw_args or {})
    except ValidationError as exc:
        raise ToolError("invalid_argument", exc.errors()[0]["msg"]) from exc
    future = _executor.submit(spec.handler, ctx, args)
    try:
        output = future.result(timeout=ctx.settings.tool_timeout_seconds)
    except FutureTimeout as exc:
        raise ToolError("timeout", f"Tool {name} exceeded {ctx.settings.tool_timeout_seconds}s") from exc
    try:
        spec.output_model.model_validate(output)
    except ValidationError as exc:
        raise ToolError("invalid_tool_response", f"Tool {name} returned an invalid payload") from exc
    return output


def issue_refund(order_id: str, amount: float, currency: str) -> dict[str, Any]:
    return {
        "refund_id": f"RF-{order_id[-4:]}-{int(time.time()) % 100000}",
        "order_id": order_id,
        "amount": amount,
        "currency": currency,
        "status": "issued",
    }


def send_reply(ticket_id: str, to: str, body: str) -> dict[str, Any]:
    return {"message_id": f"MSG-{ticket_id}-{int(time.time()) % 100000}", "to": to, "chars": len(body), "status": "sent"}
