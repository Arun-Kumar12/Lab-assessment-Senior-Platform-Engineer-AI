from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from app.data.models import Order, RefundPolicy


class PolicyOutcome(StrEnum):
    REFUND_ALLOWED = "refund_allowed"
    REFUND_DENIED = "refund_denied"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class EligibilityResult(BaseModel):
    order_id: str
    eligible: bool
    within_auto_limit: bool
    policy_outcome: PolicyOutcome
    reason_code: str
    reason: str
    days_since_delivery: int | None
    amount: float
    currency: str


def evaluate_refund(order: Order, requester_email: str, policy: RefundPolicy, now: datetime) -> EligibilityResult:
    base = dict(order_id=order.order_id, amount=order.amount, currency=order.currency, days_since_delivery=None)
    within_limit = order.amount <= policy.max_auto_refund_amount

    if requester_email.lower() != order.customer_email.lower():
        return EligibilityResult(
            **base, eligible=False, within_auto_limit=within_limit,
            policy_outcome=PolicyOutcome.MANUAL_REVIEW_REQUIRED, reason_code="customer_mismatch",
            reason="Requester email does not match the customer on the order",
        )
    if order.status not in policy.eligible_statuses or order.delivered_at is None:
        return EligibilityResult(
            **base, eligible=False, within_auto_limit=within_limit,
            policy_outcome=PolicyOutcome.REFUND_DENIED, reason_code="order_not_delivered",
            reason=f"Order status is {order.status.value}; refunds apply to delivered orders only",
        )
    days = (now - order.delivered_at).days
    base["days_since_delivery"] = days
    if days > policy.window_days:
        return EligibilityResult(
            **base, eligible=False, within_auto_limit=within_limit,
            policy_outcome=PolicyOutcome.REFUND_DENIED, reason_code="outside_return_window",
            reason=f"Delivered {days} days ago; the refund window is {policy.window_days} days",
        )
    if not within_limit:
        return EligibilityResult(
            **base, eligible=True, within_auto_limit=False,
            policy_outcome=PolicyOutcome.MANUAL_REVIEW_REQUIRED, reason_code="above_auto_limit",
            reason=f"Amount exceeds the automatic refund limit of {policy.max_auto_refund_amount:.2f}",
        )
    return EligibilityResult(
        **base, eligible=True, within_auto_limit=True,
        policy_outcome=PolicyOutcome.REFUND_ALLOWED, reason_code="eligible",
        reason=f"Delivered {days} days ago, within the {policy.window_days}-day window and the auto-refund limit",
    )
