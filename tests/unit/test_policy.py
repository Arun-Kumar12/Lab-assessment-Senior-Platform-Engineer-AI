from datetime import datetime, timedelta, timezone

import pytest

from app.data.models import Order, OrderStatus, RefundPolicy
from app.policy import PolicyOutcome, evaluate_refund

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)
POLICY = RefundPolicy(window_days=30, max_auto_refund_amount=200.0, eligible_statuses=[OrderStatus.DELIVERED])


def make_order(**overrides) -> Order:
    base = dict(order_id="ORD-1001", customer_email="alice@example.com", item="Headphones", amount=49.99, currency="USD",
                status=OrderStatus.DELIVERED, ordered_at=NOW - timedelta(days=15), delivered_at=NOW - timedelta(days=10),
                last_synced_at=NOW)
    return Order(**{**base, **overrides})


def test_eligible_order_is_refund_allowed():
    result = evaluate_refund(make_order(), "alice@example.com", POLICY, NOW)
    assert result.policy_outcome == PolicyOutcome.REFUND_ALLOWED
    assert result.eligible and result.within_auto_limit
    assert result.days_since_delivery == 10


def test_customer_mismatch_requires_manual_review():
    result = evaluate_refund(make_order(), "mallory@example.com", POLICY, NOW)
    assert result.policy_outcome == PolicyOutcome.MANUAL_REVIEW_REQUIRED
    assert result.reason_code == "customer_mismatch"


def test_email_comparison_is_case_insensitive():
    result = evaluate_refund(make_order(), "Alice@Example.com", POLICY, NOW)
    assert result.policy_outcome == PolicyOutcome.REFUND_ALLOWED


@pytest.mark.parametrize("status", [OrderStatus.SHIPPED, OrderStatus.PROCESSING, OrderStatus.CANCELLED])
def test_undelivered_order_is_denied(status):
    result = evaluate_refund(make_order(status=status, delivered_at=None), "alice@example.com", POLICY, NOW)
    assert result.policy_outcome == PolicyOutcome.REFUND_DENIED
    assert result.reason_code == "order_not_delivered"


def test_outside_window_is_denied():
    order = make_order(ordered_at=NOW - timedelta(days=40), delivered_at=NOW - timedelta(days=31))
    result = evaluate_refund(order, "alice@example.com", POLICY, NOW)
    assert result.policy_outcome == PolicyOutcome.REFUND_DENIED
    assert result.reason_code == "outside_return_window"


def test_window_boundary_is_inclusive():
    order = make_order(ordered_at=NOW - timedelta(days=40), delivered_at=NOW - timedelta(days=30))
    assert evaluate_refund(order, "alice@example.com", POLICY, NOW).policy_outcome == PolicyOutcome.REFUND_ALLOWED


def test_above_auto_limit_requires_manual_review():
    result = evaluate_refund(make_order(amount=450.0), "alice@example.com", POLICY, NOW)
    assert result.policy_outcome == PolicyOutcome.MANUAL_REVIEW_REQUIRED
    assert result.eligible and not result.within_auto_limit
    assert result.reason_code == "above_auto_limit"
