import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.data.models import Order, Ticket
from app.data.repository import FixtureError, Repository, format_amount
from tests.conftest import FIXTURES

REFERENCE = datetime(2026, 10, 20, tzinfo=timezone.utc)


def test_fixtures_load_and_cross_reference(repo):
    assert len(repo.orders) == 8 and len(repo.tickets) == 9
    referenced = {t.order_reference for t in repo.tickets.values()}
    assert referenced - set(repo.orders) == {"12345-X"}
    assert set(repo.faults) <= set(repo.orders)


def test_fixture_dates_are_timezone_aware_and_consistent(repo):
    for order in repo.orders.values():
        assert order.ordered_at.tzinfo is not None
        assert order.last_synced_at <= REFERENCE
        if order.delivered_at:
            assert order.ordered_at <= order.delivered_at <= REFERENCE


def test_scenario_coverage_in_fixtures(repo):
    stale = [o for o in repo.orders.values() if (REFERENCE - o.last_synced_at).total_seconds() > 24 * 3600]
    assert [o.order_id for o in stale] == ["ORD-1005"]
    assert repo.faults == {"ORD-1007": "timeout"}
    mismatched = [t.ticket_id for t in repo.tickets.values()
                  if t.order_reference in repo.orders and repo.orders[t.order_reference].customer_email != t.customer_email]
    assert mismatched == ["TCK-006"]


def test_ticket_view_transforms_order_fields(repo):
    view = repo.ticket_view(repo.get_ticket("TCK-003"))
    assert view.order_found is True
    assert view.order_summary == "Espresso Machine (delivered)"
    assert view.amount_display == "450.00 USD"


def test_ticket_view_for_missing_order(repo):
    view = repo.ticket_view(repo.get_ticket("TCK-009"))
    assert view.order_found is False
    assert view.order_summary is None and view.amount_display is None


@pytest.mark.parametrize("amount,currency,expected", [
    (49.99, "USD", "49.99 USD"),
    (1234.5, "EUR", "1,234.50 EUR"),
    (7, "GBP", "7.00 GBP"),
])
def test_amount_formatting(amount, currency, expected):
    assert format_amount(amount, currency) == expected


VALID_ORDER = {
    "order_id": "ORD-2001", "customer_email": "x@example.com", "item": "Thing", "amount": 10.0, "currency": "USD",
    "status": "delivered", "ordered_at": "2026-10-01T00:00:00Z", "delivered_at": "2026-10-03T00:00:00Z",
    "last_synced_at": "2026-10-19T00:00:00Z",
}


@pytest.mark.parametrize("field,value", [
    ("order_id", "ORDER-1"),
    ("customer_email", "not-an-email"),
    ("amount", -5),
    ("amount", 0),
    ("currency", "usd"),
    ("status", "returned"),
    ("item", ""),
    ("ordered_at", "yesterday"),
])
def test_malformed_order_records_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Order.model_validate({**VALID_ORDER, field: value})


def test_delivered_order_requires_delivery_timestamp():
    with pytest.raises(ValidationError, match="delivered_at"):
        Order.model_validate({**VALID_ORDER, "delivered_at": None})


def test_delivery_cannot_precede_order():
    with pytest.raises(ValidationError, match="precede"):
        Order.model_validate({**VALID_ORDER, "delivered_at": "2026-09-01T00:00:00Z"})


def test_ticket_body_length_is_bounded():
    with pytest.raises(ValidationError):
        Ticket(ticket_id="TCK-100", customer_email="a@b.co", order_reference="ORD-1001", subject="s",
               body="x" * 4001, created_at=REFERENCE)


def test_repository_rejects_corrupt_fixture(tmp_path):
    for name in ("orders.json", "tickets.json", "refund_policy.json", "faults.json"):
        (tmp_path / name).write_text((FIXTURES / name).read_text())
    bad = json.loads((tmp_path / "orders.json").read_text())
    bad[0]["amount"] = "free"
    (tmp_path / "orders.json").write_text(json.dumps(bad))
    with pytest.raises(FixtureError, match="orders.json"):
        Repository(tmp_path)


def test_repository_rejects_missing_fixture(tmp_path):
    with pytest.raises(FixtureError, match="cannot read"):
        Repository(tmp_path)
