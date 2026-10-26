import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from app.data.models import FaultType, Order, RefundPolicy, Ticket, TicketView


class FixtureError(RuntimeError):
    pass


class Repository:
    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = fixtures_dir
        self.orders = {o.order_id: o for o in self._load("orders.json", Order)}
        self.tickets = {t.ticket_id: t for t in self._load("tickets.json", Ticket)}
        self.policy = RefundPolicy.model_validate(self._read("refund_policy.json"))
        self.faults = TypeAdapter(dict[str, FaultType]).validate_python(self._read("faults.json"))

    def _read(self, name: str):
        path = self.fixtures_dir / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FixtureError(f"cannot read fixture {name}: {exc}") from exc

    def _load(self, name: str, model):
        try:
            return TypeAdapter(list[model]).validate_python(self._read(name))
        except ValidationError as exc:
            raise FixtureError(f"invalid records in {name}: {exc.errors()[0]['msg']}") from exc

    def get_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        return self.tickets.get(ticket_id)

    def list_tickets(self) -> list[Ticket]:
        return sorted(self.tickets.values(), key=lambda t: t.ticket_id)

    def fault_for(self, order_id: str) -> FaultType | None:
        return self.faults.get(order_id)

    def ticket_view(self, ticket: Ticket) -> TicketView:
        order = self.get_order(ticket.order_reference)
        return TicketView(
            **ticket.model_dump(),
            order_found=order is not None,
            order_summary=f"{order.item} ({order.status.value})" if order else None,
            amount_display=format_amount(order.amount, order.currency) if order else None,
        )


def format_amount(amount: float, currency: str) -> str:
    return f"{amount:,.2f} {currency}"
