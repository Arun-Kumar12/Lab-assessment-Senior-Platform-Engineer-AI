from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

OrderId = Annotated[str, Field(pattern=r"^ORD-\d{4}$")]
TicketId = Annotated[str, Field(pattern=r"^TCK-\d{3}$")]
Email = Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Order(BaseModel):
    order_id: OrderId
    customer_email: Email
    item: str = Field(min_length=1)
    amount: float = Field(gt=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    status: OrderStatus
    ordered_at: datetime
    delivered_at: datetime | None = None
    last_synced_at: datetime

    @model_validator(mode="after")
    def delivered_requires_timestamp(self) -> "Order":
        if self.status == OrderStatus.DELIVERED and self.delivered_at is None:
            raise ValueError("delivered orders must have delivered_at")
        if self.delivered_at and self.delivered_at < self.ordered_at:
            raise ValueError("delivered_at cannot precede ordered_at")
        return self


class Ticket(BaseModel):
    ticket_id: TicketId
    customer_email: Email
    order_reference: str = Field(min_length=1)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)
    created_at: datetime


class RefundPolicy(BaseModel):
    window_days: int = Field(gt=0)
    max_auto_refund_amount: float = Field(gt=0)
    eligible_statuses: list[OrderStatus] = Field(min_length=1)


class FaultType(StrEnum):
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"


class TicketView(BaseModel):
    ticket_id: str
    customer_email: str
    order_reference: str
    subject: str
    body: str
    created_at: datetime
    order_found: bool
    order_summary: str | None
    amount_display: str | None
