import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.config import Settings


class LLMUnavailableError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    provider: str
    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             response_schema: dict[str, Any] | None = None) -> ChatResult: ...

    def healthy(self) -> bool: ...


class OllamaClient:
    provider = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: float):
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def chat(self, messages, tools=None, response_schema=None) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if tools:
            payload["tools"] = tools
        if response_schema:
            payload["format"] = response_schema
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
        message = body.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError("Ollama response has no message")
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"_raw": arguments}
            calls.append(ToolCall(name=str(function.get("name", "")), arguments=arguments))
        return ChatResult(content=message.get("content") or "", tool_calls=calls)

    def healthy(self) -> bool:
        try:
            response = self._client.get("/api/tags", timeout=3.0)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return False
        return any(n == self.model or n.split(":")[0] == self.model for n in names)


class FakeLLM:
    """Deterministic stand-in used by tests and the evaluation harness.

    Behaves like a well-behaved agent by default; ``mode`` injects specific misbehaviours.
    """

    provider = "fake"
    model = "scripted-v1"

    def __init__(self, mode: str = "good"):
        self.mode = mode
        self.calls = 0
        self.final_attempts = 0

    def chat(self, messages, tools=None, response_schema=None) -> ChatResult:
        self.calls += 1
        if self.mode == "unavailable":
            raise LLMUnavailableError("fake provider is offline")
        order_ref = self._order_reference(messages)
        results = self._tool_results(messages)
        if response_schema is not None:
            return self._final(results, order_ref)
        if self.mode == "skip_tools":
            return ChatResult(content="I will decide without looking anything up.")
        if self.mode == "lazy_once" and self.calls == 1:
            return ChatResult(content="Let me think about this ticket first.")
        if self.mode == "unknown_tool" and self.calls == 1:
            return ChatResult(content="", tool_calls=[ToolCall("delete_order", {"order_id": order_ref})])
        if "get_order" not in results:
            return ChatResult(content="", tool_calls=[ToolCall("get_order", {"order_id": order_ref})])
        if "check_refund_eligibility" not in results:
            return ChatResult(content="", tool_calls=[ToolCall("check_refund_eligibility", {"order_id": order_ref})])
        return ChatResult(content="I have the order and eligibility details.")

    def healthy(self) -> bool:
        return self.mode != "unavailable"

    def _final(self, results, order_ref) -> ChatResult:
        self.final_attempts += 1
        if self.mode == "malformed_once" and self.final_attempts == 1:
            return ChatResult(content="Sure! Here is my decision: approve it.")
        if self.mode == "skip_tools":
            decision = {"recommendation": "approve_refund", "reason": "Customer sounds unhappy.",
                        "customer_reply": f"We have refunded order {order_ref}."}
            return ChatResult(content=json.dumps(decision))
        errors = [r for r in results.values() if isinstance(r, dict) and "error" in r]
        eligibility = results.get("check_refund_eligibility") or {}
        order = results.get("get_order") or {}
        item = order.get("item", "your item")
        if errors:
            decision = {
                "recommendation": "escalate",
                "reason": f"Tool failure ({errors[0]['error']}): {errors[0]['message']}",
                "customer_reply": f"Thanks for contacting us about order {order_ref}. We could not verify the order details automatically, so a specialist will follow up with you shortly.",
            }
        elif eligibility.get("policy_outcome") == "refund_allowed":
            decision = {
                "recommendation": "approve_refund",
                "reason": f"Order delivered {eligibility.get('days_since_delivery')} days ago and within the auto-refund limit.",
                "customer_reply": f"Thanks for reaching out about order {order_ref}. Your refund of {eligibility.get('amount')} {eligibility.get('currency')} for the {item} has been approved and will be processed within 5 business days.",
            }
        elif eligibility.get("policy_outcome") == "refund_denied":
            decision = {
                "recommendation": "deny_refund",
                "reason": eligibility.get("reason", "Policy denies this refund."),
                "customer_reply": f"Thanks for reaching out about order {order_ref}. Unfortunately this order is not eligible for a refund: {eligibility.get('reason')}.",
            }
        else:
            decision = {
                "recommendation": "escalate",
                "reason": eligibility.get("reason", "Manual review required."),
                "customer_reply": f"Thanks for contacting us about order {order_ref}. Your request needs a manual review and a specialist will follow up shortly.",
            }
        if self.mode == "terse_reply":
            decision["customer_reply"] = "Thanks for reaching out. A specialist will follow up shortly."
        return ChatResult(content=json.dumps(decision))

    @staticmethod
    def _order_reference(messages) -> str:
        for message in messages:
            if message.get("role") == "user":
                match = re.search(r"Order reference: (\S+)", message.get("content", ""))
                if match:
                    return match.group(1)
        return "ORD-0000"

    @staticmethod
    def _tool_results(messages) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for message in messages:
            if message.get("role") == "tool":
                try:
                    results[message.get("tool_name", "")] = json.loads(message.get("content", "{}"))
                except ValueError:
                    results[message.get("tool_name", "")] = {"error": "unparseable", "message": "bad tool payload"}
        return results


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLM()
    return OllamaClient(settings.ollama_base_url, settings.ollama_model, settings.ollama_timeout_seconds)
