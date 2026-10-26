import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from app.agent.guardrails import assess_request
from app.agent.llm import LLMClient, LLMResponseError, LLMUnavailableError
from app.agent.prompts import (DECISION_SCHEMA, FINAL_DECISION_PROMPT, NUDGE_PROMPT, REPLY_GUIDANCE, SYSTEM_PROMPT, Decision,
                               template_reply, ticket_message)
from app.data.repository import format_amount
from app.agent.tools import TOOLS, ToolContext, ToolError, execute_tool, issue_refund, send_reply
from app.config import Settings
from app.data.repository import Repository
from app.state.store import Proposal, Run, RunError, RunStatus, RunStore, OperatorDecision

log = logging.getLogger("agent")

POLICY_TO_RECOMMENDATION = {
    "refund_allowed": "approve_refund",
    "refund_denied": "deny_refund",
    "manual_review_required": "escalate",
}

UNSAFE_REPLY = (
    "Thanks for contacting us. We can only help with refund requests for your own order, "
    "and we cannot act on instructions to change how our systems work. "
    "If you would like help with a specific order, please reply with its order id."
)


class InvalidTransition(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, llm: LLMClient, repo: Repository, store: RunStore, settings: Settings):
        self.llm = llm
        self.repo = repo
        self.store = store
        self.settings = settings

    def start(self, ticket_id: str) -> Run:
        if self.repo.get_ticket(ticket_id) is None:
            raise KeyError(ticket_id)
        return self.store.create(ticket_id, self.llm.provider, self.llm.model)

    def execute(self, run_id: str) -> Run:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        ticket = self.repo.get_ticket(run.ticket_id)
        run.status = RunStatus.RUNNING
        self.store.save(run)
        extra = {"run_id": run.run_id, "ticket_id": run.ticket_id}
        log.info("run started", extra=extra)
        try:
            verdict = assess_request(f"{ticket.subject}\n{ticket.body}")
            run.add_step("guardrail", "assess_request", input=ticket.ticket_id, output=verdict.model_dump())
            if not verdict.safe:
                run.proposal = Proposal(recommendation="escalate", model_recommendation=None, policy_outcome=None,
                                        reason=f"Request blocked by safety guardrail: {', '.join(verdict.flags)}",
                                        customer_reply=UNSAFE_REPLY, reply_source="template", safety_flags=verdict.flags)
                log.warning("unsafe request blocked", extra={**extra, "step": "guardrail"})
            else:
                ctx = ToolContext(ticket=ticket, repo=self.repo, settings=self.settings)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ticket_message(ticket)}]
                tool_results, tool_errors = self._tool_loop(run, ctx, messages)
                verdict = self._policy_verdict(tool_results, tool_errors)
                decision = self._final_decision(run, messages, verdict, ticket.order_reference)
                run.proposal = self._enforce_policy(run, decision, verdict, tool_results, tool_errors, ticket.order_reference)
            run.status = RunStatus.AWAITING_APPROVAL
            log.info("proposal ready", extra={**extra, "step": "proposal"})
        except LLMUnavailableError as exc:
            self._fail(run, "llm_unavailable", str(exc))
        except (LLMResponseError, ValidationError) as exc:
            self._fail(run, "invalid_model_output", str(exc))
        except Exception as exc:
            log.exception("run crashed", extra=extra)
            self._fail(run, "internal_error", f"{type(exc).__name__}: {exc}")
        return self.store.save(run)

    def _tool_loop(self, run: Run, ctx: ToolContext, messages: list[dict[str, Any]]):
        tool_schemas = [spec.schema() for spec in TOOLS.values()]
        results: dict[str, Any] = {}
        errors: list[str] = []
        seen: set[str] = set()
        for _ in range(self.settings.max_agent_steps):
            result = self._chat(run, messages, tools=tool_schemas)
            if not result.tool_calls:
                if errors or "check_refund_eligibility" in results:
                    break
                messages.append({"role": "user", "content": NUDGE_PROMPT.format(order=ctx.ticket.order_reference)})
                continue
            for call in result.tool_calls:
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                if key in seen:
                    messages.append({"role": "tool", "tool_name": call.name,
                                     "content": json.dumps({"note": "already called with these arguments; use the earlier result"})})
                    continue
                seen.add(key)
                output = self._run_tool(run, ctx, call.name, call.arguments)
                if "error" in output:
                    if output["error"] not in errors:
                        errors.append(output["error"])
                else:
                    results[call.name] = output
                messages.append({"role": "tool", "tool_name": call.name, "content": json.dumps(output)})
            if errors or ("get_order" in results and "check_refund_eligibility" in results):
                break
        return results, errors

    def _run_tool(self, run: Run, ctx: ToolContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            output = execute_tool(name, arguments, ctx)
            error = None
        except ToolError as exc:
            output = exc.to_dict()
            error = exc.to_dict()
        duration = int((time.perf_counter() - started) * 1000)
        run.add_step("tool", name, input=arguments, output=output, error=error, duration_ms=duration)
        log.info("tool executed", extra={"run_id": run.run_id, "tool": name, "duration_ms": duration,
                                         "error_code": error["error"] if error else None})
        return output

    def _chat(self, run: Run, messages: list[dict[str, Any]], tools=None, response_schema=None):
        started = time.perf_counter()
        result = self.llm.chat(messages, tools=tools, response_schema=response_schema)
        duration = int((time.perf_counter() - started) * 1000)
        assistant: dict[str, Any] = {"role": "assistant", "content": result.content}
        if result.tool_calls:
            assistant["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}} for c in result.tool_calls]
        messages.append(assistant)
        run.add_step("llm", "chat", input={"messages": len(messages) - 1, "tools": bool(tools), "structured": bool(response_schema)},
                     output={"content": result.content[:500], "tool_calls": [c.name for c in result.tool_calls]},
                     duration_ms=duration)
        return result

    @staticmethod
    def _policy_verdict(tool_results: dict[str, Any], tool_errors: list[str]) -> dict[str, Any]:
        eligibility = tool_results.get("check_refund_eligibility") or {}
        policy_outcome = eligibility.get("policy_outcome")
        if tool_errors:
            expected, why = "escalate", f"tool errors occurred: {', '.join(tool_errors)}"
        elif policy_outcome in POLICY_TO_RECOMMENDATION:
            expected, why = POLICY_TO_RECOMMENDATION[policy_outcome], f"policy outcome is {policy_outcome}"
        else:
            expected, why = "escalate", "eligibility was never checked with a tool"
        return {"expected": expected, "why": why, "policy_outcome": policy_outcome, "eligibility": eligibility}

    def _final_decision(self, run: Run, messages: list[dict[str, Any]], verdict: dict[str, Any], order: str) -> Decision:
        tool_errors = [s.error["error"] for s in run.steps if s.kind == "tool" and s.error]
        evidence = [f"- tool errors: {', '.join(dict.fromkeys(tool_errors)) or 'none'}"]
        if verdict["eligibility"]:
            evidence.append(f"- policy_outcome: {verdict['policy_outcome']} ({verdict['eligibility'].get('reason')})")
        prompt = FINAL_DECISION_PROMPT.format(evidence="\n".join(evidence), expected=verdict["expected"], basis=verdict["why"],
                                              order=order, reply_guidance=REPLY_GUIDANCE[verdict["expected"]])
        messages.append({"role": "user", "content": prompt})
        last_error = "no attempts"
        for attempt in range(self.settings.max_decision_retries + 1):
            result = self._chat(run, messages, response_schema=DECISION_SCHEMA)
            try:
                return Decision.model_validate_json(_extract_json(result.content))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc).splitlines()[0]
                run.add_step("error", "decision_parse", input=attempt, error={"error": "invalid_model_output", "message": last_error})
                messages.append({"role": "user", "content": f"Your previous answer was not valid JSON matching the schema ({last_error}). Reply with JSON only."})
        raise LLMResponseError(f"model did not produce a valid decision after {self.settings.max_decision_retries + 1} attempts: {last_error}")

    def _enforce_policy(self, run: Run, decision: Decision, verdict: dict[str, Any], tool_results: dict[str, Any],
                        tool_errors: list[str], order: str) -> Proposal:
        expected, why, policy_outcome = verdict["expected"], verdict["why"], verdict["policy_outcome"]
        overridden = decision.recommendation != expected
        run.add_step("policy", "enforce_recommendation",
                     input={"model_recommendation": decision.recommendation, "policy_outcome": policy_outcome, "tool_errors": tool_errors},
                     output={"final_recommendation": expected, "overridden": overridden, "basis": why})
        if overridden:
            log.warning("model recommendation overridden", extra={"run_id": run.run_id, "step": "policy"})
        reply = decision.customer_reply
        if order not in reply and self.repo.get_order(order) is not None:
            reply = f"Regarding order {order}: {reply}"
        if overridden:
            eligibility = verdict["eligibility"]
            amount = format_amount(eligibility["amount"], eligibility["currency"]) if eligibility else None
            reply = template_reply(expected, order, amount=amount, reason=eligibility.get("reason") if eligibility else None)
        return Proposal(
            recommendation=expected,
            reason=decision.reason if not overridden else f"{decision.reason} [Overridden to {expected}: {why}]",
            customer_reply=reply,
            reply_source="template" if overridden else "model",
            model_recommendation=decision.recommendation,
            policy_outcome=policy_outcome,
            overridden=overridden,
            override_reason=why if overridden else None,
            tool_errors=tool_errors,
        )

    def decide(self, run_id: str, decision: str, note: str | None, actor: str) -> Run:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if run.status != RunStatus.AWAITING_APPROVAL or run.proposal is None:
            raise InvalidTransition(f"run is {run.status.value}, not awaiting approval")
        run.decision = OperatorDecision(decision=decision, note=note, decided_by=actor)
        ticket = self.repo.get_ticket(run.ticket_id)
        if decision == "approve":
            if run.proposal.recommendation == "approve_refund":
                order = self.repo.get_order(ticket.order_reference)
                receipt = issue_refund(order.order_id, order.amount, order.currency)
                run.actions.append({"action": "issue_refund", **receipt})
                run.add_step("action", "issue_refund", input=order.order_id, output=receipt)
            if run.proposal.recommendation == "escalate":
                handoff = {"queue": "refund-specialists", "ticket_id": ticket.ticket_id, "status": "queued"}
                run.actions.append({"action": "handoff_to_specialist", **handoff})
                run.add_step("action", "handoff_to_specialist", input=ticket.ticket_id, output=handoff)
            receipt = send_reply(ticket.ticket_id, ticket.customer_email, run.proposal.customer_reply)
            run.actions.append({"action": "send_reply", **receipt})
            run.add_step("action", "send_reply", input=ticket.customer_email, output=receipt)
            run.status = RunStatus.COMPLETED
        else:
            run.add_step("action", "rejected_by_operator", input=note)
            run.status = RunStatus.REJECTED
        log.info("operator decision applied", extra={"run_id": run.run_id, "step": decision})
        return self.store.save(run)

    def _fail(self, run: Run, code: str, message: str) -> None:
        run.status = RunStatus.FAILED
        run.error = RunError(code=code, message=message)
        run.add_step("error", code, error={"error": code, "message": message})
        log.error("run failed", extra={"run_id": run.run_id, "error_code": code})


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{"):
        return text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return text[start:end + 1]
