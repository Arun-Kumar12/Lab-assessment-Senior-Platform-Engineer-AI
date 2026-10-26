import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.llm import FakeLLM, build_llm
from app.agent.orchestrator import Orchestrator
from app.config import Settings
from app.data.repository import Repository
from app.logging_config import configure_logging
from app.state.store import RunStore
from evals.scoring import CRITERIA, CaseScore, EvalCase, aggregate, score_run

DATASET = Path(__file__).resolve().parent / "dataset.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "reports"


def load_dataset(path: Path = DATASET) -> tuple[list[EvalCase], dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(c) for c in raw["cases"]], raw["thresholds"]


def run_evals(settings: Settings, repeats: int = 1, cases: list[EvalCase] | None = None,
              thresholds: dict[str, float] | None = None, llm=None) -> dict[str, Any]:
    dataset_cases, dataset_thresholds = load_dataset()
    cases = cases or dataset_cases
    thresholds = thresholds or dataset_thresholds
    repo = Repository(settings.fixtures_dir)
    llm = llm or (FakeLLM() if settings.llm_provider == "fake" else build_llm(settings))
    orchestrator = Orchestrator(llm, repo, RunStore(), settings)
    results: list[CaseScore] = []
    for case in cases:
        ticket = repo.get_ticket(case.ticket_id)
        ticket_text = f"{ticket.subject} {ticket.body} {ticket.order_reference}"
        for repeat in range(1, repeats + 1):
            run = orchestrator.execute(orchestrator.start(case.ticket_id).run_id)
            result = score_run(case, run, repeat, ticket_text)
            results.append(result)
            print(f"{case.id} #{repeat}: {'PASS' if result.passed else 'FAIL'} "
                  f"{result.final_recommendation} tools={result.tool_calls} {result.duration_ms}ms {result.notes or ''}", flush=True)
    summary = aggregate(results, thresholds)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": llm.provider,
        "model": llm.model,
        "repeats": repeats,
        "thresholds": thresholds,
        "summary": summary,
        "results": [r.model_dump() for r in results],
    }


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "eval_report.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Evaluation report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Provider / model: `{report['provider']}` / `{report['model']}`",
        f"- Repeats per case: {report['repeats']}  |  Total runs: {summary['total_runs']}",
        f"- Overall pass rate: **{summary['pass_rate']:.0%}**  |  Thresholds met: **{'yes' if summary['passed'] else 'no'}**",
        f"- Policy-layer overrides of the model: {summary['policy_overrides']}  |  Model/policy agreement: {summary['model_policy_agreement']}",
        f"- Minimum per-case consistency across repeats: {summary['min_consistency']}",
        "",
        "## Scores by criterion",
        "",
        "| Criterion | Score | Threshold | Status |",
        "|---|---|---|---|",
    ]
    for criterion in CRITERIA:
        score = summary["by_criterion"].get(criterion, 0.0)
        threshold = report["thresholds"].get(criterion, 0.0)
        lines.append(f"| {criterion} | {score:.2f} | {threshold:.2f} | {'ok' if score >= threshold else 'BELOW'} |")
    lines += ["", "## Results by case", "", "| Case | Pass rate | Consistency | Recommendations | Avg duration |", "|---|---|---|---|---|"]
    for case_id, bucket in summary["by_case"].items():
        lines.append(f"| {case_id} | {bucket['pass_rate']:.0%} | {bucket['consistency']:.0%} | {', '.join(str(r) for r in bucket['recommendations'])} | {bucket['avg_duration_ms']} ms |")
    failures = [r for r in report["results"] if r["notes"]]
    if failures:
        lines += ["", "## Notes on imperfect runs", ""]
        for result in failures:
            lines.append(f"- `{result['case_id']}` #{result['repeat']}: " + "; ".join(result["notes"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the AI evaluation harness")
    parser.add_argument("--provider", choices=["fake", "ollama"], default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="append", help="run only the given case id (repeatable)")
    args = parser.parse_args(argv)

    overrides = {"llm_provider": args.provider} if args.provider else {}
    settings = Settings(**overrides)
    configure_logging("WARNING")
    cases, thresholds = load_dataset()
    if args.case:
        cases = [c for c in cases if c.id in args.case]
    report = run_evals(settings, repeats=args.repeats, cases=cases, thresholds=thresholds)
    write_reports(report, args.output)
    summary = report["summary"]
    print(f"\npass rate {summary['pass_rate']:.0%}; by criterion {summary['by_criterion']}")
    if not summary["passed"]:
        print(f"thresholds not met: {summary['failed_thresholds']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
