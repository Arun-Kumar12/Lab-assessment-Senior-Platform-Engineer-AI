import os

import pytest

from app.agent.llm import FakeLLM
from app.config import Settings
from evals.runner import load_dataset, run_evals, write_reports
from evals.scoring import CRITERIA


def test_dataset_covers_every_evaluation_dimension():
    cases, thresholds = load_dataset()
    assert set(thresholds) == set(CRITERIA)
    categories = {c.category for c in cases}
    assert {"happy_path", "policy_denial", "manual_review", "injected_failure", "authorization", "unsafe_request"} <= categories
    assert any(c.unsafe for c in cases)
    assert {c.expected_error for c in cases if c.expected_error} == {"stale_data", "timeout", "invalid_argument"}


def test_harness_passes_with_reference_agent(settings, tmp_path):
    report = run_evals(settings, repeats=2)
    summary = report["summary"]
    assert summary["passed"], summary["failed_thresholds"]
    assert summary["pass_rate"] == 1.0
    assert summary["min_consistency"] == 1.0
    assert summary["policy_overrides"] == 0
    write_reports(report, tmp_path)
    assert (tmp_path / "eval_report.json").exists()
    markdown = (tmp_path / "eval_report.md").read_text()
    assert "Scores by criterion" in markdown and "BELOW" not in markdown


def test_harness_detects_ungrounded_agent(settings):
    report = run_evals(settings, repeats=1, llm=FakeLLM("skip_tools"))
    summary = report["summary"]
    assert not summary["passed"]
    assert summary["by_criterion"]["tool_accuracy"] < 0.5
    assert summary["policy_overrides"] > 0
    assert summary["by_criterion"]["safety"] == 1.0


def test_harness_detects_model_outage(settings):
    report = run_evals(settings, repeats=1, llm=FakeLLM("unavailable"))
    summary = report["summary"]
    assert summary["by_criterion"]["failure_handling"] < 1.0
    assert summary["by_criterion"]["task_completion"] < 0.5


@pytest.mark.live_llm
@pytest.mark.skipif(not os.environ.get("RUN_LIVE_EVALS"), reason="set RUN_LIVE_EVALS=1 with a reachable Ollama to run")
def test_live_model_meets_thresholds():
    settings = Settings(llm_provider="ollama")
    report = run_evals(settings, repeats=int(os.environ.get("EVAL_REPEATS", "1")))
    assert report["summary"]["passed"], report["summary"]["failed_thresholds"]
