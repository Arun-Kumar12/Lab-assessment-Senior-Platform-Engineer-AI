import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import expect, sync_playwright

SCREENSHOTS = Path(__file__).resolve().parents[2] / "reports" / "screenshots"
pytestmark = pytest.mark.browser


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    env = {**os.environ, "LLM_PROVIDER": "fake", "TOOL_TIMEOUT_SECONDS": "0.2", "LOG_LEVEL": "WARNING"}
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port), "--log-level", "warning"],
                               env=env, cwd=str(Path(__file__).resolve().parents[2]))
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("app did not start")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture(scope="module")
def page(base_url):
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        yield page
        browser.close()


def test_operator_triages_and_approves_refund(page, base_url):
    page.goto(base_url)
    expect(page.get_by_test_id("ticket-TCK-001")).to_be_visible()
    page.get_by_test_id("ticket-TCK-001").click()
    expect(page.get_by_test_id("ticket-title")).to_contain_text("Refund for headphones")
    page.screenshot(path=str(SCREENSHOTS / "01-ticket-selected.png"))

    page.get_by_test_id("run-button").click()
    expect(page.get_by_test_id("run-status")).to_have_text("awaiting_approval", timeout=15000)
    expect(page.get_by_test_id("proposal-recommendation")).to_have_text("approve_refund")
    expect(page.get_by_test_id("proposal-reply")).to_contain_text("ORD-1001")
    expect(page.get_by_test_id("steps")).to_contain_text("check_refund_eligibility")
    page.screenshot(path=str(SCREENSHOTS / "02-proposal-awaiting-approval.png"), full_page=True)

    page.get_by_test_id("note").fill("Verified with the customer")
    page.get_by_test_id("approve-button").click()
    expect(page.get_by_test_id("run-status")).to_have_text("completed")
    expect(page.get_by_test_id("actions")).to_contain_text("issue_refund")
    expect(page.get_by_test_id("decision")).to_contain_text("Verified with the customer")
    page.screenshot(path=str(SCREENSHOTS / "03-refund-executed.png"), full_page=True)


def test_tool_failure_is_surfaced_in_ui(page, base_url):
    page.goto(base_url)
    page.get_by_test_id("ticket-TCK-007").click()
    page.get_by_test_id("run-button").click()
    expect(page.get_by_test_id("run-status")).to_have_text("awaiting_approval", timeout=15000)
    expect(page.get_by_test_id("proposal-recommendation")).to_have_text("escalate")
    expect(page.get_by_test_id("tool-errors")).to_contain_text("timeout")
    page.screenshot(path=str(SCREENSHOTS / "04-tool-timeout-escalated.png"), full_page=True)


def test_viewer_key_cannot_run_agent(page, base_url):
    page.goto(base_url)
    page.get_by_test_id("api-key").fill("viewer-key")
    page.get_by_test_id("api-key").dispatch_event("change")
    page.get_by_test_id("ticket-TCK-002").click()
    page.get_by_test_id("run-button").click()
    expect(page.get_by_test_id("error")).to_contain_text("operator role required")
    page.screenshot(path=str(SCREENSHOTS / "05-viewer-forbidden.png"))
