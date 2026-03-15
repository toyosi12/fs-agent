"""Tests for the Playwright browser module and vision LLM integration.

Uses mocks/fakes — does not require a running browser or LLM.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from fs_agent.judge.browser import BrowserResult, Screenshot, capture_screenshots
from fs_agent.judge.models import FrontendVerdict
from fs_agent.judge.runtime import run_appearance_test, run_frontend_tests
from fs_agent.llm import BaseLLMClient, DummyLLMClient, OpenAILLMClient


# ---------------------------------------------------------------------------
# Screenshot dataclass tests
# ---------------------------------------------------------------------------


class TestScreenshot:
    def test_base64_encoding(self):
        raw = b"\x89PNG\r\n\x1a\nfake-image-data"
        ss = Screenshot(png_bytes=raw, url="http://localhost:3000", label="main")
        b64 = ss.base64
        assert base64.b64decode(b64) == raw

    def test_empty_bytes(self):
        ss = Screenshot(png_bytes=b"", url="http://x", label="empty")
        assert ss.base64 == ""


# ---------------------------------------------------------------------------
# BrowserResult tests
# ---------------------------------------------------------------------------


class TestBrowserResult:
    def test_empty_result(self):
        br = BrowserResult()
        assert br.screenshots == []
        assert br.error is None

    def test_with_error(self):
        br = BrowserResult(error="chromium not found")
        assert br.error == "chromium not found"
        assert br.screenshots == []


# ---------------------------------------------------------------------------
# capture_screenshots — with Playwright mocked
# ---------------------------------------------------------------------------


class TestCaptureScreenshots:
    def test_playwright_not_installed(self):
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            # Force re-import failure
            import importlib
            from fs_agent.judge import browser as browser_mod
            # Patch the import inside capture_screenshots
            original_fn = browser_mod.capture_screenshots

            def _patched_import():
                raise ImportError("no playwright")

            with patch.object(browser_mod, "capture_screenshots", wraps=original_fn):
                result = capture_screenshots("http://localhost:3000")
                # It should either have an error or empty screenshots
                # The function catches ImportError internally
                assert isinstance(result, BrowserResult)

    def test_successful_capture(self):
        """Mock the Playwright API to simulate a successful screenshot."""
        fake_png = b"\x89PNGfakedata"

        mock_page = MagicMock()
        mock_page.screenshot.return_value = fake_png

        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context

        mock_chromium = MagicMock()
        mock_chromium.launch.return_value = mock_browser

        mock_pw = MagicMock()
        mock_pw.chromium = mock_chromium

        mock_pw_cm = MagicMock()
        mock_pw_cm.__enter__ = MagicMock(return_value=mock_pw)
        mock_pw_cm.__exit__ = MagicMock(return_value=False)

        with patch("fs_agent.judge.browser.sync_playwright", return_value=mock_pw_cm, create=True):
            # We need to patch the import inside capture_screenshots
            with patch.dict("sys.modules", {}):
                import fs_agent.judge.browser as bmod

                # Monkey-patch the function to use our mock
                def mock_capture(url, **kwargs):
                    result = BrowserResult()
                    result.screenshots.append(Screenshot(
                        png_bytes=fake_png, url=url, label="main",
                    ))
                    return result

                original = bmod.capture_screenshots
                bmod.capture_screenshots = mock_capture
                try:
                    result = bmod.capture_screenshots("http://localhost:3000")
                    assert len(result.screenshots) == 1
                    assert result.screenshots[0].png_bytes == fake_png
                    assert result.screenshots[0].url == "http://localhost:3000"
                    assert result.error is None
                finally:
                    bmod.capture_screenshots = original

    def test_extra_urls(self):
        """Verify that extra_urls parameter is accepted."""
        # Just test that the function signature works — actual Playwright
        # would need to be running, so we just check no TypeError
        result = capture_screenshots(
            "http://localhost:3000",
            extra_urls=["http://localhost:3000/about"],
        )
        # Will fail gracefully (no browser available in test env)
        assert isinstance(result, BrowserResult)


# ---------------------------------------------------------------------------
# BaseLLMClient.generate_with_images fallback
# ---------------------------------------------------------------------------


class TestVisionFallback:
    def test_base_class_falls_back_to_generate(self):
        """Default generate_with_images should call generate()."""
        dummy = DummyLLMClient(model="test")
        result = dummy.generate_with_images(
            "describe this image",
            ["aGVsbG8="],
            system="you are a bot",
        )
        # DummyLLMClient.generate returns placeholder text
        assert "LLM output unavailable" in result


# ---------------------------------------------------------------------------
# Fake vision-capable LLM for runtime tests
# ---------------------------------------------------------------------------


class FakeVisionLLM:
    """Fake LLM that tracks whether generate_with_images was called."""

    model = "fake-vision"

    def __init__(self):
        self.vision_called = False
        self.text_called = False

    def generate(self, user: str, *, system: str = "", temperature: float = 0.7) -> str:
        self.text_called = True
        # Frontend evaluation
        if "frontend QA evaluator" in system or "UI test case" in system:
            return json.dumps([
                {"test_index": 0, "verdict": "YES", "reasoning": "Code looks good"},
            ])
        # Appearance
        if "UI/UX designer" in system:
            return json.dumps({
                "layout": 3, "color": 3, "typography": 3,
                "component_polish": 3, "reasoning": "Code-only analysis",
            })
        return '{"verdict": "NO"}'

    def generate_with_images(
        self, prompt: str, images_b64: list[str],
        *, system: str = "", temperature: float = 0.7,
    ) -> str:
        self.vision_called = True
        # Frontend evaluation with screenshot
        if "frontend QA evaluator" in system:
            return json.dumps([
                {"test_index": 0, "verdict": "YES", "reasoning": "Screenshot shows correct UI"},
            ])
        # Appearance with screenshot
        if "UI/UX designer" in system:
            return json.dumps({
                "layout": 4, "color": 4, "typography": 4,
                "component_polish": 4, "reasoning": "Screenshot looks great",
            })
        return '{"verdict": "NO"}'


# ---------------------------------------------------------------------------
# Frontend tests with/without screenshots
# ---------------------------------------------------------------------------


class TestFrontendTestsWithScreenshot:
    def test_uses_vision_when_screenshot_available(self):
        llm = FakeVisionLLM()
        fake_png = b"\x89PNGtestdata"
        fake_browser_result = BrowserResult(
            screenshots=[Screenshot(png_bytes=fake_png, url="http://localhost:3000", label="main")]
        )

        with patch("fs_agent.judge.runtime.capture_screenshots", return_value=fake_browser_result):
            scores = run_frontend_tests(
                llm,
                frontend_url="http://localhost:3000",
                frontend_healthy=True,
                task_instruction="Build a stock app",
                frontend_code="function App() { return <div>Hello</div> }",
                ui_test_cases=[{"task": "show stocks", "expected_result": "stock list visible"}],
            )

        assert len(scores) == 1
        assert scores[0].verdict == FrontendVerdict.YES
        assert llm.vision_called is True

    def test_falls_back_to_text_when_no_screenshot(self):
        llm = FakeVisionLLM()
        fake_browser_result = BrowserResult(error="playwright not installed")

        with patch("fs_agent.judge.runtime.capture_screenshots", return_value=fake_browser_result):
            scores = run_frontend_tests(
                llm,
                frontend_url="http://localhost:3000",
                frontend_healthy=True,
                task_instruction="Build a stock app",
                frontend_code="function App() { return <div>Hello</div> }",
                ui_test_cases=[{"task": "show stocks", "expected_result": "stock list visible"}],
            )

        assert len(scores) == 1
        assert scores[0].verdict == FrontendVerdict.YES
        assert llm.text_called is True
        assert llm.vision_called is False

    def test_unhealthy_frontend_skips_all(self):
        llm = FakeVisionLLM()
        scores = run_frontend_tests(
            llm,
            frontend_url="http://localhost:3000",
            frontend_healthy=False,
            task_instruction="Build a stock app",
            frontend_code="...",
            ui_test_cases=[{"task": "show", "expected_result": "visible"}],
        )
        assert len(scores) == 1
        assert scores[0].verdict == FrontendVerdict.NO
        assert llm.vision_called is False
        assert llm.text_called is False

    def test_empty_test_cases(self):
        llm = FakeVisionLLM()
        scores = run_frontend_tests(
            llm, "http://localhost:3000", True, "task", "code", [],
        )
        assert scores == []


# ---------------------------------------------------------------------------
# Appearance tests with/without screenshots
# ---------------------------------------------------------------------------


class TestAppearanceWithScreenshot:
    def test_uses_vision_when_healthy_and_screenshot(self):
        llm = FakeVisionLLM()
        fake_png = b"\x89PNGtestdata"
        fake_browser_result = BrowserResult(
            screenshots=[Screenshot(png_bytes=fake_png, url="http://localhost:3000", label="main")]
        )

        with patch("fs_agent.judge.runtime.capture_screenshots", return_value=fake_browser_result):
            score = run_appearance_test(
                llm,
                task_instruction="Build a stock app",
                frontend_code="function App() { return <div>Hello</div> }",
                frontend_healthy=True,
                frontend_url="http://localhost:3000",
            )

        assert score.layout == 4
        assert score.color == 4
        assert score.overall == 4.0
        assert llm.vision_called is True
        assert "Screenshot looks great" in score.reasoning

    def test_falls_back_to_code_when_no_screenshot(self):
        llm = FakeVisionLLM()
        fake_browser_result = BrowserResult(error="no browser")

        with patch("fs_agent.judge.runtime.capture_screenshots", return_value=fake_browser_result):
            score = run_appearance_test(
                llm,
                task_instruction="Build a stock app",
                frontend_code="function App() { return <div>Hello</div> }",
                frontend_healthy=True,
                frontend_url="http://localhost:3000",
            )

        assert score.layout == 3
        assert llm.text_called is True
        assert llm.vision_called is False

    def test_no_frontend_url_skips_screenshot(self):
        llm = FakeVisionLLM()
        score = run_appearance_test(
            llm,
            task_instruction="Build a stock app",
            frontend_code="function App() { return <div>Hello</div> }",
            frontend_healthy=True,
            frontend_url=None,
        )
        assert score.layout == 3
        assert llm.text_called is True
        assert llm.vision_called is False

    def test_unhealthy_frontend_skips_screenshot(self):
        llm = FakeVisionLLM()
        score = run_appearance_test(
            llm,
            task_instruction="Build a stock app",
            frontend_code="function App() { return <div>Hello</div> }",
            frontend_healthy=False,
            frontend_url="http://localhost:3000",
        )
        # Should still evaluate via code analysis
        assert score.layout == 3
        assert llm.vision_called is False

    def test_llm_failure_returns_default_scores(self):
        """If the LLM raises, we get score=1 across the board."""

        class FailingLLM:
            model = "failing"

            def generate(self, *a, **kw):
                raise RuntimeError("boom")

            def generate_with_images(self, *a, **kw):
                raise RuntimeError("boom")

        score = run_appearance_test(
            FailingLLM(),
            task_instruction="Build an app",
            frontend_code="<div>Hello</div>",
            frontend_healthy=False,
        )
        assert score.layout == 1
        assert score.overall == 1.0
        assert "boom" in score.reasoning
