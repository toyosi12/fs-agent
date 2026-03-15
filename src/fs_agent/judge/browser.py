"""Playwright-based browser interaction for frontend evaluation.

Provides headless Chromium screenshot capture of running frontend
applications.  Used by the runtime evaluation pipeline to feed
GPT-4o vision for appearance and frontend functional scoring.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class Screenshot:
    """A captured screenshot with metadata."""

    png_bytes: bytes
    url: str
    label: str = ""

    @property
    def base64(self) -> str:
        """Return the screenshot as a base64-encoded PNG string."""
        return base64.b64encode(self.png_bytes).decode("ascii")


@dataclass
class BrowserResult:
    """Aggregate result from a browser session."""

    screenshots: list[Screenshot] = field(default_factory=list)
    error: str | None = None


def capture_screenshots(
    url: str,
    *,
    wait_ms: int = 3000,
    full_page: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    extra_urls: list[str] | None = None,
) -> BrowserResult:
    """Launch headless Chromium, navigate to *url*, and take screenshots.

    Parameters
    ----------
    url:
        The base URL of the running frontend (e.g. ``http://localhost:3000``).
    wait_ms:
        Milliseconds to wait after navigation for JS rendering.
    full_page:
        If True, capture the full scrollable page; otherwise viewport only.
    viewport_width / viewport_height:
        Browser viewport dimensions.
    extra_urls:
        Additional URLs to navigate and screenshot (e.g. sub-routes).

    Returns
    -------
    BrowserResult with captured screenshots (may be empty on failure).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return BrowserResult(error="playwright is not installed")

    result = BrowserResult()
    urls = [url] + (extra_urls or [])

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": viewport_width, "height": viewport_height},
            )
            page = context.new_page()

            for i, target_url in enumerate(urls):
                try:
                    page.goto(target_url, wait_until="networkidle", timeout=15000)
                    page.wait_for_timeout(wait_ms)

                    png = page.screenshot(full_page=full_page)
                    label = "main" if i == 0 else f"page_{i}"
                    result.screenshots.append(Screenshot(
                        png_bytes=png,
                        url=target_url,
                        label=label,
                    ))
                    logger.info("Captured screenshot for %s (%d bytes)", target_url, len(png))
                except Exception as exc:
                    logger.warning("Failed to screenshot %s: %s", target_url, exc)

            browser.close()
    except Exception as exc:
        result.error = str(exc)
        logger.warning("Browser session failed: %s", exc)

    return result
