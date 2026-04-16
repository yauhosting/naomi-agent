"""
NAOMI Agent - Headless Browser Automation

Provides async browser automation via Playwright for web scraping,
form filling, and structured data extraction. Uses headless Chromium
with anti-detection measures.
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger("naomi.browser")

# Defaults
DEFAULT_TIMEOUT = 30_000  # Playwright uses milliseconds
DEFAULT_VIEWPORT = {"width": 1440, "height": 900}
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
SEARCH_URL = "https://www.google.com/search?q={query}&hl=en"


@dataclass(frozen=True)
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str


class BrowserError(Exception):
    """Base exception for browser operations."""


class BrowserAgent:
    """
    Async headless browser automation using Playwright.

    The browser instance is lazily initialized on first use and
    reused across calls. Call close() to release resources.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = DEFAULT_TIMEOUT,
        viewport: Optional[Dict[str, int]] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self._headless = headless
        self._timeout = timeout
        self._viewport = viewport or DEFAULT_VIEWPORT
        self._user_agent = user_agent or DEFAULT_USER_AGENT

        # Lazy-init state
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None

    # -- Lifecycle --

    async def _ensure_browser(self) -> None:
        """Lazily start browser on first use."""
        if self._page is not None:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise BrowserError(
                "playwright is not installed. Run: pip install playwright && python -m playwright install chromium"
            )

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        self._context = await self._browser.new_context(
            viewport=self._viewport,
            user_agent=self._user_agent,
            locale="en-US",
            timezone_id="Asia/Taipei",
            # Anti-detection
            java_script_enabled=True,
            bypass_csp=False,
        )
        self._context.set_default_timeout(self._timeout)

        # Inject anti-detection script into every new page
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        """)

        self._page = await self._context.new_page()
        logger.info("Browser initialized (headless=%s)", self._headless)

    async def close(self) -> None:
        """Release all browser resources."""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as exc:
                logger.debug("Error closing browser: %s", exc)
            self._browser = None

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:
                logger.debug("Error stopping playwright: %s", exc)
            self._playwright = None

        self._context = None
        self._page = None
        logger.info("Browser closed")

    # -- Navigation --

    async def navigate(self, url: str) -> Dict[str, Any]:
        """
        Navigate to a URL. Returns page title and a text preview.
        """
        await self._ensure_browser()
        try:
            response = await self._page.goto(url, wait_until="domcontentloaded")
            status = response.status if response else 0
            title = await self._page.title()

            # Extract a text preview (first ~500 chars of visible text)
            preview = await self._page.evaluate("""
                () => {
                    const body = document.body;
                    if (!body) return '';
                    const walker = document.createTreeWalker(
                        body, NodeFilter.SHOW_TEXT, null
                    );
                    let text = '';
                    let node;
                    while ((node = walker.nextNode()) && text.length < 600) {
                        const t = node.textContent.trim();
                        if (t.length > 1) text += t + ' ';
                    }
                    return text.substring(0, 500);
                }
            """)

            return {
                "success": True,
                "url": self._page.url,
                "title": title,
                "status": status,
                "preview": preview.strip(),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "url": url}

    # -- Interaction --

    async def click(self, selector: str) -> Dict[str, Any]:
        """Click an element by CSS selector."""
        await self._ensure_browser()
        try:
            await self._page.click(selector)
            return {"success": True, "selector": selector}
        except Exception as exc:
            return {"success": False, "error": str(exc), "selector": selector}

    async def fill(self, selector: str, value: str) -> Dict[str, Any]:
        """Fill an input field."""
        await self._ensure_browser()
        try:
            await self._page.fill(selector, value)
            return {"success": True, "selector": selector}
        except Exception as exc:
            return {"success": False, "error": str(exc), "selector": selector}

    # -- Extraction --

    async def extract_text(self) -> Dict[str, Any]:
        """Get the full visible text content of the page."""
        await self._ensure_browser()
        try:
            text = await self._page.evaluate("""
                () => {
                    // Remove script, style, nav, footer elements
                    const remove = ['script', 'style', 'nav', 'footer', 'header', 'noscript'];
                    const clone = document.body.cloneNode(true);
                    remove.forEach(tag => {
                        clone.querySelectorAll(tag).forEach(el => el.remove());
                    });
                    return clone.innerText || clone.textContent || '';
                }
            """)
            return {
                "success": True,
                "text": text.strip()[:10000],
                "length": len(text.strip()),
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def extract_links(self) -> Dict[str, Any]:
        """Get all links on the current page."""
        await self._ensure_browser()
        try:
            links = await self._page.evaluate("""
                () => {
                    return Array.from(document.querySelectorAll('a[href]'))
                        .map(a => ({
                            text: (a.textContent || '').trim().substring(0, 100),
                            href: a.href,
                        }))
                        .filter(l => l.href && l.href.startsWith('http'))
                        .slice(0, 200);
                }
            """)
            return {
                "success": True,
                "links": links,
                "count": len(links),
                "url": self._page.url,
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "links": []}

    # -- Screenshot --

    async def screenshot(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Take a screenshot. Returns the file path."""
        await self._ensure_browser()
        if path is None:
            path = f"/tmp/naomi_browser_{int(time.time())}.png"

        try:
            os.makedirs(os.path.dirname(path) or "/tmp", exist_ok=True)
            await self._page.screenshot(path=path, full_page=False)
            return {
                "success": True,
                "path": path,
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # -- JavaScript evaluation --

    async def evaluate(self, js_code: str) -> Dict[str, Any]:
        """Run arbitrary JavaScript and return the result."""
        await self._ensure_browser()
        try:
            result = await self._page.evaluate(js_code)
            return {"success": True, "result": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # -- Search --

    async def search_and_extract(self, query: str,
                                 max_results: int = 5) -> Dict[str, Any]:
        """
        Search Google and extract top results as structured data.

        Returns a list of SearchResult-like dicts with title, url, snippet.
        """
        await self._ensure_browser()
        url = SEARCH_URL.format(query=quote_plus(query))

        nav_result = await self.navigate(url)
        if not nav_result.get("success"):
            return {
                "success": False,
                "error": nav_result.get("error", "Navigation failed"),
                "results": [],
                "query": query,
            }

        # Wait briefly for results to render
        try:
            await self._page.wait_for_selector("div#search", timeout=8000)
        except Exception:
            pass  # Proceed anyway -- page structure may vary

        try:
            raw_results = await self._page.evaluate("""
                (maxResults) => {
                    const results = [];
                    // Standard Google result selectors
                    const containers = document.querySelectorAll('div.g, div[data-hveid] div.g');
                    for (const el of containers) {
                        if (results.length >= maxResults) break;
                        const linkEl = el.querySelector('a[href]');
                        const titleEl = el.querySelector('h3');
                        const snippetEl = el.querySelector('div[data-sncf], div.VwiC3b, span.st');

                        if (linkEl && titleEl) {
                            const href = linkEl.href || '';
                            if (href.startsWith('http') && !href.includes('google.com/search')) {
                                results.push({
                                    title: (titleEl.textContent || '').trim(),
                                    url: href,
                                    snippet: (snippetEl ? snippetEl.textContent : '').trim().substring(0, 300),
                                });
                            }
                        }
                    }
                    return results;
                }
            """, max_results)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Extraction failed: {exc}",
                "results": [],
                "query": query,
            }

        return {
            "success": True,
            "results": raw_results,
            "count": len(raw_results),
            "query": query,
        }

    # -- Page state helpers --

    async def current_url(self) -> str:
        """Return the current page URL, or empty if browser not started."""
        if self._page is None:
            return ""
        return self._page.url

    async def wait_for_selector(self, selector: str,
                                timeout: Optional[int] = None) -> Dict[str, Any]:
        """Wait for a CSS selector to appear on the page."""
        await self._ensure_browser()
        try:
            await self._page.wait_for_selector(
                selector, timeout=timeout or self._timeout
            )
            return {"success": True, "selector": selector}
        except Exception as exc:
            return {"success": False, "error": str(exc), "selector": selector}
