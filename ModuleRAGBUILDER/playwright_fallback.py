# ModuleRAGBUILDER/playwright_fallback.py
"""
Optional Playwright fallback to click PDF/download buttons when static methods fail.
This requires playwright installed and browsers downloaded:
    pip install playwright
    python -m playwright install
The function is conservative: it attempts a few heuristics and returns the local path if download succeeded.
"""

from pathlib import Path
import re
from typing import Optional
from .utils import info, warn, USER_AGENT
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False


def playwright_fetch_pdf(landing_url: str, dest_path: Path, timeout: int = 30) -> bool:
    """
    Visit landing_url in a headless browser, try to trigger a PDF download, or find a direct .pdf link in the rendered HTML.
    Returns True on success (file written to dest_path).
    """
    if not PLAYWRIGHT_AVAILABLE:
        warn("Playwright not installed; skipping playwright fallback.")
        return False

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(user_agent=USER_AGENT, accept_downloads=True)
            page = context.new_page()
            page.goto(landing_url, timeout=timeout * 1000)
            # Try to click elements that typically trigger PDF downloads
            try:
                with page.expect_download(timeout=7000) as download_info:
                    # Try common selectors
                    for sel in ["a[href$='.pdf']", "a:has-text('PDF')", "a:has-text('Full Text')", "a[title*='PDF']"]:
                        try:
                            el = page.query_selector(sel)
                            if el:
                                el.click(timeout=3000)
                                d = download_info.value
                                path = d.path()
                                dest_path.write_bytes(Path(path).read_bytes())
                                browser.close()
                                info(f"[Playwright] Download saved: {dest_path}")
                                return True
                        except Exception:
                            continue
            except Exception:
                # No immediate download; continue to inspect page
                pass

            # Rendered HTML inspection
            html = page.content()
            m = re.search(r'href=[\'"]([^\'"]+\.pdf(?:\?[^\'"]*)?)[\'"]', html, flags=re.I)
            if m:
                pdf_url = page.url.rstrip("/") + "/" + m.group(1).lstrip("/")
                # Try to download using the page (copy cookies out) — here we let the caller use normal downloader if needed.
                # Use playwright to download as trusted fallback:
                try:
                    with page.expect_download(timeout=7000) as download_info2:
                        page.goto(pdf_url)
                        d2 = download_info2.value
                        path2 = d2.path()
                        dest_path.write_bytes(Path(path2).read_bytes())
                        browser.close()
                        info(f"[Playwright] Download via pdf link saved: {dest_path}")
                        return True
                except Exception:
                    pass

            browser.close()
    except Exception as e:
        warn(f"[Playwright] Error: {e}")
        return False

    return False
