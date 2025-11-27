# ModuleRAGBUILDER/wikipedia_fetcher.py
"""
Wikipedia fetcher module.
Keeps things minimal:
- Tries direct title fetch
- Falls back to search API
- Cleans output
- No external dependencies besides requests

Return format:
{
    "title": str,
    "url": str,
    "summary": str,
    "content": str
}
"""

import requests
from typing import Optional, Dict
from .utils import USER_AGENT, REQUESTS_TIMEOUT, info, warn, clean_filename


WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": USER_AGENT}


def _wiki_call(params: dict) -> Optional[dict]:
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        warn(f"Wikipedia API returned status {r.status_code}")
        return None
    except Exception as e:
        warn(f"Wikipedia error: {e}")
        return None


def search_wikipedia(query: str) -> Optional[str]:
    """
    Use Wikipedia search API to retrieve best matching page title.
    """
    info(f"[Wiki] Searching Wikipedia for: {query}")

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }
    data = _wiki_call(params)
    if not data or "query" not in data or not data["query"]["search"]:
        warn("[Wiki] No search result")
        return None

    best = data["query"]["search"][0]["title"]
    info(f"[Wiki] Best match: {best}")
    return best


def fetch_wikipedia_page(title: str) -> Optional[Dict]:
    """
    Fetch full Wikipedia page content using the 'extracts' API.
    """
    info(f"[Wiki] Fetching page: {title}")

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    data = _wiki_call(params)
    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    if "missing" in page:
        warn(f"[Wiki] Page missing: {title}")
        return None

    extract = page.get("extract", "")
    if not extract:
        warn(f"[Wiki] Empty extract for: {title}")

    return {
        "title": page.get("title", title),
        "url": f"https://en.wikipedia.org/wiki/{page.get('title', title).replace(' ', '_')}",
        "summary": extract[:800],   # small summary
        "content": extract,
    }


def get_wikipedia_context(query: str) -> Optional[Dict]:
    """
    High-level helper:
    - Try direct fetch
    - If fails, search for the correct title
    - Return cleaned dict
    """
    # 1. Try direct title
    page = fetch_wikipedia_page(query)
    if page:
        return page

    # 2. Fallback: search
    best_title = search_wikipedia(query)
    if not best_title:
        return None

    return fetch_wikipedia_page(best_title)
