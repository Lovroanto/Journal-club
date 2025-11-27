# ModuleRAGBUILDER/preprint_search.py
"""
Search various preprint repositories for likely matches given a title/authors.
Provides candidate PDF URLs (arXiv, bioRxiv/medRxiv, Zenodo, OSF).
These are best-effort searches; scoring and LLM verification performed elsewhere.
"""

from typing import List, Tuple, Optional, Dict
from urllib.parse import urlencode, urljoin
import requests
from .utils import USER_AGENT, REQUESTS_TIMEOUT, info, warn

ARXIV_API = "http://export.arxiv.org/api/query"
ZENODO_SEARCH = "https://zenodo.org/api/records"
OSF_SEARCH = "https://api.osf.io/v2/search/"

HEADERS = {"User-Agent": USER_AGENT}


def search_arxiv_by_title(title: str, max_results: int = 5) -> List[Dict]:
    q = urlencode({"search_query": f'ti:"{title}"', "max_results": max_results})
    url = f"{ARXIV_API}?{q}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        text = r.text
        # extract id and title roughly
        entries = []
        ids = []
        import re
        for m in re.finditer(r"<entry>(.*?)</entry>", text, flags=re.S):
            block = m.group(1)
            idm = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", block)
            tit = re.search(r"<title>([^<]+)</title>", block)
            if idm:
                aid = idm.group(1).strip()
                title_text = tit.group(1).strip() if tit else ""
                entries.append({"id": aid, "title": title_text, "pdf": f"https://arxiv.org/pdf/{aid}.pdf", "url": f"https://arxiv.org/abs/{aid}"})
        return entries
    except Exception as e:
        warn(f"[arXiv] search error: {e}")
        return []


def search_zenodo(title: str, max_results: int = 5) -> List[Dict]:
    params = {"q": title, "size": max_results}
    try:
        r = requests.get(ZENODO_SEARCH, params=params, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        hits = []
        for rec in data.get("hits", {}).get("hits", []):
            links = rec.get("links", {})
            pdf = links.get("latest_html") or links.get("self")
            # Zenodo records often include 'files' with direct links
            files = rec.get("files") or []
            file_url = None
            for f in files:
                if f.get("type") == "pdf" or f.get("key", "").lower().endswith(".pdf"):
                    file_url = f.get("links", {}).get("download") or f.get("links", {}).get("self")
                    break
            hits.append({"id": rec.get("id"), "title": rec.get("metadata", {}).get("title"), "pdf": file_url, "url": links.get("html")})
        return hits
    except Exception as e:
        warn(f"[Zenodo] search error: {e}")
        return []


def search_osf(title: str, max_results: int = 5) -> List[Dict]:
    params = {"q": title, "page[size]": max_results}
    try:
        r = requests.get(OSF_SEARCH, params=params, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        hits = []
        for item in data.get("data", []):
            attrs = item.get("attributes", {})
            hits.append({"id": item.get("id"), "title": attrs.get("title"), "url": attrs.get("url")})
        return hits
    except Exception as e:
        warn(f"[OSF] search error: {e}")
        return []


# High-level helper: aggregate
def find_preprint_candidates(title: str, authors: Optional[str] = None) -> List[Dict]:
    """
    Returns a list of candidate dicts with keys: source, title, pdf, url
    """
    results = []
    results += [{"source": "arxiv", **r} for r in search_arxiv_by_title(title)]
    results += [{"source": "zenodo", **r} for r in search_zenodo(title)]
    results += [{"source": "osf", **r} for r in search_osf(title)]
    # bioRxiv/medRxiv are not provided with stable free search endpoints here; they can be added if needed.
    return results
