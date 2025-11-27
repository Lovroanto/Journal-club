# ModuleRAGBUILDER/arxiv_search.py
"""
Arxiv preprint finder.
Given a title or DOI, try to find an arXiv match.
"""

import requests
from typing import Optional, Dict
from .utils import USER_AGENT, REQUESTS_TIMEOUT, info, warn

HEADERS = {"User-Agent": USER_AGENT}
ARXIV_SEARCH = "https://export.arxiv.org/api/query"


def arxiv_by_title(title: str) -> Optional[Dict]:
    """
    Searches arXiv by title.
    Returns {"pdf_url": ..., "entry_id": ...}
    """
    info(f"[arXiv] Searching title: {title}")

    query = f"ti:\"{title}\""
    params = {"search_query": query, "start": 0, "max_results": 1}

    try:
        r = requests.get(ARXIV_SEARCH, params=params, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        if r.status_code != 200:
            warn(f"[arXiv] status {r.status_code}")
            return None

        txt = r.text
        if "<entry>" not in txt:
            return None

        # Extract minimal
        import re
        entry_id = re.search(r"<id>(.*?)</id>", txt)
        pdf = re.search(r"<link title=\"pdf\" href=\"(.*?)\"", txt)

        return {
            "entry_id": entry_id.group(1) if entry_id else None,
            "pdf_url": pdf.group(1) if pdf else None,
        }
    except Exception as e:
        warn(f"[arXiv error]: {e}")
        return None
