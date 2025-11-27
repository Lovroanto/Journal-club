# ModuleRAGBUILDER/crossref_pdf_search.py
"""
Crossref metadata + PDF link retriever.
Minimal:
- Fetch Crossref record using works/doi API
- Extract "link" entries if present (PDF)
"""

import requests
from typing import Optional, Dict
from .utils import USER_AGENT, REQUESTS_TIMEOUT, info, warn

HEADERS = {"User-Agent": USER_AGENT}


def crossref_metadata(doi: str) -> Optional[Dict]:
    """
    Returns full Crossref metadata for a DOI.
    """
    url = f"https://api.crossref.org/works/{doi}"
    info(f"[Crossref] Fetching metadata for DOI {doi}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        if r.status_code != 200:
            warn(f"[Crossref] status {r.status_code}")
            return None
        return r.json().get("message", {})
    except Exception as e:
        warn(f"[Crossref metadata error]: {e}")
        return None


def crossref_pdf_link(doi: str) -> Optional[str]:
    """
    Extract the PDF link from Crossref metadata (if provided).
    Example metadata structure:
      "link": [
         {"URL": "...", "content-type": "application/pdf"}
      ]
    """
    md = crossref_metadata(doi)
    if not md:
        return None

    links = md.get("link", [])
    if not isinstance(links, list):
        return None

    for entry in links:
        if entry.get("content-type") == "application/pdf":
            pdf = entry.get("URL")
            if pdf:
                info(f"[Crossref] PDF link found: {pdf}")
                return pdf

    return None
