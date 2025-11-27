# ModuleRAGBUILDER/doi_negotiation.py
"""
DOI negotiation module.
Handles:
- DOI normalization
- HEAD + GET requests
- Redirect following
- PDF content-type detection
- Return canonical resolved URL and PDF candidate URL
"""

import re
import requests
from typing import Optional, Dict
from .utils import USER_AGENT, REQUESTS_TIMEOUT, info, warn

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/pdf, text/html;q=0.9,*/*;q=0.1",
}


# ---------------------------------------------------------
# Normalize DOI
# ---------------------------------------------------------

def normalize_doi(doi: str) -> str:
    """
    Remove URL wrappers and whitespace.
    Input examples:
      "https://doi.org/10.1038/abc123"
      "DOI:10.1101/2020.05.12.091397"
    Output:
      "10.1038/abc123"
    """
    doi = doi.strip()
    doi = doi.replace("DOI:", "").replace("doi:", "").strip()

    # Remove URL prefix if present
    doi = re.sub(r"https?://doi\.org/", "", doi)
    doi = doi.strip()

    return doi


# ---------------------------------------------------------
# Negotiation logic
# ---------------------------------------------------------

def _safe_head(url: str) -> Optional[requests.Response]:
    try:
        return requests.head(
            url, allow_redirects=True, headers=HEADERS, timeout=REQUESTS_TIMEOUT
        )
    except Exception as e:
        warn(f"[DOI] HEAD failed: {e}")
        return None


def _safe_get(url: str) -> Optional[requests.Response]:
    try:
        return requests.get(
            url, allow_redirects=True, headers=HEADERS, timeout=REQUESTS_TIMEOUT
        )
    except Exception as e:
        warn(f"[DOI] GET failed: {e}")
        return None


def resolve_doi(doi: str) -> Dict:
    """
    The heart of DOI negotiation.
    Steps:
    - Normalize
    - HEAD to doi.org resolver
    - Follow redirects
    - Identify final HTML landing page
    - Detect PDF link (content-type or pattern)
    """
    normalized = normalize_doi(doi)
    resolver_url = f"https://doi.org/{normalized}"

    info(f"[DOI] Resolving DOI: {normalized}")

    # HEAD request first
    head_r = _safe_head(resolver_url)
    if not head_r:
        return {
            "doi": normalized,
            "resolver_url": resolver_url,
            "landing_url": None,
            "pdf_url": None,
            "status": "head_failed",
        }

    final_url = head_r.url
    info(f"[DOI] Landing URL: {final_url}")

    # Case 1: If HEAD already indicates direct PDF
    ct = head_r.headers.get("Content-Type", "").lower()
    if "pdf" in ct:
        info("[DOI] Direct PDF via HEAD")
        return {
            "doi": normalized,
            "resolver_url": resolver_url,
            "landing_url": final_url,
            "pdf_url": final_url,
            "status": "direct_pdf",
        }

    # Case 2: GET request to inspect HTML and look for PDF
    get_r = _safe_get(final_url)
    if not get_r:
        return {
            "doi": normalized,
            "resolver_url": resolver_url,
            "landing_url": final_url,
            "pdf_url": None,
            "status": "get_failed",
        }

    ct2 = get_r.headers.get("Content-Type", "").lower()

    # If GET also returns PDF directly
    if "pdf" in ct2:
        info("[DOI] Direct PDF via GET")
        return {
            "doi": normalized,
            "resolver_url": resolver_url,
            "landing_url": final_url,
            "pdf_url": final_url,
            "status": "direct_pdf",
        }

    # Case 3: Parse HTML for PDF links
    html = get_r.text
    pdf_candidates = re.findall(r'href="([^"]+\.pdf[^"]*)"', html, re.IGNORECASE)
    if pdf_candidates:
        # Resolve relative links if needed
        pdf_url = pdf_candidates[0]
        if pdf_url.startswith("/"):
            from urllib.parse import urljoin
            pdf_url = urljoin(final_url, pdf_url)

        info(f"[DOI] Found PDF candidate in HTML: {pdf_url}")

        return {
            "doi": normalized,
            "resolver_url": resolver_url,
            "landing_url": final_url,
            "pdf_url": pdf_url,
            "status": "pdf_in_html",
        }

    # No PDF found
    warn("[DOI] No PDF found via DOI negotiation")

    return {
        "doi": normalized,
        "resolver_url": resolver_url,
        "landing_url": final_url,
        "pdf_url": None,
        "status": "no_pdf",
    }
