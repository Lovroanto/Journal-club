# ModuleRAGBUILDER/unpaywall.py
"""
Unpaywall open-access PDF finder.
Requires an email for the API.
"""

import requests
from typing import Optional, Dict
from .utils import REQUESTS_TIMEOUT, info, warn, USER_AGENT

HEADERS = {"User-Agent": USER_AGENT}


def unpaywall_pdf(doi: str, email: str) -> Optional[str]:
    """
    Query Unpaywall for open-access PDF.
    """
    url = f"https://api.unpaywall.org/v2/{doi}"
    info(f"[Unpaywall] Checking OA for {doi}")

    try:
        r = requests.get(url, params={"email": email}, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        if r.status_code != 200:
            warn(f"[Unpaywall] Status {r.status_code}")
            return None

        data = r.json()

        best_oa = data.get("best_oa_location", {})
        pdf = best_oa.get("url_for_pdf")
        if pdf:
            info(f"[Unpaywall] OA PDF: {pdf}")
            return pdf

        return None
    except Exception as e:
        warn(f"[Unpaywall error]: {e}")
        return None
