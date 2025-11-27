# ModuleRAGBUILDER/publisher_handlers.py
"""
Publisher-specific PDF handlers.
Used when DOI negotiation fails or returns no PDF.
Each handler tries to guess the correct PDF URL from a known pattern.
Called by pdf_downloader.py before using fallback scraping.
"""

import re
from urllib.parse import urljoin
from typing import Optional
import requests
from .utils import info, warn, USER_AGENT, REQUESTS_TIMEOUT

HEADERS = {"User-Agent": USER_AGENT}


# ---------------------------------------------------------
# Generic helper
# ---------------------------------------------------------

def _check_pdf(url: str) -> bool:
    """Return True if GET url returns a PDF content-type."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUESTS_TIMEOUT, stream=True)
        ct = r.headers.get("Content-Type", "").lower()
        return "pdf" in ct
    except Exception:
        return False


# ---------------------------------------------------------
# Specific publisher pattern handlers
# ---------------------------------------------------------

def handle_science_mag(url: str) -> Optional[str]:
    """
    Science / Science Advances / Science Immunology.
    Pattern:
        https://www.science.org/doi/10.1126/sciadv.xxxxx
        PDF:
        https://www.science.org/doi/pdf/10.1126/sciadv.xxxxx
    """
    if "science.org/doi" not in url:
        return None

    pdf_url = url.replace("/doi/", "/doi/pdf/")
    info(f"[Publisher] Science candidate: {pdf_url}")

    return pdf_url if _check_pdf(pdf_url) else None


def handle_aps(url: str) -> Optional[str]:
    """
    APS (Physical Review X, PRL, PRA, PRB, etc.)
    Pattern:
        https://journals.aps.org/.../abstract/...
        PDF:
        https://journals.aps.org/.../pdf/...
    """
    if "journals.aps.org" not in url:
        return None

    pdf_url = url.replace("/abstract", "/pdf")
    info(f"[Publisher] APS candidate: {pdf_url}")

    return pdf_url if _check_pdf(pdf_url) else None


def handle_nature(url: str) -> Optional[str]:
    """
    Nature / Nature Communications / Scientific Reports.
    PDF pattern usually:
        https://www.nature.com/articles/xxxxx.pdf
    or requires /pdf on top of article URL.
    """
    if "nature.com" not in url:
        return None

    # Try adding ".pdf"
    if not url.endswith(".pdf"):
        candidate = url + ".pdf"
        info(f"[Publisher] Nature candidate: {candidate}")
        if _check_pdf(candidate):
            return candidate

    # Try /pdf route
    candidate2 = urljoin(url, "pdf")
    info(f"[Publisher] Nature fallback: {candidate2}")
    if _check_pdf(candidate2):
        return candidate2

    return None


def handle_springer(url: str) -> Optional[str]:
    """
    SpringerLink:
        https://link.springer.com/article/10.1007/sxxxx
    PDF:
        https://link.springer.com/content/pdf/10.1007/sxxxx.pdf
    """
    if "link.springer.com" not in url:
        return None

    match = re.search(r"/article/(10\.\d{4,9}/[^/]+)", url)
    if not match:
        return None

    doi = match.group(1)
    pdf_url = f"https://link.springer.com/content/pdf/{doi}.pdf"
    info(f"[Publisher] Springer candidate: {pdf_url}")

    return pdf_url if _check_pdf(pdf_url) else None


def handle_elsevier(url: str) -> Optional[str]:
    """
    Elsevier / Cell / Neuron / Current Biology:
    PDF is usually served via:
        https://www.sciencedirect.com/science/article/pii/XXXXXXXX/pdfft?md5=...
    We detect pii and request /pdfft
    """
    if "sciencedirect.com" not in url:
        return None

    match = re.search(r"pii/([A-Za-z0-9]+)", url)
    if not match:
        return None

    pii = match.group(1)
    candidate = f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft"
    info(f"[Publisher] Elsevier candidate: {candidate}")

    # We check the HEAD for efficiency
    try:
        r = requests.head(candidate, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        if "pdf" in r.headers.get("Content-Type", "").lower():
            return candidate
    except:
        return None

    return None


# ---------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------

def find_publisher_pdf(url: str) -> Optional[str]:
    """
    Try each known handler in order.
    Returns PDF URL if found, otherwise None.
    """

    handlers = [
        handle_science_mag,
        handle_aps,
        handle_nature,
        handle_springer,
        handle_elsevier,
    ]

    for h in handlers:
        try:
            pdf = h(url)
            if pdf:
                info(f"[Publisher] {h.__name__} FOUND PDF")
                return pdf
        except Exception as e:
            warn(f"[Publisher] {h.__name__} error: {e}")

    return None
