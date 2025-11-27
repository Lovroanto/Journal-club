# ModuleRAGBUILDER/publisher_patterns.py
"""
Best-guess PDF heuristics based on common patterns.
Used as fallback after DOI negotiation, Crossref, Unpaywall.
"""

from typing import Optional
from .utils import info

def guess_publisher_pdf(doi: str, landing_url: Optional[str]) -> list:
    """
    Returns a *list* of possible PDF URLs.
    Caller will attempt each.

    Examples:
      - Springer: .../pdf/ + ".pdf"
      - Nature: append ".pdf"
      - Science/ScienceAdvances: use /content/.../full.pdf
    """

    guesses = []

    if landing_url:
        # generic PDF guess
        if not landing_url.endswith(".pdf"):
            guesses.append(landing_url + ".pdf")

    # Based on DOI prefixes
    prefix = doi.split("/")[0]

    # Nature & Springer style
    if "10.1038" in doi:
        guesses.append(f"https://www.nature.com/articles/{doi.split('/')[-1]}.pdf")

    if "10.1126" in doi:  # Science / SciAdv
        guesses.append(f"https://www.science.org/doi/pdf/{doi}")

    # APS
    if doi.startswith("10.1103"):
        guesses.append(f"https://link.aps.org/pdf/{doi}")

    info(f"[PublisherGuess] Generated {len(guesses)} guesses for {doi}")
    return guesses
