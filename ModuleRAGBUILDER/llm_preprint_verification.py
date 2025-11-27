# ModuleRAGBUILDER/llm_preprint_verification.py
"""
LLM fallback for preprint discovery.

Used ONLY after:
 - arXiv search = no exact match
 - preprint_search = no good candidate
 - publisher / DOI routes = failed
 - wikipedia_fetcher = insufficient metadata

The LLM suggests *probable arXiv IDs or arXiv categories*.
Real validation is done elsewhere (arxiv_search.py).

Functions:
 - llm_guess_arxiv_ids(title, abstract, llm)
 - llm_guess_arxiv_categories(title, abstract, llm)
"""

from typing import List, Optional
from .utils import info


def llm_guess_arxiv_ids(
    llm_callable,
    title: str,
    abstract: Optional[str] = None
) -> List[str]:
    """
    Ask the LLM to guess 1–3 possible arXiv IDs.
    These are only guesses; we later check validity through arxiv_search.py.
    """
    info(f"[LLM fallback] Guessing arXiv IDs for: {title}")

    prompt = f"""
You are assisting with academic paper retrieval.
A paper could not be located through standard metadata sources.

TASK:
- Guess 1 to 3 possible arXiv identifiers where this paper *might* exist.
- If uncertain, guess IDs based on the research field.
- Return ONLY arXiv IDs (e.g., 2301.01234, 2109.00112v1).
- One ID per line.
- If unsure, still guess.

TITLE:
{title}

ABSTRACT:
{abstract or "(no abstract known)"}
"""

    raw = llm_callable(prompt).strip()
    ids = [line.strip() for line in raw.split("\n") if line.strip()]

    # Only keep lines that look remotely like arXiv IDs
    filtered = [
        x for x in ids
        if any(char.isdigit() for char in x) and len(x) > 5
    ]

    return filtered[:3]


def llm_guess_arxiv_categories(
    llm_callable,
    title: str,
    abstract: Optional[str] = None
) -> List[str]:
    """
    Ask the LLM to guess arXiv categories if IDs cannot be found.
    Example outputs: ['cs.LG', 'stat.ML', 'q-bio.GN']
    These feed into preprint_search.py for broader category-based search.
    """
    info(f"[LLM fallback] Guessing arXiv categories for: {title}")

    prompt = f"""
The following paper could not be found on arXiv.
Guess which 1–3 arXiv categories this work most likely belongs to.

Examples:
 - cs.LG
 - cs.CL
 - q-bio.GN
 - physics.optics
 - math.ST
 - stat.ML

Return categories only, one per line.

TITLE:
{title}

ABSTRACT:
{abstract or "(no abstract known)"}
"""

    raw = llm_callable(prompt).strip()
    cats = [line.strip() for line in raw.split("\n") if line.strip()]

    # Very light filtering: keep items with a dot and letters
    filtered = [
        c for c in cats
        if "." in c and any(ch.isalpha() for ch in c)
    ]

    return filtered[:3]
