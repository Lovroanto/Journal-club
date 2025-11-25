"""
tei_supplementary_detector.py

A modular supplementary-boundary detector for GROBID TEI documents.

Provides a single high-level function:

    detect_supplementary_boundary(
        tei_xml: str,
        allow_user_interaction: bool = False,
        ask_user_only_if_ambiguous: bool = False,
        use_llm: bool = True,
        llm: object = None,
        supp_patterns: Optional[List[str]] = None,
    ) -> Optional[int]

It returns a character index in a unified plain-text representation (full_text)
that marks the start of the supplementary material, or None if not found.

The module implements multiple methods and returns the earliest candidate:

 - METHOD A: TEI heading-based detection (checks <head> and short <p> as headings)
 - METHOD B: Raw full-text heading detection (line-based)
 - METHOD C: Reference-based detection (find last <biblStruct> and LLM-assisted
             extraction of the next sentence if needed)
 - METHOD D: Optional user override (interactive; only when allow_user_interaction)

The module includes a light LLM helper that integrates with
`langchain_ollama.OllamaLLM` (or a plain `ollama` invocation) —
if `llm` is provided, it will be consulted to disambiguate the boundary or
extract the most-likely first supplementary sentence when METHOD C leaves
an ambiguous candidate.

Design choices & returned index:
----------------------------------
- The returned index refers to a location inside `full_text` which is a
  unified text produced by `soup.get_text("\n", strip=True)`; it can be used
  to slice `full_text[:index]` and `full_text[index:]`.

Notes:
- This module is intentionally self-contained and defensive about XML quirks.
- It will not make network calls; the LLM call must be present as a local
  object passed to the function (e.g. `llm = OllamaLLM(model="llama3.1")`).

"""

from typing import List, Optional, Tuple, Any
from bs4 import BeautifulSoup
import re
import html

# Default patterns (user-provided list can override/extend)
DEFAULT_SUPP_PATTERNS = [
    r"Supplementary Material",
    r"Supplementary Information",
    r"Supplemental Material",
    r"Supporting Information",
    r"Additional Information",
    r"Supplementary Methods",
    r"Supplementary Figures",
    r"Supplementary Table",
    r"Supplementary Text",
    r"SUPPLEMENTARY",
    r"Supplementary materials",
    r"Methods",
]

# ----------------------
# UTILITIES
# ----------------------

def _clean_text_for_search(t: str) -> str:
    """Normalize whitespace for searching/matching."""
    return re.sub(r"\s+", " ", t).strip()


def _extract_first_sentence(text: str) -> Optional[str]:
    """Return a best-effort first sentence (terminate at .!?)."""
    if not text:
        return None
    # Protect common abbreviations by a quick heuristic
    protected = re.sub(r"\b(e\.g\.|i\.e\.|Fig\.|Fig\b|etc\.)", lambda m: m.group(0).replace('.', '@@@'), text)
    m = re.search(r'([A-Z0-9][^.!?]{10,}?[.!?])', protected)
    if not m:
        # fallback: take up to first newline or 200 chars
        s = text.strip().split('\n', 1)[0][:200].strip()
        return s if len(s) > 10 else None
    sentence = m.group(1).replace('@@@', '.')
    sentence = _clean_text_for_search(sentence)
    return sentence


def _fuzzy_find(full_text: str, snippet: str, min_len: int = 30) -> Optional[int]:
    """Try to find snippet in full_text.

    Strategy:
    - try exact match of snippet
    - try matching first min_len chars of snippet
    - try case-insensitive search
    - return character index or None
    """
    if not snippet:
        return None
    s = _clean_text_for_search(snippet)
    if not s:
        return None

    # Exact match
    idx = full_text.find(s)
    if idx != -1:
        return idx

    # Case-insensitive
    idx = full_text.lower().find(s.lower())
    if idx != -1:
        return idx

    # Partial start match
    if len(s) >= min_len:
        part = s[:min_len]
        idx = full_text.find(part)
        if idx != -1:
            return idx
        idx = full_text.lower().find(part.lower())
        if idx != -1:
            return idx

    return None


# ----------------------
# METHOD A: TEI heading detection
# ----------------------

def detect_by_heading_tei(soup: BeautifulSoup, patterns: List[str]) -> Optional[int]:
    """Search <head> and short <p> tags for heading-like supplementary keywords.

    Returns None or the character index in the unified full_text (computed by
    finding the heading text inside full_text later by the caller).
    """
    # We will return the heading text (caller will resolve to an index)
    headings = []
    for tag_name in ("head", "p"):
        for node in soup.find_all(tag_name):
            txt = node.get_text(" ", strip=True)
            if not txt:
                continue
            # Keep only short headings/paragraphs
            tokens = txt.split()
            if len(tokens) > 6:
                continue
            # Check patterns anchored at start
            for pat in patterns:
                try:
                    if re.search(rf"^{pat}\b", txt, flags=re.IGNORECASE):
                        # Return the raw text so the caller can perform fuzzy find
                        return txt
                except re.error:
                    # If user-supplied pattern is invalid, fall back to substring
                    if txt.lower().startswith(pat.lower()):
                        return txt
    return None


# ----------------------
# METHOD B: Full-text heading detection
# ----------------------

def detect_by_heading_fulltext(full_text: str, patterns: List[str]) -> Optional[int]:
    """Look for short lines in the full_text that start with a supplementary pattern.

    Returns the matching line text (caller will resolve to index via fuzzy find).
    """
    for line in full_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) > 6:
            continue
        for pat in patterns:
            try:
                if re.search(rf"^{pat}\b", line, flags=re.IGNORECASE):
                    return line
            except re.error:
                if line.lower().startswith(pat.lower()):
                    return line
    return None


# ----------------------
# METHOD C: Reference-based detection
# ----------------------

def detect_after_last_reference(soup: BeautifulSoup, full_text: str, llm: Any = None, use_llm: bool = True) -> Optional[int]:
    """Find a candidate boundary after the last <biblStruct>.

    Strategy:
    - Locate all <biblStruct> tags
    - Pick the one with largest numeric suffix in xml:id (bNN)
    - Find the next paragraph/element after that biblStruct in the TEI
    - Attempt to extract the first sentence from that element
    - If extraction fails or is ambiguous, optionally ask the LLM

    Returns a character index in full_text or None.
    """
    bibl_structs = soup.find_all('biblStruct')
    if not bibl_structs:
        return None

    def numeric_id(b):
        # try various attribute names
        for name in ("xml:id", "id", "{http://www.w3.org/XML/1998/namespace}id"):
            val = b.get(name)
            if val:
                m = re.search(r"(\d+)", val)
                if m:
                    return int(m.group(1))
        # fallback: try xml:id as 'b12' style
        val = b.get('xml:id') or b.get('id') or b.get('n') or ''
        m = re.search(r"(\d+)", val)
        return int(m.group(1)) if m else -1

    try:
        last_b = max(bibl_structs, key=numeric_id)
    except Exception:
        last_b = bibl_structs[-1]

    # find next paragraph or head after this bibliographic element
    nxt = last_b.find_next()
    # walk forward until we find a paragraph-like element
    while nxt is not None and nxt.name not in ("p", "div", "head", "listBibl"):
        nxt = nxt.find_next()
    if nxt is None:
        return None

    # If the following element is the start of references again (listBibl), no boundary
    if nxt.name == 'listBibl' or nxt.find_parent('listBibl'):
        return None

    # Extract text to attempt to get first sentence
    nxt_text = _clean_text_for_search(nxt.get_text(" ", strip=True))
    first_sent = _extract_first_sentence(nxt_text)

    if first_sent:
        idx = _fuzzy_find(full_text, first_sent)
        if idx is not None:
            return idx

    # If we get here, we failed to reliably find the sentence; consult LLM if allowed
    if use_llm and llm is not None:
        prompt = (
            "You are given a small fragment of an article TEI (end of references) and the following\n"
            "short text that follows the references. Extract a probable beginning sentence of the\n"
            "supplementary material, returning only the single sentence. If you cannot find it,\n"
            "return the empty string.\n\n"
            f"Following text:\n{nxt_text[:2000]}\n\nPlease respond with a single sentence (no extra explanation)."
        )
        try:
            # Support both LangChain OllamaLLM and direct ollama usage
            if hasattr(llm, '__call__'):
                resp = llm(prompt)
                candidate = resp if isinstance(resp, str) else getattr(resp, 'text', str(resp))
            elif hasattr(llm, 'generate'):
                gen = llm.generate([prompt])
                # LangChain generate objects vary - try to extract a usable text
                try:
                    candidate = gen.generations[0][0].text
                except Exception:
                    candidate = str(gen)
            elif hasattr(llm, 'predict'):
                candidate = llm.predict(prompt)
            else:
                # try simple attribute
                candidate = llm.chat(prompt) if hasattr(llm, 'chat') else None
            if candidate:
                candidate = _clean_text_for_search(candidate.strip())
                if candidate:
                    idx = _fuzzy_find(full_text, candidate)
                    if idx is not None:
                        return idx
        except Exception:
            # If LLM fails, silently continue
            pass

    return None


# ----------------------
# METHOD D: User override
# ----------------------

def detect_by_user(full_text: str, prompt_text: str = "Enter a phrase/sentence that marks the start of the supplementary (or leave blank): ") -> Optional[int]:
    try:
        user_input = input(prompt_text)
    except Exception:
        return None
    if not user_input or not user_input.strip():
        return None
    return _fuzzy_find(full_text, user_input.strip())


# ----------------------
# FINAL INTEGRATOR
# ----------------------

def detect_supplementary_boundary(
    tei_xml: str,
    allow_user_interaction: bool = False,
    ask_user_only_if_ambiguous: bool = False,
    use_llm: bool = True,
    llm: Any = None,
    supp_patterns: Optional[List[str]] = None,
) -> Optional[int]:
    """Run all detection methods and return earliest candidate index in full_text.

    Parameters
    ----------
    tei_xml: str
        The full raw TEI xml text returned by GROBID (raw_tei.xml or corrected TEI).
    allow_user_interaction: bool
        If True, prompt the user for an override (interactive use). Default False.
    ask_user_only_if_ambiguous: bool
        If True, only ask the user when no strong candidate is found.
    use_llm: bool
        If True and llm is provided, use the llm in METHOD C for disambiguation.
    llm: Any
        An LLM object (e.g. OllamaLLM(model='llama3.1')) that supports __call__ or generate.
    supp_patterns: List[str]
        Optional list of supplementary heading patterns (regular expressions).

    Returns
    -------
    int or None
        Character offset into 'full_text' marking the start of the supplementary.
    """
    if supp_patterns is None:
        patterns = DEFAULT_SUPP_PATTERNS
    else:
        patterns = supp_patterns

    soup = BeautifulSoup(tei_xml, 'lxml-xml')
    full_text = soup.get_text("\n", strip=True)

    # Candidates will be integer indices in full_text
    candidates: List[int] = []

    # METHOD A: TEI heading
    heading_text = detect_by_heading_tei(soup, patterns)
    if heading_text:
        idx = _fuzzy_find(full_text, heading_text)
        if idx is not None:
            candidates.append(idx)

    # METHOD B: full-text heading
    heading_full = detect_by_heading_fulltext(full_text, patterns)
    if heading_full:
        idx = _fuzzy_find(full_text, heading_full)
        if idx is not None:
            candidates.append(idx)

    # METHOD C: after last reference
    idx_ref = detect_after_last_reference(soup, full_text, llm=llm, use_llm=use_llm)
    if idx_ref is not None:
        candidates.append(idx_ref)

    # METHOD D: user override
    user_idx = None
    if allow_user_interaction and not ask_user_only_if_ambiguous:
        user_idx = detect_by_user(full_text)
        if user_idx is not None:
            candidates.append(user_idx)

    # If ambiguous and user allowed, ask user
    if allow_user_interaction and ask_user_only_if_ambiguous and not candidates:
        user_idx = detect_by_user(full_text)
        if user_idx is not None:
            candidates.append(user_idx)

    # Choose earliest candidate if any
    candidates = [c for c in candidates if isinstance(c, int) and c >= 0]
    if not candidates:
        return None
    supp_start = min(candidates)
    return supp_start


# ----------------------
# Convenience helper: example integration with extract_sections()
# ----------------------

def split_main_and_supp_by_boundary(tei_xml: str, boundary_idx: Optional[int]) -> Tuple[str, str]:
    """Return (main_text, supp_text) by splitting the TEI-derived full_text.

    Note: full_text is the TEI text with newlines preserved.
    """
    soup = BeautifulSoup(tei_xml, 'lxml-xml')
    full_text = soup.get_text("\n", strip=True)
    if boundary_idx is None:
        return full_text, ''
    return full_text[:boundary_idx].strip(), full_text[boundary_idx:].strip()


__all__ = [
    'detect_supplementary_boundary',
    'split_main_and_supp_by_boundary',
]
