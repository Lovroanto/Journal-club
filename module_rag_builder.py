"""
RAG Builder — Full aggressive (legal) discovery + download pipeline
- Aggressive DOI discovery: Crossref (bibliographic), Semantic Scholar (optional), arXiv, SerpAPI (optional)
- Unpaywall-first PDF discovery
- Crossref publisher link fallback + publisher-specific URL patterns
- Landing-page scanning for PDF hrefs
- Robust session-based downloader (preflight GET, browser headers, Referer, %PDF check)
- Optional Playwright fallback (requires user-provided institutional cookies / login)
- Caching of API results to speed repeated runs
- Parallel downloads, metadata & debug logging per paper

USAGE
1) Install dependencies:
   pip install requests wikipedia unidecode tqdm
   # Playwright is optional for fallback:
   # pip install playwright
   # python -m playwright install

2) Set environment variables (recommended):
   - UNPAYWALL_EMAIL (required to get Unpaywall OA links reliably)
   - SEMANTIC_SCHOLAR_API_KEY (optional)
   - SERPAPI_API_KEY (optional)

3) Prepare CSV with at least columns: notion,title,authors,year,journal
   and a notions.txt with notion lines (first line optional header).

4) Run:
    python rag_builder_full_search.py

NOTES
- This tool avoids Sci-Hub. It aims to be aggressive but legal: first Unpaywall, then Crossref/publisher heuristics, then landing-page scans, then optional browser fallback if you have credentials.
- If many PDFs still fail, enable Playwright fallback and provide institutional login/cookies in an interactive browser session.

"""

import os
import re
import json
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from urllib.parse import urljoin, urlparse, urlencode

import requests
import wikipedia
from tqdm import tqdm
from unidecode import unidecode

# Optional Playwright (fallback only)
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# ---------------------------
# Configuration
# ---------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
REQUESTS_TIMEOUT = 20
CROSSREF_API = "https://api.crossref.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
ARXIV_API = "http://export.arxiv.org/api/query"

SESSION_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
PDF_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# ---------------------------
# Utilities
# ---------------------------

def clean_filename(s: str, max_len: int = 120) -> str:
    s = unidecode(s or "")
    s = re.sub(r"[^\w\s\-\(\)\.\[\]]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len]


def stable_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ensure_str(x):
    if x is None:
        return None
    if isinstance(x, list) and x:
        return x[0]
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return None

# ---------------------------
# Json disk cache
# ---------------------------
class JsonCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        makedirs(cache_dir)

    def _path(self, key: str) -> Path:
        name = clean_filename(key)
        return self.cache_dir / (name + ".json")

    def get(self, key: str):
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, key: str, value) -> None:
        p = self._path(key)
        try:
            p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

# ---------------------------
# Crossref search (aggressive)
# ---------------------------

def crossref_search_aggressive(title: str, authors: Optional[str], rows: int = 30, cache: Optional[JsonCache] = None):
    key = f"crossref_bib::{title}::{authors or ''}::rows{rows}"
    if cache:
        cached = cache.get(key)
        if cached is not None:
            return cached

    params = {
        "query.bibliographic": title,
        "rows": rows,
        "sort": "score",
    }
    if authors:
        params["query.author"] = authors.split(",")[0].strip()

    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(CROSSREF_API, params=params, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
    except Exception as e:
        print(f"[crossref_aggr] error for '{title}': {e}")
        items = []

    if cache:
        cache.set(key, items)
    return items

# ---------------------------
# Semantic Scholar (optional)
# ---------------------------

def semantic_scholar_search(title: str, api_key: Optional[str], cache: Optional[JsonCache] = None):
    if not api_key:
        return []
    key = f"semanticscholar::{title}"
    if cache:
        c = cache.get(key)
        if c is not None:
            return c

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title, "limit": 6, "fields": "title,authors,year,doi,url"}
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        r = requests.get(url, params=params, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        papers = data.get("data") or data.get("papers") or []
    except Exception as e:
        print(f"[semantic_scholar] failed: {e}")
        papers = []

    if cache:
        cache.set(key, papers)
    return papers

# ---------------------------
# arXiv search by title
# ---------------------------

def arxiv_search_by_title(title: str, cache: Optional[JsonCache] = None) -> List[str]:
    key = f"arxiv::{title}"
    if cache:
        c = cache.get(key)
        if c is not None:
            return c
    q = urlencode({"search_query": f"ti:\"{title}\"", "max_results": 5})
    url = f"{ARXIV_API}?{q}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        text = r.text
        ids = re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", text)
    except Exception as e:
        print(f"[arxiv] error for '{title}': {e}")
        ids = []
    if cache:
        cache.set(key, ids)
    return ids

# ---------------------------
# SerpAPI optional web search
# ---------------------------

def serpapi_search_title(title: str, serpapi_key: Optional[str], engine: str = "google") -> List[str]:
    if not serpapi_key:
        return []
    url = "https://serpapi.com/search.json"
    params = {"engine": engine, "q": title, "api_key": serpapi_key, "num": 5}
    try:
        r = requests.get(url, params=params, timeout=REQUESTS_TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        resp = r.json()
        results = []
        for r0 in (resp.get("organic_results", []) + resp.get("inline_links", [])):
            link = r0.get("link") or r0.get("url")
            if link:
                results.append(link)
    except Exception as e:
        print(f"[serpapi] failed: {e}")
        results = []
    return results

# ---------------------------
# Unpaywall lookup (legal OA)
# ---------------------------

def unpaywall_pdf_url(doi: str, email: str, cache: Optional[JsonCache] = None) -> Optional[str]:
    if not doi or not email:
        return None
    key = f"unpaywall::{doi}"
    if cache:
        v = cache.get(key)
        if v is not None:
            return ensure_str(v.get("best_pdf")) if isinstance(v, dict) else None
    try:
        url = f"{UNPAYWALL_API}/{doi}"
        r = requests.get(url, params={"email": email}, headers={"User-Agent": USER_AGENT}, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        info = r.json()
        best = info.get("best_oa_location") or (info.get("oa_locations") or [None])[0]
        pdf = None
        if best and best.get("url_for_pdf"):
            pdf = ensure_str(best.get("url_for_pdf"))
        else:
            pdf = ensure_str(info.get("url_for_pdf"))
    except Exception as e:
        print(f"[unpaywall] lookup failed for DOI {doi}: {e}")
        pdf = None
        info = {}
    if cache:
        cache.set(key, {"best_pdf": pdf, "raw": info})
    return pdf

# ---------------------------
# Crossref helpers
# ---------------------------

def pick_crossref_pdf_link(item: dict) -> Optional[str]:
    links = item.get("link") or []
    for l in links:
        url = ensure_str(l.get("URL") or l.get("url") or l.get("href"))
        ctype = ensure_str(l.get("content-type")) or ""
        if url and ("pdf" in ctype.lower() or url.lower().endswith(".pdf")):
            return url
    # fallback
    if links:
        return ensure_str(links[0].get("URL") or links[0].get("url"))
    return None

# ---------------------------
# Candidate resolver (publisher patterns)
# ---------------------------

def resolve_pdf_candidates(doi: Optional[str], crossref_item: Optional[dict], title: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    candidates: List[Tuple[str, Optional[str]]] = []
    seen = set()
    def add(u: str, ref: Optional[str] = None):
        if not u:
            return
        u = u.strip()
        if u in seen:
            return
        seen.add(u)
        candidates.append((u, ref))

    # Crossref explicit links first
    if crossref_item:
        for l in (crossref_item.get("link") or []):
            url = ensure_str(l.get("URL") or l.get("url") or l.get("href"))
            if url:
                add(url, ref=f"https://doi.org/{doi}" if doi else None)

    # arXiv detection
    if doi and "arxiv" in doi.lower():
        aid = doi.split("/")[-1]
        if aid:
            add(f"https://arxiv.org/pdf/{aid}.pdf", None)
    if crossref_item:
        possible = []
        if crossref_item.get("URL"):
            possible.append(ensure_str(crossref_item.get("URL")))
        for l in (crossref_item.get("link") or []):
            u = ensure_str(l.get("URL"))
            if u:
                possible.append(u)
        for u in possible:
            if u and "arxiv.org" in u:
                m = re.search(r"(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", u)
                if m:
                    add(f"https://arxiv.org/pdf/{m.group(1)}.pdf", None)

    # APS family
    if doi and doi.startswith("10.1103/"):
        add(f"https://link.aps.org/doi/{doi}", ref=f"https://doi.org/{doi}")
        add(f"https://journals.aps.org/pdf/{doi}", ref=f"https://doi.org/{doi}")

    # Science / AAAS
    if doi:
        add(f"https://www.science.org/doi/pdf/{doi}", ref=f"https://doi.org/{doi}")
        add(f"https://www.science.org/doi/full/{doi}", ref=f"https://doi.org/{doi}")

    # Nature family
    if doi and doi.startswith("10.1038/"):
        suffix = doi.split("/", 1)[1]
        add(f"https://www.nature.com/articles/{suffix}.pdf", ref=f"https://doi.org/{doi}")

    # Springer
    if doi:
        add(f"https://link.springer.com/content/pdf/{doi}.pdf", ref=f"https://doi.org/{doi}")

    # Wiley
    if doi:
        add(f"https://onlinelibrary.wiley.com/doi/pdf/{doi}", ref=f"https://doi.org/{doi}")
        add(f"https://onlinelibrary.wiley.com/doi/epdf/{doi}", ref=f"https://doi.org/{doi}")

    # Elsevier (doi.org will usually redirect to ScienceDirect)
    if doi:
        add(f"https://doi.org/{doi}", ref=None)

    # PLOS
    if doi and crossref_item and ("plos" in (crossref_item.get("publisher") or "").lower()):
        add(f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable", ref=f"https://doi.org/{doi}")

    # fallback to crossref landing
    if crossref_item and crossref_item.get("URL"):
        add(ensure_str(crossref_item.get("URL")), None)

    return candidates

# ---------------------------
# Session-based PDF downloader
# ---------------------------

def download_pdf_session(url: str, dest: Path, referer: Optional[str] = None, max_retries: int = 3) -> bool:
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    # preflight: root + parent
    try:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        for u in (root, url.rsplit("/", 1)[0]):
            try:
                session.get(u, timeout=8)
            except Exception:
                pass
    except Exception:
        pass

    headers = PDF_HEADERS.copy()
    if referer:
        headers["Referer"] = referer

    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=REQUESTS_TIMEOUT, allow_redirects=True) as r:
                r.raise_for_status()
                raw = r.raw.read(4)
                ctype = r.headers.get("Content-Type", "") or ""
                if raw != b"%PDF" and "pdf" not in ctype.lower():
                    raise Exception(f"Missing %PDF signature and content-type {ctype!r}")
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    f.write(raw)
                    for chunk in r.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            return True
        except Exception as e:
            print(f"Download attempt {attempt} failed for {url}: {e}")
            time.sleep(0.8 * attempt)
            continue
    return False

# ---------------------------
# Playwright fallback (optional)
# ---------------------------

def playwright_fetch_pdf(landing_url: str, dest_path: Path, timeout: int = 30) -> bool:
    if not PLAYWRIGHT_AVAILABLE:
        print("Playwright not available. Install playwright if you want this fallback.")
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(user_agent=USER_AGENT, accept_downloads=True)
            page = context.new_page()
            page.goto(landing_url, timeout=timeout * 1000)
            # try to detect download
            try:
                with page.expect_download(timeout=6000) as d:
                    # heuristics: try clicking common pdf links
                    for sel in ["a[title*='PDF']", "a.pdf", "a[href*='.pdf']", "a:has-text('PDF')"]:
                        try:
                            el = page.query_selector(sel)
                            if el:
                                el.click(timeout=2000)
                                download = d.value
                                path = download.path()
                                dest_path.write_bytes(Path(path).read_bytes())
                                browser.close()
                                return True
                        except Exception:
                            continue
            except Exception:
                pass
            # fallback: search page HTML for pdf link
            html = page.content()
            m = re.search(r'href=[\'\"]([^\'\"]+\.pdf(?:\?[^\'\"]*)?)[\'\"]', html, flags=re.I)
            if m:
                pdf_url = urljoin(page.url, m.group(1))
                ok = download_pdf_session(pdf_url, dest_path, referer=page.url)
                browser.close()
                return ok
            browser.close()
    except Exception as e:
        print(f"Playwright fallback error: {e}")
        return False
    return False

# ---------------------------
# DOI / landing discovery
# ---------------------------

def find_doi_for_paper(title: str, authors: Optional[str], cache: JsonCache, semantic_scholar_key: Optional[str] = None, serpapi_key: Optional[str] = None):
    debug = {"attempts": []}
    # 1) Crossref bibliographic
    cr_items = crossref_search_aggressive(title, authors, rows=30, cache=cache)
    debug["attempts"].append({"source": "crossref_bib", "found": len(cr_items)})
    doi_candidates = []
    landing_candidates = []
    for it in (cr_items or []):
        cand_title = ensure_str((it.get("title") or [""])[0]) if it.get("title") else ""
        score = stable_similarity(title, cand_title)
        if score > 0.60:
            d = ensure_str(it.get("DOI"))
            if d:
                doi_candidates.append((d, score, it))
            u = ensure_str(it.get("URL"))
            if u:
                landing_candidates.append(u)
    if doi_candidates:
        doi_candidates.sort(key=lambda x: x[1], reverse=True)
        debug["doi_candidates"] = [(d, s) for (d, s, _) in doi_candidates]
        debug["landing_candidates"] = landing_candidates
        return doi_candidates[0][0], landing_candidates, debug

    # 2) Semantic Scholar
    if semantic_scholar_key:
        ss = semantic_scholar_search(title, semantic_scholar_key, cache=cache)
        debug["attempts"].append({"source": "semantic_scholar", "found": len(ss)})
        for p in (ss or []):
            cand_title = ensure_str(p.get("title"))
            score = stable_similarity(title, cand_title)
            if score > 0.60:
                doi = ensure_str(p.get("doi"))
                url = ensure_str(p.get("url"))
                if url:
                    landing_candidates.append(url)
                if doi and doi.startswith("10."):
                    return doi, landing_candidates, {"found_via": "semantic_scholar", "score": score}

    # 3) arXiv
    arx = arxiv_search_by_title(title, cache=cache)
    debug["attempts"].append({"source": "arxiv", "found": len(arx)})
    if arx:
        aid = arx[0]
        return f"arXiv:{aid}", [f"https://arxiv.org/abs/{aid}", f"https://arxiv.org/pdf/{aid}.pdf"], {"found_via": "arxiv", "id": aid}

    # 4) SerpAPI (optional)
    if serpapi_key:
        serp = serpapi_search_title(title, serpapi_key)
        debug["attempts"].append({"source": "serpapi", "found": len(serp)})
        if serp:
            return None, serp, debug

    # 5) Crossref plain query fallback
    cr_plain = crossref_search_aggressive(title, authors, rows=15, cache=cache)
    debug["attempts"].append({"source": "crossref_plain", "found": len(cr_plain)})
    for it in (cr_plain or []):
        cand_title = ensure_str((it.get("title") or [""])[0]) if it.get("title") else ""
        score = stable_similarity(title, cand_title)
        if score > 0.65:
            doi = ensure_str(it.get("DOI"))
            if doi:
                return doi, [ensure_str(it.get("URL"))] if it.get("URL") else [], {"found_via": "crossref_plain", "score": score}

    return None, landing_candidates, debug

# ---------------------------
# Process paper (integrated)
# ---------------------------

def process_paper(
    notion: str,
    paper: dict,
    pdf_dir: Path,
    rag_dir: Path,
    cache: JsonCache,
    unpaywall_email: Optional[str] = None,
    semantic_scholar_key: Optional[str] = None,
    serpapi_key: Optional[str] = None,
    use_playwright_fallback: bool = False,
) -> Tuple[str, Optional[str]]:
    title = paper.get("title") or ""
    authors = paper.get("authors") or ""
    year = paper.get("year") or ""
    journal = paper.get("journal") or ""

    # discovery
    doi, landing_candidates, debug_find = find_doi_for_paper(title, authors, cache, semantic_scholar_key, serpapi_key)

    chosen_crossref = None
    if doi and not doi.lower().startswith("arxiv:"):
        # find chosen crossref item if available in cache
        cr_items = crossref_search_aggressive(title, authors, rows=10, cache=cache)
        best = None
        best_score = 0.0
        for it in (cr_items or []):
            cand_title = ensure_str((it.get("title") or [""])[0]) if it.get("title") else ""
            s = stable_similarity(title, cand_title)
            if s > best_score:
                best_score = s
                best = it
        chosen_crossref = best

    safe_name = clean_filename(f"{(authors.split(',')[0] if authors else 'author')}_{year}_{title[:80]}")
    meta_path = rag_dir / f"{safe_name}.txt"
    pdf_saved = None
    tried = []

    # 1) Unpaywall first (if DOI present)
    unpay_pdf = None
    if doi and unpaywall_email and not str(doi).lower().startswith("arxiv:"):
        unpay_pdf = unpaywall_pdf_url(doi, unpaywall_email, cache=cache)
    if unpay_pdf:
        tried.append((unpay_pdf, f"https://doi.org/{doi}" if doi else None, "unpaywall"))
        p = pdf_dir / f"{safe_name}.pdf"
        if not p.exists():
            if download_pdf_session(unpay_pdf, p, referer=f"https://doi.org/{doi}" if doi else None):
                pdf_saved = str(p)
        else:
            pdf_saved = str(p)

    # 2) Crossref link best
    if not pdf_saved and chosen_crossref:
        cr_pdf = pick_crossref_pdf_link(chosen_crossref)
        if cr_pdf:
            tried.append((cr_pdf, f"https://doi.org/{doi}" if doi else None, "crossref_link"))
            p = pdf_dir / f"{safe_name}.pdf"
            if not p.exists():
                if download_pdf_session(cr_pdf, p, referer=f"https://doi.org/{doi}" if doi else None):
                    pdf_saved = str(p)
            else:
                pdf_saved = str(p)

    # 3) If arXiv DOI discovered
    if not pdf_saved and doi and str(doi).lower().startswith("arxiv:"):
        aid = doi.split(":", 1)[1]
        arx_url = f"https://arxiv.org/pdf/{aid}.pdf"
        tried.append((arx_url, None, "arxiv"))
        p = pdf_dir / f"{safe_name}.pdf"
        if not p.exists():
            if download_pdf_session(arx_url, p, referer=None):
                pdf_saved = str(p)
        else:
            pdf_saved = str(p)

    # 4) If no pdf yet: generate publisher candidates
    if not pdf_saved:
        candidates = []
        # include any landing candidates discovered earlier
        for lc in (landing_candidates or []):
            if lc:
                candidates.append((lc, None))
        # include resolver candidates from crossref item
        resolver_cands = resolve_pdf_candidates(doi, chosen_crossref, title)
        for c in resolver_cands:
            candidates.append(c)

        # try candidates sequentially
        session = requests.Session()
        session.headers.update(SESSION_HEADERS)
        for (cand_url, referer) in candidates:
            if not cand_url:
                continue
            tried.append((cand_url, referer, "candidate"))
            p = pdf_dir / f"{safe_name}.pdf"
            try:
                # direct pdf URL quick attempt
                if cand_url.lower().endswith('.pdf'):
                    if download_pdf_session(cand_url, p, referer=referer):
                        pdf_saved = str(p)
                        break
                # GET landing & scan
                r = session.get(cand_url, allow_redirects=True, timeout=12)
                final = r.url
                if final and final.lower().endswith('.pdf'):
                    if download_pdf_session(final, p, referer=referer or cand_url):
                        pdf_saved = str(p)
                        break
                text = r.text or ''
                # look for pdf links
                for m in re.finditer(r'href=[\'\"]([^\'\"]+\.pdf(?:\?[^\'\"]*)?)[\'\"]', text, flags=re.I):
                    absolute = urljoin(final, m.group(1))
                    tried.append((absolute, final, 'scan'))
                    if download_pdf_session(absolute, p, referer=final):
                        pdf_saved = str(p)
                        break
                if pdf_saved:
                    break
            except Exception:
                continue

    # 5) DOI landing scan as last-ditch
    if not pdf_saved and doi and not str(doi).lower().startswith('arxiv:'):
        tried.append((f"https://doi.org/{doi}", None, 'doi-landing'))
        try:
            s = requests.Session()
            s.headers.update(SESSION_HEADERS)
            r = s.get(f"https://doi.org/{doi}", timeout=12)
            final = r.url
            text = r.text or ''
            m = re.search(r'href=[\'\"]([^\'\"]+\.pdf(?:\?[^\'\"]*)?)[\'\"]', text, flags=re.I)
            if m:
                absolute = urljoin(final, m.group(1))
                p = pdf_dir / f"{safe_name}.pdf"
                tried.append((absolute, final, 'doi-scan'))
                if download_pdf_session(absolute, p, referer=final):
                    pdf_saved = str(p)
        except Exception:
            pass

    # 6) Optional Playwright fallback
    if not pdf_saved and use_playwright_fallback:
        # try a few landing pages: chosen_crossref URL, doi.org, first candidate
        fallbacks = []
        if chosen_crossref and chosen_crossref.get('URL'):
            fallbacks.append(ensure_str(chosen_crossref.get('URL')))
        if doi:
            fallbacks.append(f"https://doi.org/{doi}")
        for u, _ in resolve_pdf_candidates(doi, chosen_crossref, title):
            fallbacks.append(u)
        for landing in fallbacks:
            if not landing:
                continue
            tried.append((landing, None, 'playwright'))
            p = pdf_dir / f"{safe_name}.pdf"
            if playwright_fetch_pdf(landing, p):
                pdf_saved = str(p)
                break

    # write metadata
    meta = {
        "Notion": notion,
        "Title": title,
        "Authors": authors,
        "Year": year,
        "Journal": journal,
        "DOI": doi,
        "PDF_path": pdf_saved,
        "Tried": tried,
        "Discovery": debug_find,
    }
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e:
        print(f"Failed to write metadata for {title}: {e}")

    return str(meta_path), pdf_saved

# ---------------------------
# Wikipedia helpers (reuse earlier functions)
# ---------------------------

def fetch_wikipedia_best(notion: str, cache: Optional[JsonCache] = None, lang: str = 'en') -> Tuple[Optional[str], Optional[str]]:
    key = f"wikipedia::{notion}::{lang}"
    if cache:
        cached = cache.get(key)
        if cached is not None:
            return cached.get('title'), cached.get('content')
    wikipedia.set_lang(lang)
    try:
        candidates = wikipedia.search(notion, results=8)
    except Exception as e:
        print(f"Wikipedia search failed for '{notion}': {e}")
        candidates = []
    best = None
    best_score = 0.0
    notion_tokens = set(re.findall(r"\w+", notion.lower()))
    for cand in candidates:
        title = cand
        title_tokens = set(re.findall(r"\w+", title.lower()))
        overlap = len(notion_tokens & title_tokens) / max(1, len(notion_tokens))
        seq = stable_similarity(notion, title)
        score = 0.7 * overlap + 0.3 * seq
        if score > best_score:
            best_score = score
            best = title
    if best and best_score >= 0.45:
        try:
            page = wikipedia.page(best, auto_suggest=False, redirect=True)
            content = page.content
            if cache:
                cache.set(key, {"title": page.title, "content": content})
            return page.title, content
        except Exception as e:
            print(f"Failed to fetch wikipedia page '{best}': {e}")
            return None, None
    return None, None


def clean_wikipedia_text(raw_text: str) -> str:
    if not raw_text:
        return ''
    text = raw_text
    text = re.sub(r"\[citation needed\]", "", text, flags=re.I)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\{\{[^}]{0,400}\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\$[^$]{0,200}\$", "", text)
    text = re.sub(r"\\\[[^\\\]]{0,400}\\\]", "", text)
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    clean_paras = []
    for p in paras:
        if len(p) < 50:
            continue
        alpha_ratio = len(re.findall(r"[A-Za-z0-9]", p)) / max(1, len(p))
        if alpha_ratio < 0.25:
            continue
        clean_paras.append(re.sub(r"\s+", " ", p))
    return '\n\n'.join(clean_paras)


def split_into_clean_chunks(text: str, notion: str, max_chunk_size: int = 3500) -> List[str]:
    prefix = f"Notion: {notion}\n\n"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = []
    current = ''
    for s in sentences:
        cand = (current + ' ' + s).strip() if current else s
        if len(prefix) + len(cand) <= max_chunk_size:
            current = cand
        else:
            if current:
                chunks.append(prefix + current)
            if len(prefix) + len(s) > max_chunk_size:
                chunks.append(prefix + s[: max_chunk_size - len(prefix) - 3] + '...')
                current = ''
            else:
                current = s
    if current:
        chunks.append(prefix + current)
    return chunks

# ---------------------------
# Top-level: build_rag_dataset
# ---------------------------

def build_rag_dataset(
    csv_input: str,
    notions_txt: str,
    rag_folder: str,
    pdf_folder: str,
    cache_folder: str = './.cache',
    max_chunk_size: int = 3500,
    max_articles_per_notion: int = 4,
    use_wikipedia: bool = True,
    download_pdfs: bool = True,
    unpaywall_email: Optional[str] = None,
    semantic_scholar_key: Optional[str] = None,
    serpapi_key: Optional[str] = None,
    parallel_workers: int = 6,
    use_playwright_fallback: bool = False,
):
    rag_dir = Path(rag_folder)
    pdf_dir = Path(pdf_folder)
    cache_dir = Path(cache_folder)
    makedirs(rag_dir)
    makedirs(pdf_dir)
    makedirs(cache_dir)
    cache = JsonCache(cache_dir)

    # read notions
    with open(notions_txt, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    notions = lines[1:] if len(lines) and ('notion' in lines[0].lower() or lines[0].startswith('#')) else lines

    # read csv
    papers_by_notion = {n: [] for n in notions}
    with open(csv_input, newline='', encoding='utf-8') as f:
        reader = __import__('csv').DictReader(f)
        for row in reader:
            notion = row.get('notion')
            if notion in papers_by_notion:
                papers_by_notion[notion].append(row)
    for n in papers_by_notion:
        papers_by_notion[n].sort(key=lambda x: int(x.get('year') or 0), reverse=True)

    # Wikipedia
    if use_wikipedia:
        print('\n=== WIKIPEDIA ===')
        for notion in tqdm(notions, desc='Wikipedia'):
            title, raw = fetch_wikipedia_best(notion, cache=cache)
            if not raw:
                continue
            clean_text = clean_wikipedia_text(raw)
            if not clean_text.strip():
                continue
            chunks = split_into_clean_chunks(clean_text, notion, max_chunk_size)
            base = clean_filename(notion)
            for i, chunk in enumerate(chunks, 1):
                (rag_dir / f"{base}_wiki_chunk{i}.txt").write_text(chunk, encoding='utf-8')
            time.sleep(0.15)

    # PDF downloads
    if download_pdfs:
        if not unpaywall_email:
            print('WARNING: UNPAYWALL_EMAIL not provided; Unpaywall step will be skipped (less effective).')
        print('\n=== PDF DOWNLOAD (Aggressive legal pipeline) ===')
        futures = {}
        with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
            for notion in notions:
                count = 0
                for paper in papers_by_notion.get(notion, []):
                    if count >= max_articles_per_notion:
                        break
                    fut = ex.submit(process_paper, notion, paper, pdf_dir, rag_dir, cache, unpaywall_email, semantic_scholar_key, serpapi_key, use_playwright_fallback)
                    futures[fut] = (notion, paper)
                    count += 1

            total = len(futures)
            for fut in tqdm(as_completed(futures), total=total, desc='Downloading PDFs'):
                try:
                    meta_path, pdf_path = fut.result()
                except Exception as e:
                    print(f'Paper processing failed: {e}')

    print('\nRAG BUILD COMPLETE')
    print(f'  Texts -> {rag_dir}')
    print(f'  PDFs  -> {pdf_dir}')


if __name__ == '__main__':
    # quick demo run - edit paths & env vars as needed
    build_rag_dataset(
        csv_input='relevant_references_balanced.csv',
        notions_txt='notions.txt',
        rag_folder='./RAG_data',
        pdf_folder='./PDFs',
        cache_folder='./.cache',
        max_chunk_size=3500,
        max_articles_per_notion=4,
        use_wikipedia=True,
        download_pdfs=True,
        unpaywall_email=os.getenv('UNPAYWALL_EMAIL'),  # set this env var
        semantic_scholar_key=os.getenv('SEMANTIC_SCHOLAR_API_KEY'),
        serpapi_key=os.getenv('SERPAPI_API_KEY'),
        parallel_workers=6,
        use_playwright_fallback=False,
    )
