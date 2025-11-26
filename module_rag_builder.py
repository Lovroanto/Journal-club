# rag_builder_final.py
"""
RAG dataset builder - Final improved version (Option B)
- Crossref -> DOI metadata
- Unpaywall -> OA PDF URL (preferred)
- Fallback: DOI resolver (https://doi.org/...), Crossref links
- Robust session-based PDF downloading with preflight HTML get & browser headers
- Disk caching (JSON)
- Parallel downloads
- Wikipedia fetching + cleaning + chunking
- Produces metadata .txt files per downloaded paper in the RAG folder

Usage:
    from rag_builder_final import build_rag_dataset

    build_rag_dataset(
        csv_input="relevant_references_balanced.csv",
        notions_txt="notions.txt",
        rag_folder="./RAG_data",
        pdf_folder="./PDFs",
        cache_folder="./.cache",
        max_chunk_size=3500,
        max_articles_per_notion=4,
        use_wikipedia=True,
        download_pdfs=True,
        unpaywall_email="your-email@example.com",
        parallel_workers=6,
    )
"""

import csv
import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from urllib.parse import urljoin, urlparse

import requests
import wikipedia
from tqdm import tqdm
from unidecode import unidecode

# ---------------------------
# Configuration / Constants
# ---------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
REQUESTS_TIMEOUT = 20
CROSSREF_API = "https://api.crossref.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2"

# Browser-like headers
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


def makedirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Simple JSON disk cache
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

    def set(self, key: str, value):
        p = self._path(key)
        try:
            p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


# ---------------------------
# Crossref search
# ---------------------------
def crossref_search(title: str, authors: Optional[str], rows: int, cache: Optional[JsonCache]):
    key = f"crossref::{title}::{authors or ''}::rows{rows}"
    if cache:
        cached = cache.get(key)
        if cached is not None:
            return cached

    params = {"query.title": title, "rows": rows}
    if authors:
        first = authors.split(",")[0].strip()
        if first:
            params["query.author"] = first

    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(CROSSREF_API, params=params, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
    except Exception as e:
        print(f"Crossref error for '{title}': {e}")
        items = []

    if cache:
        cache.set(key, items)
    return items


# ---------------------------
# Unpaywall lookup
# ---------------------------
def unpaywall_pdf_url(doi: str, email: str, cache: Optional[JsonCache]) -> Optional[str]:
    if not doi or not email:
        return None
    key = f"unpaywall::{doi}"
    if cache:
        cached = cache.get(key)
        if cached is not None:
            return cached.get("best_pdf")

    url = f"{UNPAYWALL_API}/{doi}"
    params = {"email": email}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        info = r.json()
        best = info.get("best_oa_location") or (info.get("oa_locations") or [None])[0]
        pdf = None
        if best and best.get("url_for_pdf"):
            pdf = best.get("url_for_pdf")
        else:
            pdf = info.get("url_for_pdf")
    except Exception as e:
        print(f"Unpaywall lookup failed for DOI {doi}: {e}")
        pdf = None
        info = {}

    if cache:
        cache.set(key, {"best_pdf": pdf, "raw": info})
    return pdf


# ---------------------------
# Helpers to resolve possible PDF URLs
# ---------------------------
def try_doi_resolver(doi: str, session: requests.Session) -> Optional[str]:
    """
    Try resolving https://doi.org/<doi> and follow redirects to find PDF URL.
    Returns final URL or None.
    """
    if not doi:
        return None
    doi_url = f"https://doi.org/{doi}"
    try:
        r = session.head(doi_url, allow_redirects=True, timeout=REQUESTS_TIMEOUT)
        # if head works, check final url
        final = r.url
        # some sites require GET to provide cookies; caller will do GET
        return final
    except Exception:
        try:
            r = session.get(doi_url, allow_redirects=True, timeout=REQUESTS_TIMEOUT)
            return r.url
        except Exception:
            return None


def pick_crossref_pdf_link(item: dict) -> Optional[str]:
    """
    Crossref item may contain 'link' entries. Prefer 'application/pdf' links.
    """
    links = item.get("link") or []
    for l in links:
        content_type = (l.get("content-type") or "").lower()
        url = l.get("URL") or l.get("url") or l.get("URL")
        if not url:
            continue
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            return url
    # fallback: first link URL
    if links:
        return links[0].get("URL") or links[0].get("url")
    return None


# ---------------------------
# Session-based PDF downloader (Option B)
# ---------------------------
def download_pdf_session(url: str, dest_path: Path, referer: Optional[str] = None, max_retries: int = 3) -> bool:
    """
    Robust download using session with browser headers + preflight HTML get.
    Validates PDF signature '%PDF' at start.
    """
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    # Preflight: load parent HTML page to get cookies
    try:
        parsed = urlparse(url)
        page_root = f"{parsed.scheme}://{parsed.netloc}"
        # Try the actual page root first, then parent path
        try_urls = [page_root, url.rsplit("/", 1)[0]]
        for t in try_urls:
            try:
                session.get(t, timeout=10)
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
                # read first 4 bytes to validate PDF signature
                raw = r.raw.read(4)
                if raw != b"%PDF":
                    # if first bytes not PDF, maybe the server returns HTML or a redirect page; treat as failure
                    # but also check content-type header as a small tolerance
                    ctype = r.headers.get("Content-Type", "")
                    if "pdf" not in (ctype or "").lower():
                        raise Exception(f"Missing %PDF signature and content-type {ctype!r}")
                    # else continue writing (some servers may stream differently)
                # write file: include header we already read
                tmp = dest_path.with_suffix(dest_path.suffix + ".part")
                with open(tmp, "wb") as f:
                    f.write(raw)
                    for chunk in r.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest_path)
            return True
        except Exception as e:
            print(f"Download attempt {attempt} failed for {url}: {e}")
            time.sleep(1 + attempt * 0.5)
            continue
    return False


# ---------------------------
# Wikipedia helpers (lightweight)
# ---------------------------
def fetch_wikipedia_best(notion: str, cache: Optional[JsonCache], lang: str = "en") -> Tuple[Optional[str], Optional[str]]:
    key = f"wikipedia::{notion}::{lang}"
    if cache:
        cached = cache.get(key)
        if cached:
            return cached.get("title"), cached.get("content")

    wikipedia.set_lang(lang)
    try:
        candidates = wikipedia.search(notion, results=8)
    except Exception as e:
        print(f"Wikipedia search error for '{notion}': {e}")
        candidates = []

    best = None
    best_score = 0.0
    notion_tokens = set(re.findall(r"\w+", notion.lower()))
    for cand in candidates:
        title_tokens = set(re.findall(r"\w+", cand.lower()))
        overlap = len(notion_tokens & title_tokens) / max(1, len(notion_tokens))
        seq = stable_similarity(notion, cand)
        score = 0.7 * overlap + 0.3 * seq
        if score > best_score:
            best_score = score
            best = cand

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
        return ""
    text = raw_text
    text = re.sub(r"\[citation needed\]", "", text, flags=re.I)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\{\{[^}]{0,400}\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\$[^$]{0,200}\$", "", text)
    text = re.sub(r"\\\[[^\\\]]{0,400}\\\]", "", text)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    clean_paras = []
    for p in paras:
        if len(p) < 50:
            continue
        alpha_ratio = len(re.findall(r"[A-Za-z0-9]", p)) / max(1, len(p))
        if alpha_ratio < 0.25:
            continue
        clean_paras.append(re.sub(r"\s+", " ", p))
    return "\n\n".join(clean_paras)


def split_into_clean_chunks(text: str, notion: str, max_chunk_size: int = 3500) -> List[str]:
    prefix = f"Notion: {notion}\n\n"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = []
    current = ""
    for s in sentences:
        cand = (current + " " + s).strip() if current else s
        if len(prefix) + len(cand) <= max_chunk_size:
            current = cand
        else:
            if current:
                chunks.append(prefix + current)
            if len(prefix) + len(s) > max_chunk_size:
                chunks.append(prefix + s[: max_chunk_size - len(prefix) - 3] + "...")
                current = ""
            else:
                current = s
    if current:
        chunks.append(prefix + current)
    return chunks


# ---------------------------
# Orchestration: process paper
# ---------------------------
# ---------- ADD THIS HELPER (put near other helpers) ----------
def resolve_pdf_candidates(doi: Optional[str], crossref_item: Optional[dict], title: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    """
    Returns a prioritized list of (pdf_url, referer) candidate tuples to try.
    Uses DOI, Crossref 'link' entries, and common publisher URL patterns.
    The referer should be set to a likely landing page (often doi.org/<doi>).
    """
    candidates: List[Tuple[str, Optional[str]]] = []
    seen = set()

    def add(url: str, referer: Optional[str] = None):
        if not url:
            return
        if isinstance(url, str):
            u = url.strip()
        else:
            return
        if not u:
            return
        if u in seen:
            return
        seen.add(u)
        candidates.append((u, referer))

    # 0) From crossref item 'link' entries (prefer PDF)
    if crossref_item:
        links = crossref_item.get("link") or []
        for l in links:
            url = l.get("URL") or l.get("url") or l.get("href")
            ctype = (l.get("content-type") or "").lower()
            if url:
                add(url, referer=f"https://doi.org/{doi}" if doi else None)
                # often there's a pdf link and a landing link; we add both

    # 1) Unpaywall/Direct OA handled before calling this helper (so omitted here)

    # 2) arXiv — detect id from DOI or title or crossref
    # Common arXiv DOI pattern: 10.48550/arXiv.<id> or arXiv:<id>
    if doi and "arxiv" in doi.lower():
        # 10.48550/arXiv.XXXX
        arx = doi.split("/")[-1]
        if arx.lower().startswith("arxiv."):
            aid = arx.split(".", 1)[1]
            add(f"https://arxiv.org/pdf/{aid}.pdf", referer=None)
    # Also try title / crossref URL for arXiv id
    # Try scanning crossref 'URL' or 'resource' fields:
    if crossref_item:
        possible_urls = []
        if crossref_item.get("URL"):
            possible_urls.append(crossref_item.get("URL"))
        for link in (crossref_item.get("link") or []):
            if link.get("URL"):
                possible_urls.append(link.get("URL"))
        for u in possible_urls:
            if u and "arxiv.org" in u:
                # get id
                m = re.search(r"(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\\d+)?)", u)
                if m:
                    aid = m.group(1)
                    add(f"https://arxiv.org/pdf/{aid}.pdf", referer=None)

    # 3) APS family (Physical Review, PRX, PRL etc.)
    # Many APS PDFs are accessible via journals.aps.org/<journal>/pdf/<doi>
    if doi and doi.startswith("10.1103/"):
        # determine journal short name from DOI suffix like PhysRevX.8.021036
        # Use raw doi as suffix in APS pdf url
        add(f"https://journals.aps.org/{''}pdf/{doi}", referer=f"https://doi.org/{doi}")
        # Another APS format (sometimes): https://link.aps.org/doi/{doi}?pdf=true
        add(f"https://link.aps.org/doi/{doi}", referer=f"https://doi.org/{doi}")

    # 4) AAAS / Science (Science, Science Advances)
    # pattern: https://www.science.org/doi/pdf/<doi>
    if doi and ("science" in (crossref_item.get("publisher") or "").lower() or "sciadv" in (crossref_item.get("title") or "").lower() or "sciadv" in (title or "").lower()):
        add(f"https://www.science.org/doi/pdf/{doi}", referer=f"https://doi.org/{doi}")
    # also try generic
    if doi:
        add(f"https://www.science.org/doi/pdf/{doi}", referer=f"https://doi.org/{doi}")

    # 5) Nature family (nature, npj, etc)
    # Nature article pages often map: https://www.nature.com/articles/<suffix>.pdf
    # Extract suffix from DOI: doi:10.1038/s41586-019-1666-5 -> s41586-019-1666-5
    if doi and doi.startswith("10.1038/"):
        suffix = doi.split("/", 1)[1]
        add(f"https://www.nature.com/articles/{suffix}.pdf", referer=f"https://doi.org/{doi}")

    # 6) SpringerLink common pattern
    # Springer: often PDF like https://link.springer.com/content/pdf/<doi>.pdf or /pdf/<something>.pdf
    if doi and "springer" in (crossref_item.get("publisher") or "").lower():
        add(f"https://link.springer.com/content/pdf/{doi}.pdf", referer=f"https://doi.org/{doi}")

    # 7) Wiley
    # Wiley often uses: https://onlinelibrary.wiley.com/doi/pdf/<doi>
    if doi and "wiley" in (crossref_item.get("publisher") or "").lower():
        add(f"https://onlinelibrary.wiley.com/doi/pdf/{doi}", referer=f"https://doi.org/{doi}")
        add(f"https://onlinelibrary.wiley.com/doi/epdf/{doi}", referer=f"https://doi.org/{doi}")

    # 8) Elsevier / ScienceDirect
    # Elsevier often requires session, but try direct PDF via science direct / SD content URL patterns
    if doi and ("elsevier" in (crossref_item.get("publisher") or "").lower() or "sciencedirect" in (crossref_item.get("URL") or "")):
        # try doi.org resolution — session will follow
        add(f"https://doi.org/{doi}", referer=None)

    # 9) Taylor & Francis (tandfonline) pdf pattern
    if doi and "tandfonline" in (crossref_item.get("publisher") or "").lower():
        add(f"https://www.tandfonline.com/doi/pdf/{doi}", referer=f"https://doi.org/{doi}")

    # 10) PLOS
    if doi and ("plos" in (crossref_item.get("publisher") or "").lower() or (crossref_item.get("container-title") and any('plos' in ct.lower() for ct in (crossref_item.get("container-title") or [])))):
        add(f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable", referer=f"https://doi.org/{doi}")

    # 11) Generic DOI landing page (we'll GET it and scan for .pdf links)
    if doi:
        add(f"https://doi.org/{doi}", referer=None)

    # 12) Last resort: try crossref item 'URL' field (landing)
    if crossref_item:
        url = crossref_item.get("URL")
        if url:
            add(url, referer=None)

    return candidates


# ---------- process_paper----------
def process_paper(
    notion: str,
    paper: dict,
    pdf_dir: Path,
    rag_dir: Path,
    cache: JsonCache,
    unpaywall_email: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Enhanced: tries Unpaywall first, then publisher-specific candidate URLs generated by resolve_pdf_candidates,
    then Crossref links and finally scans landing HTML for PDF hrefs.
    """
    title = paper.get("title") or ""
    authors = paper.get("authors") or ""
    year = paper.get("year") or ""
    journal = paper.get("journal") or ""

    # Crossref search to get DOI and any links
    items = crossref_search(title, authors, rows=5, cache=cache)
    chosen = None
    best_score = 0.0
    for it in items:
        cand_title = (it.get("title") or [""])[0]
        score = stable_similarity(title, cand_title)
        cand_year = None
        if it.get("issued") and isinstance(it.get("issued").get("date-parts"), list):
            try:
                cand_year = int(it.get("issued").get("date-parts")[0][0])
            except Exception:
                cand_year = None
        if year and cand_year and str(cand_year) == str(year):
            score += 0.12
        if score > best_score:
            best_score = score
            chosen = it

    doi = chosen.get("DOI") if chosen else None

    safe_name = clean_filename(f"{(authors.split(',')[0] if authors else 'author')}_{year}_{title[:80]}")
    meta_path = rag_dir / f"{safe_name}.txt"
    pdf_saved_path = None
    tried_candidates = []

    # 1) Try Unpaywall -> OA PDF (best)
    pdf_url = None
    if doi and unpaywall_email:
        pdf_url = unpaywall_pdf_url(doi, unpaywall_email, cache=cache)

    if pdf_url:
        # Try downloading directly
        tried_candidates.append((pdf_url, f"https://doi.org/{doi}" if doi else None, "unpaywall"))
        pdf_path = pdf_dir / f"{safe_name}.pdf"
        if not pdf_path.exists():
            ok = download_pdf_session(pdf_url, pdf_path, referer=f"https://doi.org/{doi}" if doi else None)
            if ok:
                pdf_saved_path = str(pdf_path)
        else:
            pdf_saved_path = str(pdf_path)

    # 2) Generate publisher-specific candidates and try them in order
    if not pdf_saved_path:
        candidates = resolve_pdf_candidates(doi, chosen, title)
        # ensure Unpaywall-produced pdf_url (if any) isn't duplicated
        if pdf_url:
            candidates = [(u, r) for (u, r) in candidates if u != pdf_url]
        session = requests.Session()
        session.headers.update(SESSION_HEADERS)

        for (candidate_url, referer) in candidates:
            tried_candidates.append((candidate_url, referer, "resolver"))
            # If candidate looks like a DOI landing page (doi.org), we will GET it and try to extract PDF links
            # but first try direct download if url endswith .pdf
            pdf_path = pdf_dir / f"{safe_name}.pdf"
            if candidate_url.lower().endswith(".pdf"):
                ok = download_pdf_session(candidate_url, pdf_path, referer=referer)
                if ok:
                    pdf_saved_path = str(pdf_path)
                    break
                else:
                    continue
            # If candidate is a doi.org or landing page, attempt to GET and scan for pdf links
            try:
                r = session.get(candidate_url, allow_redirects=True, timeout=12)
                # If server responded with final location that ends with .pdf, try that
                final_url = r.url
                if final_url and final_url.lower().endswith(".pdf"):
                    ok = download_pdf_session(final_url, pdf_path, referer=referer or candidate_url)
                    if ok:
                        pdf_saved_path = str(pdf_path)
                        break
                # scan HTML for typical pdf hrefs
                text = r.text or ""
                # look for obvious pdf links (href="/...pdf" or href="...pdf")
                m_iter = re.finditer(r'href=[\'"]([^\'"]+\\.pdf(?:\\?[^\'"]*)?)[\'"]', text, flags=re.I)
                found_any = False
                for m in m_iter:
                    found_any = True
                    candidate = m.group(1)
                    absolute = urljoin(final_url, candidate)
                    tried_candidates.append((absolute, final_url, "scan"))
                    ok = download_pdf_session(absolute, pdf_path, referer=final_url)
                    if ok:
                        pdf_saved_path = str(pdf_path)
                        break
                if pdf_saved_path:
                    break
                # some pages provide meta refresh to pdf or javascript links; try to see if it contains 'pdf' token anywhere
                if not found_any:
                    # try search for '/pdf/' or 'downloadPdf' or 'pdf?' tokens
                    m2 = re.search(r'/(?:pdf|download|downloads)/[^\'" >]+', text, flags=re.I)
                    if m2:
                        candidate = urljoin(final_url, m2.group(0))
                        tried_candidates.append((candidate, final_url, "heuristic"))
                        ok = download_pdf_session(candidate, pdf_path, referer=final_url)
                        if ok:
                            pdf_saved_path = str(pdf_path)
                            break
            except Exception as e:
                # ignore and move to next candidate
                # note: we don't raise, just record tried candidate
                # print(f"Resolver GET failed for {candidate_url}: {e}")
                continue

    # 3) As a last resort: try Crossref link list again with direct GET to each link
    if not pdf_saved_path and chosen:
        links = chosen.get("link") or []
        for l in links:
            url = l.get("URL") or l.get("url")
            if not url:
                continue
            pdf_path = pdf_dir / f"{safe_name}.pdf"
            tried_candidates.append((url, f"https://doi.org/{doi}" if doi else None, "crossref-link"))
            ok = download_pdf_session(url, pdf_path, referer=f"https://doi.org/{doi}" if doi else None)
            if ok:
                pdf_saved_path = str(pdf_path)
                break

    # 4) final scan of DOI landing page (if nothing else worked)
    if not pdf_saved_path and doi:
        session = requests.Session()
        session.headers.update(SESSION_HEADERS)
        try:
            r = session.get(f"https://doi.org/{doi}", timeout=12)
            final = r.url
            text = r.text or ""
            # try to find .pdf link
            m = re.search(r'href=[\'"]([^\'"]+\\.pdf(?:\\?[^\'"]*)?)[\'"]', text, flags=re.I)
            if m:
                absolute = urljoin(final, m.group(1))
                pdf_path = pdf_dir / f"{safe_name}.pdf"
                tried_candidates.append((absolute, final, "doi-scan"))
                ok = download_pdf_session(absolute, pdf_path, referer=final)
                if ok:
                    pdf_saved_path = str(pdf_path)
        except Exception:
            pass

    # 5) Write metadata (always)
    meta = {
        "Notion": notion,
        "Title": title,
        "Authors": authors,
        "Year": year,
        "Journal": journal,
        "DOI": doi,
        "PDF_path": pdf_saved_path,
        "Tried_candidates": tried_candidates,
    }
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Failed to write metadata for {title}: {e}")

    return str(meta_path), pdf_saved_path



# ---------------------------
# Top-level function
# ---------------------------
def build_rag_dataset(
    csv_input: str,
    notions_txt: str,
    rag_folder: str,
    pdf_folder: str,
    cache_folder: str = "./.cache",
    max_chunk_size: int = 3500,
    max_articles_per_notion: int = 4,
    use_wikipedia: bool = True,
    download_pdfs: bool = True,
    unpaywall_email: Optional[str] = None,
    parallel_workers: int = 6,
):
    rag_dir = Path(rag_folder)
    pdf_dir = Path(pdf_folder)
    cache_dir = Path(cache_folder)
    makedirs(rag_dir)
    makedirs(pdf_dir)
    makedirs(cache_dir)
    cache = JsonCache(cache_dir)

    # Load notions
    with open(notions_txt, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if len(lines) and ("notion" in lines[0].lower() or lines[0].startswith("#")):
        notions = lines[1:]
    else:
        notions = lines

    # Load CSV
    papers_by_notion = {n: [] for n in notions}
    with open(csv_input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            notion = row.get("notion")
            if notion in papers_by_notion:
                papers_by_notion[notion].append(row)

    # sort by year desc
    for n in papers_by_notion:
        papers_by_notion[n].sort(key=lambda x: int(x.get("year") or 0), reverse=True)

    # Wikipedia
    if use_wikipedia:
        print("\n=== WIKIPEDIA ===")
        for notion in tqdm(notions, desc="Wikipedia"):
            title, raw = fetch_wikipedia_best(notion, cache=cache)
            if not raw:
                continue
            clean_text = clean_wikipedia_text(raw)
            if not clean_text.strip():
                continue
            chunks = split_into_clean_chunks(clean_text, notion, max_chunk_size)
            base = clean_filename(notion)
            for i, chunk in enumerate(chunks, 1):
                (rag_dir / f"{base}_wiki_chunk{i}.txt").write_text(chunk, encoding="utf-8")
            time.sleep(0.15)

    # PDF downloads
    if download_pdfs:
        if not unpaywall_email:
            print("WARNING: No Unpaywall email provided; Unpaywall lookups will be skipped (best-effort fallback used).")
        print("\n=== PDF DOWNLOAD (Crossref + Unpaywall preferred) ===")

        jobs = []
        with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
            futures = {}
            for notion in notions:
                count = 0
                for paper in papers_by_notion.get(notion, []):
                    if count >= max_articles_per_notion:
                        break
                    fut = ex.submit(process_paper, notion, paper, pdf_dir, rag_dir, cache, unpaywall_email)
                    futures[fut] = (notion, paper)
                    count += 1

            total = len(futures)
            for fut in tqdm(as_completed(futures), total=total, desc="Downloading PDFs"):
                try:
                    meta_path, pdf_path = fut.result()
                    # optionally print or aggregate stats here
                except Exception as e:
                    print(f"Paper processing failed: {e}")

    print("\nRAG DATASET COMPLETE!")
    print(f"  Texts -> {rag_dir}")
    print(f"  PDFs  -> {pdf_dir}")


# ---------------------------
# If run directly, demo call (edit file paths / email)
# ---------------------------
if __name__ == "__main__":
    build_rag_dataset(
        csv_input="relevant_references_balanced.csv",
        notions_txt="notions.txt",
        rag_folder="./RAG_data",
        pdf_folder="./PDFs",
        cache_folder="./.cache",
        max_chunk_size=3500,
        max_articles_per_notion=4,
        use_wikipedia=True,
        download_pdfs=True,
        unpaywall_email=None,  # set your email here for Unpaywall
        parallel_workers=6,
    )
