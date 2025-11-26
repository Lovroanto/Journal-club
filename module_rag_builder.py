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
def process_paper(
    notion: str,
    paper: dict,
    pdf_dir: Path,
    rag_dir: Path,
    cache: JsonCache,
    unpaywall_email: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Try to obtain a PDF for the given paper metadata dict from CSV.
    Returns: (saved_metadata_txt_path, downloaded_pdf_path_or_None)
    """
    title = paper.get("title") or ""
    authors = paper.get("authors") or ""
    year = paper.get("year") or ""
    journal = paper.get("journal") or ""

    # 1) Crossref to obtain DOI + links
    items = crossref_search(title, authors, rows=5, cache=cache)
    chosen = None
    best_score = 0.0
    for it in items:
        cand_title = (it.get("title") or [""])[0]
        score = stable_similarity(title, cand_title)
        # small boost for year match
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

    doi = None
    if chosen:
        doi = chosen.get("DOI")

    # prepare metadata output path (always write a small JSON/TXT describing the attempt)
    safe_name = clean_filename(f"{(authors.split(',')[0] if authors else 'author')}_{year}_{title[:80]}")
    meta_path = rag_dir / f"{safe_name}.txt"

    pdf_saved_path = None

    # 2) Prefer Unpaywall (if email provided and DOI found)
    pdf_url = None
    if doi and unpaywall_email:
        pdf_url = unpaywall_pdf_url(doi, unpaywall_email, cache=cache)

    # 3) If Unpaywall didn't give a pdf_url, try crossref link items
    if not pdf_url and chosen:
        pdf_url = pick_crossref_pdf_link(chosen)

    # 4) If still nothing and DOI exists, try doi resolver redirect
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)
    if not pdf_url and doi:
        resolved = try_doi_resolver(doi, session)
        # If resolver gives a direct .pdf url or a domain likely serving pdfs, use that
        if resolved and (resolved.lower().endswith(".pdf") or "pdf" in resolved.lower()):
            pdf_url = resolved
        else:
            # Sometimes doi resolves to the HTML landing page that contains a PDF link.
            # We'll set referer to resolved and then attempt to find candidate pdfs:
            # Use chosen['link'] as fallback - already tried. If resolved is a publisher landing page,
            # try to do a GET and look for common pdf href patterns.
            try:
                r = session.get(resolved or "", timeout=8)
                text = r.text if r is not None else ""
                # naive regex to extract .pdf links from HTML
                m = re.search(r'href=[\'"]([^\'"]+\\.pdf)[\'"]', text, flags=re.I)
                if m:
                    candidate = m.group(1)
                    pdf_url = urljoin(resolved, candidate)
            except Exception:
                pass

    # 5) Final attempt: if pdf_url found, download using session-based downloader
    if pdf_url:
        fname = clean_filename(f"{(authors.split(',')[0] if authors else 'author')}_{year}_{title[:60]}")
        pdf_path = pdf_dir / f"{fname}.pdf"
        # If file exists, reuse it
        if not pdf_path.exists():
            # Use referer = DOI landing page or chosen publisher site if available
            referer = None
            if doi:
                referer = f"https://doi.org/{doi}"
            # Try download
            ok = download_pdf_session(pdf_url, pdf_path, referer=referer, max_retries=3)
            if ok:
                pdf_saved_path = str(pdf_path)
            else:
                pdf_saved_path = None
        else:
            pdf_saved_path = str(pdf_path)

    # 6) write metadata .txt (JSON inside) to rag_dir (even when pdf not found)
    meta = {
        "Notion": notion,
        "Title": title,
        "Authors": authors,
        "Year": year,
        "Journal": journal,
        "DOI": doi,
        "PDF_URL_used": pdf_url,
        "PDF_path": pdf_saved_path,
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
