# ModuleRAGBUILDER/rag_builder.py
"""
Main orchestrator for RAG dataset building.
Exposed function: build_rag_dataset(...)
This file ties together:
 - wikipedia_fetcher
 - crossref DOI discovery (internal)
 - unpaywall lookup (internal)
 - preprint search & llm verification
 - pdf_downloader
"""

import os
import csv
import time
from pathlib import Path
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import makedirs, JsonCache, clean_filename, info, warn
from .wikipedia_fetcher import get_wikipedia_context
from .pdf_downloader import download_pdf_for_paper
from .preprint_search import find_preprint_candidates
from .llm_preprint_verification import generate_preprint_search_queries, verify_preprint_candidate

import requests

CROSSREF_API = "https://api.crossref.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2"
USER_AGENT = "rag-builder/1.0"


def crossref_search_aggressive(title: str, authors: Optional[str], rows: int = 30):
    params = {"query.bibliographic": title, "rows": rows, "sort": "score"}
    if authors:
        params["query.author"] = authors.split(",")[0].strip()
    try:
        r = requests.get(CROSSREF_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        return r.json().get("message", {}).get("items", [])
    except Exception as e:
        warn(f"[Crossref] error: {e}")
        return []


def unpaywall_pdf_url(doi: str, email: str):
    if not doi or not email:
        return None
    try:
        r = requests.get(f"{UNPAYWALL_API}/{doi}", params={"email": email}, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        info(f"[Unpaywall] hit for DOI {doi}")
        data = r.json()
        best = data.get("best_oa_location") or (data.get("oa_locations") or [None])[0]
        if best and best.get("url_for_pdf"):
            return best.get("url_for_pdf")
        return data.get("url_for_pdf")
    except Exception as e:
        warn(f"[Unpaywall] failed: {e}")
        return None


def find_doi_for_paper(title: str, authors: Optional[str], cache: JsonCache, semantic_scholar_key: Optional[str] = None, serpapi_key: Optional[str] = None):
    """
    Aggressive DOI discovery using Crossref (bibliographic) and optional Semantic Scholar / SerpAPI.
    Returns (doi_or_None, landing_candidates_list, debug_info)
    """
    debug = {"attempts": []}
    items = crossref_search_aggressive(title, authors, rows=30)
    debug["crossref_found"] = len(items)
    for it in items:
        cand_title = (it.get("title") or [""])[0]
        score = clean_string_similarity(title, cand_title)
        if score > 0.62:
            doi = it.get("DOI")
            landing = it.get("URL")
            return doi, [landing] if landing else [], {"found_via": "crossref", "score": score}
    # fallback: preprint candidates
    preprints = find_preprint_candidates(title, authors)
    debug["preprints_found"] = len(preprints)
    if preprints:
        # return the first preprint candidate (caller will verify)
        return None, [p.get("pdf") or p.get("url") for p in preprints if p.get("pdf") or p.get("url")], {"found_via": "preprints", "candidates": preprints}
    return None, [], debug


def clean_string_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


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
    llm_callable: Optional[Any] = None,
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

    # load notions
    with open(notions_txt, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    notions = lines[1:] if len(lines) and ("notion" in lines[0].lower() or lines[0].startswith("#")) else lines

    # load CSV
    papers_by_notion = {n: [] for n in notions}
    with open(csv_input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            notion = row.get("notion")
            if notion in papers_by_notion:
                papers_by_notion[notion].append(row)

    for n in papers_by_notion:
        papers_by_notion[n].sort(key=lambda x: int(x.get("year") or 0), reverse=True)

    # Wikipedia
    if use_wikipedia:
        info("=== WIKIPEDIA ===")
        for notion in notions:
            page = get_wikipedia_context(notion)
            if not page:
                continue
            clean_text = page.get("content") or ""
            if not clean_text.strip():
                continue
            # chunking: simple sentence split (kept minimal here)
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", clean_text) if s.strip()]
            prefix = f"Notion: {notion}\\n\\n"
            chunks = []
            cur = ""
            for s in sentences:
                cand = (cur + " " + s).strip() if cur else s
                if len(prefix) + len(cand) <= max_chunk_size:
                    cur = cand
                else:
                    if cur:
                        chunks.append(prefix + cur)
                    cur = s
            if cur:
                chunks.append(prefix + cur)
            base = clean_filename(notion)
            for i, c in enumerate(chunks, 1):
                (rag_dir / f"{base}_wiki_chunk{i}.txt").write_text(c, encoding="utf-8")

    # PDF downloads
    if download_pdfs:
        info("=== PDF DOWNLOAD ===")
        jobs = []
        with ThreadPoolExecutor(max_workers=parallel_workers) as ex:
            futures = {}
            for notion in notions:
                count = 0
                for paper in papers_by_notion.get(notion, []):
                    if count >= max_articles_per_notion:
                        break
                    title = paper.get("title") or ""
                    authors = paper.get("authors") or ""
                    year = paper.get("year") or ""
                    journal = paper.get("journal") or ""
                    # 1) discovery: DOI / landing candidates
                    doi, landing_cands, debug = find_doi_for_paper(title, authors, cache, None, None)
                    # 2) try Unpaywall if DOI found
                    unpay_pdf = None
                    if doi and unpaywall_email:
                        unpay_pdf = unpaywall_pdf_url(doi, unpaywall_email)
                    # 3) schedule download
                    fut = ex.submit(
                        download_pdf_for_paper,
                        title, authors, year, journal,
                        doi, pdf_dir, rag_dir, cache,
                        unpay_pdf,
                        use_playwright_fallback
                    )
                    futures[fut] = (notion, paper, doi, landing_cands, debug)
                    count += 1

            total = len(futures)
            for fut in as_completed(futures):
                meta, pdf_path = fut.result()
                notion, paper, doi, landing_cands, debug = futures[fut]
                # optionally record summary CSV line
                # Save metadata file is already done inside downloader (or we add here)
                # For now, save meta as rag_dir/_summary.json per paper
                # Write also a human-readable small file
                safe = clean_filename(f\"{(paper.get('authors') or '').split(',')[0]}_{paper.get('year')}_{(paper.get('title') or '')[:60]}\")\n                try:\n                    (rag_dir / f\"{safe}_meta.json\").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')\n                except Exception:\n                    pass\n\n    info('RAG build finished')\n"""
