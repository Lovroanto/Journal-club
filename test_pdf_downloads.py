#!/usr/bin/env python3
"""
Test downloader on a CSV of known papers.
Focus: given metadata -> download PDF.

Hardcoded paths:
- CSV input:
  /home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary/06_relevant_references.csv
- Output:
  /home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary/Downloads/
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from ModuleRAGBUILDER.utils import (
    JsonCache,
    makedirs,
    info,
    warn,
    stable_similarity,
)
from ModuleRAGBUILDER.doi_negotiation import normalize_doi
from ModuleRAGBUILDER.unpaywall import unpaywall_pdf
from ModuleRAGBUILDER.pdf_downloader import download_pdf_for_paper


# ------------------------------------------------------------------
# HARD-CODED PATHS
# ------------------------------------------------------------------

CSV_PATH = Path(
    "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary/06_relevant_references.csv"
)

OUT_BASE = Path(
    "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary/Downloads/"
)

PDF_DIR = OUT_BASE / "pdfs"
META_DIR = OUT_BASE / "meta"
CACHE_DIR = OUT_BASE / ".cache"

# Optional settings
UNPAYWALL_EMAIL: Optional[str] = None   # e.g. "you@domain.com"
USE_PLAYWRIGHT_FALLBACK = False
PARALLEL_WORKERS = 6

# ------------------------------------------------------------------

CROSSREF_WORKS = "https://api.crossref.org/works"
UA = "pdf-test/0.1"


def _norm_key(k: str) -> str:
    return (k or "").strip().lower()


def _get(row: Dict[str, str], *keys: str, default: str = "") -> str:
    row_lc = {_norm_key(k): v for k, v in row.items()}
    for k in keys:
        if _norm_key(k) in row_lc and row_lc[_norm_key(k)]:
            return row_lc[_norm_key(k)].strip()
    return default


def crossref_find_doi(
    title: str, authors: str = "", rows: int = 10
) -> Tuple[Optional[str], Dict[str, Any]]:
    params = {"query.bibliographic": title, "rows": rows, "sort": "score"}
    first_author = (authors.split(",")[0] or "").strip()
    if first_author:
        params["query.author"] = first_author

    debug: Dict[str, Any] = {}

    try:
        r = requests.get(
            CROSSREF_WORKS,
            params=params,
            headers={"User-Agent": UA},
            timeout=20,
        )
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", []) or []
    except Exception as e:
        debug["error"] = str(e)
        return None, debug

    best = None
    best_score = -1.0
    for it in items:
        cand_title = (it.get("title") or [""])[0]
        score = stable_similarity(title, cand_title)
        if score > best_score:
            best_score = score
            best = it

    debug["best_score"] = best_score
    if best and best_score >= 0.70:
        return best.get("DOI"), debug

    return None, debug


@dataclass
class Job:
    idx: int
    title: str
    authors: str
    journal: str
    year: str
    doi: Optional[str]


def process_one(job: Job, cache: JsonCache) -> Dict[str, Any]:
    title, authors, journal, year = job.title, job.authors, job.journal, job.year
    doi = normalize_doi(job.doi) if job.doi else None

    result: Dict[str, Any] = {
        "idx": job.idx,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "doi_final": "",
        "pdf_path": "",
        "status": "",
        "error": "",
    }

    # DOI discovery if missing
    if not doi:
        doi, _ = crossref_find_doi(title, authors)

    result["doi_final"] = doi or ""

    # Unpaywall
    unpay_pdf = None
    if doi and UNPAYWALL_EMAIL:
        unpay_pdf = unpaywall_pdf(doi, UNPAYWALL_EMAIL)

    # Download
    try:
        meta, pdf_path = download_pdf_for_paper(
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            pdf_dir=PDF_DIR,
            rag_dir=META_DIR,
            cache=cache,
            unpaywall_pdf_url=unpay_pdf,
            use_playwright=USE_PLAYWRIGHT_FALLBACK,
        )
        result["status"] = meta.get("doi_resolve", {}).get("status", "")
        result["pdf_path"] = pdf_path or ""
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)

    return result


def main():
    makedirs(OUT_BASE)
    makedirs(PDF_DIR)
    makedirs(META_DIR)
    makedirs(CACHE_DIR)

    cache = JsonCache(CACHE_DIR)

    jobs: List[Job] = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            title = _get(row, "title")
            authors = _get(row, "authors")
            journal = _get(row, "journal")
            year = _get(row, "year")
            doi = _get(row, "doi") or None

            if not title:
                warn(f"[CSV] Row {i}: missing title, skipped")
                continue

            jobs.append(Job(i, title, authors, journal, year, doi))

    info(f"Loaded {len(jobs)} papers")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
        futures = [ex.submit(process_one, j, cache) for j in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            info(
                f"[DONE] {res['idx']:02d} "
                f"PDF={'YES' if res['pdf_path'] else 'NO'} "
                f"DOI={res['doi_final']}"
            )

    # Write summary CSV
    summary_csv = OUT_BASE / "results.csv"
    fieldnames = sorted({k for r in results for k in r.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(results, key=lambda x: x["idx"]):
            w.writerow(r)

    # ---- Summary stats ----
    total = len(results)
    downloaded = sum(1 for r in results if (r.get("pdf_path") or "").strip())
    failed = total - downloaded
    pct = (100.0 * downloaded / total) if total else 0.0

    info(f"Finished. Results written to {summary_csv}")
    info(f"PDFs saved in {PDF_DIR}")
    info(f"Downloaded PDFs: {downloaded}/{total} ({pct:.1f}%)  |  Failed: {failed}")


if __name__ == "__main__":
    main()
