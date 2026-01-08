# ModuleRAGBUILDER/pdf_downloader.py
"""
PDF downloading orchestration.
Uses:
 - doi_negotiation.resolve_doi
 - publisher_handlers.find_publisher_pdf
 - playwright_fallback.playwright_fetch_pdf (optional)
 - institution_gateway.download_pdf_via_gateway (optional)
Exposes:
 - download_pdf_for_paper(...) -> (meta_dict, pdf_path_or_None)
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import requests

from .utils import USER_AGENT, REQUESTS_TIMEOUT, info, warn, JsonCache, clean_filename
from .doi_negotiation import resolve_doi, normalize_doi
from .publisher_handlers import find_publisher_pdf
from .playwright_fallback import playwright_fetch_pdf
from .institution_gateway import download_pdf_via_gateway

# -----------------------
# Configure gateway here
# -----------------------
# Put your private base string here. It must be a prefix, the DOI gets appended.
# Example: "https://myinfoirmation/"
INSTITUTION_GATEWAY_BASE: Optional[str] = "https://sci-hub.se/"
# If your gateway needs URL-encoded DOI, keep True (safe default).
INSTITUTION_GATEWAY_URL_ENCODE_DOI: bool = True


PDF_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
}

SESSION_HEADERS = {"User-Agent": USER_AGENT}


# Simple HTTP download with %PDF validation
def http_stream_download(
    url: str,
    dest: Path,
    referer: Optional[str] = None,
    max_retries: int = 3,
) -> bool:
    s = requests.Session()
    s.headers.update(SESSION_HEADERS)
    headers = PDF_HEADERS.copy()
    if referer:
        headers["Referer"] = referer

    for attempt in range(1, max_retries + 1):
        try:
            with s.get(
                url,
                headers=headers,
                stream=True,
                timeout=REQUESTS_TIMEOUT,
                allow_redirects=True,
            ) as r:
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

            info(f"[Download] Saved PDF: {dest}")
            return True

        except Exception as e:
            warn(f"[Download] attempt {attempt} failed for {url}: {e}")
            import time
            time.sleep(0.8 * attempt)

    return False


def doi_content_negotiation_download(doi: str, dest: Path) -> Optional[str]:
    """
    Try GET https://doi.org/<doi> with Accept: application/pdf.
    If successful, file is saved to dest and dest path returned.
    """
    norm = normalize_doi(doi)
    url = f"https://doi.org/{norm}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf"}

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            allow_redirects=True,
            timeout=REQUESTS_TIMEOUT,
        ) as r:
            r.raise_for_status()
            raw = r.raw.read(4)
            ct = r.headers.get("Content-Type", "") or ""
            if raw != b"%PDF" and "pdf" not in ct.lower():
                raise Exception(f"Content negotiation didn't yield PDF; content-type {ct}")

            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                f.write(raw)
                for chunk in r.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)

        info(f"[DOINeg] Saved PDF via content negotiation: {dest}")
        return str(dest)

    except Exception as e:
        warn(f"[DOINeg] failed for {doi}: {e}")
        return None


def download_pdf_for_paper(
    title: str,
    authors: str,
    year: str,
    journal: str,
    doi: Optional[str],
    pdf_dir: Path,
    rag_dir: Path,
    cache: JsonCache,
    unpaywall_pdf_url: Optional[str] = None,
    use_playwright: bool = False,
) -> Tuple[Dict, Optional[str]]:
    """
    High-level function to attempt downloads and return metadata dict + pdf_path (or None).
    Steps:
      - Try unpaywall pdf (if provided)
      - Try institution gateway (if configured and DOI exists)
      - Try doi content negotiation
      - Use resolve_doi to get landing_url and embedded pdf
      - Try publisher-specific handler on landing_url
      - Playwright fallback if enabled
    """
    meta: Dict = {"title": title, "authors": authors, "year": year, "journal": journal, "doi": doi}
    safe = clean_filename(f"{(authors.split(',')[0] if authors else 'author')}_{year}_{title[:80]}")
    pdf_path = pdf_dir / f"{safe}.pdf"

    # IMPORTANT: avoid "local variable 'r' referenced before assignment"
    r: Optional[Dict] = None

    # 1) Unpaywall PDF preferred
    if unpaywall_pdf_url:
        info("[DL] Trying Unpaywall URL")
        meta.setdefault("tried", []).append(("unpaywall", unpaywall_pdf_url))
        if http_stream_download(unpaywall_pdf_url, pdf_path, referer=None):
            return meta, str(pdf_path)

    # 1.5) Institution gateway (if configured)
    if doi and INSTITUTION_GATEWAY_BASE:
        info("[DL] Trying institution gateway")
        meta.setdefault("tried", []).append(("institution_gateway", INSTITUTION_GATEWAY_BASE))
        gw = download_pdf_via_gateway(
            doi=doi,
            dest=pdf_path,
            gateway_base=INSTITUTION_GATEWAY_BASE,
            url_encode_doi=INSTITUTION_GATEWAY_URL_ENCODE_DOI,
        )
        if gw:
            info("[DL] Institution gateway: FOUND PDF")
            return meta, gw
        else:
            info("[DL] Institution gateway: no PDF")

    # 2) DOI content negotiation
    if doi:
        info("[DL] Trying DOI content negotiation")
        meta.setdefault("tried", []).append(("doi_content_negotiation", doi))
        cn = doi_content_negotiation_download(doi, pdf_path)
        if cn:
            return meta, cn

    # 3) Resolve DOI landing and inspect
    if doi:
        info("[DL] Resolving DOI landing")
        r = resolve_doi(doi)
        meta["doi_resolve"] = r
        if r.get("pdf_url"):
            meta.setdefault("tried", []).append(("doi_resolve_pdf", r.get("pdf_url")))
            if http_stream_download(r.get("pdf_url"), pdf_path, referer=r.get("landing_url")):
                return meta, str(pdf_path)

    # 4) Publisher-specific handler on landing url(s)
    landing_candidates: List[str] = []
    if r and r.get("landing_url"):
        landing_candidates.append(r.get("landing_url"))

    for lc in landing_candidates:
        if not lc:
            continue
        meta.setdefault("tried", []).append(("landing_publish_handler", lc))
        pub_pdf = find_publisher_pdf(lc)
        if pub_pdf:
            meta.setdefault("tried", []).append(("publisher_pdf", pub_pdf))
            if http_stream_download(pub_pdf, pdf_path, referer=lc):
                return meta, str(pdf_path)

    # 5) Playwright fallback
    if use_playwright:
        info("[DL] Trying Playwright fallback")
        meta.setdefault("tried", []).append(("playwright", landing_candidates[0] if landing_candidates else None))
        ok = playwright_fetch_pdf(landing_candidates[0] if landing_candidates else "about:blank", pdf_path)
        if ok:
            return meta, str(pdf_path)

    info("[DL] No PDF found for paper")
    return meta, None
