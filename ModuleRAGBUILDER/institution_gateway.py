# ModuleRAGBUILDER/institution_gateway.py
"""
Institution gateway downloader.

You provide a private base string (gateway prefix) that works like:
    full_url = BASE + DOI

This module:
- builds the URL
- downloads the PDF
- validates it's actually a PDF
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from .utils import USER_AGENT, REQUESTS_TIMEOUT, info, warn
from .doi_negotiation import normalize_doi


def download_pdf_via_gateway(
    doi: str,
    dest: Path,
    gateway_base: str,
    url_encode_doi: bool = True,
    max_retries: int = 2,
) -> Optional[str]:
    """
    Build URL = gateway_base + doi (normalized) and download the PDF there.

    Returns: str(dest) on success, else None.

    Notes:
    - Works only if your environment is already authorized (VPN/proxy/cookies).
    - gateway_base is intentionally not stored here; pass it in from your code/env.
    """
    if not doi or not gateway_base:
        return None

    d = normalize_doi(doi)
    d_part = quote(d, safe="") if url_encode_doi else d
    url = f"{gateway_base}{d_part}"

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})

    for attempt in range(1, max_retries + 1):
        try:
            with s.get(url, stream=True, allow_redirects=True, timeout=REQUESTS_TIMEOUT) as r:
                r.raise_for_status()

                # Validate PDF (either signature or content-type)
                raw = r.raw.read(4)
                ctype = (r.headers.get("Content-Type", "") or "").lower()
                if raw != b"%PDF" and "pdf" not in ctype:
                    raise Exception(f"Not a PDF (ctype={ctype!r}, first4={raw!r})")

                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    f.write(raw)
                    for chunk in r.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)

            info(f"[Gateway] Saved PDF: {dest}")
            return str(dest)

        except Exception as e:
            warn(f"[Gateway] attempt {attempt} failed for {url}: {e}")

    return None
