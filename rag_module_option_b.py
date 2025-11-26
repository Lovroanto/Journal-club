# Option B: Session-based PDF downloader with correct headers
import requests
import re
from pathlib import Path
from urllib.parse import urlparse

# ------------------------------------------------------------------
# CONSTANT HEADERS (Mimic Chrome)
# ------------------------------------------------------------------
SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

PDF_HEADERS = {
    "User-Agent": SESSION_HEADERS["User-Agent"],
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Accept-Language": SESSION_HEADERS["Accept-Language"],
    "Connection": "keep-alive",
}

# ------------------------------------------------------------------
# Helper: Clean filename
# ------------------------------------------------------------------
def clean_filename(s: str) -> str:
    s = re.sub(r"[^\w\s\-()]+", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:120]

# ------------------------------------------------------------------
# MAIN FUNCTION: Robust PDF downloader
# ------------------------------------------------------------------
def download_pdf_with_session(url: str, save_path: str, referer: str = None, max_retries: int = 3) -> bool:
    """
    Downloads a PDF using a persistent session with browser-like headers.
    Works far better than direct requests.get() or Selenium for many publishers.
    """

    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    # 1) PRE-FLIGHT: load the HTML page for cookies
    parent_url = url.rsplit("/", 1)[0]
    try:
        session.get(parent_url, timeout=10)
    except Exception:
        pass

    # 2) Attempt PDF download
    headers = PDF_HEADERS.copy()
    if referer:
        headers["Referer"] = referer

    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=20) as r:
                r.raise_for_status()

                # Validate PDF signature
                head = r.raw.read(4)
                if head != b"%PDF":
                    raise Exception("File is not a PDF (missing %PDF signature)")

                # Open file and re-write the header + content
                with open(save_path, "wb") as f:
                    f.write(head)
                    for chunk in r.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)

            print(f"✓ PDF saved → {save_path}")
            return True

        except Exception as e:
            print(f"Download attempt {attempt} failed for {url}: {e}")

    return False

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    pdfs = [
        "https://link.aps.org/pdf/10.1103/PhysRevX.8.021036",
        "https://arxiv.org/pdf/2102.12345.pdf",
    ]
    out_dir = Path("./test_pdfs")
    out_dir.mkdir(exist_ok=True)

    for url in pdfs:
        name = clean_filename(Path(urlparse(url).path).name) or "paper"
        target = out_dir / f"{name}.pdf"
        download_pdf_with_session(url, target)
