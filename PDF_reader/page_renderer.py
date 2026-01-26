from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional, List


def _which(cmd: str) -> Optional[str]:
    """Return path to executable or None."""
    try:
        out = subprocess.run(["which", cmd], capture_output=True, text=True, check=False)
        p = out.stdout.strip()
        return p if p else None
    except Exception:
        return None


def render_pdf_to_png_pages(
    pdf_path: str,
    out_dir: Path,
    dpi: int = 300,
    prefix: str = "page",
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
) -> Dict:
    """
    Render each PDF page to a PNG using pdftocairo.

    Output files look like:
      out_dir / f"{prefix}-1.png"
      out_dir / f"{prefix}-2.png"
      ...

    Returns:
      {
        "pages_dir": "...",
        "dpi": 300,
        "page_files": [".../page-1.png", ...],
        "num_pages": N
      }
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _which("pdftocairo") is None:
        raise RuntimeError(
            "pdftocairo not found. Install poppler-utils: sudo apt-get install poppler-utils"
        )

    # pdftocairo writes prefix + "-1.png", "-2.png", ...
    out_prefix = out_dir / prefix

    cmd: List[str] = [
        "pdftocairo",
        "-png",
        "-r",
        str(dpi),
    ]

    # Optional page range
    if first_page is not None:
        cmd += ["-f", str(first_page)]
    if last_page is not None:
        cmd += ["-l", str(last_page)]

    cmd += [str(pdf_path), str(out_prefix)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "pdftocairo failed:\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    # Collect generated files
    page_files = sorted(out_dir.glob(f"{prefix}-*.png"))
    return {
        "pages_dir": str(out_dir),
        "dpi": dpi,
        "page_files": [str(p) for p in page_files],
        "num_pages": len(page_files),
    }
