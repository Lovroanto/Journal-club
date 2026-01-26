from __future__ import annotations

from pathlib import Path
import subprocess
import shutil
from typing import Dict, List


def extract_embedded_images_from_pdf(pdf_path: str, out_dir: Path) -> Dict:
    """
    Uses poppler's pdfimages to extract *all* embedded images.
    Produces files like: out-000.png / out-001.jpg / ...
    Returns a dict with output directory and list of image paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure pdfimages exists
    if shutil.which("pdfimages") is None:
        raise RuntimeError("pdfimages not found. Install with: sudo apt-get install poppler-utils")

    prefix = out_dir / "img"
    cmd = ["pdfimages", "-all", pdf_path, str(prefix)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pdfimages failed: {proc.stderr.strip() or proc.stdout.strip()}")

    # collect extracted images
    images: List[str] = []
    for p in sorted(out_dir.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".ppm", ".pbm", ".tif", ".tiff", ".jbig2", ".jp2"}:
            images.append(str(p))

    return {
        "images_dir": str(out_dir),
        "images": images,
        "count": len(images),
    }
