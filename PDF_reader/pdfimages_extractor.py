# PDF_reader/pdfimages_extractor.py
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def extract_images_pdfimages(
    pdf_path: str,
    output_dir: Path,
    *,
    prefix: str = "img",
) -> Dict:
    """
    Extract all embedded images using poppler's pdfimages.
    Writes:
      - output_dir/pdfimages_list.txt   (from `pdfimages -list`)
      - output_dir/{prefix}-*.*         (from `pdfimages -all`)
    Returns a dict with paths and file list.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    list_txt = output_dir / "pdfimages_list.txt"
    out_prefix = output_dir / prefix

    # 1) list
    res_list = _run(["pdfimages", "-list", pdf_path])
    if res_list.returncode != 0:
        raise RuntimeError(f"pdfimages -list failed:\n{res_list.stderr}")
    list_txt.write_text(res_list.stdout, encoding="utf-8")

    # 2) extract
    res_ext = _run(["pdfimages", "-all", pdf_path, str(out_prefix)])
    if res_ext.returncode != 0:
        raise RuntimeError(f"pdfimages -all failed:\n{res_ext.stderr}")

    files = sorted(output_dir.glob(f"{prefix}-*.*"))

    return {
        "images_dir": str(output_dir),
        "pdfimages_list": str(list_txt),
        "num_images": len(files),
        "images": [str(p) for p in files],
    }
