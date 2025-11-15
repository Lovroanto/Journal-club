#!/usr/bin/env python3
"""
grobid_preprocessor.py

Preprocess PDFs via GROBID:
- Extracts title, authors, abstract
- Separates main text and supplementary cleanly
- Supplementary starts ONLY at <head> in a <div>
- Dynamic chunking based on paragraph aggregation
- Tracks figures per chunk
- Writes figure summary files and inverse mappings
"""

import os
import re
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict
import requests

DEFAULT_CHUNK_SIZE = 4000

SUPP_PATTERNS = [
    r"Supplementary Material",
    r"Supplementary Information",
    r"Supplemental Material",
    r"Supporting Information",
    r"Additional Information"
]


# -----------------------
# Utility
# -----------------------
def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# -----------------------
# Metadata extraction
# -----------------------
def extract_metadata_from_tei(tei_xml: str):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    tei_header = soup.find("teiHeader")
    if tei_header is None:
        return {"title": "", "authors": [], "abstract": ""}

    # Title
    title_tag = tei_header.find("title", {"level": "a", "type": "main"})
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Authors
    authors = []
    analytic = tei_header.find("analytic")
    if analytic:
        for author in analytic.find_all("author"):
            forename = author.find("forename", {"type": "first"})
            surname = author.find("surname")
            authors.append({
                "first": forename.get_text(strip=True) if forename else "",
                "last": surname.get_text(strip=True) if surname else "",
            })

    # Abstract
    abstract_tag = tei_header.find("abstract")
    abstract_text = ""
    if abstract_tag:
        abstract_text = " ".join(
            p.get_text(" ", strip=True) for p in abstract_tag.find_all("p")
        )

    return {"title": title, "authors": authors, "abstract": abstract_text}


# -----------------------
# Clean paragraph text and extract figures
# -----------------------
def clean_paragraph(p_tag):
    # Remove bibliography refs
    for b in p_tag.find_all("ref", {"type": "bibr"}):
        b.decompose()

    # Rewrite and collect figure refs
    figures = []
    for f in p_tag.find_all("ref", {"type": "figure"}):
        fig_text = f.get_text(strip=True)
        fig_id = f.get("target", "")
        label = f"{fig_id}: {fig_text}"
        figures.append(label)
        f.string = f"(fig {label})"

    return p_tag.get_text(" ", strip=True), figures


# -----------------------
# Extract sections and chunks
# -----------------------
def extract_sections(tei_xml: str, chunk_size=DEFAULT_CHUNK_SIZE):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    divs = soup.find_all("div")

    main_chunks = []
    supp_chunks = []

    current_section = "main"
    suppl_started = False

    chunk_text = ""
    chunk_figures = set()

    def finalize_chunk():
        nonlocal chunk_text, chunk_figures
        if not chunk_text.strip():
            return
        if chunk_figures:
            chunk_text += "\nFigures used in this chunk: " + ", ".join(sorted(chunk_figures))
        if current_section == "main":
            main_chunks.append({"text": chunk_text, "figures": list(chunk_figures)})
        else:
            supp_chunks.append({"text": chunk_text, "figures": list(chunk_figures)})
        chunk_text = ""
        chunk_figures = set()

    for div in divs:

        head_tag = div.find("head")
        head_text = head_tag.get_text(strip=True) if head_tag else ""

        # --------------------- SUPPLEMENTARY DETECTION ---------------------
        if (not suppl_started) and head_tag:
            if any(re.search(pat, head_text, re.IGNORECASE) for pat in SUPP_PATTERNS):
                suppl_started = True
                finalize_chunk()
                current_section = "supplementary"

        # --------------------- SECTION TITLE ---------------------
        if head_tag:
            if chunk_text:
                finalize_chunk()
            chunk_text = f"{head_text}\n"

        # --------------------- PARAGRAPHS ---------------------
        for p in div.find_all("p"):
            para_text, figures = clean_paragraph(p)

            if len(chunk_text) + len(para_text) > chunk_size and chunk_text:
                finalize_chunk()
                chunk_text = ""

            chunk_text += para_text + "\n"
            chunk_figures.update(figures)

    finalize_chunk()

    return main_chunks, supp_chunks


# -----------------------
# Process PDF
# -----------------------
def preprocess_pdf(pdf_path: str, output_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) Run GROBID ----------------
    url = "http://localhost:8070/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        r = requests.post(url, files={"input": f})
    tei_xml = r.text

    raw_tei_path = output_dir / "raw_tei.xml"
    save_text(raw_tei_path, tei_xml)

    # ---------------- 2) Metadata ----------------
    meta = extract_metadata_from_tei(tei_xml)
    title_txt_path = output_dir / "title.txt"
    with open(title_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {meta['title']}\n")
        f.write("Authors:\n")
        for a in meta['authors']:
            f.write(f"{a['first']} {a['last']}\n")
        f.write(f"\nAbstract:\n{meta['abstract']}\n")

    # ---------------- 3) Sections & chunks ----------------
    main_chunks, supp_chunks = extract_sections(tei_xml, chunk_size=chunk_size)

    # ---------------- Save chunks ----------------
    main_dir = output_dir / "main"
    supp_dir = output_dir / "supplementary"
    main_dir.mkdir(exist_ok=True)
    supp_dir.mkdir(exist_ok=True)

    # Track all figures
    main_figs = set()
    supp_figs = set()
    main_fig_map = {}
    supp_fig_map = {}

    for i, c in enumerate(main_chunks, start=1):
        fname = main_dir / f"main_chunk{str(i).zfill(2)}.txt"
        save_text(fname, c["text"])

        for fig in c["figures"]:
            main_figs.add(fig)
            main_fig_map.setdefault(fig, []).append(str(i).zfill(2))

    for i, c in enumerate(supp_chunks, start=1):
        fname = supp_dir / f"sup_chunk{str(i).zfill(2)}.txt"
        save_text(fname, c["text"])

        for fig in c["figures"]:
            supp_figs.add(fig)
            supp_fig_map.setdefault(fig, []).append(str(i).zfill(2))

    # ---------------- Save figure lists ----------------
    save_text(output_dir / "figmain.txt", "\n".join(sorted(main_figs)))
    save_text(output_dir / "figsup.txt", "\n".join(sorted(supp_figs)))

    # ---------------- Save inverse mappings ----------------
    with open(output_dir / "figmain_map.txt", "w", encoding="utf-8") as f:
        for fig in sorted(main_fig_map):
            chunks = ", ".join(main_fig_map[fig])
            f.write(f"{fig} → {chunks}\n")

    with open(output_dir / "figsup_map.txt", "w", encoding="utf-8") as f:
        for fig in sorted(supp_fig_map):
            chunks = ", ".join(supp_fig_map[fig])
            f.write(f"{fig} → {chunks}\n")

    return {
        "raw_tei": str(raw_tei_path),
        "title_txt": str(title_txt_path),
        "main_chunks": len(main_chunks),
        "supp_chunks": len(supp_chunks)
    }
