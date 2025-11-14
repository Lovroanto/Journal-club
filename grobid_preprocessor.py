#!/usr/bin/env python3
"""
grobid_preprocessor.py

Preprocess PDFs via GROBID:
- Extracts title, authors, abstract
- Separates main text and supplementary
- Aggregates paragraphs into chunks dynamically
- Tracks figure references per chunk
"""

import os
import re
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict
import requests

DEFAULT_CHUNK_SIZE = 4000  # default max characters per chunk

SUPP_PATTERNS = [
    r"Supplementary Material",
    r"Supplementary Information",
    r"Supplemental Material",
    r"Supporting Information",
    r"Additional Information"
]


# -----------------------
# Utility: save text
# -----------------------
def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# -----------------------
# Extract metadata (title, authors, abstract)
# -----------------------
def extract_metadata_from_tei(tei_xml: str):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    tei_header = soup.find("teiHeader")
    if tei_header is None:
        return {"title": "", "authors": [], "abstract": ""}

    # Title
    title_tag = tei_header.find("title", {"level": "a", "type": "main"})
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Authors (only in analytic)
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
    # remove bibliography references
    for b in p_tag.find_all("ref", {"type": "bibr"}):
        b.decompose()

    # rewrite figure references and collect them
    figures = []
    for f in p_tag.find_all("ref", {"type": "figure"}):
        fig_text = f.get_text(strip=True)
        fig_id = f.get("target", "")
        figures.append(f"{fig_id}: {fig_text}")
        f.string = f"(fig {fig_id}: {fig_text})"

    return p_tag.get_text(" ", strip=True), figures


# -----------------------
# Extract main and supplementary chunks
# -----------------------
def extract_sections(tei_xml: str, chunk_size=DEFAULT_CHUNK_SIZE):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    divs = soup.find_all("div", {"xmlns": "http://www.tei-c.org/ns/1.0"})

    main_chunks = []
    supp_chunks = []
    current_section_type = "main"

    chunk_text = ""
    chunk_figures = set()

    for div in divs:
        div_text = div.get_text(" ", strip=True)

        # Detect if this is supplementary section
        if any(re.search(pat, div_text, re.IGNORECASE) for pat in SUPP_PATTERNS):
            current_section_type = "supplementary"

        # Section title
        head_tag = div.find("head")
        section_title = head_tag.get_text(strip=True) if head_tag else None
        if section_title:
            if chunk_text:
                # finalize previous chunk
                if chunk_figures:
                    chunk_text += "\nFigures used in this chunk: " + ", ".join(sorted(chunk_figures))
                target_list = main_chunks if current_section_type == "main" else supp_chunks
                target_list.append({"text": chunk_text, "figures": list(chunk_figures)})
                chunk_text = ""
                chunk_figures = set()
            # start new chunk with section title
            chunk_text += f"{section_title}\n"

        # Paragraphs
        for p in div.find_all("p"):
            para_text, figures = clean_paragraph(p)

            # if adding this paragraph exceeds chunk_size and current chunk is non-empty
            if len(chunk_text) + len(para_text) > chunk_size and chunk_text:
                # finalize chunk
                if chunk_figures:
                    chunk_text += "\nFigures used in this chunk: " + ", ".join(sorted(chunk_figures))
                target_list = main_chunks if current_section_type == "main" else supp_chunks
                target_list.append({"text": chunk_text, "figures": list(chunk_figures)})
                chunk_text = ""
                chunk_figures = set()

            # add paragraph
            chunk_text += para_text + "\n"
            chunk_figures.update(figures)

    # finalize last chunk
    if chunk_text:
        if chunk_figures:
            chunk_text += "\nFigures used in this chunk: " + ", ".join(sorted(chunk_figures))
        target_list = main_chunks if current_section_type == "main" else supp_chunks
        target_list.append({"text": chunk_text, "figures": list(chunk_figures)})

    return main_chunks, supp_chunks


# -----------------------
# Preprocess a single PDF
# -----------------------
def preprocess_pdf(pdf_path: str, output_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict:
    """
    pdf_path: path to PDF
    output_dir: where to save processed files
    chunk_size: max chars per chunk (paragraphs aggregated)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Send PDF to GROBID
    url = "http://localhost:8070/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        r = requests.post(url, files={"input": f})
    tei_xml = r.text

    # save raw TEI
    raw_tei_path = output_dir / "raw_tei.xml"
    save_text(raw_tei_path, tei_xml)

    # 2) Extract metadata
    meta = extract_metadata_from_tei(tei_xml)
    title_txt_path = output_dir / "title.txt"
    with open(title_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {meta['title']}\n")
        f.write("Authors:\n")
        for a in meta['authors']:
            f.write(f"{a['first']} {a['last']}\n")
        f.write(f"\nAbstract:\n{meta['abstract']}\n")

    # 3) Extract main and supplementary chunks
    main_chunks, supp_chunks = extract_sections(tei_xml, chunk_size=chunk_size)

    main_dir = output_dir / "main"
    supp_dir = output_dir / "supplementary"
    main_dir.mkdir(exist_ok=True)
    supp_dir.mkdir(exist_ok=True)

    for i, c in enumerate(main_chunks):
        save_text(main_dir / f"chunk_{i+1}.txt", c["text"])
    for i, c in enumerate(supp_chunks):
        save_text(supp_dir / f"chunk_{i+1}.txt", c["text"])

    return {
        "raw_tei": str(raw_tei_path),
        "title_txt": str(title_txt_path),
        "main_chunks": len(main_chunks),
        "supp_chunks": len(supp_chunks)
    }
