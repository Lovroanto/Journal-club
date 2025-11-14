#!/usr/bin/env python3
"""
grobid_preprocessor.py

Preprocess PDFs via GROBID:
- Extract metadata (title, authors, abstract)
- Separate main and supplementary sections
- Aggregate paragraphs into logical chunks
- Track figure references per chunk
"""

import os
import re
from pathlib import Path
from typing import Dict
import requests
from bs4 import BeautifulSoup

CHUNK_SIZE = 1500  # max chars per chunk

SUPP_PATTERNS = [
    r"Supplementary Material",
    r"Supplementary Information",
    r"Supplemental Material",
    r"Supporting Information",
    r"Additional Information"
]

# -----------------------
# Utilities
# -----------------------
def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def clean_paragraph(p_tag):
    """
    Remove bibliography refs and rewrite figure references
    """
    figures = []

    # remove bibliography references
    for b in p_tag.find_all("ref", {"type": "bibr"}):
        b.decompose()

    # rewrite figure references
    for f in p_tag.find_all("ref", {"type": "figure"}):
        fig_text = f.get_text(strip=True)
        fig_id = f.get("target", "")
        figures.append(f"{fig_id}: {fig_text}")
        f.string = f"(fig {fig_id}: {fig_text})"

    para_text = p_tag.get_text(" ", strip=True)
    return para_text, figures

# -----------------------
# Extract metadata
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
                "last": surname.get_text(strip=True) if surname else ""
            })

    # Abstract
    abstract_tag = tei_header.find("abstract")
    if abstract_tag:
        abstract_text = " ".join(
            p.get_text(" ", strip=True) for p in abstract_tag.find_all("p")
        )
    else:
        abstract_text = ""

    return {"title": title, "authors": authors, "abstract": abstract_text}

# -----------------------
# Extract sections and chunks
# -----------------------
def extract_sections(tei_xml: str, chunk_size=CHUNK_SIZE):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    divs = soup.find_all("div", {"xmlns": "http://www.tei-c.org/ns/1.0"})

    main_chunks = []
    supplementary_chunks = []
    current_section_type = "main"

    chunk_paragraphs = []
    chunk_figures = set()

    def flush_chunk():
        if chunk_paragraphs:
            chunk_text = "\n".join(chunk_paragraphs)
            if chunk_figures:
                chunk_text += f"\n\nFigures used in this chunk: {', '.join(sorted(chunk_figures))}"
            target_list = main_chunks if current_section_type == "main" else supplementary_chunks
            target_list.append({"text": chunk_text, "figures": list(chunk_figures)})
            chunk_paragraphs.clear()
            chunk_figures.clear()

    for div in divs:
        div_text = div.get_text(" ", strip=True)
        if any(re.search(pat, div_text, re.IGNORECASE) for pat in SUPP_PATTERNS):
            current_section_type = "supplementary"

        # section title
        head_tag = div.find("head")
        section_title = head_tag.get_text(strip=True) if head_tag else None
        if section_title:
            flush_chunk()
            chunk_paragraphs.append(section_title)

        # paragraphs
        for p in div.find_all("p"):
            para_text, figures = clean_paragraph(p)
            chunk_paragraphs.append(para_text)
            chunk_figures.update(figures)

            # check if total length exceeds chunk_size
            if sum(len(par) for par in chunk_paragraphs) > chunk_size:
                # backtrack last paragraph
                last_par = chunk_paragraphs.pop()
                flush_chunk()
                chunk_paragraphs.append(last_par)

    flush_chunk()
    return main_chunks, supplementary_chunks

# -----------------------
# Main preprocessing
# -----------------------
def preprocess_pdf(pdf_path: str, output_dir: str, chunk_size: int = CHUNK_SIZE) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Preprocessing PDF: {pdf_path}")

    # Send PDF to GROBID
    url = "http://localhost:8070/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        r = requests.post(url, files={"input": f})
    tei_xml = r.text

    # Save raw TEI
    raw_tei_path = output_dir / "raw_tei.xml"
    save_text(raw_tei_path, tei_xml)

    # Extract metadata
    meta = extract_metadata_from_tei(tei_xml)
    title_txt_path = output_dir / "title.txt"
    with open(title_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {meta['title']}\n")
        f.write("Authors:\n")
        for a in meta['authors']:
            f.write(f"{a['first']} {a['last']}\n")
        f.write(f"\nAbstract:\n{meta['abstract']}\n")

    # Extract chunks
    main_chunks, supp_chunks = extract_sections(tei_xml, chunk_size=chunk_size)

    main_dir = output_dir / "main"
    supp_dir = output_dir / "supplementary"
    main_dir.mkdir(exist_ok=True)
    supp_dir.mkdir(exist_ok=True)

    for i, c in enumerate(main_chunks):
        save_text(main_dir / f"chunk_{i+1}.txt", c["text"])
    for i, c in enumerate(supp_chunks):
        save_text(supp_dir / f"chunk_{i+1}.txt", c["text"])

    print(f"✅ Preprocessing complete!")
    print(f"Main chunks: {len(main_chunks)}, Supplementary chunks: {len(supp_chunks)}")
    return {
        "raw_tei": str(raw_tei_path),
        "title_txt": str(title_txt_path),
        "main_chunks": len(main_chunks),
        "supp_chunks": len(supp_chunks)
    }
