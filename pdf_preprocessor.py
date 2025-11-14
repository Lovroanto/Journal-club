#!/usr/bin/env python3
"""
grobid_preprocessor.py

High-quality scientific PDF preprocessing using GROBID.
- Extracts body sections
- Extracts supplementary sections
- Extracts figure captions
- Extracts references (clean formatted)
- Performs smart text chunking
- Saves everything into an RAG-ready folder structure

Dependencies:
    pip install requests beautifulsoup4 lxml
GROBID server must be running:
    ./gradlew run
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from textwrap import wrap


# ------------------------------------------------------------
# 1. CALL GROBID TO GET TEI XML
# ------------------------------------------------------------

def grobid_fulltext(pdf_path, grobid_url="http://localhost:8070/api/processFulltextDocument"):
    """Send PDF to GROBID and return TEI XML string."""
    with open(pdf_path, "rb") as f:
        files = {"input": f}
        response = requests.post(grobid_url, files=files)

    if response.status_code != 200:
        raise RuntimeError(f"GROBID error {response.status_code}: {response.text}")

    return response.text


# ------------------------------------------------------------
# 2. PARSE TEI XML (BODY, SUPP, FIGURES, REFERENCES)
# ------------------------------------------------------------

def extract_body_sections(soup):
    """Extract main text sections with <head> and <p>."""
    body = soup.find("body")
    if body is None:
        return []

    sections = []
    for div in body.find_all("div", recursive=False):
        header = div.find("head")
        title = header.get_text().strip() if header else "Untitled Section"

        paragraphs = [p.get_text(" ", strip=True) for p in div.find_all("p")]
        text = "\n".join(paragraphs)

        if text.strip():
            sections.append((title, text))

    return sections


def extract_supplementary_sections(soup):
    """Detect supplementary / appendix / annex sections."""
    supp_sections = []

    possible_tags = soup.find_all(
        ["div"],
        attrs={"type": ["annex", "appendix", "supplementary-material", "supplementary"]}
    )

    # Also detect supplementary by section title
    for div in soup.find_all("div"):
        head = div.find("head")
        if head and re.search(r"(supplementary|appendix|annex)", head.get_text(), re.I):
            possible_tags.append(div)

    for div in possible_tags:
        head = div.find("head")
        title = head.get_text().strip() if head else "Supplementary Section"

        paragraphs = [p.get_text(" ", strip=True) for p in div.find_all("p")]
        text = "\n".join(paragraphs)

        if text.strip():
            supp_sections.append((title, text))

    return supp_sections


def extract_figures(soup):
    """Extract all figure captions."""
    figures = []
    idx = 1

    for fig in soup.find_all("figure"):
        caption = ""
        head = fig.find("head")
        desc = fig.find("figdesc") or fig.find("figDesc")

        if head:
            caption += head.get_text(" ", strip=True) + ". "
        if desc:
            caption += desc.get_text(" ", strip=True)

        caption = caption.strip()

        if caption:
            figures.append((idx, caption))
            idx += 1

    return figures


def extract_references(soup):
    """Extract structured references from <listBibl>."""
    refs = []
    list_bibl = soup.find("listbibl")

    if not list_bibl:
        return refs

    for item in list_bibl.find_all("biblstruct"):
        raw = item.get_text(" ", strip=True)
        raw = re.sub(r"\s+", " ", raw)
        refs.append(raw)

    return refs


# ------------------------------------------------------------
# 3. SMART CHUNKING
# ------------------------------------------------------------

def chunk_text(text, max_chars=1500):
    """Split long text into meaningful chunks."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []

    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += sent + " "
        else:
            chunks.append(current.strip())
            current = sent + " "
    if current.strip():
        chunks.append(current.strip())

    return chunks


# ------------------------------------------------------------
# 4. SAVE TO DISK
# ------------------------------------------------------------

def save_sections(base_dir, sections, prefix):
    """Save section chunks to disk."""
    saved_files = []

    for i, (title, text) in enumerate(sections, start=1):
        chunks = chunk_text(text)
        section_dir = Path(base_dir) / f"{prefix}{i:02d}_{title.replace(' ', '_')}"

        os.makedirs(section_dir, exist_ok=True)

        for j, chunk in enumerate(chunks, start=1):
            filepath = section_dir / f"chunk_{j:02d}.txt"
            filepath.write_text(chunk)
            saved_files.append(str(filepath))

    return saved_files


def save_figures(base_dir, figures):
    os.makedirs(base_dir, exist_ok=True)
    out_index = []

    for idx, caption in figures:
        fp = Path(base_dir) / f"figure_{idx:02d}.txt"
        fp.write_text(caption)
        out_index.append(fp)

    return out_index


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------

def preprocess_pdf(pdf_path, output_dir):
    """
    Full preprocessing pipeline.
    Returns dict with all saved paths for RAG.
    """

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract TEI
    print("📄 Sending PDF to GROBID…")
    tei_xml = grobid_fulltext(pdf_path)

    tei_path = output_dir / "raw.tei.xml"
    tei_path.write_text(tei_xml)

    # 2. Parse TEI
    soup = BeautifulSoup(tei_xml, "xml")

    body_sections = extract_body_sections(soup)
    supp_sections = extract_supplementary_sections(soup)
    figures = extract_figures(soup)
    references = extract_references(soup)

    # 3. Save outputs
    paths = {
        "raw_tei": str(tei_path),
        "body_chunks": save_sections(output_dir / "body", body_sections, "section_"),
        "supplementary_chunks": save_sections(output_dir / "supplementary", supp_sections, "supp_"),
        "figure_captions": save_figures(output_dir / "figures", figures),
        "references_txt": str((output_dir / "references.txt").write_text("\n".join(references)))
    }

    return paths
