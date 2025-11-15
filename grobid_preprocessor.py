#!/usr/bin/env python3
"""
grobid_preprocessor.py

Preprocess PDFs via GROBID with:
- Metadata extraction
- Main/supplementary separation
- Dynamic chunking
- Figure extraction
- Optional modes:
  - Standard mode: all features including references, with/without keeping references in chunks
  - RAG mode: only chunks + figures, names prefixed with article_name
"""

import os
import re
import requests
from pathlib import Path
from typing import Dict
from bs4 import BeautifulSoup

DEFAULT_CHUNK_SIZE = 4000
SUPP_PATTERNS = [
    r"Supplementary Material",
    r"Supplementary Information",
    r"Supplemental Material",
    r"Supporting Information",
    r"Additional Information",
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
    title_tag = tei_header.find("title", {"level": "a", "type": "main"})
    title = title_tag.get_text(strip=True) if title_tag else ""
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
    abstract_tag = tei_header.find("abstract")
    abstract_text = ""
    if abstract_tag:
        abstract_text = " ".join(p.get_text(" ", strip=True) for p in abstract_tag.find_all("p"))
    return {"title": title, "authors": authors, "abstract": abstract_text}


# -----------------------
# Clean paragraph and extract figures
# -----------------------
def clean_paragraph(p_tag):
    for b in p_tag.find_all("ref", {"type": "bibr"}):
        b.decompose()
    figures = []
    for f in p_tag.find_all("ref", {"type": "figure"}):
        fig_text = f.get_text(strip=True)
        fig_id = f.get("target", "")
        label = f"{fig_id}: {fig_text}"
        figures.append(label)
        f.string = f"(fig {label})"
    return p_tag.get_text(" ", strip=True), figures


# -----------------------
# Extract figures
# -----------------------
def extract_figures(tei_xml: str, output_dir: Path, rag_mode=False, article_name: str = None):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    figure_tags = soup.find_all("figure")

    # In RAG mode, keep figures in output_dir; else in 'figures/' subfolder
    fig_dir = output_dir if rag_mode else output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for fig in figure_tags:
        fig_id = fig.get("xml:id", "unknown_id")

        # Extract <head> (figure number/title inside article)
        head_tag = fig.find("head")
        fig_title = head_tag.get_text(" ", strip=True) if head_tag else ""

        # Extract <figDesc>
        desc_tag = fig.find("figDesc")
        fig_desc = desc_tag.get_text(" ", strip=True) if desc_tag else ""

        # Compose output
        content = f"FIGURE ID: {fig_id}\nTITLE IN ARTICLE: {fig_title}\nDESCRIPTION:\n{fig_desc}\n"

        # Determine file name
        if rag_mode and article_name:
            file_name = f"{article_name}_{fig_id}.txt"
        else:
            file_name = f"{fig_id}.txt"

        save_text(fig_dir / file_name, content)


# -----------------------
# Extract references
# -----------------------
def extract_references(tei_xml: str, output_dir: Path):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    list_bibl = soup.find("listBibl")
    out_file = output_dir / "references.txt"
    if list_bibl is None:
        save_text(out_file, "NO REFERENCES FOUND\n")
        return
    out_lines = []
    for bibl in list_bibl.find_all("biblStruct"):
        ref_id = bibl.get("xml:id", "unknown_id")
        title_tag = bibl.find("title", {"level": "a", "type": "main"})
        title = title_tag.get_text(" ", strip=True) if title_tag else ""
        authors = []
        for pers in bibl.find_all("persName"):
            fn = pers.find("forename", {"type": "first"})
            sn = pers.find("surname")
            first = fn.get_text(strip=True) if fn else ""
            last = sn.get_text(strip=True) if sn else ""
            if first or last:
                authors.append(f"{first} {last}".strip())
        authors_str = ", ".join(authors)
        journal_tag = bibl.find("title", {"level": "j"})
        journal = journal_tag.get_text(" ", strip=True) if journal_tag else ""
        date_tag = bibl.find("date", {"type": "published"})
        year = date_tag.get("when") if date_tag else (date_tag.get_text(strip=True) if date_tag else "")
        idnos = []
        for idno in bibl.find_all("idno"):
            id_type = idno.get("type", "unknown")
            value = idno.get_text(strip=True)
            idnos.append(f"type={id_type} → {value}")
        entry = [
            f"ID: {ref_id}",
            f"Title: {title}",
            f"Authors: {authors_str}",
            f"Journal: {journal}",
            f"Year: {year}",
            "IDNOS:"]
        if idnos:
            entry.extend([f" - {n}" for n in idnos])
        else:
            entry.append(" - None")
        entry.append("-" * 40)
        out_lines.append("\n".join(entry))
    save_text(out_file, "\n".join(out_lines))


# -----------------------
# Extract sections & chunks
# -----------------------
def extract_sections(tei_xml: str, chunk_size=DEFAULT_CHUNK_SIZE):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    divs = soup.find_all("div")
    main_chunks, supp_chunks = [], []
    current_section = "main"
    suppl_started_by_header = False
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
        if (not suppl_started_by_header) and head_tag:
            if any(re.search(pat, head_text, re.IGNORECASE) for pat in SUPP_PATTERNS):
                suppl_started_by_header = True
                finalize_chunk()
                current_section = "supplementary"
        if head_tag:
            if chunk_text:
                finalize_chunk()
            chunk_text = f"{head_text}\n"
        for p in div.find_all("p"):
            para_text, figures = clean_paragraph(p)
            if len(chunk_text) + len(para_text) > chunk_size and chunk_text:
                finalize_chunk()
            chunk_text += para_text + "\n"
            chunk_figures.update(figures)
    finalize_chunk()
    return main_chunks, supp_chunks


# -----------------------
# Main processing
# -----------------------
def preprocess_pdf(pdf_path: str, output_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                   rag_mode: bool = False, article_name: str = None, keep_references: bool = False) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) GROBID ----
    url = "http://localhost:8070/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        r = requests.post(url, files={"input": f})
    tei_xml = r.text

    raw_tei_path = output_dir / "raw_tei.xml"
    save_text(raw_tei_path, tei_xml)

    # ---- 2) Metadata ----
    meta = extract_metadata_from_tei(tei_xml)
    title_txt_path = output_dir / "title.txt"
    with open(title_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {meta['title']}\nAuthors:\n")
        for a in meta["authors"]:
            f.write(f"{a['first']} {a['last']}\n")
        f.write(f"\nAbstract:\n{meta['abstract']}\n")

    # ---- 3) Figures ----
    extract_figures(tei_xml, output_dir, rag_mode=rag_mode, article_name=article_name)

    # ---- 4) References (only in standard mode) ----
    if not rag_mode:
        extract_references(tei_xml, output_dir)

    # ---- 5) Sections & chunks ----
    main_chunks, supp_chunks = extract_sections(tei_xml, chunk_size)

    # ---- 6) Save chunks ----
    if rag_mode:
        # All in one folder
        main_dir = supp_dir = output_dir
        prefix_main = f"{article_name}_chunk_main" if article_name else "main_chunk"
        prefix_sup = f"{article_name}_chunk_sup" if article_name else "sup_chunk"
        # Remove raw TEI to avoid confusion
        if raw_tei_path.exists():
            raw_tei_path.unlink()
    else:
        main_dir = output_dir / "main"
        supp_dir = output_dir / "supplementary"
        main_dir.mkdir(exist_ok=True)
        supp_dir.mkdir(exist_ok=True)
        prefix_main = "main_chunk"
        prefix_sup = "sup_chunk"

    main_figs, supp_figs = set(), set()
    main_fig_map, supp_fig_map = {}, {}

    # ---- Save main chunks ----
    for i, c in enumerate(main_chunks, 1):
        fname = main_dir / f"{prefix_main}{str(i).zfill(2)}.txt"
        save_text(fname, c["text"])
        for fig in c["figures"]:
            main_figs.add(fig)
            main_fig_map.setdefault(fig, []).append(str(i).zfill(2))

    # ---- Save supplementary chunks ----
    for i, c in enumerate(supp_chunks, 1):
        fname = supp_dir / f"{prefix_sup}{str(i).zfill(2)}.txt"
        save_text(fname, c["text"])
        for fig in c["figures"]:
            supp_figs.add(fig)
            supp_fig_map.setdefault(fig, []).append(str(i).zfill(2))

    # ---- Save figure lists & maps (only standard mode) ----
    if not rag_mode:
        save_text(output_dir / "figmain.txt", "\n".join(sorted(main_figs)))
        save_text(output_dir / "figsup.txt", "\n".join(sorted(supp_figs)))
        with open(output_dir / "figmain_map.txt", "w", encoding="utf-8") as f:
            for fig in sorted(main_fig_map):
                f.write(f"{fig} → {', '.join(main_fig_map[fig])}\n")
        with open(output_dir / "figsup_map.txt", "w", encoding="utf-8") as f:
            for fig in sorted(supp_fig_map):
                f.write(f"{fig} → {', '.join(supp_fig_map[fig])}\n")

    # ---- Optional: save chunks WITH references ----
    if keep_references and not rag_mode:
        wr_dir = output_dir / "with_references"
        wr_main_dir = wr_dir / "main"
        wr_supp_dir = wr_dir / "supplementary"
        wr_main_dir.mkdir(parents=True, exist_ok=True)
        wr_supp_dir.mkdir(parents=True, exist_ok=True)
        references_text = Path(output_dir / "references.txt").read_text() if (output_dir / "references.txt").exists() else ""
        for i, c in enumerate(main_chunks, 1):
            fname = wr_main_dir / f"{prefix_main}{str(i).zfill(2)}.txt"
            save_text(fname, c["text"] + "\n" + references_text)
        for i, c in enumerate(supp_chunks, 1):
            fname = wr_supp_dir / f"{prefix_sup}{str(i).zfill(2)}.txt"
            save_text(fname, c["text"] + "\n" + references_text)

    return {
        "raw_tei": str(raw_tei_path) if raw_tei_path.exists() else None,
        "title_txt": str(title_txt_path),
        "main_chunks": len(main_chunks),
        "supp_chunks": len(supp_chunks),
    }
