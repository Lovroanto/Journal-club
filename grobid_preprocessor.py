#!/usr/bin/env python3
"""
grobid_preprocessor.py

Processes scientific PDFs using GROBID:
- Extracts main text sections
- Extracts figures & captions
- Extracts references
- Extracts supplementary material
- Splits large sections into chunks (for RAG)

Author: ChatGPT (2025)
"""

import os
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# ----------------------------
# GROBID CONFIG
# ----------------------------
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# ----------------------------
# UTILITY
# ----------------------------

def clean_filename(s):
    s = re.sub(r"[^a-zA-Z0-9_\- ]", "", s)
    s = s.replace(" ", "_")
    return s[:80]


def chunk_text(text, max_words=1800):
    words = text.split()
    chunks = []
    current = []

    for w in words:
        current.append(w)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks

# ----------------------------
# GROBID CALL
# ----------------------------

def grobid_process(pdf_path):
    with open(pdf_path, "rb") as f:
        files = {"input": f}
        response = requests.post(
            GROBID_URL,
            files=files,
            data={"consolidateHeader": 1, "consolidateCitations": 1},
            timeout=300
        )

    if response.status_code != 200:
        raise RuntimeError(f"GROBID error: {response.status_code}")

    return response.text

# ----------------------------
# XML PARSING
# ----------------------------

def extract_sections(xml_text):
    soup = BeautifulSoup(xml_text, "xml")

    # ----------------------------
    # MAIN TEXT
    # ----------------------------
    main_sections = []
    body = soup.find("body")

    if body:
        for div in body.find_all("div"):
            head = div.find("head")
            title = head.text.strip() if head else "Untitled"

            paragraphs = " ".join(p.text.strip() for p in div.find_all("p"))
            if paragraphs.strip():
                main_sections.append((title, paragraphs))

    # ----------------------------
    # FIGURES
    # ----------------------------
    figures = []
    for fig in soup.find_all("figure"):
        caption = ""
        if fig.figDesc:
            caption = fig.figDesc.text.strip()
        elif fig.find("label"):
            caption = fig.find("label").text.strip()

        if caption:
            figures.append(caption)

    # ----------------------------
    # REFERENCES
    # ----------------------------
    references = []
    biblio = soup.find("listBibl")

    if biblio:
        for b in biblio.find_all("biblStruct"):
            ref_text = " ".join(t.text.strip() for t in b.find_all())
            references.append(ref_text)

    # ----------------------------
    # SUPPLEMENTARY SECTIONS
    # (div type="annex", "appendix", "supplementaryMaterial")
    # ----------------------------
    supplementary_sections = []

    for sup in soup.find_all("div"):
        t = (sup.get("type") or "").lower()
        if t in ["annex", "appendix", "supplementarymaterial", "supplementary"]:
            title = sup.head.text.strip() if sup.head else "Supplementary"
            text = " ".join(p.text.strip() for p in sup.find_all("p"))
            if text.strip():
                supplementary_sections.append((title, text))

    return main_sections, figures, references, supplementary_sections


# ----------------------------
# SAVE OUTPUT
# ----------------------------

def preprocess_pdf(pdf_path, output_dir):

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Processing PDF with GROBID: {pdf_path}")

    # 1. Run GROBID
    xml_text = grobid_process(pdf_path)

    # 2. Extract sections
    main_sections, figures, references, supp_sections = extract_sections(xml_text)

    # -----------------------------------------
    # SAVE MAIN SECTIONS (chunked)
    # -----------------------------------------
    main_out = output_dir / "main"
    main_out.mkdir(exist_ok=True)

    main_files = []
    for title, text in main_sections:
        chunks = chunk_text(text)

        base = clean_filename(title)
        for i, chunk in enumerate(chunks):
            fn = main_out / f"{base}_chunk{i+1}.txt"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(chunk)
            main_files.append(str(fn))

    # -----------------------------------------
    # SAVE FIGURES
    # -----------------------------------------
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig_index = fig_dir / "figure_captions_index.txt"
    with open(fig_index, "w", encoding="utf-8") as f:
        for i, caption in enumerate(figures, 1):
            fname = f"figure_{i}.txt"
            with open(fig_dir / fname, "w", encoding="utf-8") as cf:
                cf.write(caption)
            f.write(f"{fname}\n")

    # -----------------------------------------
    # SAVE REFERENCES
    # -----------------------------------------
    ref_file = output_dir / "references.txt"
    with open(ref_file, "w", encoding="utf-8") as f:
        for r in references:
            f.write(r + "\n\n")

    # -----------------------------------------
    # SAVE SUPPLEMENTARY
    # -----------------------------------------
    supp_dir = output_dir / "supplementary"
    supp_dir.mkdir(exist_ok=True)

    supp_files = []
    for title, text in supp_sections:
        chunks = chunk_text(text)
        base = clean_filename(title)

        for i, chunk in enumerate(chunks):
            fn = supp_dir / f"{base}_chunk{i+1}.txt"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(chunk)
            supp_files.append(str(fn))

    # FULL supplementary (merged)
    full_supp = supp_dir / "all_supplementary.txt"
    with open(full_supp, "w", encoding="utf-8") as f:
        for _, text in supp_sections:
            f.write(text + "\n\n")

    # -----------------------------------------
    # RETURN SUMMARY
    # -----------------------------------------
    return {
        "main_chunks": main_files,
        "num_main_chunks": len(main_files),
        "figures": len(figures),
        "references_file": str(ref_file),
        "supplementary_chunks": supp_files,
        "num_supplementary_chunks": len(supp_files),
        "supplementary_full": str(full_supp),
    }
