#!/usr/bin/env python3
"""
grobid_preprocessor.py

Preprocess PDFs via GROBID with improved main vs supplementary detection,
paragraph-aware chunking (keep paragraphs together unless single paragraph
exceeds chunk_size), figure reference normalization and per-chunk figure lists.

Usage:
    from grobid_preprocessor import preprocess_pdf
    results = preprocess_pdf(pdf_path, output_dir, chunk_size=4000, rag_mode=False)

Returns:
    dict with keys: raw_tei, title_txt, main_chunks, supp_chunks
"""

import os
import re
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup

# -----------------------
# Defaults & patterns
# -----------------------
DEFAULT_CHUNK_SIZE = 4000
SUPP_PATTERNS = [
    r"Supplementary Material",
    r"Supplementary Information",
    r"Supplemental Material",
    r"Supporting Information",
    r"Additional Information",
    r"Supplementary Methods",
    r"Supplementary Figures",
    r"Supplementary Table",
    r"Supplementary Text",
    r"SUPPLEMENTARY",
    r"Supplementary materials",
]

# -----------------------
# Utilities
# -----------------------
def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def clean_filename(text: str, maxlen: int = 120) -> str:
    s = re.sub(r'[\\/*?:"<>|]', "", text)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:maxlen]

# -----------------------
# TEI metadata extraction
# -----------------------
def extract_metadata_from_tei(tei_xml: str) -> Dict:
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    tei_header = soup.find("teiHeader")
    title = ""
    authors = []
    abstract_text = ""

    if tei_header:
        # Title
        t = tei_header.find("title", {"level": "a", "type": "main"})
        if not t:
            t = tei_header.find("title")
        title = t.get_text(" ", strip=True) if t else ""

        # Authors (analytic)
        analytic = tei_header.find("analytic")
        if analytic:
            for author in analytic.find_all("author"):
                forename = author.find("forename", {"type": "first"})
                surname = author.find("surname")
                first = forename.get_text(strip=True) if forename else ""
                last = surname.get_text(strip=True) if surname else ""
                if first or last:
                    authors.append({"first": first, "last": last})

        # Abstract
        abstract_tag = tei_header.find("abstract")
        if abstract_tag:
            # join possible <p> within abstract
            paras = abstract_tag.find_all("p")
            if paras:
                abstract_text = " ".join(p.get_text(" ", strip=True) for p in paras)
            else:
                abstract_text = abstract_tag.get_text(" ", strip=True)

    return {"title": title, "authors": authors, "abstract": abstract_text}

# -----------------------
# Extract figures (textual metadata)
# -----------------------
def extract_figures(tei_xml: str, output_dir: Path, rag_mode: bool = False, article_name: str = None):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    figures = soup.find_all("figure")
    fig_dir = output_dir if rag_mode else (output_dir / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    for fig in figures:
        fig_id = fig.get("xml:id") or fig.get("id") or "fig_unknown"
        head = fig.find("head")
        fig_title = head.get_text(" ", strip=True) if head else ""
        fig_desc = ""
        figDesc = fig.find("figDesc")
        if figDesc:
            fig_desc = figDesc.get_text(" ", strip=True)
        content = f"FIGURE ID: {fig_id}\nTITLE_IN_ARTICLE: {fig_title}\nDESCRIPTION:\n{fig_desc}\n"
        fname = f"{article_name}_{fig_id}.txt" if rag_mode and article_name else f"{fig_id}.txt"
        save_text(fig_dir / fname, content)

# -----------------------
# Extract references
# -----------------------
def extract_references(tei_xml: str, output_dir: Path):
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    list_bibl = soup.find("listBibl")
    out_file = output_dir / "references.txt"
    if not list_bibl:
        save_text(out_file, "NO REFERENCES FOUND\n")
        return
    out_lines = []
    for bibl in list_bibl.find_all("biblStruct"):
        ref_id = bibl.get("xml:id", "unknown_id")
        # title
        title_tag = bibl.find("title", {"level": "a", "type": "main"}) or bibl.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""
        # authors
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
        date_tag = bibl.find("date", {"type": "published"}) or bibl.find("date")
        year = ""
        if date_tag:
            year = date_tag.get("when") or date_tag.get_text(" ", strip=True)
        idnos = []
        for idno in bibl.find_all("idno"):
            id_type = idno.get("type", "unknown")
            value = idno.get_text(" ", strip=True)
            idnos.append(f"{id_type}:{value}")
        entry = [
            f"ID: {ref_id}",
            f"Title: {title}",
            f"Authors: {authors_str}",
            f"Journal: {journal}",
            f"Year: {year}",
            "IDNOS:"
        ]
        if idnos:
            entry += [f" - {x}" for x in idnos]
        else:
            entry.append(" - None")
        entry.append("-" * 50)
        out_lines.append("\n".join(entry))
    save_text(out_file, "\n\n".join(out_lines))

# -----------------------
# Paragraph cleaning and figure ref normalization
# -----------------------
def clean_paragraph_and_extract_figures(p_tag) -> Tuple[str, List[str]]:
    """
    - remove bibliographic inline refs (<ref type="bibr">)
    - normalize figure refs (<ref type="figure" target="#fig_1">3(a)</ref>) -> (fig fig_1: 3(a))
    - collect fig ids in a list
    """
    # remove bibliography refs
    for b in p_tag.find_all("ref", {"type": "bibr"}):
        b.decompose()

    figures = []
    for f in p_tag.find_all("ref", {"type": "figure"}):
        fig_text = f.get_text(" ", strip=True)
        fig_target = f.get("target", "").lstrip("#")
        if not fig_target:
            # fallback: try to extract number from text
            m = re.search(r"fig(?:ure)?\s*[_\s]?(\d+)", fig_text, re.IGNORECASE)
            fig_target = m.group(1) if m else "unknown"
        label = f"{fig_target}: {fig_text}"
        figures.append(label)
        # replace node content with normalized text
        f.string = f"(fig {fig_target}: {fig_text})"

    # get cleaned text
    cleaned = p_tag.get_text(" ", strip=True)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, sorted(set(figures))

# -----------------------
# Main vs supplementary detection helpers
# -----------------------
def count_paragraphs_in_div(div) -> int:
    return len(div.find_all("p"))

def detect_supplementary_by_keyword(div_text: str) -> bool:
    for pat in SUPP_PATTERNS:
        if re.search(pat, div_text, re.IGNORECASE):
            return True
    return False

# -----------------------
# Extract sections & chunking with the 3-step logic
# -----------------------
def extract_sections(tei_xml: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (main_chunks, supplementary_chunks)
    Each chunk: {"text": "...", "figures": [...], "section_title": "..."}
    """
    soup = BeautifulSoup(tei_xml, "lxml-xml")

    # focus on the <body> if present
    body = soup.find("body")
    divs = body.find_all("div") if body else soup.find_all("div")

    # If no divs found, fallback: treat entire text as single main div
    if not divs:
        full_text = soup.get_text(" ", strip=True)
        cleaned_text = re.sub(r"\s+", " ", full_text).strip()
        return ([{"text": cleaned_text, "figures": [], "section_title": None}], [])

    # STEP 1: structural detection using first div(s)
    div_paragraph_counts = [count_paragraphs_in_div(d) for d in divs]
    main_div_index = None
    supp_div_index = None
    # Rule: if div0 has >=5 paragraphs -> div0=main, next sizable div -> supplementary
    if div_paragraph_counts[0] >= 5:
        main_div_index = 0
        # find first subsequent div with >=3 paragraphs as supplementary
        for i in range(1, len(divs)):
            if div_paragraph_counts[i] >= 3:
                supp_div_index = i
                break
    else:
        # div0 small: try div1 as main
        if len(div_paragraph_counts) > 1 and div_paragraph_counts[1] >= 5:
            main_div_index = 1
            # next sizable as supplementary
            for i in range(2, len(divs)):
                if div_paragraph_counts[i] >= 3:
                    supp_div_index = i
                    break

    # STEP 2: if structural failed, use keyword fallback scanning div-by-div
    if main_div_index is None:
        # find first div that does not look like supplementary — prefer large ones
        for i, d in enumerate(divs):
            text = d.get_text(" ", strip=True)
            if not detect_supplementary_by_keyword(text) and div_paragraph_counts[i] >= 3:
                main_div_index = i
                break
        # if still none, fallback: choose first div as main
        if main_div_index is None:
            main_div_index = 0

        # find first div after main that matches supplementary pattern
        for j in range(main_div_index + 1, len(divs)):
            if detect_supplementary_by_keyword(divs[j].get_text(" ", strip=True)) or div_paragraph_counts[j] >= 3:
                supp_div_index = j
                break

    # STEP 3: default: if still no supplementary, leave supp_div_index None
    # Now build lists of divs for main and supplementary based on indices
    main_divs = []
    supp_divs = []
    if supp_div_index is not None:
        # everything between main_div_index..(supp_div_index-1) -> main
        for i in range(main_div_index, supp_div_index):
            main_divs.append(divs[i])
        # everything from supp_div_index..end -> supplementary
        for i in range(supp_div_index, len(divs)):
            supp_divs.append(divs[i])
    else:
        # no supplementary found -> every div from main_div_index..end is main
        for i in range(main_div_index, len(divs)):
            main_divs.append(divs[i])

    # Helper to produce chunks from a list of divs
    def produce_chunks_from_divs(div_list: List, label: str) -> List[Dict]:
        chunks = []
        current_chunk_text = ""
        current_chunk_figs = set()
        current_section_title = None

        def flush_chunk():
            nonlocal current_chunk_text, current_chunk_figs, current_section_title
            if not current_chunk_text.strip():
                return
            # append figure summary line
            if current_chunk_figs:
                fig_line = "\nFigures used in this chunk: " + ", ".join(sorted(current_chunk_figs))
                current_chunk_text = current_chunk_text.rstrip() + "\n\n" + fig_line + "\n"
            chunks.append({
                "text": current_chunk_text.strip(),
                "figures": sorted(current_chunk_figs),
                "section_title": current_section_title
            })
            current_chunk_text = ""
            current_chunk_figs = set()
            current_section_title = None

        for div in div_list:
            # section title if any
            head = div.find("head")
            head_text = head.get_text(" ", strip=True) if head else None
            if head_text:
                # if current chunk is non-empty and adding the title would exceed chunk_size then flush first
                if current_chunk_text and (len(current_chunk_text) + len(head_text) > chunk_size):
                    flush_chunk()
                # start a new section title in the chunk
                if current_chunk_text:
                    current_chunk_text += "\n" + head_text + "\n"
                else:
                    current_chunk_text = head_text + "\n"
                current_section_title = head_text

            # iterate paragraphs in this div
            paragraphs = div.find_all("p")
            for p in paragraphs:
                cleaned_para, figs = clean_paragraph_and_extract_figures(p)
                # if single paragraph itself bigger than chunk_size: we will place it alone (can't avoid splitting)
                if len(cleaned_para) > chunk_size:
                    # flush existing chunk first
                    if current_chunk_text:
                        flush_chunk()
                    # split long paragraph into sub-parts while keeping words intact
                    start = 0
                    while start < len(cleaned_para):
                        part = cleaned_para[start:start+chunk_size]
                        # attempt not to cut mid-word: move end back to last space
                        if start + chunk_size < len(cleaned_para):
                            last_space = part.rfind(" ")
                            if last_space > 100:  # only if reasonable
                                part = part[:last_space]
                        chunk_text = part.strip()
                        chunk_fig_set = set(figs)
                        # append figure list if any
                        if chunk_fig_set:
                            chunk_text = chunk_text + "\n\nFigures used in this chunk: " + ", ".join(sorted(chunk_fig_set)) + "\n"
                        chunks.append({"text": chunk_text, "figures": sorted(chunk_fig_set), "section_title": current_section_title})
                        start += len(part)
                    # reset current chunk
                    current_chunk_text = ""
                    current_chunk_figs = set()
                else:
                    # try to append paragraph to current chunk (prefer grouping multiple small paras)
                    if not current_chunk_text:
                        # start new chunk with this paragraph
                        current_chunk_text = cleaned_para + "\n"
                        current_chunk_figs.update(figs)
                    else:
                        # if appending this paragraph would exceed chunk_size, flush current chunk and start new
                        if len(current_chunk_text) + len(cleaned_para) > chunk_size:
                            flush_chunk()
                            current_chunk_text = cleaned_para + "\n"
                            current_chunk_figs.update(figs)
                        else:
                            # safe to append
                            current_chunk_text += cleaned_para + "\n"
                            current_chunk_figs.update(figs)

            # after finishing div, continue to next div (do not auto-flush unless needed)
        # end div loop
        # flush final chunk
        flush_chunk()
        return chunks

    main_chunks = produce_chunks_from_divs(main_divs, "main")
    supp_chunks = produce_chunks_from_divs(supp_divs, "supplementary") if supp_divs else []

    return main_chunks, supp_chunks

# -----------------------
# Main preprocessing entrypoint
# -----------------------
def preprocess_pdf(pdf_path: str, output_dir: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                   rag_mode: bool = False, article_name: str = None, keep_references: bool = False) -> Dict:
    """
    Preprocess a PDF via a running GROBID instance.

    Args:
        pdf_path: input PDF path
        output_dir: directory where outputs will be saved
        chunk_size: target maximum characters per chunk (paragraph-aware)
        rag_mode: if True, save figure files into output_dir and prefix files for RAG usage
        article_name: optional prefix when rag_mode=True
        keep_references: if True, also save a version of chunks with full references appended

    Returns:
        dict with summary information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Preprocessing PDF with GROBID: {pdf_path}")
    # 1) send to GROBID
    url = "http://localhost:8070/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        r = requests.post(url, files={"input": f})
    tei_xml = r.text

    # save raw TEI
    raw_tei_path = output_dir / "raw_tei.xml"
    save_text(raw_tei_path, tei_xml)
    print(f"📄 Raw TEI saved: {raw_tei_path}")

    # 2) extract metadata
    meta = extract_metadata_from_tei(tei_xml)
    title_txt_path = output_dir / "title.txt"
    with open(title_txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Title: {meta.get('title','')}\n\n")
        fh.write("Authors:\n")
        for a in meta.get("authors", []):
            fh.write(f" - {a.get('first','')} {a.get('last','')}\n")
        fh.write(f"\nAbstract:\n{meta.get('abstract','')}\n")
    print(f"📝 Metadata saved: {title_txt_path}")

    # 3) extract figures (textual metadata)
    extract_figures(tei_xml, output_dir, rag_mode=rag_mode, article_name=article_name)
    print("📷 Figure metadata extracted.")

    # 4) extract references (unless rag mode)
    if not rag_mode:
        extract_references(tei_xml, output_dir)
        print("📚 References extracted.")

    # 5) extract sections & chunking
    main_chunks, supp_chunks = extract_sections(tei_xml, chunk_size=chunk_size)
    print(f"✂️  Chunking done: {len(main_chunks)} main chunks, {len(supp_chunks)} supplementary chunks (chunk_size={chunk_size})")

    # 6) save chunks
    if rag_mode:
        # single folder outputs for RAG: prefix files with article_name
        if article_name:
            prefix_main = f"{clean_filename(article_name)}_main_chunk"
            prefix_sup = f"{clean_filename(article_name)}_sup_chunk"
        else:
            prefix_main = "main_chunk"
            prefix_sup = "sup_chunk"
        for i, c in enumerate(main_chunks, 1):
            fname = output_dir / f"{prefix_main}{str(i).zfill(3)}.txt"
            save_text(fname, c["text"])
        for i, c in enumerate(supp_chunks, 1):
            fname = output_dir / f"{prefix_sup}{str(i).zfill(3)}.txt"
            save_text(fname, c["text"])
        # optionally remove raw tei to keep folder tidy
    else:
        main_dir = output_dir / "main"
        supp_dir = output_dir / "supplementary"
        main_dir.mkdir(parents=True, exist_ok=True)
        supp_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(main_chunks, 1):
            fname = main_dir / f"main_chunk{str(i).zfill(3)}.txt"
            save_text(fname, c["text"])
        for i, c in enumerate(supp_chunks, 1):
            fname = supp_dir / f"sup_chunk{str(i).zfill(3)}.txt"
            save_text(fname, c["text"])

    # 7) save fig lists and maps (standard mode)
    if not rag_mode:
        main_figs = set()
        supp_figs = set()
        main_map = {}
        supp_map = {}
        for idx, c in enumerate(main_chunks, 1):
            for fig in c.get("figures", []):
                main_figs.add(fig)
                main_map.setdefault(fig, []).append(str(idx).zfill(3))
        for idx, c in enumerate(supp_chunks, 1):
            for fig in c.get("figures", []):
                supp_figs.add(fig)
                supp_map.setdefault(fig, []).append(str(idx).zfill(3))
        save_text(output_dir / "figmain.txt", "\n".join(sorted(main_figs)))
        save_text(output_dir / "figsup.txt", "\n".join(sorted(supp_figs)))
        with open(output_dir / "figmain_map.txt", "w", encoding="utf-8") as fm:
            for k in sorted(main_map):
                fm.write(f"{k} -> {', '.join(main_map[k])}\n")
        with open(output_dir / "figsup_map.txt", "w", encoding="utf-8") as fm:
            for k in sorted(supp_map):
                fm.write(f"{k} -> {', '.join(supp_map[k])}\n")

    # 8) optional: save chunks with references appended
    if keep_references and not rag_mode:
        refs_text = (output_dir / "references.txt").read_text(encoding="utf-8") if (output_dir / "references.txt").exists() else ""
        wr_dir = output_dir / "with_references"
        wr_main = wr_dir / "main"
        wr_supp = wr_dir / "supplementary"
        wr_main.mkdir(parents=True, exist_ok=True)
        wr_supp.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(main_chunks, 1):
            fname = wr_main / f"main_chunk{str(i).zfill(3)}.txt"
            save_text(fname, c["text"] + "\n\n" + refs_text)
        for i, c in enumerate(supp_chunks, 1):
            fname = wr_supp / f"sup_chunk{str(i).zfill(3)}.txt"
            save_text(fname, c["text"] + "\n\n" + refs_text)

    print("✅ Preprocessing complete.")

    return {
        "raw_tei": str(raw_tei_path) if raw_tei_path.exists() else None,
        "title_txt": str(title_txt_path),
        "main_chunks": len(main_chunks),
        "supp_chunks": len(supp_chunks),
    }
