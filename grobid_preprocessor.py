#!/usr/bin/env python3
"""
grobid_preprocessor.py

Preprocess PDFs via GROBID with improved main vs supplementary detection,
paragraph-aware chunking (keep paragraphs together unless single paragraph
exceeds chunk_size), figure reference normalization and per-chunk figure lists.

New options:
 - preclean_pdf (bool): if True, attempt to pre-clean the PDF using Ghostscript/qpdf
                        and send a temporary cleaned PDF to GROBID (temp file in output_dir).
 - use_grobid_consolidation (bool): if True, request GROBID consolidation params
                                   (consolidateHeader=1&consolidateCitations=1).

Usage:
    results = preprocess_pdf(pdf_path, output_dir,
                             chunk_size=4000,
                             rag_mode=False,
                             article_name=None,
                             keep_references=False,
                             preclean_pdf=True,
                             use_grobid_consolidation=True)
"""

import os
import re
import shutil
import subprocess
import requests
import fitz
from typing import Union, Tuple, Optional
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
    r"Methods",
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
# PDF pre-clean helpers (temporary cleaned file)
# -----------------------
def tool_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def try_ghostscript_clean(input_pdf: str, output_pdf: str) -> bool:
    """
    Use Ghostscript to rebuild the PDF (often fixes structure).
    Returns True on success.
    """
    if not tool_exists("gs"):
        return False
    try:
        # -dPDFSETTINGS=/prepress ensures quality, -dNOPAUSE -dBATCH for noninteractive
        gs_cmd = [
            "gs", "-q",
            "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.7",
            "-dPDFSETTINGS=/prepress",
            "-sOutputFile=" + str(output_pdf),
            str(input_pdf)
        ]
        subprocess.run(gs_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def try_qpdf_linearize(input_pdf: str, output_pdf: str) -> bool:
    """
    Use qpdf to linearize / rewrite PDF.
    Returns True on success.
    """
    if not tool_exists("qpdf"):
        return False
    try:
        qpdf_cmd = ["qpdf", "--linearize", str(input_pdf), str(output_pdf)]
        subprocess.run(qpdf_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        # fallback: try simple qpdf --recompress-streams rewrite
        try:
            qpdf_cmd2 = ["qpdf", str(input_pdf), str(output_pdf)]
            subprocess.run(qpdf_cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

def produce_temp_cleaned_pdf(original_pdf: str, output_dir: Path) -> str:
    """
    Create a temporary cleaned PDF in output_dir/tmp_cleaned.pdf.
    Try Ghostscript first, then qpdf. If none available, copy original.
    Return full path to the file to use (string).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_pdf = output_dir / "tmp_cleaned.pdf"

    # 1) Try Ghostscript
    if try_ghostscript_clean(original_pdf, tmp_pdf):
        return str(tmp_pdf)

    # 2) Try qpdf
    if try_qpdf_linearize(original_pdf, tmp_pdf):
        return str(tmp_pdf)

    # 3) Fallback: copy original to temp location
    try:
        shutil.copyfile(original_pdf, tmp_pdf)
        return str(tmp_pdf)
    except Exception:
        # last-resort: return original path
        return str(original_pdf)

# -----------------------
# This part was added afterwards
# This is for checking the grobid output

def extract_pdf_text(pdf_path: Union[str, Path]) -> str:
    """Extract all text from PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text").strip() for page in doc]
    doc.close()
    return "\n\n".join(pages)


# ------------------------------------------------------------------ #
# 1. Caption extractor (your original – unchanged)
# ------------------------------------------------------------------ #
def extract_and_remove_figure_captions(text: str, out_dir: Path) -> Tuple[str, int]:
    pattern = re.compile(
        r"(?:(?:^|\n)(?:Fig(?:ure)?\.?|Extended Data Fig\.|Supplementary Fig(?:ure)?\.?)\s*\d+[a-zA-Z]?(?:[.:)]|[\s|–-]))"
        r"[\s\S]*?(?=\n\s*\n|(?:(?:Fig(?:ure)?|Extended Data Fig\.|Supplementary Fig\.|References|Supplementary|Methods)\b)|\Z)",
        flags=re.MULTILINE,
    )
    captions = [m.group(0).strip() for m in pattern.finditer(text)]
    for i, cap in enumerate(captions, 1):
        (out_dir / f"figure_{i:03d}.txt").write_text(cap + "\n", encoding="utf-8")
    return pattern.sub("\n", text), len(captions)


# ------------------------------------------------------------------ #
# 2. Axis-label / equation junk remover (your original – unchanged)
# ------------------------------------------------------------------ #
def remove_figure_axis_labels(text: str) -> str:
    lines = text.splitlines()
    new_lines = []
    for l in lines:
        stripped = l.strip()
        if re.fullmatch(r"[\d\s\.\(\)××eE\-\+]+", stripped):
            continue
        if re.search(r"\b(kHz|Hz|s|ms|nm|μm|cm|dB|mol|K|°C|×10)\b", stripped):
            continue
        if len(stripped) <= 2:
            continue
        tokens = stripped.split()
        if len(tokens) >= 3 and sum(len(t) < 3 for t in tokens) > len(tokens) / 2:
            continue
        new_lines.append(l)
    return "\n".join(new_lines)


# ------------------------------------------------------------------ #
# 3. Extra safety for pure-math lines
# ------------------------------------------------------------------ #
def is_equation_block(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 100:
        return False
    if re.fullmatch(r"[±√∞α-ωΑ-Ω0-9δΔ∇Γ±=+\-*/^√∞≈∑∏∫≤≥≠±÷√∞πθφμλστΩΔ∇⋅×·∑∏∫]+", s):
        return True
    if re.search(r"[α-ωΑ-Ω]", s) and len(re.findall(r"\w", s)) < 3:
        return True
    if s.count("=") > 1 or s.count("+") > 2 or s.count("–") > 1:
        return True
    return False


# ------------------------------------------------------------------ #
# 4. Detect References block by heading + consecutive numbers
# ------------------------------------------------------------------ #
def find_references_block(text: str) -> Optional[Tuple[int, int]]:
    """
    Returns (start, end) indices of the References block, or None.
    """
    # 1. Find possible headings
    heading_pat = re.compile(
        r"(?mi)^.*\b("
        r"References|Bibliography|Literature|References\s+and\s+notes|"
        r"Reference|References?[\s:]?"
        r")\b.*$"
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if heading_pat.search(line):
            # 2. Look at the next 15 lines for consecutive numbers
            snippet = "\n".join(lines[i + 1 : i + 16])
            numbers = re.findall(r"^\s*(\d+)[\.\)]\s", snippet, flags=re.MULTILINE)
            if len(numbers) >= 3:
                # at least 3 numbers, check if they are roughly consecutive
                nums = [int(n) for n in numbers]
                diffs = [nums[j + 1] - nums[j] for j in range(len(nums) - 1)]
                if any(0 < d <= 3 for d in diffs):  # allow small gaps
                    # Found a real reference list
                    start = text.find(line)
                    # Find end: next major section or EOF
                    rest = text[start:]
                    end_match = re.search(
                        r"(?mi)^(\s*(Supplementary|Appendix|Methods|Acknowledgments|Author|Data)\b)",
                        rest,
                        flags=re.MULTILINE,
                    )
                    if end_match:
                        block_end = start + end_match.start()
                    else:
                        block_end = len(text)
                    return start, block_end
    return None


# ------------------------------------------------------------------ #
# 5. MAIN FUNCTION
# ------------------------------------------------------------------ #
def prepare_pdf_for_grobid_check(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    preview_lines: int = 30,
    save_cleaned: bool = True,
    cleaned_filename: str = "cleaned_article.txt",
) -> str:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figure_from_pdf"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------- #
    # 1. Extract pages + save images
    # -------------------------------------------------- #
    doc = fitz.open(pdf_path)
    raw_pages = []
    img_cnt = 0
    for page in doc:
        raw_pages.append(page.get_text("text"))
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            img_cnt += 1
            img_id = f"fig_{img_cnt:04d}"
            (figure_dir / f"{img_id}.{base['ext']}").write_bytes(base["image"])
    doc.close()
    full_text = "\n\n".join(raw_pages)

    # -------------------------------------------------- #
    # 2. REMOVE REFERENCES BLOCK (smart detection)
    # -------------------------------------------------- #
    refs_block = find_references_block(full_text)
    if refs_block:
        start, end = refs_block
        body_text = full_text[:start]
        post_refs = full_text[end:]  # KEEP EVERYTHING AFTER
    else:
        body_text = full_text
        post_refs = ""

    # -------------------------------------------------- #
    # 3. Clean a section
    # -------------------------------------------------- #
    def clean_section(txt: str) -> str:
        if not txt.strip():
            return ""
        txt, _ = extract_and_remove_figure_captions(txt, figure_dir)
        txt = remove_figure_axis_labels(txt)

        lines = txt.splitlines()
        kept = []
        for line in lines:
            if is_equation_block(line):
                continue
            s = line.strip()
            if len(s) < 100 and (
                s.isdigit()
                or re.match(r"^https?://", s)
                or re.match(r"^[A-Z]{3,}$", s)
                or re.match(r"^\d{1,3}$", s)
            ):
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    cleaned_body = clean_section(body_text)
    cleaned_post = clean_section(post_refs) if post_refs else ""

    # -------------------------------------------------- #
    # 4. Re-assemble
    # -------------------------------------------------- #
    final_text = cleaned_body
    if cleaned_post:
        final_text += "\n\n" + cleaned_post

    # -------------------------------------------------- #
    # 5. Save to disk
    # -------------------------------------------------- #
    if save_cleaned:
        out_path = output_dir / cleaned_filename
        out_path.write_text(final_text, encoding="utf-8")
        print(f"Cleaned article saved → {out_path.resolve()}")

    # -------------------------------------------------- #
    # 6. Short preview only
    # -------------------------------------------------- #
    preview = "\n".join(final_text.splitlines()[:preview_lines])
    print("\n" + "=" * 80)
    print(f"PREVIEW – first {preview_lines} lines")
    print("=" * 80)
    print(preview)
    if len(final_text.splitlines()) > preview_lines:
        print(f"… ({len(final_text.splitlines()) - preview_lines} more lines)")
    supp = "SUPPLEMENTARY INCLUDED" if cleaned_post else "No supplementary"
    print(f"\n{supp}")
    print(f"Images: {img_cnt} → {figure_dir.resolve()}")
    print("=" * 80 + "\n")

    # -------------------------------------------------- #
    # 7. Rename caption files
    # -------------------------------------------------- #
    for i in range(1, img_cnt + 1):
        old = figure_dir / f"figure_{i:03d}.txt"
        if old.exists():
            old.rename(figure_dir / f"fig_{i:04d}_caption.txt")

    return final_text

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
        # Title (try main-level title then fallback)
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
        fname = f"{clean_filename(article_name)}_{fig_id}.txt" if rag_mode and article_name else f"{fig_id}.txt"
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
        title_tag = bibl.find("title", {"level": "a", "type": "main"}) or bibl.find("title")
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
            m = re.search(r"fig(?:ure)?\s*[_\s]?(\d+)", fig_text, re.IGNORECASE)
            fig_target = m.group(1) if m else "unknown"
        label = f"{fig_target}: {fig_text}"
        figures.append(label)
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
        for i in range(1, len(divs)):
            if div_paragraph_counts[i] >= 3:
                supp_div_index = i
                break
    else:
        # div0 small: try div1 as main
        if len(div_paragraph_counts) > 1 and div_paragraph_counts[1] >= 5:
            main_div_index = 1
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
    main_divs = []
    supp_divs = []
    if supp_div_index is not None:
        for i in range(main_div_index, supp_div_index):
            main_divs.append(divs[i])
        for i in range(supp_div_index, len(divs)):
            supp_divs.append(divs[i])
    else:
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
                current_chunk_text = ""
                current_chunk_figs = set()
                current_section_title = None
                return
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
            head = div.find("head")
            head_text = head.get_text(" ", strip=True) if head else None
            if head_text:
                if current_chunk_text and (len(current_chunk_text) + len(head_text) > chunk_size):
                    flush_chunk()
                if current_chunk_text:
                    current_chunk_text += "\n" + head_text + "\n"
                else:
                    current_chunk_text = head_text + "\n"
                current_section_title = head_text

            paragraphs = div.find_all("p")
            for p in paragraphs:
                cleaned_para, figs = clean_paragraph_and_extract_figures(p)
                # very long paragraph: split into parts
                if len(cleaned_para) > chunk_size:
                    if current_chunk_text:
                        flush_chunk()
                    start = 0
                    while start < len(cleaned_para):
                        part = cleaned_para[start:start+chunk_size]
                        if start + chunk_size < len(cleaned_para):
                            last_space = part.rfind(" ")
                            if last_space > 100:
                                part = part[:last_space]
                        chunk_text = part.strip()
                        chunk_fig_set = set(figs)
                        if chunk_fig_set:
                            chunk_text = chunk_text + "\n\nFigures used in this chunk: " + ", ".join(sorted(chunk_fig_set)) + "\n"
                        chunks.append({"text": chunk_text, "figures": sorted(chunk_fig_set), "section_title": current_section_title})
                        start += len(part)
                    current_chunk_text = ""
                    current_chunk_figs = set()
                else:
                    # try to append paragraph to chunk (group small paras)
                    if not current_chunk_text:
                        current_chunk_text = cleaned_para + "\n"
                        current_chunk_figs.update(figs)
                    else:
                        if len(current_chunk_text) + len(cleaned_para) > chunk_size:
                            flush_chunk()
                            current_chunk_text = cleaned_para + "\n"
                            current_chunk_figs.update(figs)
                        else:
                            current_chunk_text += cleaned_para + "\n"
                            current_chunk_figs.update(figs)
        flush_chunk()
        return chunks

    main_chunks = produce_chunks_from_divs(main_divs, "main")
    supp_chunks = produce_chunks_from_divs(supp_divs, "supplementary") if supp_divs else []

    return main_chunks, supp_chunks

# -----------------------
# Main preprocessing entrypoint
# -----------------------
def preprocess_pdf(pdf_path: str,
                   output_dir: str,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   rag_mode: bool = False,
                   article_name: str = None,
                   keep_references: bool = False,
                   preclean_pdf: bool = False,
                   use_grobid_consolidation: bool = False) -> Dict:
    """
    Preprocess a PDF via a running GROBID instance.

    Args:
        pdf_path: input PDF path
        output_dir: directory where outputs will be saved
        chunk_size: target maximum characters per chunk (paragraph-aware)
        rag_mode: if True, save figure files into output_dir and prefix files for RAG usage
        article_name: optional prefix when rag_mode=True
        keep_references: if True, also save a version of chunks with full references appended
        preclean_pdf: if True, attempt to preclean PDF (ghostscript/qpdf) and send temp_cleaned.pdf
        use_grobid_consolidation: if True, add consolidation params to the GROBID request

    Returns:
        dict with summary information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # decide which PDF to send (temporary cleaned or original)
    pdf_to_send = str(pdf_path)
    tmp_pdf_path = None
    if preclean_pdf:
        tmp_pdf_path = produce_temp_cleaned_pdf(str(pdf_path), output_dir)
        pdf_to_send = tmp_pdf_path

    # build GROBID endpoint and params
    base_url = "http://localhost:8070/api/processFulltextDocument"
    params = {}
    if use_grobid_consolidation:
        params["consolidateHeader"] = "1"
        params["consolidateCitations"] = "1"

    # send to GROBID
    with open(pdf_to_send, "rb") as f:
        r = requests.post(base_url, params=params, files={"input": f}, timeout=120)
    tei_xml = r.text

    # save raw TEI
    raw_tei_path = output_dir / "raw_tei.xml"
    save_text(raw_tei_path, tei_xml)

    # 2. Run the sanity-check on the *original* PDF
    # -------------------------------------------------
    cleaned_text = prepare_pdf_for_grobid_check(
        pdf_path=pdf_path,
        output_dir=output_dir,
        preview_lines=30,      # change if you want more/less preview
        save_cleaned=True
    )

    # extract metadata
    meta = extract_metadata_from_tei(tei_xml)
    title_txt_path = output_dir / "title.txt"
    with open(title_txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Title: {meta.get('title','')}\n\n")
        fh.write("Authors:\n")
        for a in meta.get("authors", []):
            fh.write(f" - {a.get('first','')} {a.get('last','')}\n")
        fh.write(f"\nAbstract:\n{meta.get('abstract','')}\n")

    # figures & references
    extract_figures(tei_xml, output_dir, rag_mode=rag_mode, article_name=article_name)
    if not rag_mode:
        extract_references(tei_xml, output_dir)

    # chunking
    main_chunks, supp_chunks = extract_sections(tei_xml, chunk_size=chunk_size)

    # save chunks
    if rag_mode:
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

    # save figure lists & maps (standard mode)
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

    # optional: save chunks with references appended
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

    # cleanup temporary cleaned pdf if we created one (and it is not the original)
    if preclean_pdf and tmp_pdf_path:
        try:
            tmp = Path(tmp_pdf_path)
            orig = Path(pdf_path)
            # only remove tmp if it's inside the output_dir tmp_cleaned.pdf
            if tmp.exists() and tmp.name == "tmp_cleaned.pdf" and tmp.parent.resolve() == Path(output_dir).resolve():
                tmp.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "raw_tei": str(raw_tei_path) if raw_tei_path.exists() else None,
        "title_txt": str(title_txt_path),
        "main_chunks": len(main_chunks),
        "supp_chunks": len(supp_chunks),
    }
