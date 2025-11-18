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
import xml.etree.ElementTree as ET
from typing import Union, Tuple, Optional
from pathlib import Path
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher, HtmlDiff
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
import html

MODEL = "llama3.1"  # same as your chunk summary
SIMILARITY_THRESHOLD = 0.7
MAX_DIFF_TO_SHOW = 200   # safety – don’t flood the report
HTML_DIFF = HtmlDiff(wrapcolumn=80)   # pretty word-level diff
MIN_CHAR_DIFF = 15          # ← ignore <15 char differences
DEFAULT_AUTHOR_RATIO = 0.60   # ← 60% = perfect balance, change anytime
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
# 4. Detect References block
# ------------------------------------------------------------------ #
def find_references_block(text: str) -> Optional[Tuple[int, int]]:
    heading_pat = re.compile(
        r"(?mi)^.*\b("
        r"References|Bibliography|Literature|References\s+and\s+notes|"
        r"Reference|References?[\s:]?"
        r")\b.*$"
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if heading_pat.search(line):
            snippet = "\n".join(lines[i + 1 : i + 16])
            numbers = re.findall(r"^\s*(\d+)[\.\)]\s", snippet, flags=re.MULTILINE)
            if len(numbers) >= 3:
                nums = [int(n) for n in numbers]
                diffs = [nums[j + 1] - nums[j] for j in range(len(nums) - 1)]
                if any(0 < d <= 3 for d in diffs):
                    start = text.find(line)
                    rest = text[start:]
                    end_match = re.search(
                        r"(?mi)^(\s*(Supplementary|Appendix|Methods|Acknowledgments|Author|Data)\b)",
                        rest, flags=re.MULTILINE,
                    )
                    block_end = start + end_match.start() if end_match else len(text)
                    return start, block_end
    return None

# ------------------------------------------------------------------ #
# 5. BEST METADATA / DATE LINE REMOVER (your improved version)
# ------------------------------------------------------------------ #
def is_metadata_date_line(line: str) -> bool:
    """
    Removes:
    • Received/Accepted/Published lines with dates
    • Volume/Issue/Page lines
    • Any line containing an email address (with @)
    • Keeps everything else (like "Methods", "Introduction", etc.)
    """
    s = line.strip()
    if not s:
        return False

    # ———————— 1. Remove any line with an email (@ symbol) ————————
    if "@" in s:
        return True

    # ———————— 2. Classic submission/publication dates ————————
    if re.search(r"\b(Received|Accepted|Submitted|Revised|Published|Publication|Online|Posted|Issued)\b", s, re.I):
        if re.search(r"\d{1,4}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}", s, re.I):
            return True
        if re.search(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,4},?\s+\d{4}", s, re.I):
            return True
        if re.search(r"\d{4}-\d{2}-\d{2}", s):
            return True
        if re.search(r"\d{1,2}\s+[\w]+\s+\d{4}", s):
            return True
        # Even without full date, if keyword + year → still remove
        if re.search(r"\d{4}", s):
            return True

    # ———————— 3. Journal / Volume / Issue / Pages ————————
    if re.search(r"\b(?:Volume|Vol\.?|Issue|pp?\.?\s*\d+[–-]\d+|\|\s*\d+\s*[–-]\d+)\b", s):
        return True

    return False

# ------------------------------------------------------------------ #
# 6. FINAL AUTHOR / AFFILIATION KILLER — uppercase letters ÷ words
# ------------------------------------------------------------------ #
def is_author_or_affiliation_line(line: str, ratio: float = DEFAULT_AUTHOR_RATIO) -> bool:
    """
    Returns True → delete line (author/affiliation block)
    Only triggers if:
      • uppercase_letters / word_count >= ratio
      • AND there are strictly more than 1 uppercase letter
    → This keeps "Methods", "Results", "Introduction", etc.
    → But kills "Vera M. Schäfer", "1JILA, NIST...", etc.
    """
    s = line.strip()
    if not s or len(s) > 600:
        return False

    # Count words (any token with letters)
    words = [w for w in re.findall(r'\b[\w\.\-]+\b', s) if any(c.isalpha() for c in w)]
    if not words:
        return False

    word_count = len(words)
    uppercase_count = sum(1 for c in s if c.isupper())

    # NEW SAFETY: require at least 2 uppercase letters
    # → "Methods" has only 1 → kept
    # → "Vera M. Schäfer" has many → deleted
    if uppercase_count <= 1:
        return False

    return uppercase_count / word_count >= ratio

# ------------------------------------------------------------------ #
# 7. MAIN FUNCTION — with configurable ratio + merging
# ------------------------------------------------------------------ #
def prepare_pdf_for_grobid_check(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    preview_lines: int = 30,
    save_cleaned: bool = True,
    cleaned_filename: str = "cleaned_article.txt",
    author_ratio: float = DEFAULT_AUTHOR_RATIO,   # ← FULLY CONFIGURABLE
) -> str:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    figure_dir = output_dir / "figure_from_pdf"
    figure_dir.mkdir(parents=True, exist_ok=True)

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

    refs_block = find_references_block(full_text)
    body_text = full_text[:refs_block[0]] if refs_block else full_text
    post_refs = full_text[refs_block[1]:] if refs_block else ""

    def clean_section(txt: str) -> str:
        if not txt.strip():
            return ""
        txt, _ = extract_and_remove_figure_captions(txt, figure_dir)
        txt = remove_figure_axis_labels(txt)
        lines = txt.splitlines()
        kept = []
        buffer = []

        for line in lines:
            s = line.strip()

            # Buffer short capital-heavy lines
            if s and len(s) < 180:
                letters = [c for c in s if c.isalpha()]
                if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.45:
                    buffer.append(line)
                    continue

            if buffer:
                merged = " ".join(b.strip() for b in buffer if b.strip())
                if (is_author_or_affiliation_line(merged, ratio=author_ratio) or
                    is_metadata_date_line(merged)):
                    buffer = []
                    continue
                kept.extend(buffer)
                buffer = []

            if (is_equation_block(line) or
                is_metadata_date_line(line) or
                is_author_or_affiliation_line(line, ratio=author_ratio)):
                continue

            if len(s) < 100 and (
                s.isdigit() or re.match(r"^https?://", s) or
                re.match(r"^[A-Z]{3,}$", s) or re.match(r"^\d{1,3}$", s)):
                continue

            kept.append(line)

        if buffer:
            merged = " ".join(b.strip() for b in buffer if b.strip())
            if not (is_author_or_affiliation_line(merged, ratio=author_ratio) or
                    is_metadata_date_line(merged)):
                kept.extend(buffer)

        return "\n".join(kept).strip()

    cleaned_body = clean_section(body_text)
    cleaned_post = clean_section(post_refs) if post_refs else ""
    final_text = cleaned_body + ("\n\n" + cleaned_post if cleaned_post else "")

    if save_cleaned:
        out_path = output_dir / cleaned_filename
        out_path.write_text(final_text, encoding="utf-8")
        print(f"Cleaned article saved → {out_path.resolve()}")

    preview = "\n".join(final_text.splitlines()[:preview_lines])
    print("\n" + "="*80)
    print(f"PREVIEW – first {preview_lines} lines")
    print("="*80)
    print(preview)
    if len(final_text.splitlines()) > preview_lines:
        print(f"… ({len(final_text.splitlines()) - preview_lines} more lines)")
    print(f"\n{'SUPPLEMENTARY INCLUDED' if cleaned_post else 'No supplementary'}")
    print(f"Images: {img_cnt} → {figure_dir.resolve()}")
    print("="*80 + "\n")

    for i in range(1, img_cnt + 1):
        old = figure_dir / f"figure_{i:03d}.txt"
        if old.exists():
            old.rename(figure_dir / f"fig_{i:04d}_caption.txt")

    return final_text

def extract_all_sentences_to_single_file(
    text: str,
    output_path: Union[Path, str],
    min_words: int = 6,
    marker_start: str = "SENTENCE",
    separator: str = " ::: ",
    end_marker: str = " ////"
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Normalize whitespace but keep real paragraph breaks
    # ------------------------------------------------------------------
    text = re.sub(r'[ \t]+', ' ', text)      # collapse spaces
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)   # max two blank lines

    # ------------------------------------------------------------------
    # 2. Temporarily mark places where a sentence-ending dot is NOT allowed
    #     → only inside common abbreviations that are NOT followed by space + capital
    # ------------------------------------------------------------------
    # These are the only patterns where a dot must NOT end a sentence
    protected_dots = [
        r'\b(Fig|Figs|Eq|Eqs|Ref|Refs|Tab|Table)\.\s+\d+',   # Fig. 5
        r'\b(et al|i\.e|e\.g|vs|cf|ca)\.',                    # et al.
        r'\b(Dr|Mr|Mrs|Ms|Prof)\.',                           # Dr.
        r'[a-zA-Z]\.[a-zA-Z]\.',                              # J. P. Smith (middle initials)
    ]

    for pattern in protected_dots:
        text = re.sub(pattern, lambda m: m.group(0).replace('.', '___DOT___'), text, flags=re.I)

    # ------------------------------------------------------------------
    # 3. NOW split on real sentence endings:
    #     → dot (or ?!) + whitespace + capital letter
    # ------------------------------------------------------------------
    sentence_split = re.compile(r'''
        ([.!?])          # punctuation
        \s*              # optional spaces
        (?:\n\s*)*       # optional newlines
        \s+              # at least one whitespace
        (?=[A-Z])        # next sentence starts with capital
    ''', re.VERBOSE)

    sentences = []
    start = 0
    for m in sentence_split.finditer(text):
        end = m.start()
        sent = text[start:end].strip()
        if sent:
            sentences.append(sent + m.group(1))   # include the final .!? 
        start = m.end()

    # Don't forget the last sentence
    last = text[start:].strip()
    if last:
        # If it ends with punctuation, great. Otherwise just take it.
        if re.search(r'[.!?]$', last):
            sentences.append(last)
        else:
            # Try to find the last real punctuation
            match = re.search(r'([.!?])\s*$', last)
            if match:
                sentences.append(last[:match.end(1)])
            else:
                sentences.append(last)

    # ------------------------------------------------------------------
    # 4. Restore protected dots
    # ------------------------------------------------------------------
    restored = []
    for s in sentences:
        s = s.replace('___DOT___', '.')
        s = re.sub(r' +', ' ', s)        # clean extra spaces
        s = s.strip()
        if not s:
            continue
        restored.append(s)

    # ------------------------------------------------------------------
    # 5. Filter + write
    # ------------------------------------------------------------------
    with output_path.open('w', encoding='utf-8') as f:
        count = 0
        for sent in restored:
            words = sent.split()
            if len(words) < min_words:
                continue
            if not re.search(r'[.!?]$', sent):
                continue

            count += 1
            f.write(f"{marker_start} {count:03d}{separator}{sent}{end_marker}\n\n")

        print(f"Extracted {count} clean sentences → {output_path.resolve()}")

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


# -------------------------------------------------- #
# 2. Load GROBID plain text
# -------------------------------------------------- #
def extract_grobid_text(tei_path: Path) -> str:
    tree = ET.parse(tei_path)
    root = tree.getroot()
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    body = root.find('.//tei:body', ns)
    if body is None:
        return ""
    text = ET.tostring(body, encoding='unicode', method='text')
    return re.sub(r'\s+', ' ', text).strip()


# -------------------------------------------------- #
# 3. Load cleaned article
# -------------------------------------------------- #
def load_cleaned_text(txt_path: Path) -> str:
    return txt_path.read_text(encoding='utf-8')


# -------------------------------------------------- #
# 4. Split into sentences
# -------------------------------------------------- #
def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s) > 15]


# -------------------------------------------------- #
# 5. Similarity
# -------------------------------------------------- #
def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# -------------------------------------------------- #
# 6. Character difference count
# -------------------------------------------------- #
def char_diff_count(a: str, b: str) -> int:
    matcher = SequenceMatcher(None, a, b)
    diff = 0
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete', 'insert'):
            diff += j2 - j1
    return diff


# -------------------------------------------------- #
# 7. Diff: cleaned vs grobid (only missing + big misplaced)
# -------------------------------------------------- #
def sentence_diff(cleaned_sents: List[str], grobid_sents: List[str]) -> List[Dict[str, Any]]:
    matcher = SequenceMatcher(None, cleaned_sents, grobid_sents)
    diffs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'delete':  # CLEANED has more → GROBID is missing
            for i in range(i1, i2):
                diffs.append({
                    'type': 'missing_in_grobid',
                    'sentence': cleaned_sents[i],
                    'index_cleaned': i
                })
        elif tag == 'insert':  # GROBID has extra → ignore
            continue
        elif tag == 'replace':
            for i in range(i1, i2):
                for j in range(j1, j2):
                    c_sent = cleaned_sents[i]
                    g_sent = grobid_sents[j]
                    if similarity(c_sent, g_sent) > SIMILARITY_THRESHOLD:
                        char_diff = char_diff_count(c_sent, g_sent)
                        if char_diff >= MIN_CHAR_DIFF:
                            diffs.append({
                                'type': 'possible_misplaced',
                                'cleaned_sent': c_sent,
                                'grobid_sent': g_sent,
                                'index_cleaned': i,
                                'index_grobid': j,
                                'score': similarity(c_sent, g_sent),
                                'char_diff': char_diff
                            })
                        break
    return diffs


# -------------------------------------------------- #
# 8. Context helper
# -------------------------------------------------- #
def get_context(sents: List[str], idx: int, n: int = 2) -> str:
    start = max(0, idx - n)
    end = min(len(sents), idx + n + 1)
    return " | ".join(sents[start:end])


# -------------------------------------------------- #
# 9. Word-level HTML diff
# -------------------------------------------------- #
def html_word_diff(a: str, b: str) -> str:
    return HTML_DIFF.make_table(
        [a], [b],
        fromdesc="CLEANED", todesc="GROBID",
        context=False, numlines=0
    ).replace("<table", '<table style="font-size:0.9em; margin:8px 0;"')


# -------------------------------------------------- #
# 10. MAIN – AI validation (only missing + big changes)
# -------------------------------------------------- #
def ai_validate_grobid(
    tei_path: Path,
    cleaned_txt_path: Path,
    output_dir: Path,
    model: str = MODEL,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    grobid_text = extract_grobid_text(tei_path)
    cleaned_text = load_cleaned_text(cleaned_txt_path)

    cleaned_sents = split_into_sentences(cleaned_text)
    grobid_sents = split_into_sentences(grobid_text)

    print(f"GROBID sentences: {len(grobid_sents)}")
    print(f"Cleaned sentences: {len(cleaned_sents)}")

    # Diff
    raw_diffs = sentence_diff(cleaned_sents, grobid_sents)
    print(f"Real missing/misplaced differences: {len(raw_diffs)} (extra in GROBID ignored)")

    # AI + logging
    llm = OllamaLLM(model=model)
    decisions = []

    for idx, diff in enumerate(raw_diffs):
        if idx >= MAX_DIFF_TO_SHOW:
            print(f"[INFO] Stopping at {MAX_DIFF_TO_SHOW} diffs (report will show all).")
            break

        sentence = diff.get('sentence') or diff.get('cleaned_sent') or "[NO SENTENCE]"
        context_idx = diff.get('index_cleaned', 0)
        context = get_context(cleaned_sents, context_idx, n=2)

        if diff['type'] == 'missing_in_grobid':
            prompt = f"""
You are a scientific text validator.

CLEANED (gold standard):
"{sentence}"

Context:
{context}

This sentence is MISSING in GROBID.

Is it:
1. Real scientific content?
2. Formula, citation, figure label, or noise?

Answer with ONE of:
- ADD_TO_GROBID
- IGNORE
"""
        else:  # possible_misplaced
            prompt = f"""
Compare these two versions (≥{MIN_CHAR_DIFF} chars differ):

CLEANED:
"{diff['cleaned_sent']}"

GROBID:
"{diff['grobid_sent']}"

Are they the same meaning but in wrong order?
Answer with ONE of:
- MOVE_IN_GROBID
- IGNORE
"""

        try:
            raw_answer = llm.invoke(prompt).strip()
        except Exception as e:
            print(f"[LLM error] {e}")
            raw_answer = "IGNORE"

        decision = "IGNORE"
        if "ADD_TO_GROBID" in raw_answer.upper():
            decision = "ADD_TO_GROBID"
        elif "MOVE_IN_GROBID" in raw_answer.upper():
            decision = "MOVE_IN_GROBID"

        log = {
            **diff,
            'sentence': sentence,
            'prompt': prompt.strip(),
            'llm_raw_answer': raw_answer,
            'ai_decision': decision,
            'context': context,
            'word_diff_html': (
                html_word_diff(diff['cleaned_sent'], diff['grobid_sent'])
                if diff['type'] == 'possible_misplaced' else None
            )
        }
        decisions.append(log)

        print(f"\n--- Diff #{idx+1} [{diff['type']}] ---")
        print(f"AI → {decision}")

    # Apply corrections
    tree = ET.parse(tei_path)
    root = tree.getroot()
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    ET.register_namespace('', ns['tei'])

    body = root.find('.//tei:body', ns)
    for d in decisions:
        if d['ai_decision'] == 'ADD_TO_GROBID':
            note = ET.Element("{http://www.tei-c.org/ns/1.0}note")
            note.set('type', 'ai-correction')
            note.text = f"AI ADDED: {d['sentence']}"
            p = ET.SubElement(body, "{http://www.tei-c.org/ns/1.0}p")
            p.text = d['sentence']
            p.append(note)

    # Save
    corrected_path = output_dir / "ai_corrected_tei.xml"
    with open(corrected_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)
    print(f"\nAI-corrected GROBID → {corrected_path}")

    # Full HTML report
    report_path = output_dir / "ai_validation_report.html"
    generate_full_report(decisions, report_path)
    print(f"Full AI report → {report_path}")

    return corrected_path


# -------------------------------------------------- #
# 11. Full HTML Report
# -------------------------------------------------- #
def generate_full_report(decisions: List[Dict], out_path: Path):
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:Arial;margin:40px;line-height:1.6;background:#f9f9f9;}",
        ".diff{padding:15px;margin:12px 0;border-radius:8px;background:white;box-shadow:0 1px 3px rgba(0,0,0,0.1);}",
        ".missing{background:#ffebee;border-left:5px solid #e74c3c;}",
        ".move{background:#fff3cd;border-left:5px solid #f39c12;}",
        ".prompt{font-family:monospace;font-size:0.9em;background:#f0f0f0;padding:10px;margin:8px 0;border-radius:4px;}",
        ".answer{font-style:italic;color:#2c3e50;}",
        ".context{color:#555;font-size:0.9em;}",
        "table.diff {font-family:monospace;font-size:0.9em;}",
        "td.diff_add {background:#e6ffed;}",
        "td.diff_sub {background:#ffeef0;}",
        "</style></head><body>",
        f"<h1>AI-Powered GROBID Validation</h1>",
        f"<p><strong>Only missing content (cleaned > GROBID)</strong> and big changes (≥{MIN_CHAR_DIFF} chars) are shown.</p>",
        "<p>Extra content in GROBID is ignored (usually correct).</p>",
        "<hr>",
    ]

    for i, d in enumerate(decisions):
        sent = html.escape(d['sentence'])
        cls = "missing" if d['type'] == 'missing_in_grobid' else "move"
        title = "MISSING in GROBID" if d['type'] == 'missing_in_grobid' else f"MOVED ({d.get('char_diff', 0)} chars differ)"

        lines.append(
            f'<div class="diff {cls}">'
            f'<strong>#{i+1} – {title}</strong><br>'
            f'<strong>Sentence:</strong> "{sent}"<br>'
            f'<div class="context"><strong>Context:</strong> {html.escape(d["context"])}</div>'
        )

        if d.get('word_diff_html'):
            lines.append(f"<strong>Word-level diff:</strong>")
            lines.append(d['word_diff_html'])

        lines.append(
            f'<details><summary><strong>Prompt to LLM</strong></summary>'
            f'<div class="prompt">{html.escape(d["prompt"])}</div></details>'
            f'<details><summary><strong>LLM Raw Answer</strong></summary>'
            f'<div class="answer">{html.escape(d["llm_raw_answer"])}</div></details>'
            f'<strong>Final Decision:</strong> <code>{d["ai_decision"]}</code>'
            f'</div>'
        )

    lines.append("</body></html>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


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
    output_dir_path = output_dir

    # 2. Run the sanity-check on the *original* PDF
    # -------------------------------------------------
    cleaned_text = prepare_pdf_for_grobid_check(
        pdf_path=pdf_path,
        output_dir=output_dir_path,
        preview_lines=30,
        save_cleaned=True
    )

    extract_all_sentences_to_single_file(
        text=cleaned_text,   # use the returned string directly!
        output_path=output_dir_path / "checktxt" / "sentences_all_in_one.txt",
        min_words=6
    )

    corrected_tei = ai_validate_grobid(
        tei_path=output_dir / "raw_tei.xml",
        cleaned_txt_path=output_dir / "cleaned_article.txt",
        output_dir=output_dir,
        model="llama3.1"  # same as your chunk summary
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
