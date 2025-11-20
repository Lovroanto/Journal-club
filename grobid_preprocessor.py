
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
from lxml import etree as ET
from typing import Union, Tuple, Optional
from pathlib import Path
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher, HtmlDiff
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from sentence_alignment import align_clean_to_grobid, save_alignment_report
import html

MODEL = "llama3.1"  # same as your chunk summary
SIMILARITY_THRESHOLD = 0.7
MAX_DIFF_TO_SHOW = 200   # safety – don’t flood the report
HTML_DIFF = HtmlDiff(wrapcolumn=80)   # pretty word-level diff
MIN_CHAR_DIFF = 15          # ← ignore <15 char differences
DEFAULT_AUTHOR_RATIO = 0.60   # ← 60% = perfect balance, change anytime

_figure_counter = 0
_figure_lock = False  # prevents accidental reset if imported multiple times
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


def reset_figure_counter():
    global _figure_counter
    _figure_counter = 0

def _get_next_figure_number() -> int:
    global _figure_counter
    _figure_counter += 1
    return _figure_counter



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
    if not text.strip():
        return text, 0

    pattern = re.compile(
        r"^((?:Fig(?:ure)?\.?|Extended Data Fig\.?|Supplementary Fig(?:ure)?\.?)\s*"
        r"\d+[a-zA-Z]?(?:[.:)\s]|[\s–—-]))"
        r"([\s\S]*?)"
        r"(?=\Z|"
        r"^(?:Fig(?:ure)?|Extended\s+Data\s+Fig|Supplementary\s+Fig|References|Methods|Acknowledgements)\b|"
        r"\n\s*\n)",
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return text, 0

    cleaned_text = text
    offset = 0
    local_count = 0

    print("Extracting figure captions...")
    for match in matches:
        local_count += 1
        idx = _get_next_figure_number()

        label = match.group(1).strip()
        caption_body = match.group(2).strip()
        first_line = caption_body.split("\n", 1)[0] if caption_body else ""
        preview = (first_line[:40] + "..." if len(first_line) > 40 else first_line)

        print(f"  → figure_{idx:03d}.txt  |  {label:<20} {preview}")

        filename = out_dir / f"figure_{idx:03d}.txt"
        filename.write_text(f"# Original label: {label}\n{caption_body}\n", encoding="utf-8")

        start = match.start() - offset
        end = match.end() - offset
        cleaned_text = cleaned_text[:start] + "\n" + cleaned_text[end:]
        offset += end - start - 1

    print(f"Extracted {local_count} caption(s) → figure_{_figure_counter-local_count+1:03d} to figure_{_figure_counter:03d}.txt")
    return cleaned_text, local_count

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
    # CLEAN ONCE PER PDF → this fixes everything
    for temp_file in figure_dir.glob("figure_*.txt"):
        temp_file.unlink(missing_ok=True)

    # Reset counter for this new PDF
    reset_figure_counter()

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
    # 1. Normalize whitespace
    # ------------------------------------------------------------------
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ------------------------------------------------------------------
    # 2. Protect abbreviations
    # ------------------------------------------------------------------
    protected_dots = [
        r'\b(Fig|Figs|Eq|Eqs|Ref|Refs|Tab|Table)\.\s+\d+',
        r'\b(et al|i\.e|e\.g|vs|cf|ca)\.',
        r'\b(Dr|Mr|Mrs|Ms|Prof)\.',
        r'[a-zA-Z]\.[a-zA-Z]\.',
    ]
    for pattern in protected_dots:
        text = re.sub(pattern, lambda m: m.group(0).replace('.', '___DOT___'), text, flags=re.I)

    # ------------------------------------------------------------------
    # 3. MAIN split — your original reliable rule (kept 100%)
    # ------------------------------------------------------------------
    sentence_split = re.compile(r'''
        ([.!?])          # punctuation
        \s*              # optional spaces
        (?:\n\s*)*       # optional newlines
        \s+              # at least one whitespace
        (?=[A-Z])        # next sentence starts with capital letter
    ''', re.VERBOSE)

    sentences = []
    start = 0
    for m in sentence_split.finditer(text):
        end = m.start()
        sent = text[start:end].strip()
        if sent:
            sentences.append(sent + m.group(1))
        start = m.end()

    # Last sentence
    last = text[start:].strip()
    if last:
        if re.search(r'[.!?]$', last):
            sentences.append(last)
        else:
            match = re.search(r'([.!?])\s*$', last)
            if match:
                sentences.append(last[:match.end(1)])
            else:
                sentences.append(last)

    # ------------------------------------------------------------------
    # 3B. EXTRA RULE: split on ). / ] / " / ' followed by newline
    #     This fixes exactly your "detectors.) \n g(2)" case
    # ------------------------------------------------------------------
    extra_rule = re.compile(r'''
        ([.!?])          # punctuation
        \s*              # spaces
        ([\)\]"'\]}›»]*) # optional closing brackets/quotes
        \s*              # more spaces
        (?:\n\s*)+       # at least one newline
    ''', re.VERBOSE)

    # Apply extra rule inside each already-split sentence
    final_sentences = []
    for sent in sentences:
        if extra_rule.search(sent):
            # Split this sentence further using the extra rule
            parts = extra_rule.split(sent)
            current = ""
            for i, part in enumerate(parts):
                if i == 0:
                    current = part
                    continue
                # part is either punctuation or closing delimiters or empty
                if part.strip() == "":
                    continue
                # Reconstruct
                if re.match(r'^[.!?]', part):
                    current += part[0]  # the punctuation
                    remaining = extra_rule.sub('', sent[len(current):], count=1).lstrip()
                    if remaining:
                        current += " " + remaining.split("\n", 1)[0].strip()
                    final_sentences.append(current.strip())
                    sent = sent[len(current):].lstrip()
                    current = sent.lstrip()
                elif part in "}])>'\"":
                    current += part
            if current.strip():
                final_sentences.append(current.strip())
        else:
            final_sentences.append(sent.strip())

    sentences = final_sentences

    # ------------------------------------------------------------------
    # 4. Restore protected dots
    # ------------------------------------------------------------------
    restored = []
    for s in sentences:
        s = s.replace('___DOT___', '.')
        s = re.sub(r' +', ' ', s).strip()
        if s:
            restored.append(s)

    # ------------------------------------------------------------------
    # 5. Filter + write
    # ------------------------------------------------------------------
    with output_path.open('w', encoding='utf-8') as f:
        count = 0
        for sent in restored:
            if len(sent.split()) < min_words:
                continue
            # FIXED LINE — properly escaped quotes
            if not re.search(r"[.!?][)\]\"'\]}›»]*$", sent):
                continue

            count += 1
            f.write(f"{marker_start} {count:03d}{separator}{sent}{end_marker}\n\n")

    print(f"Extracted {count} clean sentences → {output_path.resolve()}")

def _extract_sentences_exact_logic(text: str, min_words: int = 3) -> List[str]:
    """
    Internal helper: extracts sentences using EXACTLY the same logic
    as your original extract_all_sentences_to_single_file function.
    Returns list of clean, valid sentences.
    """
    if not text.strip():
        return []

    # ------------------------------------------------------------------
    # 1. Normalize whitespace
    # ------------------------------------------------------------------
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ------------------------------------------------------------------
    # 2. Protect abbreviations
    # ------------------------------------------------------------------
    protected_dots = [
        r'\b(Fig|Figs|Eq|Eqs|Ref|Refs|Tab|Table)\.\s+\d+',
        r'\b(et al|i\.e|e\.g|vs|cf|ca)\.',
        r'\b(Dr|Mr|Mrs|Ms|Prof)\.',
        r'[a-zA-Z]\.[a-zA-Z]\.',
    ]
    for pattern in protected_dots:
        text = re.sub(pattern, lambda m: m.group(0).replace('.', '___DOT___'), text, flags=re.IGNORECASE)

    # ------------------------------------------------------------------
    # 3. MAIN split — identical to your original
    # ------------------------------------------------------------------
    sentence_split = re.compile(r'''
        ([.!?])          # punctuation
        \s*              # optional spaces
        (?:\n\s*)*       # optional newlines
        \s+              # at least one whitespace
        (?=[A-Z])        # next sentence starts with capital letter
    ''', re.VERBOSE)

    sentences = []
    start = 0
    for m in sentence_split.finditer(text):
        end = m.start()
        sent = text[start:end].strip()
        if sent:
            sentences.append(sent + m.group(1))
        start = m.end()

    # Last sentence
    last = text[start:].strip()
    if last:
        if re.search(r'[.!?]$', last):
            sentences.append(last)
        else:
            match = re.search(r'([.!?])\s*$', last)
            if match:
                sentences.append(last[:match.end(1)])
            else:
                sentences.append(last)

    # ------------------------------------------------------------------
    # 3B. EXTRA RULE: split on ). / ] / " / ' followed by newline
    # ------------------------------------------------------------------
    extra_rule = re.compile(r'''
        ([.!?])                  # punctuation
        \s*                      # spaces
        ([\)\]"'\]}›»]*)         # optional closing brackets/quotes
        \s*                      # more spaces
        (?:\n\s*)+               # at least one newline
    ''', re.VERBOSE)

    final_sentences = []
    for sent in sentences:
        if extra_rule.search(sent):
            parts = extra_rule.split(sent)
            current = ""
            i = 0
            while i < len(parts):
                part = parts[i]
                if i == 0:
                    current = part
                    i += 1
                    continue
                if part.strip() == "":
                    i += 1
                    continue
                if re.match(r'^[.!?]', part):
                    current += part[0]
                    # Consume everything until next split point
                    remaining = "".join(parts[i+1:])
                    if remaining.strip():
                        current += " " + remaining.split("\n", 1)[0].strip()
                    final_sentences.append(current.strip())
                    # Restart from remainder
                    sent = remaining
                    parts = extra_rule.split(sent)
                    i = 0
                    current = ""
                else:
                    current += part
                    i += 1
            if current.strip():
                final_sentences.append(current.strip())
        else:
            final_sentences.append(sent.strip())
    sentences = final_sentences

    # ------------------------------------------------------------------
    # 4. Restore protected dots
    # ------------------------------------------------------------------
    restored = []
    for s in sentences:
        s = s.replace('___DOT___', '.')
        s = re.sub(r' +', ' ', s).strip()
        if s:
            restored.append(s)

    # ------------------------------------------------------------------
    # 5. Final filtering (same as original)
    # ------------------------------------------------------------------
    clean_sentences = []
    for sent in restored:
        if len(sent.split()) < min_words:
            continue
        if not re.search(r"[.!?][)\]\"'\]}›»]*$", sent):
            continue
        clean_sentences.append(sent)

    return clean_sentences


def append_figure_captions_to_sentence_file(
    figures_folder: Union[str, Path],
    sentence_file: Union[str, Path],
    marker_start: str = "SENTENCE",
    separator: str = " ::: ",
    end_marker: str = " ////",
    min_words_caption: int = 3,
) -> None:
    """Appends figure captions using YOUR exact sentence logic. Now bulletproof."""
    figures_folder = Path(figures_folder)
    sentence_file = Path(sentence_file)

    print(f"\nLooking for figure captions in: {figures_folder.resolve()}")
    if not figures_folder.exists():
        print("ERROR: Figures folder does not exist!")
        return
    if not figures_folder.is_dir():
        print("ERROR: Path is not a directory!")
        return

    all_files = list(figures_folder.iterdir())
    print(f"Found {len(all_files)} items in the folder")

    # Robust detection: any .txt file that contains "fig" or "figure" and a number
    candidate_files = []
    for p in all_files:
        if p.is_file() and p.suffix.lower() == ".txt":
            name_low = p.name.lower()
            if ('fig' in name_low or 'figure' in name_low) and re.search(r'\d+', name_low):
                candidate_files.append(p)

    print(f"Detected {len(candidate_files)} figure caption file(s):")
    for p in candidate_files:
        print(f"  → {p.name}")

    if not candidate_files:
        print("No figure caption files found — nothing to do.")
        return

    # Sort by the number in the filename
    def sort_key(p):
        m = re.search(r'\d+', p.name)
        return int(m.group()) if m else 0

    candidate_files.sort(key=sort_key)

    # Find current max sentence number
    max_num = 0
    if sentence_file.exists():
        with sentence_file.open("r", encoding="utf-8") as f:
            for line in f:
                m = re.search(rf"{re.escape(marker_start)}\s*(\d+)", line)
                if m:
                    max_num = max(max_num, int(m.group(1)))

    print(f"Current highest sentence number: {max_num}")
    next_num = max_num + 1
    added = 0

    with sentence_file.open("a", encoding="utf-8") as f:
        for fig_path in candidate_files:
            # Extract number and create FIGxxx
            num_match = re.search(r'\d+', fig_path.name)
            fig_num = f"{int(num_match.group()):03d}" if num_match else "000"
            fig_id = f"FIG{fig_num}"

            print(f"Processing {fig_path.name} → {fig_id}")

            caption = fig_path.read_text(encoding="utf-8")
            if not caption.strip():
                print("  → Empty file, skipping")
                continue

            sentences = _extract_sentences_exact_logic(caption, min_words=min_words_caption)
            print(f"  → Extracted {len(sentences)} sentence(s)")

            for sent in sentences:
                f.write(f"{marker_start}{next_num:03d};{fig_id}{separator}{sent}{end_marker}\n\n")
                next_num += 1
                added += 1

    print(f"\nSUCCESS! Appended {added} figure-caption sentences")
    print(f"New total sentences: {next_num - 1}")
    print(f"Updated file: {sentence_file.resolve()}\n")

def correct_overmerged_sentences(
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    marker_start: str = "SENTENCE",
    separator: str = " ::: ",
    end_marker: str = " ////",
    min_current_line_length: int = 30,   # current line must be long enough
    max_prev_line_length: int = 49       # previous line must be SHORT (< 50)
) -> None:
    """
    Keeps only the last real sentence that starts after a short line (title/author/page number).
    Perfect for cleaning the first over-merged block in scientific PDFs.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capital_start_re = re.compile(r'^\s*[A-Z][a-z]{2,}')

    new_lines = []
    fixed = 0

    with input_path.open('r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(rf'({marker_start}\s+\d+{re.escape(separator)})', content)

    i = 1
    while i < len(blocks):
        marker = blocks[i]
        if i + 1 >= len(blocks):
            break
        raw_block = blocks[i + 1]
        raw = re.sub(rf'{re.escape(end_marker)}\s*$', '', raw_block).strip()
        if not raw:
            i += 2
            continue

        lines = [line.rstrip() for line in raw.splitlines()]
        if len(lines) <= 1:
            new_lines.append(marker + raw + end_marker)
            i += 2
            continue

        valid_starts = []

        for idx in range(len(lines)):
            line = lines[idx]
            stripped = line.strip()

            # 1. Must start with real capitalized word
            if not capital_start_re.match(line):
                continue

            # 2. Current line must be long enough
            if len(stripped) < min_current_line_length:
                continue

            # 3. Previous line must be SHORT (or it's the first line)
            if idx == 0:
                # Allow first line only if it's very clearly the start of text
                if len(stripped) >= 50:
                    valid_starts.append(idx)
                continue

            prev_stripped = lines[idx - 1].strip()
            if len(prev_stripped) > max_prev_line_length:
                continue  # previous line is long → flowing text → not a new paragraph

            # All 3 conditions passed → real new paragraph start
            valid_starts.append(idx)

        # Use the LAST valid start
        if valid_starts:
            start_idx = valid_starts[-1]
            clean = ' '.join(line.strip() for line in lines[start_idx:])
            clean = re.sub(r'\s+', ' ', clean).strip()
            new_lines.append(marker + clean + end_marker)
            if start_idx > 0:
                fixed += 1
        else:
            # Fallback: keep original
            new_lines.append(marker + raw + end_marker)

        i += 2

    with output_path.open('w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + "\n\n")

    print(f"Fixed {fixed} over-merged sentences.")
    print(f"Clean output → {output_path.resolve()}")


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


def extract_structured_sentences_from_tei(
    tei_path: Path,
    output_txt_path: Path
) -> None:
    """
    Extracts every full sentence from the <body> of a GROBID TEI file,
    correctly handling sentences that span multiple <p> tags.
    Outputs one sentence per line with DIV and PARA provenance.
    """
    tree = ET.parse(str(tei_path))
    root = tree.getroot()
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    # Find the main body
    body = root.find('.//tei:body', ns)
    if body is None:
        output_txt_path.write_text("ERROR: No <body> found in TEI file.\n")
        return

    # We'll collect sentences with their provenance
    sentences_with_loc = []  # List[Tuple[str, div_idx, para_idx]]

    # Track current open text buffer (for cross-paragraph sentences)
    buffer = ""
    current_div_idx = 0
    current_para_idx = 0

    # Walk through all <div> in order
    for div_idx, div in enumerate(body.findall('.//tei:div', ns), start=1):
        current_div_idx = div_idx
        current_para_idx = 0

        for para_idx, p in enumerate(div.xpath('.//tei:p', namespaces=ns), start=1):
            current_para_idx = para_idx

            # Extract raw text from this <p> (preserves spaces better than .text)
            p_text = " ".join(p.xpath('.//text()')).strip()
            if not p_text:
                continue

            # Append space if we're continuing a sentence across paragraphs
            if buffer and not buffer.endswith(" "):
                buffer += " "
            buffer += p_text + " "

            # Now try to extract complete sentences from the growing buffer
            while True:
                sentence_match = extract_next_sentence(buffer)
                if sentence_match is None:
                    break  # No complete sentence yet → wait for more text

                sentence, end_pos = sentence_match
                sentence = sentence.strip()

                # Heuristic cleaning: remove trailing junk like page numbers in sentences
                sentence = re.sub(r'\s*\[\d+\]\s*$', '', sentence)  # citations like [12]
                sentence = re.sub(r'\s*\(Formula\)\s*$', '', sentence)  # fake formula placeholders
                sentence = re.sub(r'\s*-\s*$', '', sentence)  # broken hyphens

                if len(sentence) > 20:  # filter garbage
                    provenance = f"DIV{current_div_idx:02d}PARA{current_para_idx:02d}"
                    sentences_with_loc.append((sentence, current_div_idx, current_para_idx))

                # Remove the extracted sentence from buffer
                buffer = buffer[end_pos:].lstrip()

            # After processing paragraph, keep buffer for next one (cross-paragraph sentences)

        # After each <div>, if buffer has leftover incomplete sentence, discard or warn
        # (usually rare and garbage)
        if buffer.strip():
            # Optional: save incomplete trailing sentence?
            pass
        buffer = ""  # reset between divs? Or keep? → better reset, most papers don't span divs

    # ------------------------------------------------------------------
    # Write final structured output
    # ------------------------------------------------------------------
    lines = []
    for idx, (sent, div_idx, para_idx) in enumerate(sentences_with_loc, start=1):
        prov = f"DIV{div_idx:02d}PARA{para_idx:02d}"
        line = f"SENTENCE {idx:03d} {prov} ::: {sent} ////"
        lines.append(line)

    output_txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Extracted {len(lines)} structured sentences → {output_txt_path}")

def append_figure_captions_from_tei(
    tei_path: Union[str, Path],
    sentence_file: Union[str, Path],
    marker_start: str = "SENTENCE",
    separator: str = " ::: ",
    end_marker: str = " ////",
    min_words_caption: int = 3,
) -> None:
    tei_path = Path(tei_path)
    sentence_file = Path(sentence_file)

    print(f"\n=== EXTRACTING FIGURE CAPTIONS FROM TEI ===")
    print(f"TEI file: {tei_path}")

    tree = ET.parse(str(tei_path))
    root = tree.getroot()
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    figures = root.findall('.//tei:figure', ns)
    print(f"Found {len(figures)} <figure> elements")

    # Get current max sentence number
    max_num = 0
    if sentence_file.exists():
        with open(sentence_file, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(rf"{re.escape(marker_start)}\s*(\d+)", line)
                if m:
                    max_num = max(max_num, int(m.group(1)))
    print(f"Current highest sentence number: {max_num}")
    next_num = max_num + 1
    added = 0

    with open(sentence_file, "a", encoding="utf-8") as f:
        for i, fig in enumerate(figures, 1):
            print(f"\n--- Figure {i} ---")

            # ROBUST xml:id extraction (handles namespaced attributes)
            xml_id = fig.get("xml:id") or fig.get("{http://www.tei-c.org/ns/1.0}xml:id") or fig.get("id") or fig.get("{http://www.w3.org/XML/1998/namespace}id")
            print(f"  Detected xml:id = '{xml_id}'")

            if not xml_id or not xml_id.startswith("fig_"):
                print("  → Skipped (no valid fig_* xml:id)")
                continue

            try:
                fig_num = int(xml_id.split("_")[-1])
                fig_id = f"FIG{fig_num:03d}"
            except:
                print("  → Cannot parse number from xml:id")
                continue

            # Extract caption from <figDesc>
            caption = ""
            figDesc = fig.find(".//tei:figDesc", ns)
            if figDesc is not None:
                parts = [figDesc.text or ""]
                for elem in figDesc:
                    if elem.text:
                        parts.append(elem.text)
                    if elem.tail:
                        parts.append(elem.tail)
                caption = "".join(parts).strip()

            if not caption:
                head = fig.find("tei:head", ns)
                if head is not None:
                    caption = "".join(head.itertext()).strip()

            # Clean prefix
            cleaned = re.sub(r'^Fig\.?\s*\d+\s*[\|–:—]\s*', '', caption, flags=re.I).strip()

            if not cleaned:
                print(f"  → {fig_id}: Empty caption after cleaning")
                continue

            print(f"  → {fig_id}: Caption found ({len(cleaned.split())} words)")
            print(f"     >>>{cleaned[:200]}{'...' if len(cleaned)>200 else ''}<<<")

            # Extract sentences
            sentences = _extract_sentences_exact_logic(cleaned, min_words=1)

            final_sentences = []
            for s in sentences:
                s = s.strip()
                if len(s.split()) < min_words_caption:
                    continue
                if not s.endswith(('.', '!', '?')):
                    s += "."
                final_sentences.append(s)

            print(f"     Extracted {len(final_sentences)} final sentence(s)")

            for sent in final_sentences:
                f.write(f"{marker_start}{next_num:03d};{fig_id}{separator}{sent}{end_marker}\n\n")
                next_num += 1
                added += 1

    print(f"\n=== SUCCESS ===")
    print(f"Appended {added} figure-caption sentences!")
    print(f"New total: {next_num - 1}")
    print(f"File: {sentence_file}\n")

# ------------------------------------------------------------------
# Core: extract the next complete sentence from text buffer
# ------------------------------------------------------------------
SENTENCE_END_PATTERN = re.compile(
    r"""(?x)                    # verbose mode
    (                           # Capture group 1: the sentence
        [A-Z]                   # Starts with capital letter
        [^.!?]*                 # Anything that's not a sentence end
        [.!?]                   # Ends with . ! or ?
    )
    (?:\s|$)                    # Followed by space or end of string
    """
)

def extract_next_sentence(text: str) -> Optional[Tuple[str, int]]:
    """
    Returns (sentence, end_position) of the first complete sentence found,
    or None if no complete sentence yet.
    Handles abbreviations like "Fig.", "eq.", "e.g.", "i.e.", etc.
    """
    # Temporary: avoid breaking on known abbreviations
    protected = text
    protected = re.sub(r'\b(Fig|FIGS|Eq|eq|eqs|Eqs|e\.g\.|i\.e\.|vs\.|cf\.|Dr\.|Prof\.|Mr\.|Mrs\.)\s+', r'\1@@@', protected)

    match = SENTENCE_END_PATTERN.search(protected)
    if not match:
        return None

    raw_sentence = match.group(1)
    end_pos = match.end()

    # Restore protected abbreviations
    sentence = raw_sentence.replace("@@@", " ")

    # Final cleanup
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    return sentence, end_pos

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
    

    correct_overmerged_sentences(
        output_dir_path / "checktxt" / "sentences_all_in_one.txt",
        output_dir_path / "checktxt" / "sentences_corrected.txt",
        min_current_line_length=25,
        max_prev_line_length=49
    )
    append_figure_captions_to_sentence_file(
        figures_folder=output_dir_path / "figure_from_pdf",
        sentence_file=output_dir_path / "checktxt" / "sentences_corrected.txt"
    )

    extract_structured_sentences_from_tei(
        output_dir_path / "raw_tei.xml", 
        output_dir_path / "checktxt" / "tei_corrected.txt",
    )

    append_figure_captions_from_tei(
        tei_path=output_dir_path / "raw_tei.xml", 
        sentence_file=output_dir_path / "checktxt" / "tei_corrected.txt",
    )

    checktxt_dir = output_dir_path / "checktxt"
    clean_file   = checktxt_dir / "sentences_corrected.txt"      # your gold-standard
    grobid_file  = checktxt_dir / "tei_corrected.txt"           # GROBID output
    report_file  = checktxt_dir / "SENTENCE_ALIGNMENT_FULL.txt" # where you want the report

    threshold = 0.20   # change freely: 0.15, 0.25, 0.30, 0.40 …

    # 1. Run alignment
    alignment_results = align_clean_to_grobid(
        clean_txt_path=clean_file,
        grobid_txt_path=grobid_file,
        min_score_to_accept=threshold
    )

    # 2. Save beautiful report in the right place
    save_alignment_report(
        alignment_results=alignment_results,
        output_path=report_file,           # ← now correct Path object!
        threshold_used=threshold
    )

    print(f"Alignment done → report saved to: {report_file}")


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
