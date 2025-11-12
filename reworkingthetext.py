#!/usr/bin/env python3
"""
preprocess_extract_figures_and_refs_v2.py

Step 0 — Clean and structure the article text:
1) Extract all figure captions (Figure, Fig., Supplementary Figure, etc.)
2) Remove them from the main text
3) Remove headers/footers, author names, DOIs, page numbers
4) Remove figure label blocks (axis text, single letters, numeric scales)
5) Separate References section
"""

import os
import re
import fitz  # PyMuPDF

BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
PDF_PATH = os.path.join(BASE_DIR, "main", "s41567-025-02854-4.pdf")

OUT_BODY = os.path.join(BASE_DIR, "body_cleaned.txt")
OUT_FIGS = os.path.join(BASE_DIR, "figure_captions.txt")
OUT_REFS = os.path.join(BASE_DIR, "references.txt")

# --------- Helpers ---------
def extract_pdf_text(pdf_path):
    """Extract plain text from PDF pages and join with page breaks."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)

def separate_references(full_text):
    """Split off References section."""
    patterns = [
        r"(?mi)\n\s*References\s*\n",
        r"(?mi)\n\s*REFERENCES\s*\n",
        r"(?mi)\n\s*Bibliography\s*\n"
    ]
    for pat in patterns:
        m = re.search(pat, full_text)
        if m:
            idx = m.end()
            return full_text[:idx], full_text[idx:]
    cutoff = int(len(full_text) * 0.8)
    return full_text[:cutoff], full_text[cutoff:]

def extract_and_remove_figure_captions(text):
    """Find and remove all figure captions."""
    caption_pattern = re.compile(
        r"(?:(?:^|\n)(?:Fig(?:ure)?\.?|Extended Data Fig\.|Supplementary Fig(?:ure)?\.?)\s*\d+[a-zA-Z]?(?:[.:)]|[\s|–-]))"
        r"[\s\S]*?(?=\n\s*\n|(?:(?:Fig(?:ure)?|Extended Data Fig\.|Supplementary Fig\.|References)\b)|\Z)",
        flags=re.MULTILINE
    )
    captions = [m.group(0).strip() for m in caption_pattern.finditer(text)]
    cleaned_text = caption_pattern.sub("\n", text)
    return cleaned_text, captions

def remove_page_headers_and_footers(text):
    """Remove repetitive headers, footers, DOIs, author names, page numbers."""
    lines = text.splitlines()
    cleaned = []
    seen_headers = set()
    for line in lines:
        l = line.strip()
        # Skip empty lines later
        if not l:
            cleaned.append("")
            continue
        # Common header/footer patterns
        if re.match(r"^https?://|^DOI|^doi|^www\.|^Nature|^Science|^arXiv", l):
            continue
        if re.match(r"^\d{1,3}\s*$", l):  # pure page number
            continue
        if re.match(r"^\(?\d{4}\)?$", l):  # year line
            continue
        if re.match(r"^[A-Z][A-Za-z,\-\s]+et al\.$", l):  # author repeating
            continue
        if re.match(r"^[A-Z][A-Za-z\-\s]+[A-Z][A-Za-z\-\s]+$", l):  # repeated name headers
            if l in seen_headers:
                continue
            seen_headers.add(l)
            continue
        cleaned.append(l)
    return "\n".join(cleaned)

def remove_figure_axis_labels(text):
    """Remove blocks that look like figure axis text."""
    # Detect blocks of mostly numbers, single letters, units, or short fragments
    lines = text.splitlines()
    new_lines = []
    buffer = []
    for l in lines:
        if re.fullmatch(r"[\d\s\.\(\)××eE\-\+]+", l.strip()):
            continue  # purely numeric
        if re.search(r"\b(kHz|Hz|s|ms|nm|μm|cm|dB|mol|K|°C|×10)\b", l):
            continue
        if len(l.strip()) <= 2:  # isolated letters like a, b, c, d
            continue
        # discard lines with many short tokens (<3 chars)
        tokens = l.strip().split()
        if len(tokens) >= 3 and sum(len(t) < 3 for t in tokens) > len(tokens) / 2:
            continue
        new_lines.append(l)
    return "\n".join(new_lines)

def collapse_blank_lines(text):
    """Collapse multiple blank lines."""
    return re.sub(r"\n{3,}", "\n\n", text)

# --------- main process ---------
def main():
    print(f"Reading {PDF_PATH} ...")
    full_text = extract_pdf_text(PDF_PATH)
    print(f"Total length: {len(full_text)} characters")

    # 1. Remove headers, footers, page numbers, author repetitions
    cleaned = remove_page_headers_and_footers(full_text)

    # 2. Separate references
    body_text, refs_text = separate_references(cleaned)
    print(f"Body length: {len(body_text)}, Refs length: {len(refs_text)}")

    # 3. Extract figure captions
    body_no_figs, captions = extract_and_remove_figure_captions(body_text)
    print(f"Found {len(captions)} potential figure captions")

    # 4. Remove figure axis labels / noise
    body_no_noise = remove_figure_axis_labels(body_no_figs)

    # 5. Collapse blank lines
    final_body = collapse_blank_lines(body_no_noise)

    # 6. Save outputs
    with open(OUT_BODY, "w") as f:
        f.write(final_body.strip() + "\n")
    with open(OUT_FIGS, "w") as f:
        for i, cap in enumerate(captions, 1):
            f.write(f"=== Figure Caption {i} ===\n{cap}\n\n")
    with open(OUT_REFS, "w") as f:
        f.write(refs_text.strip() + "\n")

    print("✅ Preprocessing done:")
    print(f"  → Cleaned body: {OUT_BODY}")
    print(f"  → Captions: {OUT_FIGS}")
    print(f"  → References: {OUT_REFS}")

if __name__ == "__main__":
    main()
