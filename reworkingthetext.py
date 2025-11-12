#!/usr/bin/env python3
"""
preprocess_extract_figures_and_refs_v3.py

Improved preprocessor:
1. Extract all figure captions (each in its own file)
2. Remove figure captions, axis noise, and headers/footers
3. Separate References and Supplementary materials
4. Save everything in ~/ai_data/Journal_Club/First/main_article/
"""

import os
import re
import fitz  # PyMuPDF

# ---------- CONFIG ----------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First/main_article")
PDF_PATH = os.path.join(BASE_DIR, "../main/s41567-025-02854-4.pdf")

os.makedirs(BASE_DIR, exist_ok=True)

OUT_BODY = os.path.join(BASE_DIR, "body_cleaned.txt")
OUT_CAPTIONS_INDEX = os.path.join(BASE_DIR, "figure_captions_index.txt")
OUT_REFS = os.path.join(BASE_DIR, "references.txt")
OUT_SUPPL = os.path.join(BASE_DIR, "supplementary.txt")

# ---------- HELPERS ----------

def extract_pdf_text(pdf_path):
    """Extract all text from PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text").strip() for page in doc]
    doc.close()
    return "\n\n".join(pages)

def remove_page_headers_and_footers(text):
    """Remove page numbers, author names, DOIs, etc."""
    lines = text.splitlines()
    cleaned = []
    seen_headers = set()
    for line in lines:
        l = line.strip()
        if not l:
            cleaned.append("")
            continue
        # Common junk patterns
        if re.match(r"^https?://|^DOI|^doi|^www\.|^Nature|^Science|^arXiv", l):
            continue
        if re.match(r"^\d{1,3}\s*$", l):  # page number
            continue
        if re.match(r"^\(?\d{4}\)?$", l):  # year line
            continue
        if re.match(r"^[A-Z][A-Za-z,\-\s]+et al\.$", l):  # author line
            continue
        if re.match(r"^[A-Z][A-Za-z\-\s]+[A-Z][A-Za-z\-\s]+$", l):  # name header
            if l in seen_headers:
                continue
            seen_headers.add(l)
            continue
        cleaned.append(l)
    return "\n".join(cleaned)

def remove_figure_axis_labels(text):
    """Remove numeric and short symbol lines typical of axis labels."""
    lines = text.splitlines()
    new_lines = []
    for l in lines:
        stripped = l.strip()
        if re.fullmatch(r"[\d\s\.\(\)××eE\-\+]+", stripped):
            continue
        if re.search(r"\b(kHz|Hz|s|ms|nm|μm|cm|dB|mol|K|°C|×10)\b", stripped):
            continue
        if len(stripped) <= 2:  # single letters
            continue
        tokens = stripped.split()
        if len(tokens) >= 3 and sum(len(t) < 3 for t in tokens) > len(tokens) / 2:
            continue
        new_lines.append(l)
    return "\n".join(new_lines)

def collapse_blank_lines(text):
    """Collapse multiple blank lines."""
    return re.sub(r"\n{3,}", "\n\n", text)

def extract_and_remove_figure_captions(text, out_dir):
    """
    Extract figure captions and remove them from main text.
    Each caption saved as figure_XXX.txt.
    """
    caption_pattern = re.compile(
        r"(?:(?:^|\n)(?:Fig(?:ure)?\.?|Extended Data Fig\.|Supplementary Fig(?:ure)?\.?)\s*\d+[a-zA-Z]?(?:[.:)]|[\s|–-]))"
        r"[\s\S]*?(?=\n\s*\n|(?:(?:Fig(?:ure)?|Extended Data Fig\.|Supplementary Fig\.|References)\b)|\Z)",
        flags=re.MULTILINE
    )
    captions = [m.group(0).strip() for m in caption_pattern.finditer(text)]
    for i, cap in enumerate(captions, 1):
        with open(os.path.join(out_dir, f"figure_{i:03d}.txt"), "w") as f:
            f.write(cap + "\n")
    cleaned_text = caption_pattern.sub("\n", text)
    return cleaned_text, captions

def separate_refs_and_supplementary(full_text):
    """
    Separate body, references, and supplementary materials.
    """
    refs_match = re.search(r"(?mi)\n\s*References\s*\n", full_text)
    if not refs_match:
        return full_text, "", ""

    refs_start = refs_match.start()
    refs_text = full_text[refs_start:]
    body = full_text[:refs_start]

    # Now detect supplementary material after references
    supp_match = re.search(r"(?mi)\n\s*(Supplementary|Extended Data|Methods)\s", refs_text)
    if supp_match:
        supp_start = supp_match.start()
        refs_only = refs_text[:supp_start]
        supp_only = refs_text[supp_start:]
        return body, refs_only, supp_only
    else:
        return body, refs_text, ""

# ---------- MAIN ----------
def main():
    print(f"Reading PDF from {PDF_PATH} ...")
    raw_text = extract_pdf_text(PDF_PATH)
    print(f"Extracted {len(raw_text)} characters.")

    # Remove headers/footers first
    cleaned = remove_page_headers_and_footers(raw_text)

    # Separate body, references, and supplementary
    body, refs_text, supp_text = separate_refs_and_supplementary(cleaned)
    print(f"Body: {len(body)}, Refs: {len(refs_text)}, Supp: {len(supp_text)}")

    # Extract and remove figure captions (each as .txt)
    body_no_figs, captions = extract_and_remove_figure_captions(body, BASE_DIR)
    print(f"Extracted {len(captions)} figure captions.")

    # Remove numeric figure axis junk
    body_no_noise = remove_figure_axis_labels(body_no_figs)
    final_body = collapse_blank_lines(body_no_noise)

    # Save files
    with open(OUT_BODY, "w") as f:
        f.write(final_body.strip() + "\n")
    with open(OUT_CAPTIONS_INDEX, "w") as f:
        for i, cap in enumerate(captions, 1):
            f.write(f"=== Figure {i} ===\n{cap}\n\n")
    if refs_text:
        with open(OUT_REFS, "w") as f:
            f.write(refs_text.strip() + "\n")
    if supp_text:
        with open(OUT_SUPPL, "w") as f:
            f.write(supp_text.strip() + "\n")

    print("\n✅ Done preprocessing:")
    print(f" → Cleaned body: {OUT_BODY}")
    print(f" → {len(captions)} figure caption files in {BASE_DIR}")
    print(f" → References: {OUT_REFS}")
    if supp_text:
        print(f" → Supplementary: {OUT_SUPPL}")

if __name__ == "__main__":
    main()
