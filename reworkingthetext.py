#!/usr/bin/env python3
"""
reworkingthetext.py

Preprocess the PDF for Journal Club:
1. Extract figure captions (each in its own file)
2. Remove figure captions, axis noise, and headers/footers from main body
3. Separate References
4. Split Supplementary material into sections (improved title detection)
5. Save all outputs in ~/ai_data/Journal_Club/First/main_article/
"""

import os
import re
import fitz  # PyMuPDF

# ---------- CONFIG ----------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First/main_article")
PDF_PATH = os.path.join(BASE_DIR, "../main/s41567-025-02854-4.pdf")

os.makedirs(BASE_DIR, exist_ok=True)

OUT_BODY = os.path.join(BASE_DIR, "body_cleaned.txt")
OUT_REFS = os.path.join(BASE_DIR, "references.txt")
OUT_CAPTIONS_INDEX = os.path.join(BASE_DIR, "figure_captions_index.txt")
SUPP_SECTIONS_DIR = os.path.join(BASE_DIR, "supplementary_sections")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SUPP_SECTIONS_DIR, exist_ok=True)

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
        if re.match(r"^https?://|^DOI|^doi|^www\.|^Nature|^Science|^arXiv", l):
            continue
        if re.match(r"^\d{1,3}\s*$", l):  # page number
            continue
        if re.match(r"^\(?\d{4}\)?$", l):  # year line
            continue
        if re.match(r"^[A-Z][A-Za-z,\-\s]+et al\.$", l):
            continue
        if re.match(r"^[A-Z][A-Za-z\-\s]+[A-Z][A-Za-z\-\s]+$", l):
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
        if len(stripped) <= 2:
            continue
        tokens = stripped.split()
        if len(tokens) >= 3 and sum(len(t) < 3 for t in tokens) > len(tokens)/2:
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
        r"[\s\S]*?(?=\n\s*\n|(?:(?:Fig(?:ure)?|Extended Data Fig\.|Supplementary Fig\.|References|Supplementary|Methods)\b)|\Z)",
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
    Returns: body_text, references_text, list of (section_number_title, section_text)
    """
    # Extract references first
    refs_match = re.search(r"(?mi)\n\s*References\s*\n", full_text)
    if refs_match:
        refs_start = refs_match.start()
        refs_text = full_text[refs_start:]
        body_text = full_text[:refs_start]
    else:
        body_text = full_text
        refs_text = ""

    # Extract supplementary material (everything after keywords)
    supp_match = re.search(r"(?mi)\n\s*(Supplementary|Extended Data|Methods)\b", refs_text)
    if supp_match:
        supp_start = supp_match.start()
        supp_text = refs_text[supp_start:]
        refs_only = refs_text[:supp_start]
    else:
        supp_text = ""
        refs_only = refs_text

    # Heuristic section split
    supp_sections = []
    lines = supp_text.splitlines()
    current_title = None
    current_content = []
    section_counter = 1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            current_content.append("")
            continue

        # Heuristic for section title
        is_short = len(stripped.split()) <= 4  # max 4 words
        is_capitalized = stripped[0].isupper()
        no_period = not stripped.endswith(".")
        has_alpha = any(c.isalpha() for c in stripped)
        keyword_match = re.search(r"Section\s+\d+|Supplementary\s+Figure|Extended Data\s+Figure|Methods|Homodyne|Experimental", stripped, re.IGNORECASE)
        paragraph_start_words = {"Both", "This", "We", "In", "At", "For", "Typical"}

        first_word = stripped.split()[0] if stripped.split() else ""
        likely_paragraph_start = first_word in paragraph_start_words

        if (is_short and is_capitalized and no_period and has_alpha and (keyword_match or not likely_paragraph_start)):
            # Save previous section
            if current_title and current_content:
                section_name = f"{section_counter:02d}_{current_title}"
                supp_sections.append((section_name, "\n".join(current_content).strip()))
                section_counter += 1
            current_title = re.sub(r"[^\w\d]+", "_", stripped)
            current_content = []
        else:
            current_content.append(line)

    # Add last section
    if current_title and current_content:
        section_name = f"{section_counter:02d}_{current_title}"
        supp_sections.append((section_name, "\n".join(current_content).strip()))

    return body_text, refs_only, supp_sections

# ---------- MAIN ----------
def main():
    print(f"Reading PDF from {PDF_PATH} ...")
    raw_text = extract_pdf_text(PDF_PATH)
    print(f"Extracted {len(raw_text)} characters.")

    # Remove headers/footers first
    cleaned = remove_page_headers_and_footers(raw_text)

    # Separate body, references, and supplementary
    body, refs_text, supp_sections = separate_refs_and_supplementary(cleaned)
    print(f"Body: {len(body)}, Refs: {len(refs_text)}, Supplementary sections: {len(supp_sections)}")

    # Save full supplementary text for debugging
    full_supp_path = os.path.join(BASE_DIR, "supplementary_full.txt")
    with open(full_supp_path, "w", encoding="utf-8") as f:
        full_supp_text = "\n\n".join([sec_text for _, sec_text in supp_sections])
        f.write(full_supp_text.strip() + "\n")
    print(f"→ Full supplementary saved to {full_supp_path}")

    # Extract and remove figure captions (each as .txt)
    body_no_figs, captions = extract_and_remove_figure_captions(body, FIGURES_DIR)
    print(f"Extracted {len(captions)} figure captions.")

    # Remove numeric figure axis junk
    body_no_noise = remove_figure_axis_labels(body_no_figs)
    final_body = collapse_blank_lines(body_no_noise)

    # Save cleaned main body
    with open(OUT_BODY, "w", encoding="utf-8") as f:
        f.write(final_body.strip() + "\n")

    # Save figure captions index
    with open(OUT_CAPTIONS_INDEX, "w", encoding="utf-8") as f:
        for i, cap in enumerate(captions, 1):
            f.write(f"=== Figure {i} ===\n{cap}\n\n")

    # Save references
    if refs_text:
        with open(OUT_REFS, "w", encoding="utf-8") as f:
            f.write(refs_text.strip() + "\n")

    # Save each supplementary section separately
    os.makedirs(SUPP_SECTIONS_DIR, exist_ok=True)
    for title, text in supp_sections:
        section_path = os.path.join(SUPP_SECTIONS_DIR, f"{title}.txt")
        with open(section_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    print("\n✅ Done preprocessing:")
    print(f" → Cleaned body: {OUT_BODY}")
    print(f" → {len(captions)} figure caption files in {FIGURES_DIR}")
    print(f" → References: {OUT_REFS}")
    print(f" → Supplementary sections saved in {SUPP_SECTIONS_DIR}")
    print(f" → Full supplementary saved: {full_supp_path}")

if __name__ == "__main__":
    main()
