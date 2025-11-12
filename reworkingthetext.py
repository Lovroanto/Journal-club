#!/usr/bin/env python3
"""
preprocess_extract_figures_and_refs.py

Step 0 — Preprocess the paper:
1) Extract all figure captions (Figure, Fig., Supplementary Figure, etc.)
2) Remove them from the main text
3) Separate References section
"""

import os
import re
import fitz  # PyMuPDF

BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
PDF_PATH = os.path.join(BASE_DIR, "main", "s41567-025-02854-4.pdf")

OUT_BODY = os.path.join(BASE_DIR, "body_cleaned.txt")
OUT_FIGS = os.path.join(BASE_DIR, "figure_captions.txt")
OUT_REFS = os.path.join(BASE_DIR, "references.txt")

# --------- helper functions ---------
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text

def separate_references(full_text):
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
    # fallback: last 20% assumed as refs
    cutoff = int(len(full_text) * 0.8)
    return full_text[:cutoff], full_text[cutoff:]

def extract_and_remove_figure_captions(text):
    """
    Find figure captions and remove them.
    Captions usually look like:
      "Figure 1 | ..." or "Fig. 2. ..." or "Extended Data Fig. 3 | ..."
    """
    # regex handles multiline captions ending before a blank line or next Figure
    caption_pattern = re.compile(
        r"(?:(?:^|\n)(?:Fig(?:ure)?\.?|Extended Data Fig\.|Supplementary Fig(?:ure)?\.?)\s*\d+[a-zA-Z]?(?:[.:)]|[\s|–-]))"
        r"[\s\S]*?(?=\n\s*\n|(?:(?:Fig(?:ure)?|Extended Data Fig\.|Supplementary Fig(?:ure)?|References)\b)|\Z)",
        flags=re.MULTILINE
    )

    captions = caption_pattern.findall(text)  # Only first token captured
    full_captions = caption_pattern.findall(text)
    # Actually we want full matches:
    full_captions = [m.group(0).strip() for m in caption_pattern.finditer(text)]

    cleaned_text = caption_pattern.sub("\n", text)
    return cleaned_text, full_captions

# --------- main process ---------
def main():
    print(f"Reading {PDF_PATH} ...")
    full_text = extract_pdf_text(PDF_PATH)
    print(f"Total length: {len(full_text)} characters")

    # 1) Split off references
    body_text, refs_text = separate_references(full_text)
    print(f"Body length: {len(body_text)}, Refs length: {len(refs_text)}")

    # 2) Extract and remove figure captions
    cleaned_body, captions = extract_and_remove_figure_captions(body_text)
    print(f"Found {len(captions)} potential figure captions")

    # 3) Write results
    with open(OUT_BODY, "w") as f:
        f.write(cleaned_body.strip() + "\n")
    with open(OUT_FIGS, "w") as f:
        for i, cap in enumerate(captions, 1):
            f.write(f"=== Figure Caption {i} ===\n{cap}\n\n")
    with open(OUT_REFS, "w") as f:
        f.write(refs_text.strip() + "\n")

    print("✅ Preprocessing done:")
    print(f"  → {OUT_BODY}")
    print(f"  → {OUT_FIGS}")
    print(f"  → {OUT_REFS}")

if __name__ == "__main__":
    main()