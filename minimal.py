#!/usr/bin/env python3
"""
minimal.py — robust runner for grobid_preprocessor.preprocess_pdf

This version tolerates different return dict keys produced by different
versions of grobid_preprocessor.py and prints a clean summary.
"""

from pathlib import Path
import pprint

# import the preprocess function (must be in the same folder or in PYTHONPATH)
from grobid_preprocessor import preprocess_pdf

# ----------------------------
# CONFIG
# ----------------------------
pdf_path = "/home/lbarisic/ai_data/Journal_Club/First/summaries/references/articles/2012.08885v5.pdf"
output_dir = "/home/lbarisic/ai_data/Journal_Club/First/test_article/"

print("🚀 Running grobid_preprocessor.preprocess_pdf() ...\n")

results = preprocess_pdf(pdf_path, output_dir)

print("\n🎉 Preprocessing finished. Detected output keys:")
pprint.pprint(list(results.keys()))
print("\n----- Human-friendly summary -----\n")

# Helper to safely print lists/paths
def print_list(label, value):
    print(f"{label}:")
    if not value:
        print("   (none)")
        return
    if isinstance(value, (list, tuple)):
        for v in value[:200]:
            print(f"   {v}")
    else:
        print(f"   {value}")

# Try to handle different possible key names
# 1) Raw TEI xml
raw_tei_keys = ["raw_tei", "tei_path", "raw_tei_path"]
for k in raw_tei_keys:
    if k in results:
        print(f"Raw TEI XML: {results[k]}")
        break

# 2) Body / main chunks
body_keys = ["body_chunks", "main_chunks", "main_files"]
found = False
for k in body_keys:
    if k in results:
        print_list("Body / main chunks", results[k])
        found = True
        break
if not found:
    print("Body / main chunks: (none found)")

# 3) Supplementary
supp_keys = ["supplementary_chunks", "supplementary", "supp_files"]
found = False
for k in supp_keys:
    if k in results:
        print_list("Supplementary chunks", results[k])
        found = True
        break
if not found:
    print("Supplementary chunks: (none found)")

# 4) Figures
fig_keys = ["figure_captions", "figures", "figures_list", "figures_dir"]
found = False
for k in fig_keys:
    if k in results:
        print_list("Figure captions / files", results[k])
        found = True
        break
if not found:
    print("Figure captions / files: (none found)")

# 5) References
ref_keys = ["references_txt", "references_file", "references", "references_txt_path"]
for k in ref_keys:
    if k in results:
        print(f"References file / text saved at: {results[k]}")
        break
else:
    # maybe results contains the list itself
    if "references_list" in results:
        print_list("References (list)", results["references_list"])
    elif "references" in results and isinstance(results["references"], list):
        print_list("References (list)", results["references"])
    else:
        print("References: (none found)")

# 6) Misc / counts
if "num_main_chunks" in results:
    print(f"\nNumber of main chunks: {results['num_main_chunks']}")
if "num_supplementary_chunks" in results:
    print(f"Number of supplementary chunks: {results['num_supplementary_chunks']}")
if "figures" in results and isinstance(results["figures"], int):
    print(f"Number of figures (reported): {results['figures']}")

print("\n----------------------------------")
print("If you expected a particular file but it's missing, check the raw TEI (if present)\nand inspect which tags GROBID produced. You can also paste the keys listed above\nhere and I will help map them to the pipeline.")
