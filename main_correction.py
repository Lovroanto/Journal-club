
#!/usr/bin/env python3
"""
test_grobid_modes.py
Test the latest grobid_preprocessor with:
→ fake-reference protection
→ fast mode (correct_grobid=False)
→ no supplementary material
"""

from pathlib import Path
from grobid_preprocessor import preprocess_pdf
from exercise_splitter import run_tagger, collect_all_text, split_exercises
# from chunksummary_module import run_chunk_summary   # uncomment when ready

# ------------------------------------------------------------------
# Your PDF and output folder
# ------------------------------------------------------------------
pdf_file = "/home/lbarisic/ai_data/CorrectedTD/First/PDF/td4_251121_132609.pdf"
output_dir_article = Path("/home/lbarisic/ai_data/CorrectedTD/First/Decomposition/")

# ------------------------------------------------------------------
# 1) STANDARD MODE – fast + clean (recommended for journal club)
# ------------------------------------------------------------------
print("Running preprocess_pdf (fast & smart mode)...")
result = preprocess_pdf(
    pdf_path=pdf_file,
    output_dir=output_dir_article,
    chunk_size=2000,
    correct_grobid=False,          # ← fast, no heavy correction
    process_supplementary=False,   # ← ignore supplementary
    keep_references=True,          # ← will ONLY create folder if REAL references exist
    preclean_pdf=True,             # ← optional but recommended
    use_grobid_consolidation=True  # ← better header/citations when they exist
)

# ------------------------------------------------------------------
# Print what actually happened
# ------------------------------------------------------------------
print("\nSTANDARD MODE FINISHED!")
print(f"   Main chunks    : {result['main_chunks']}")
print(f"   Supp chunks    : {result['supp_chunks']}")
print(f"   Final TEI      : {result['final_tei']}")
print(f"   Title file     : {result['title_txt']}")
print(f"   Real references? → {result['has_references']}")

if result["has_references"]:
    print("   → references.txt and with_references/ folder created")
else:
    print("   → No real references → clean output, no fake files")
    
# Step 1 — Tag the chunks
run_tagger(
    input_dir=output_dir_article/"main",
    output_dir=output_dir_article /"Tagged",
    model="llama3.1"
)

# Step 2 — Parse into exercises
full_text = collect_all_text(output_dir_article /"Tagged")
split_exercises(full_text, output_dir_article / "Exercises")

# ------------------------------------------------------------------
# (Optional) Run summary when you want
# ------------------------------------------------------------------
# output_dir_summary = "/home/lbarisic/ai_data/CorrectedTD/First/Summary/"
# run_chunk_summary(output_dir_article, output_dir_summary)



