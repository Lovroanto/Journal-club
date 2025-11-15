
#!/usr/bin/env python3
"""
test_grobid_modes.py

Minimal wrapper to test grobid_preprocessor in both modes:
- Standard mode (all features)
- RAG mode (only chunks + figures, optional article_name)
"""

from pathlib import Path
from grobid_preprocessor import preprocess_pdf
from chunksummary_module import run_chunk_summary

pdf_file = "/home/lbarisic/ai_data/Journal_Club/First/main/s41567-025-02854-4.pdf"           # Path to your PDF
output_dir_article = "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/"
#output_dir_rag = "/home/lbarisic/ai_data/Journal_Club/First/test_article/rag_mode/"

# --------------------------
# 1) Standard mode
# --------------------------
result_standard = preprocess_pdf(
    pdf_path=pdf_file,
    output_dir=output_dir_article,
    chunk_size=4000,
    rag_mode=False,          # standard mode
    keep_references=True     # also produce with_references folder
)
print("STANDARD MODE RESULTS:")
print(result_standard)
output_dir_summary = "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary/"
run_chunk_summary(output_dir_article, output_dir_summary)



