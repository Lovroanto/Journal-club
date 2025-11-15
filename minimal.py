
#!/usr/bin/env python3
"""
test_grobid_modes.py

Minimal wrapper to test grobid_preprocessor in both modes:
- Standard mode (all features)
- RAG mode (only chunks + figures, optional article_name)
"""

from pathlib import Path
from grobid_preprocessor import preprocess_pdf

pdf_file = "/home/lbarisic/ai_data/Journal_Club/First/summaries/references/articles/2012.08885v5.pdf"           # Path to your PDF
output_dir_standard = "/home/lbarisic/ai_data/Journal_Club/First/test_article/"
output_dir_rag = "/home/lbarisic/ai_data/Journal_Club/First/test_article/rag_mode/"

# --------------------------
# 1) Standard mode
# --------------------------
result_standard = preprocess_pdf(
    pdf_path=pdf_file,
    output_dir=output_dir_standard,
    chunk_size=4000,
    rag_mode=False,          # standard mode
    keep_references=True     # also produce with_references folder
)
print("STANDARD MODE RESULTS:")
print(result_standard)

# --------------------------
# 2) RAG mode
# --------------------------
article_name = "ExampleArticle"
result_rag = preprocess_pdf(
    pdf_path=pdf_file,
    output_dir=output_dir_rag,
    chunk_size=4000,
    rag_mode=True,           # RAG mode
    article_name=article_name
)
print("RAG MODE RESULTS:")
print(result_rag)
