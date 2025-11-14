#!/usr/bin/env python3

from grobid_preprocessor import preprocess_pdf

pdf_path = "/home/lbarisic/ai_data/Journal_Club/First/summaries/references/articles/2012.08885v5.pdf"
output_dir = "/home/lbarisic/ai_data/Journal_Club/First/test_article/"
results = preprocess_pdf(pdf_path, output_dir, chunk_size=5000)
print("✅ Preprocessing complete!")
print(f"Title + abstract saved at: {results['title_txt']}")
print(f"Main chunks: {results['main_chunks']}, Supplementary chunks: {results['supp_chunks']}")
