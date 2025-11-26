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
from global_summary_module import generate_global_and_plan
from langchain_ollama import OllamaLLM   # ← import LLM class
from extract_referenced_papers import extract_referenced_papers_balanced
from module_rag_builder import build_rag_dataset
#from slide_enrichment_module import enrich_presentation_plan
from notion_extractor_module import extract_specialized_notions

pdf_file = "/home/lbarisic/ai_data/Journal_Club/First/main/s41567-025-02854-4.pdf"
output_dir_article = "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/"

# --------------------------
# 0) Instantiate LLM
# --------------------------
LLM_MODEL = "llama3.1"
llm = OllamaLLM(model="llama3.1")  # ← create LLM object once

if (False):
    # --------------------------
    # 1) Standard mode
    # --------------------------
    result_standard = preprocess_pdf(
        pdf_path=pdf_file,
        output_dir=output_dir_article,
        chunk_size=4000,
        rag_mode=False,
        keep_references=True,
        preclean_pdf=True,
        use_grobid_consolidation=True,
        correct_grobid=True,
        process_supplementary=True,
        ask_user_for_supplementary=False,  # ← enables interactive prompt
        llm=llm,                          # ← LLM is passed
    )
    print("STANDARD MODE RESULTS:")
    print(result_standard)


output_dir_summary = "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary/"
if(False):
    run_chunk_summary(output_dir_article, output_dir_summary, llm=llm)
if(False):
    generate_global_and_plan(output_dir_summary, output_dir_summary, desired_slides=0, llm=llm)
if(False):
    extract_specialized_notions(
        Path(output_dir_summary) /"04_presentation_plan.txt",
        Path(output_dir_summary) /"05_Extracted_notions.txt",
        model="llama3.1:latest",
        num_repeats=5,
        min_occurrences=3
    )
if(False):
    extract_referenced_papers_balanced(
        notions_txt=Path(output_dir_summary) /"05_Extracted_notions.txt",
        grobid_tei=Path(output_dir_article) /"NEWraw_tei.xml",
        bib_tt=Path(output_dir_article) /"references.txt",
        output_csv=Path(output_dir_summary) /"06_relevant_references.csv",
        fuzzy_threshold=90
    )
if(True):
    build_rag_dataset(
        csv_input=Path(output_dir_summary) /"06_relevant_references.csv",
        notions_txt=Path(output_dir_summary) /"05_Extracted_notions.txt",
        rag_folder=Path(output_dir_article)  /"RAG/literature_rag",
        pdf_folder=Path(output_dir_article)  /"pdfs_litterature",
        cache_folder="./.cache",
        max_chunk_size=3500,
        max_articles_per_notion=4,
        use_wikipedia=True,
        download_pdfs=True,
        unpaywall_email="myemail@ens.fr", # set your email here for Unpaywall
        parallel_workers=6,
    )
#    build_rag_dataset(
#        csv_input=Path(output_dir_summary) /"06_relevant_references.csv",
#        notions_txt=Path(output_dir_summary) /"05_Extracted_notions.txt",
#        rag_folder=Path(output_dir_article)  /"RAG/literature_rag",
#        pdf_folder=Path(output_dir_article)  /"pdfs_litterature",
#        max_chunk_size=4000,
#        max_articles_per_notion=4,
#        use_wikipedia=True,        # set False if you already have them
#        download_pdfs=True,
#        llm_model_name=llm
#    )
#if (False):
#    enrich_presentation_plan(
#        summaries_dir=output_dir_summary,
#        plan_path=Path(output_dir_summary) /"04_presentation_plan.txt",
#        output_dir=output_dir_summary /"final_presentation",
#        llm=llm
#    )
