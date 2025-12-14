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
from build_literature_rag import build_literature_rag
from rag_ingester import create_or_update_vectorstore
from notion_extractor_module import extract_specialized_notions
from slide_enrichment_module import enrich_presentation_plan
from supplementary_linker import integrate_supplementary
from generate_slides import generate_slides

pdf_file = "/home/lbarisic/ai_data/Journal_Club/First/main/s41567-025-02854-4.pdf"
output_dir_article = "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/"

# --------------------------
# 0) Instantiate LLM
# --------------------------
LLM_MODEL = "llama3.1"
llm = OllamaLLM(model="llama3.1")  # ← create LLM object once





main_folder = "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/main/"
supp_folder = "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/supplementary/"
if (False):
    # --------------------------
    # 1) Standard mode
    # --------------------------
    result_standard = preprocess_pdf(
        pdf_path=pdf_file,
        output_dir=output_dir_article,
        chunk_size=4000,
        overlap = 200,   
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
    print("PREPROCESSING DONE!")
    print(f"→ Title file       : {result_standard['title_txt']}")
    print(f"→ Main chunks      : {result_standard['main_chunks']} chunks")
    print(f"→ Supp chunks      : {result_standard['supp_chunks']} chunks")
    print(f"→ Has references   : {result_standard['has_references']}")

    # THE MOST IMPORTANT ONES FOR RAG:
    main_folder = result_standard["main_chunks_dir"]
    supp_folder = result_standard["supp_chunks_dir"]


output_dir_summary = "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary/"
# --------------------------
# doing the summaey of each chunck
# --------------------------
if(False):
    run_chunk_summary(output_dir_article, output_dir_summary, llm=llm)
# --------------------------
# generating the global summary and presentation plan
# --------------------------
if(False):
    generate_global_and_plan(
        summaries_dir=output_dir_summary,
        output_dir=output_dir_summary,
        slides_output_dir=str(Path(output_dir_summary) / "slides"),   # ← this activates the split!
        desired_slides=0,
        llm=llm
    )

if(False):
    integrate_supplementary(
        chunks_dir=str(Path(output_dir_summary) / "supplementary_chunks"),
        #chunks_dir=supp_folder, # this is only if the process pdf was done
        summary_output_file=str(Path(output_dir_summary) / "06_supplementary_global.txt"),
        slides_dir=str(Path(output_dir_summary) / "slides"),
        llm=llm
    )

# --------------------------
# extracting notions from presentation plan
# --------------------------
if(False):
    extract_specialized_notions(
        Path(output_dir_summary) /"03_main_contextualized.txt",
        Path(output_dir_summary) /"05_Extracted_notions.txt",
        model="llama3.1:latest",
        num_repeats=5,
        min_occurrences=3
    )


# --------------------------
# extracting referenced papers balanced
# --------------------------
if(False):
    extract_referenced_papers_balanced(
        notions_txt=Path(output_dir_summary) /"05_Extracted_notions.txt",
        grobid_tei=Path(output_dir_article) /"NEWraw_tei.xml",
        bib_tt=Path(output_dir_article) /"references.txt",
        output_csv=Path(output_dir_summary) /"06_relevant_references.csv",
        fuzzy_threshold=90
    )
# --------------------------
# building RAG dataset
# --------------------------
if(False):
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
# --------------------------
# building literature RAG
# --------------------------
if(False):
    build_literature_rag(
        pdf_folder=Path(output_dir_article)  /"pdfs_litterature",
        rag_folder=Path(output_dir_article)  /"RAG/literature_rag",
        max_chunk_size=3500,
        max_workers=1,          # safe default (GROBID runs locally)
        skip_existing=True      # won't reprocess if chunks already exist
    )
# --------------------------
# creating vectorstores for main and supp folders
# --------------------------
if (False):
    create_or_update_vectorstore(
        data_folder=Path(main_folder),
        persist_directory=str(Path(main_folder) / "chroma_db"),   # ← convert to str
        embedding_model_name="BAAI/bge-small-en-v1.5",
        include_pdfs=True,
    )
    create_or_update_vectorstore(
        data_folder=Path(supp_folder),
        persist_directory=str(Path(supp_folder) / "chroma_db"),
        embedding_model_name="BAAI/bge-small-en-v1.5",
        include_pdfs=True,
    )
    create_or_update_vectorstore(
        data_folder=Path(output_dir_article)  /"RAG/literature_rag",
        persist_directory=str(Path(output_dir_article)  /"RAG/literature_rag" / "chroma_db"),
        embedding_model_name="BAAI/bge-small-en-v1.5",
        include_pdfs=True,
    )
    create_or_update_vectorstore(
        data_folder=Path(output_dir_article)  /"figures",
        persist_directory=str(Path(output_dir_article)  /"figures" / "chroma_db"),
        embedding_model_name="BAAI/bge-small-en-v1.5",
        include_pdfs=True,
    )
if(True):
    
    slide_group_file = str(Path(output_dir_summary) /"slides/003_Introduction_to_Continuous_Lasing.txt")    # contains Slide X + context
    global_context_file = str(Path(output_dir_summary) /"04_presentation_plan.txt"  )      # summary of whole paper (you already generated this earlier) 
    db_paths = {
        "main": str(Path(main_folder) / "chroma_db"),
        "figs": str(Path(output_dir_article)  /"figures" / "chroma_db"),
        "sup": str(Path(supp_folder) / "chroma_db"),
        "lit": str(Path(output_dir_article)  /"RAG/literature_rag" / "chroma_db")
    }

    output_dir_slides = str(Path(output_dir_summary) /"Finalplan_slides")    # one .txt per slide will appear here
    generate_slides(
        slide_group_file=slide_group_file,
        global_context_file=global_context_file,
        db_paths=db_paths,
        output_dir=output_dir_slides,
        model=LLM_MODEL
    )








#if (True):
#    enrich_presentation_plan(
#        summaries_dir=output_dir_summary,
#        plan_path=Path(output_dir_summary) /"04_presentation_plan.txt",
#        output_dir=output_dir_summary /"07_final_presentation",
#        llm=llm
#    )



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

