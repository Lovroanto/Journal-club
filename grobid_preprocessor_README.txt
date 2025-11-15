GROBID Preprocessor Module
==========================

Description:
------------
This module preprocesses PDF files using GROBID (http://grobid.readthedocs.io/) to extract structured text, metadata, figures, references, and create text chunks suitable for downstream processing (e.g., NLP, RAG pipelines). It supports two modes of operation:

1. Standard Mode:
   - Extracts all features:
     - Metadata (title, authors, abstract)
     - Main and supplementary sections
     - Figures (captions and descriptions)
     - References
   - Optional: create chunks that include references appended at the end.

2. RAG Mode:
   - Simplified extraction for retrieval-augmented generation:
     - Only chunks and figures are extracted
     - References, figure maps, raw TEI are omitted
     - Chunks and figures are named with the article name prefix
     - All outputs are saved in a single folder (no subfolders)

Modes are controlled via:
- `rag_mode: bool` → True activates RAG mode, False is standard mode
- `article_name: str` → used for file naming in RAG mode
- `keep_references: bool` → optional in standard mode, keeps references appended to chunks

--------------------------------------------------------------------------------
Workflow Overview:
--------------------------------------------------------------------------------
1. Run GROBID fulltext processing on PDF → TEI XML.
2. Extract metadata:
   - Title
   - Authors
   - Abstract
3. Extract figures:
   - Figure ID (`xml:id`)
   - Title in article (`<head>`)
   - Description (`<figDesc>`)
   - Saved as individual `.txt` files
4. Extract references (standard mode only):
   - Reference ID
   - Title
   - Authors
   - Journal
   - Year
   - Additional IDs (arXiv, DOI, etc.)
5. Extract main and supplementary sections:
   - Split into chunks of ~4000 characters
   - Track which figures are mentioned in each chunk
6. Save all outputs according to mode and options.

--------------------------------------------------------------------------------
Output Structure (Standard Mode):
--------------------------------------------------------------------------------
output_dir/
├── raw_tei.xml                ← full TEI XML from GROBID
├── title.txt                  ← metadata (title, authors, abstract)
├── references.txt             ← all extracted references
├── figmain.txt                ← list of figures appearing in main text
├── figsup.txt                 ← list of figures appearing in supplementary
├── figmain_map.txt            ← mapping of figures → chunk numbers (main)
├── figsup_map.txt             ← mapping of figures → chunk numbers (supp)
├── main/                      ← main text chunks
│   ├── main_chunk01.txt
│   ├── main_chunk02.txt
│   └── ...
├── supplementary/             ← supplementary text chunks
│   ├── sup_chunk01.txt
│   ├── sup_chunk02.txt
│   └── ...
├── figures/                   ← figure descriptions
│   ├── fig_0.txt
│   ├── fig_1.txt
│   └── ...
└── with_references/           ← optional: chunks including references
    ├── main/
    └── supplementary/

Notes:
- Each chunk file contains text from the section, plus a list of figure IDs mentioned in that chunk.
- Figure files contain:
    FIGURE ID, Title in Article, Description
- References are fully structured with ID, Title, Authors, Journal, Year, and other identifiers.

--------------------------------------------------------------------------------
Output Structure (RAG Mode):
--------------------------------------------------------------------------------
output_dir/
├── title.txt                  ← metadata
├── main_chunk01.txt           ← chunk from main text
├── main_chunk02.txt
├── sup_chunk01.txt            ← chunk from supplementary
├── sup_chunk02.txt
├── fig_0.txt                  ← figure descriptions
├── fig_1.txt
└── ...

Notes:
- All files are in a single folder (no `main`, `supplementary`, or `figures` subfolders)
- Figure file names and chunk file names can include the article name prefix
  (e.g., `MyArticle_chunk_main01.txt`, `MyArticle_fig_0.txt`)
- `raw_tei.xml`, figure maps, and reference files are omitted
- Suitable for feeding directly into RAG or similar retrieval systems

--------------------------------------------------------------------------------
Usage Summary:
--------------------------------------------------------------------------------
- Standard Mode:
    preprocess_pdf(pdf_path, output_dir, rag_mode=False, keep_references=True)
- RAG Mode:
    preprocess_pdf(pdf_path, output_dir, rag_mode=True, article_name="MyArticle")

--------------------------------------------------------------------------------
End of Document
--------------------------------------------------------------------------------
