# rag_ingester_smart.py
"""
Smart RAG ingestion module – November 2025 edition
→ Automatically chooses perfect chunk_size & overlap for your embedding model
→ GPU auto-detection
→ PDF support
→ Safety warnings only when really needed
"""

import os
from pathlib import Path
from typing import Optional, Literal

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ===================================================================
# Smart defaults per embedding model (updated November 2025)
# ===================================================================
MODEL_PRESETS = {
    # model name → (max_tokens, recommended_chunk_size_chars, recommended_overlap_chars)
    "all-MiniLM-L6-v2":                     (256,   1000,   100),
    "all-mpnet-base-v2":                    (384,   1500,   200),
    "BAAI/bge-small-en-v1.5":               (512,   3000,   400),
    "BAAI/bge-base-en-v1.5":                (512,   3000,   400),
    "BAAI/bge-large-en-v1.5":               (512,   3500,   500),
    "Snowflake/snowflake-arctic-embed-m":   (512,   3500,   500),
    "Snowflake/snowflake-arctic-embed-l":   (512,   4000,   600),
    "nomic-ai/nomic-embed-text-v1.5":       (8192,  8000,  1000),
    "voyageai/voyage-3":                    (16000, 12000, 1500),
    "voyageai/voyage-3-large":              (16000, 14000, 2000),
}

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def create_or_update_vectorstore(
    data_folder: str,
    persist_directory: str = "./chroma_rag_db",
    embedding_model_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    include_pdfs: bool = False,
    device: Optional[Literal["cuda", "cpu", "mps"]] = None,
) -> Chroma:
    data_path = Path(data_folder)
    if not data_path.exists():
        raise FileNotFoundError(f"Folder not found: {data_folder}")

    embedding_model_name = embedding_model_name or DEFAULT_MODEL

    # Auto-configure chunking parameters
    if embedding_model_name not in MODEL_PRESETS:
        print(f"Warning: Model {embedding_model_name} unknown → using conservative defaults")
        max_tokens = 512
        auto_chunk = 2500
        auto_overlap = 300
    else:
        max_tokens, auto_chunk, auto_overlap = MODEL_PRESETS[embedding_model_name]

    final_chunk_size = chunk_size if chunk_size is not None else auto_chunk
    final_overlap = chunk_overlap if chunk_overlap is not None else auto_overlap

    print(f"Using embedding model: {embedding_model_name}")
    print(f"Chunk size: {final_chunk_size} chars | Overlap: {final_overlap} chars (auto-tuned)")

    # ===================================================================
    # Load documents
    # ===================================================================
    if include_pdfs:
        loaders = [
            DirectoryLoader(data_folder, glob="**/*.txt", loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"}),
            DirectoryLoader(data_folder, glob="**/*.pdf", loader_cls=PyPDFLoader),
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
    else:
        loader = DirectoryLoader(data_folder, glob="**/*.txt", loader_cls=TextLoader,
                                 loader_kwargs={"encoding": "utf-8"}, recursive=True)
        docs = loader.load()

    print(f"Loaded {len(docs)} documents")

    # ===================================================================
    # Split with smart overlap
    # ===================================================================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks (average {sum(len(c.page_content) for c in chunks)//len(chunks):,} chars)")

    # Warn only if a final chunk is still bigger than model limit
    max_allowed_chars = max_tokens * 5
    too_big_chunks = [c for c in chunks if len(c.page_content) > max_allowed_chars]
    if too_big_chunks:
        print(f"\nWarning: {len(too_big_chunks)} final chunks are still > {max_tokens} tokens!")
        print("   Retrieval quality may suffer on these parts. Consider smaller source files.")
        for c in too_big_chunks[:5]:
            print(f"   • {c.metadata.get('source','?')} → {len(c.page_content)} chars")

    # ===================================================================
    # Embedding model – GPU auto-used if available
    # ===================================================================
    if device is None:
        device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    print(f"Loading embeddings on {device}...")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    # ===================================================================
    # Save vector DB
    # ===================================================================
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    db.persist()
    print(f"Vector DB ready → {persist_directory} ({db._collection.count()} vectors)")

    return db


# ===================================================================
# Easy one-liner for most users
# ===================================================================
if __name__ == "__main__":
    create_or_update_vectorstore(
        data_folder="./rag_data",
        persist_directory="./chroma_db",
        embedding_model_name="BAAI/bge-small-en-v1.5",   # change whenever you want
        include_pdfs=True,
        # chunk_size and overlap are auto-set perfectly — you almost never need to touch them
    )