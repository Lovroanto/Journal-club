# Presentation_module/rag_loader.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class RAGStores:
    main: Chroma
    figs: Chroma
    sup: Optional[Chroma] = None
    lit: Optional[Chroma] = None


def load_chroma_db(path: Union[str, Path], *, embedding_model: str) -> Chroma:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chroma DB not found: {p}")
    return Chroma(
        persist_directory=str(p),
        embedding_function=HuggingFaceEmbeddings(model_name=embedding_model),
    )


def load_rag_stores(
    paths: Dict[str, Union[str, Path]],
    *,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> RAGStores:
    main = load_chroma_db(paths["main"], embedding_model=embedding_model)
    figs = load_chroma_db(paths["figs"], embedding_model=embedding_model)

    sup = load_chroma_db(paths["sup"], embedding_model=embedding_model) if "sup" in paths and paths["sup"] else None
    lit = load_chroma_db(paths["lit"], embedding_model=embedding_model) if "lit" in paths and paths["lit"] else None

    return RAGStores(main=main, figs=figs, sup=sup, lit=lit)
