# build_literature_rag.py
import logging
from pathlib import Path
from typing import Union                     # ← only this import needed for old Python
from concurrent.futures import ProcessPoolExecutor, as_completed

from grobid_preprocessor import preprocess_pdf


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_single_pdf(
    pdf_path: Path,
    rag_folder: Path,
    max_chunk_size: int = 3500,
) -> None:
    """
    Process one PDF using the perfect RAG-only literature mode.
    """
    if not pdf_path.is_file():
        logger.warning(f"File not found, skipping: {pdf_path}")
        return

    article_name = pdf_path.stem  # e.g. "Smith_2024_Nature"

    try:
        logger.info(f"Processing → {pdf_path.name}")

        preprocess_pdf(
            pdf_path=str(pdf_path),
            output_dir=str(rag_folder),
            rag_mode=True,
            correct_grobid=True,              # best text quality
            process_supplementary=False,      # no supplementary noise
            keep_references=False,
            preclean_pdf=True,
            chunk_size=max_chunk_size,
            article_name=article_name,
        )
        logger.info(f"Done → {pdf_path.name} | chunks saved as {article_name}_main_chunkXXX.txt")

    except Exception as e:
        logger.error(f"Failed → {pdf_path.name} | Error: {e}", exc_info=True)


def build_literature_rag(
    pdf_folder: Union[str, Path],
    rag_folder: Union[str, Path],
    max_chunk_size: int = 3500,
    max_workers: int = 6,
    skip_existing: bool = True,
) -> None:
    """
    Convert all PDFs in pdf_folder → clean, traceable RAG chunks in rag_folder.
    One flat folder. No figures. No supplementary. Perfect for literature RAG.
    """
    pdf_folder = Path(pdf_folder)
    rag_folder = Path(rag_folder)
    rag_folder.mkdir(parents=True, exist_ok=True)

    if not pdf_folder.is_dir():
        raise ValueError(f"pdf_folder not found: {pdf_folder}")

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {pdf_folder}")
        return

    logger.info(f"Found {len(pdf_files)} PDFs in {pdf_folder}")
    logger.info(f"Output RAG folder → {rag_folder}")
    logger.info(f"Using {max_workers} parallel workers | chunk_size={max_chunk_size}")

    # Skip already processed papers
    if skip_existing:
        existing_prefixes = {
            p.name.split("_main_chunk")[0]
            for p in rag_folder.glob("*_main_chunk*.txt")
            if "_main_chunk" in p.name
        }
        pdf_files = [p for p in pdf_files if p.stem not in existing_prefixes]
        logger.info(f"After skip_existing: {len(pdf_files)} PDFs left to process")

    if not pdf_files:
        logger.info("All PDFs already processed. Nothing to do!")
        return

    # Parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_pdf, pdf_path, rag_folder, max_chunk_size): pdf_path
            for pdf_path in pdf_files
        }

        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Unhandled exception for {pdf_path.name}: {e}", exc_info=True)

    logger.info(f"All done! Your RAG literature corpus is ready in → {rag_folder}")


# ——— CLI entrypoint (run with: python build_literature_rag.py --pdf_folder ... ) ———
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a clean RAG literature corpus from a folder of PDFs"
    )
    parser.add_argument("--pdf_folder", type=str, required=True, help="Folder with literature PDFs")
    parser.add_argument("--rag_folder", type=str, required=True, help="Where to save all text chunks")
    parser.add_argument("--max_chunk_size", type=int, default=3500, help="Max tokens per chunk")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers (6 is safe default)")

    args = parser.parse_args()

    build_literature_rag(
        pdf_folder=args.pdf_folder,
        rag_folder=args.rag_folder,
        max_chunk_size=args.max_chunk_size,
        max_workers=args.workers,
    )