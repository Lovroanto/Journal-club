# file: module_rag_builder.py
"""
Module 2 – RAG Dataset Builder (FINAL CLEAN & WORKING VERSION)
Clean Wikipedia + working PDF download
"""

import re
import time
import urllib.parse
from pathlib import Path
from typing import List, Dict
import csv
import wikipedia
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm


def clean_filename(s: str) -> str:
    s = re.sub(r'[^\w\s\-\(\)]', '', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    return s[:100]


# ──────────────────────────────────────────────────────────────
# CLEAN & ROBUST WIKIPEDIA
# ──────────────────────────────────────────────────────────────
def clean_wikipedia_text(raw_text: str) -> str:
    """Remove math, references, short lines, templates, etc."""
    lines = raw_text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue
        if re.match(r'^[\[\]0-9=+\-*/^_(){}\\/|]+$', line):
            continue
        if line.startswith('=') and line.endswith('='):
            continue
        # Remove citations [1], [edit], etc.
        line = re.sub(r'\[\d+\]', '', line)
        line = re.sub(r'\[edit\].*', '', line)
        # Remove LaTeX
        line = re.sub(r'\$\$?.*?\$\$?', '', line)
        line = re.sub(r'\\\[.*?\\\]', '', line, flags=re.DOTALL)
        line = re.sub(r'\\\(.*?\\\)', '', line)
        # Remove templates
        line = re.sub(r'\{\{[^}]{0,300}\}\}', '', line)
        # Remove HTML
        line = re.sub(r'<[^>]+>', '', line)
        # Collapse spaces
        line = re.sub(r'\s+', ' ', line).strip()

        if len(line) >= 10:
            cleaned_lines.append(line)

    # Rebuild paragraphs
    paragraphs = []
    current = []
    for line in cleaned_lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))

    # Keep only meaningful paragraphs
    good_paragraphs = [
        p for p in paragraphs
        if len(p) >= 50 and len(re.findall(r'[a-zA-Z]', p)) / len(p) > 0.3
    ]

    return "\n\n".join(good_paragraphs)


def split_into_clean_chunks(text: str, notion: str, max_chunk_size: int = 3500) -> List[str]:
    """Smart chunking with prefix"""
    prefix = f"Notion: {notion}\n\n"
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if not para.endswith(('.', '!', '?', '"', "'")):
            para += "."

        candidate = current + ("\n\n" if current else "") + para

        if len(prefix + candidate) <= max_chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(prefix + current)
            current = para

    if current.strip():
        chunks.append(prefix + current)

    return chunks


def fetch_wikipedia_strict(notion: str) -> str:
    """
    Very strict Wikipedia lookup:
    - Tries exact phrase + small variations
    - Only accepts a page if ≥80% of the notion's words appear in the page title
    - Never returns generic pages
    """
    wikipedia.set_lang("en")
    
    # Normalize notion for comparison
    notion_lower = notion.lower()
    notion_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', notion_lower))  # words ≥3 letters
    if len(notion_words) < 2:
        min_match = 1
    else:
        min_match = max(2, int(0.8 * len(notion_words)))  # at least 80%

    # List of allowed variants (only small changes)
    variants = [
        notion,
        notion.split('(')[0].strip(),                    # remove (RIR)
        notion.replace('-', ' '),                        # hyphen → space
        notion.replace('high-finesse', 'high finesse'),  # common physics variant
        re.sub(r'\s*\([^)]*\)', '', notion).strip(),     # remove anything in parentheses
    ]
    variants = list(dict.fromkeys(variants))  # deduplicate & preserve order

    for variant in variants:
        try:
            print(f"   Trying Wikipedia: '{variant}'")
            page = wikipedia.page(variant, auto_suggest=False, redirect=True)
            
            # Check if page title is relevant enough
            title_lower = page.title.lower()
            title_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', title_lower))
            matched_words = title_words & notion_words
            
            if len(matched_words) >= min_match:
                print(f"   MATCH! '{page.title}' ({len(matched_words)}/{len(notion_words)} words)")
                return page.content
            else:
                print(f"   Too generic: '{page.title}' → only {len(matched_words)} word(s) match")
                
        except wikipedia.DisambiguationError as e:
            # Try first 3 disambiguation options that contain most of our words
            for opt in e.options[:3]:
                opt_lower = opt.lower()
                opt_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', opt_lower))
                matched = opt_words & notion_words
                if len(matched) >= min_match:
                    try:
                        page = wikipedia.page(opt)
                        print(f"   MATCH via disambig: '{page.title}'")
                        return page.content
                    except:
                        continue
        except wikipedia.PageError:
            continue
        except Exception as e:
            print(f"   Error: {e}")
            continue

    print(f"   No good Wikipedia page found for: {notion}")
    return ""


# ──────────────────────────────────────────────────────────────
# PDF DOWNLOAD
# ──────────────────────────────────────────────────────────────
def download_pdf_smart(driver, title: str, authors: str, year: str, notion: str) -> bool:
    first_author = authors.split(',')[0].strip() if authors.strip() else ""
    queries = [
        f"{title} {year} PDF",
        f"{title} {first_author} {year} filetype:pdf",
        f"{title} {year}",
        f"{notion} {year} PDF",
        title,
    ]

    for q in queries:
        try:
            print(f"     Search → {q}")
            url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(q)}"
            driver.get(url)
            time.sleep(5)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.gs_r"))
            )

            links = driver.find_elements(By.PARTIAL_LINK_TEXT, "[PDF]")
            links += driver.find_elements(By.CSS_SELECTOR, "div.gs_or_ggsm a")

            for link in links:
                href = link.get_attribute("href")
                if href and (href.lower().endswith(".pdf") or "pdf" in href.lower()):
                    print(f"     DOWNLOADING → {href}")
                    driver.get(href)
                    time.sleep(8)
                    return True
        except Exception as e:
            print(f"     Failed: {e}")
            continue

    print("     No PDF found")
    return False


# ──────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────────
def build_rag_dataset(
    csv_input: str,
    notions_txt: str,
    rag_folder: str,
    pdf_folder: str,
    max_chunk_size: int = 3500,
    max_articles_per_notion: int = 4,
    use_wikipedia: bool = True,
    download_pdfs: bool = True,
    llm_model_name: str = "grok"
):
    rag_dir = Path(rag_folder)
    pdf_dir = Path(pdf_folder)
    rag_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Load notions
    with open(notions_txt, "r", encoding="utf-8") as f:
        notions = [line.strip() for line in f.readlines()[1:] if line.strip()]

    # Load papers
    papers_by_notion: Dict[str, List[Dict]] = {n: [] for n in notions}
    with open(csv_input, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['notion'] in papers_by_notion:
                papers_by_notion[row['notion']].append(row)

    for n in papers_by_notion:
        papers_by_notion[n].sort(key=lambda x: int(x['year'] or "0"), reverse=True)

    # === WIKIPEDIA (STRICT & CLEAN) ===
    if use_wikipedia:
        print("\n=== WIKIPEDIA (STRICT 80% MATCH REQUIRED) ===")
        for notion in tqdm(notions, desc="Wikipedia"):
            raw = fetch_wikipedia_strict(notion)
            if not raw:
                continue
            clean_text = clean_wikipedia_text(raw)
            if not clean_text.strip():
                continue
            chunks = split_into_clean_chunks(clean_text, notion, max_chunk_size)
            base = clean_filename(notion)
            for i, chunk in enumerate(chunks, 1):
                (rag_dir / f"{base}_wiki_chunk{i}.txt").write_text(chunk, encoding="utf-8")
            print(f"   Saved {len(chunks)} clean chunk(s)")

    # === PDF DOWNLOAD ===
    if download_pdfs:
        print("\n=== PDF DOWNLOAD ===")
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        # Remove --headless to see browser (optional)
        # options.add_argument("--headless=new")
        options.add_experimental_option("prefs", {
            "download.default_directory": str(pdf_dir.absolute()),
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True,
        })
        driver = webdriver.Chrome(options=options)
        already_downloaded = set()

        for notion in tqdm(notions, desc="Notions"):
            print(f"\nNOTION: {notion}")
            count = 0
            for paper in papers_by_notion[notion]:
                if count >= max_articles_per_notion:
                    break
                key = f"{paper['title']}_{paper['year']}"
                if key in already_downloaded:
                    continue

                print(f"   {paper['year']} → {paper['title'][:80]}...")
                if download_pdf_smart(driver, paper['title'], paper['authors'], paper['year'], notion):
                    safe_name = clean_filename(f"{paper['authors'].split(',')[0]}_{paper['year']}_{paper['title'][:60]}")
                    (rag_dir / f"{safe_name}.txt").write_text(
                        f"Notion: {notion}\n\n"
                        f"Title: {paper['title']}\n"
                        f"Authors: {paper['authors']}\n"
                        f"Year: {paper['year']}\n"
                        f"Journal: {paper['journal']}\n"
                        f"PDF: Downloaded\n",
                        encoding="utf-8"
                    )
                    already_downloaded.add(key)
                    count += 1
                    time.sleep(4)

        driver.quit()

    print("\nRAG DATASET COMPLETE!")
    print(f"   Texts → {rag_dir}")
    print(f"   PDFs  → {pdf_dir}")
    print(f"   Model → {llm_model_name}")


if __name__ == "__main__":
    build_rag_dataset(
        csv_input="relevant_references_balanced.csv",
        notions_txt="notions.txt",
        rag_folder="./RAG_data",
        pdf_folder="./PDFs",
        max_chunk_size=3500,
        max_articles_per_notion=4,
        use_wikipedia=True,
        download_pdfs=True,
        llm_model_name="llama3.1:70b"
    )