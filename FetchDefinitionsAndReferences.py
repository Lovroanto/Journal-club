#!/usr/bin/env python3
"""
FetchDefinitionsAndReferences.py

Fetches:
1. Definitions for scientific notions from Wikipedia.
2. Article references:
   - Uses DOI if possible.
   - Attempts to download PDFs via Chrome/Selenium.
   - Falls back to arXiv if not freely available.

Outputs:
- definitions/<notion>.txt
- references/<firstauthor>_<year>_<shorttitle>.txt
- references/articles/<firstauthor>_<year>_<shorttitle>.pdf (if PDF available)
"""

import os
import re
import json
import time
import urllib.parse
import requests
import wikipedia
from pathlib import Path

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ---------------- CONFIG ----------------
BASE_DIR = Path.home() / "ai_data/Journal_Club/First"
SUMMARIES_DIR = BASE_DIR / "summaries"
DEFINITIONS_DIR = SUMMARIES_DIR / "definitions"
REFERENCES_DIR = SUMMARIES_DIR / "references"
REFERENCES_ARTICLES_DIR = SUMMARIES_DIR / "references/articles"

NOTIONS_FILE = SUMMARIES_DIR / "notions_clean.txt"
REFERENCES_FILE = SUMMARIES_DIR / "referencetosearch.txt"

# Wikipedia language
wikipedia.set_lang("en")

# Create directories
DEFINITIONS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)

# Debug flags
SKIP_WIKIPEDIA = True  # set to False to fetch Wikipedia definitions


# ---------------- Utility ----------------
def clean_filename(text):
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = text.replace(" ", "_")
    return text[:80]


# ---------------- Wikipedia ----------------
def fetch_wikipedia_definition(notion):
    try:
        page = wikipedia.page(notion)
        return page.content
    except wikipedia.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0])
            return page.content
        except:
            return ""
    except wikipedia.PageError:
        return ""
    except Exception as e:
        print(f"Error fetching Wikipedia page: {e}")
        return ""


def process_notions():
    if SKIP_WIKIPEDIA:
        print("✅ Skipping Wikipedia definitions (debug mode).")
        return

    if not NOTIONS_FILE.exists():
        print(f"No notions file: {NOTIONS_FILE}")
        return

    with open(NOTIONS_FILE, "r", encoding="utf-8") as f:
        notions = [n.strip() for n in f if n.strip()]

    for notion in notions:
        print(f"🔹 Fetching definition: {notion}")
        content = fetch_wikipedia_definition(notion)
        if content:
            filename = DEFINITIONS_DIR / f"{clean_filename(notion)}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
        time.sleep(1)


# ---------------- ArXiv ----------------
def fetch_reference_arxiv(title):
    query = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and "<entry>" in r.text:
            match_title = re.search(r"<title>(.*?)</title>", r.text, re.DOTALL)
            match_pdf = re.search(r"<link title='pdf' href='(.*?)'/>", r.text)
            match_author = re.findall(r"<name>(.*?)</name>", r.text)
            match_year = re.search(r"<published>(\d{4})", r.text)
            if match_title and match_pdf:
                return {
                    "title": match_title.group(1).strip(),
                    "pdf": match_pdf.group(1),
                    "authors": match_author,
                    "year": match_year.group(1) if match_year else "",
                }
    except:
        return None
    return None


# ---------------- Chrome Selenium PDF downloader ----------------
def setup_chrome(download_dir):
    print("🔧 Setting up Chrome driver...")

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # ❗ Make Chrome visible so you can see what it is doing
    # chrome_options.add_argument("--headless=new")   # DISABLED for debugging

    chrome_options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True
        }
    )

    driver = webdriver.Chrome(options=chrome_options)
    return driver


def download_pdf_chrome(driver, search_query, filename_path):
    """
    Opens Google Scholar, finds the first PDF link, downloads it.
    Prints debugging information.
    """

    try:
        query = urllib.parse.quote(search_query)
        url = f"https://scholar.google.com/scholar?q={query}"

        print(f"   🔍 Opening Scholar: {url}")
        driver.get(url)
        time.sleep(4)

        print("   🔍 Looking for PDF links…")

        # Modern Google Scholar PDF links are inside: <div class="gs_or_ggsm"><a href="...">
        pdf_wrappers = driver.find_elements(By.CSS_SELECTOR, "div.gs_or_ggsm a")

        print(f"   🔍 Found {len(pdf_wrappers)} potential PDF links")

        for link in pdf_wrappers:
            href = link.get_attribute("href")
            print(f"      → {href}")

        if pdf_wrappers:
            pdf_url = pdf_wrappers[0].get_attribute("href")
            print(f"   📥 Downloading PDF: {pdf_url}")
            driver.get(pdf_url)
            time.sleep(6)
            return True

        print("   ❌ No PDF links found on the page.")
        return False

    except Exception as e:
        print(f"   ❌ Chrome PDF download failed: {e}")
        return False


# ---------------- Process references ----------------
def process_references():
    if not REFERENCES_FILE.exists():
        print(f"No references file: {REFERENCES_FILE}")
        return

    with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
        references = [r.strip() for r in f if r.strip()]

    driver = setup_chrome(REFERENCES_ARTICLES_DIR)

    for ref in references:
        print(f"🔹 Searching reference: {ref}")
        filename_base = clean_filename(ref)
        txt_file = REFERENCES_DIR / f"{filename_base}.txt"
        pdf_file = REFERENCES_ARTICLES_DIR / f"{filename_base}.pdf"

        # Try DOI metadata first
        doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", ref, re.I)
        if doi_match:
            doi = doi_match.group(1)
            headers = {"Accept": "application/vnd.citationstyles.csl+json"}
            try:
                r = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    with open(txt_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
            except:
                pass

        # Try Chrome PDF download
        success = download_pdf_chrome(driver, ref, pdf_file)

        if not success:
            # fallback arXiv
            arxiv_info = fetch_reference_arxiv(ref)
            if arxiv_info:
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(f"Title: {arxiv_info['title']}\n")
                    f.write(f"Authors: {', '.join(arxiv_info['authors'])}\n")
                    f.write(f"Year: {arxiv_info['year']}\n")
                    f.write(f"PDF: {arxiv_info['pdf']}\n")
            else:
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(ref)

        time.sleep(2)

    driver.quit()


# ---------------- Main ----------------
def main():
    print("=== Step 1: Fetch notion definitions from Wikipedia ===")
    process_notions()

    print("\n=== Step 2: Fetch references and PDFs via Chrome / arXiv ===")
    process_references()

    print("\n✅ All definitions and references fetched.")


if __name__ == "__main__":
    main()
