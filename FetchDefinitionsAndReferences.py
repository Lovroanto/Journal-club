#!/usr/bin/env python3
"""
FetchDefinitionsAndReferences.py

Fetches:
1. Definitions for scientific notions from Wikipedia.
2. Article references:
   - Uses DOI if possible.
   - Attempts to download PDFs via Firefox/Selenium.
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
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
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
    if not NOTIONS_FILE.exists():
        print(f"No notions file: {NOTIONS_FILE}")
        return
    with open(NOTIONS_FILE, "r", encoding="utf-8") as f:
        notions = [n.strip() for n in f if n.strip()]
    for notion in notions:
        print(f"Fetching definition: {notion}")
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

# ---------------- Selenium Firefox PDF downloader ----------------
def setup_firefox(download_dir):
    options = Options()
    options.headless = True
    profile = webdriver.FirefoxProfile()
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.dir", str(download_dir))
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
    driver = webdriver.Firefox(options=options, firefox_profile=profile)
    return driver

def download_pdf_firefox(driver, search_query, filename_path):
    try:
        query = urllib.parse.quote(search_query)
        driver.get(f"https://scholar.google.com/scholar?q={query}")
        time.sleep(3)
        # try to find [PDF] link
        pdf_links = driver.find_elements(By.XPATH, "//a[contains(text(), '[PDF]')]")
        if pdf_links:
            pdf_url = pdf_links[0].get_attribute("href")
            driver.get(pdf_url)
            time.sleep(5)  # wait download
            print(f"Downloaded PDF via Firefox: {filename_path}")
            return True
    except Exception as e:
        print(f"Firefox PDF download failed: {e}")
    return False

# ---------------- Process references ----------------
def process_references():
    if not REFERENCES_FILE.exists():
        print(f"No references file: {REFERENCES_FILE}")
        return
    with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
        references = [r.strip() for r in f if r.strip()]

    driver = setup_firefox(REFERENCES_ARTICLES_DIR)

    for ref in references:
        print(f"Searching reference: {ref}")
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

        # Try Firefox download
        success = download_pdf_firefox(driver, ref, pdf_file)
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

    print("\n=== Step 2: Fetch references and PDFs via Firefox / arXiv ===")
    process_references()

    print("\n✅ All definitions and references fetched.")

if __name__ == "__main__":
    main()
