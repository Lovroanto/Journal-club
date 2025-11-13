#!/usr/bin/env python3
"""
FetchDefinitionsAndReferences.py

Fetches:
1. Definitions for scientific notions from Wikipedia.
2. Article references, using DOI if possible, or fallback to arXiv.

Outputs:
- definitions/<notion>.txt
- references/<firstauthor>_<year>_<shorttitle>.txt
- references_pdf/<firstauthor>_<year>_<shorttitle>.pdf (if PDF available)
"""

import os
import re
import json
import time
import urllib.parse
import requests
import wikipedia
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path.home() / "ai_data/Journal_Club/First"
SUMMARIES_DIR = BASE_DIR / "summaries"
DEFINITIONS_DIR = SUMMARIES_DIR / "definitions"
REFERENCES_DIR = SUMMARIES_DIR / "references"
REFERENCES_PDF_DIR = SUMMARIES_DIR / "references_pdf"

NOTIONS_FILE = SUMMARIES_DIR / "notions_clean.txt"
REFERENCES_FILE = SUMMARIES_DIR / "referencetosearch.txt"

# Wikipedia language
wikipedia.set_lang("en")

# Create directories if not exist
DEFINITIONS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_PDF_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utility functions ----------------
def clean_filename(text):
    """Sanitize text to be used as a filename."""
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = text.replace(" ", "_")
    return text[:80]  # limit length

# ---------------- Notion definitions ----------------
def fetch_wikipedia_definition(notion):
    try:
        page = wikipedia.page(notion)
        content = page.content
        return content
    except wikipedia.DisambiguationError as e:
        print(f"⚠️ Disambiguation for notion '{notion}': using first option {e.options[0]}")
        try:
            page = wikipedia.page(e.options[0])
            return page.content
        except Exception as e2:
            print(f"❌ Failed fetching page for {e.options[0]}: {e2}")
            return ""
    except wikipedia.PageError:
        print(f"❌ Wikipedia page not found for: {notion}")
        return ""
    except Exception as e:
        print(f"❌ Error fetching Wikipedia page for {notion}: {e}")
        return ""

def process_notions():
    if not NOTIONS_FILE.exists():
        print(f"❌ Notions file not found: {NOTIONS_FILE}")
        return
    with open(NOTIONS_FILE, "r", encoding="utf-8") as f:
        notions = [n.strip() for n in f.readlines() if n.strip()]

    for notion in notions:
        print(f"🔹 Fetching definition for: {notion}")
        content = fetch_wikipedia_definition(notion)
        if content:
            filename = DEFINITIONS_DIR / f"{clean_filename(notion)}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
        time.sleep(1)  # be polite to Wikipedia

# ---------------- Reference handling ----------------
def fetch_reference_arxiv(title):
    """
    Search arXiv for a title and return a link to PDF or abstract.
    """
    query = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and "<entry>" in r.text:
            # parse minimal info
            match_title = re.search(r"<title>(.*?)</title>", r.text, re.DOTALL)
            match_pdf = re.search(r"<link title='pdf' href='(.*?)'/>", r.text)
            match_author = re.findall(r"<name>(.*?)</name>", r.text)
            match_year = re.search(r"<published>(\d{4})", r.text)
            if match_title and match_pdf:
                pdf_url = match_pdf.group(1)
                return {
                    "title": match_title.group(1).strip(),
                    "pdf": pdf_url,
                    "authors": match_author,
                    "year": match_year.group(1) if match_year else "",
                }
    except Exception as e:
        print(f"❌ Error searching arXiv for {title}: {e}")
    return None

def process_references():
    if not REFERENCES_FILE.exists():
        print(f"❌ References file not found: {REFERENCES_FILE}")
        return
    with open(REFERENCES_FILE, "r", encoding="utf-8") as f:
        references = [r.strip() for r in f.readlines() if r.strip()]

    for ref in references:
        print(f"🔹 Searching reference: {ref}")
        # Try to extract DOI
        doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", ref, re.I)
        filename_base = clean_filename(ref)
        txt_file = REFERENCES_DIR / f"{filename_base}.txt"
        if doi_match:
            doi = doi_match.group(1)
            print(f"   → Found DOI: {doi}")
            headers = {"Accept": "application/vnd.citationstyles.csl+json"}
            try:
                r = requests.get(f"https://doi.org/{doi}", headers=headers, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    with open(txt_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    continue
            except Exception as e:
                print(f"❌ Failed fetching DOI info: {e}")

        # fallback to arXiv
        arxiv_info = fetch_reference_arxiv(ref)
        if arxiv_info:
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"Title: {arxiv_info['title']}\n")
                f.write(f"Authors: {', '.join(arxiv_info['authors'])}\n")
                f.write(f"Year: {arxiv_info['year']}\n")
                f.write(f"PDF: {arxiv_info['pdf']}\n")
        else:
            # fallback: just save the original reference text
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(ref)

        time.sleep(1)  # be polite to servers

# ---------------- Main ----------------
def main():
    print("=== Step 1: Fetch notion definitions from Wikipedia ===")
    process_notions()

    print("\n=== Step 2: Fetch references metadata / arXiv PDFs ===")
    process_references()

    print("\n✅ All definitions and references fetched and saved.")

if __name__ == "__main__":
    main()
