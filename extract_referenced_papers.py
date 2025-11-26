# file: extract_referenced_papers.py

import re
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from typing import List, Set, Dict, Tuple

def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', text.lower().replace('-', ' '))

def load_notion_list(notions_path: Path) -> List[Tuple[str, int]]:
    """Load notions, skip first line, return list of (notion, line_number)"""
    notions = []
    with open(notions_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines[1:], start=2):  # start from line 2
        notion = line.strip()
        if notion:
            notions.append((notion, i))
    return notions

def load_bib_from_tt(tt_path: Path) -> Dict[str, Dict]:
    """Parse your custom .tt bibliography file"""
    bib_dict = {}
    current_id = None
    with open(tt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = re.split(r'-{10,}', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        entry = {}
        for line in lines:
            if line.startswith('ID: '):
                current_id = line[4:].strip()
            elif line.startswith('Title: '):
                entry['title'] = line[7:].strip()
            elif line.startswith('Authors: '):
                entry['authors'] = line[9:].strip()
            elif line.startswith('Journal: '):
                entry['journal'] = line[9:].strip()
            elif line.startswith('Year: '):
                entry['year'] = line[6:].strip()
        if current_id and 'title' in entry:
            bib_dict[current_id] = entry
    return bib_dict

def extract_referenced_papers(
    notions_txt: str,
    grobid_tei: str,
    bib_tt: str,
    output_csv: str = "references_for_notion.csv",
    fuzzy_threshold: int = 90
):
    notions_path = Path(notions_txt)
    tei_path = Path(grobid_tei)
    tt_path = Path(bib_tt)

    # Load notions with line numbers
    notions_with_line = load_notion_list(notions_path)
    normalized_notions = {normalize(notion): (notion, line) for notion, line in notions_with_line}

    # Load bibliography
    bib_db = load_bib_from_tt(tt_path)

    # Parse TEI
    with open(tei_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')

    body = soup.find('body')
    if not body:
        raise ValueError("No <body> found in TEI file")

    # Will collect results
    results = []
    seen = set()  # to avoid exact duplicates in same sentence (optional)

    # Process all paragraphs
    for p in body.find_all('p'):
        text_content = p.get_text(separator=' ')
        refs_in_p = p.find_all('ref', type='bibr')

        if not refs_in_p:
            continue

        # Split paragraph into sentences (simple but robust)
        sentences = re.split(r'(?<=[.!?])\s+', text_content)
        sentence_start = 0
        for sentence in sentences:
            if not sentence.strip():
                continue

            # Find which refs belong to this sentence
            sentence_refs: List[str] = []
            sentence_end = sentence_start + len(sentence)

            for ref in refs_in_p:
                ref_text = ref.get_text(strip=True)
                ref_pos = text_content.find(ref_text, sentence_start)
                if ref_pos != -1 and ref_pos < sentence_start + len(sentence) + 20:  # tolerance
                    target = ref.get('target', '')
                    if target.startswith('#'):
                        bib_id = target[1:]  # #b15 → b15
                        sentence_refs.append(bib_id)

            # Check if sentence contains any notion
            norm_sentence = normalize(sentence)
            matched_notion = None
            matched_original = None
            matched_line = None

            for norm_notion, (orig_notion, line) in normalized_notions.items():
                if fuzz.partial_ratio(norm_notion, norm_sentence) >= fuzzy_threshold:
                    matched_notion = norm_notion
                    matched_original = orig_notion
                    matched_line = line
                    break
                # Also try token set ratio for better multi-word matching
                if fuzz.token_set_ratio(norm_notion, norm_sentence) >= fuzzy_threshold:
                    matched_notion = norm_notion
                    matched_original = orig_notion
                    matched_line = line
                    break

            if matched_original and sentence_refs:
                for bib_id in sentence_refs:
                    key = (matched_original, bib_id)
                    if key in seen:
                        continue
                    seen.add(key)

                    entry = bib_db.get(bib_id, {})
                    results.append({
                        'notion': matched_original,
                        'line_in_notions_txt': matched_line,
                        'bib_id': bib_id,
                        'title': entry.get('title', 'UNKNOWN'),
                        'authors': entry.get('authors', 'UNKNOWN'),
                        'journal': entry.get('journal', 'UNKNOWN'),
                        'year': entry.get('year', 'UNKNOWN')
                    })

            # Update cursor
            sentence_start += len(sentence) + 1

    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'notion', 'line_in_notions_txt', 'bib_id',
            'title', 'authors', 'journal', 'year'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Found {len(results)} reference-notion links → {output_csv}")

# ==================== USAGE ====================
if __name__ == "__main__":
    extract_referenced_papers(
        notions_txt="notions.txt",
        grobid_tei="paper.tei.xml",
        bib_tt="extracted_notions.tt",
        output_csv="relevant_references.csv",
        fuzzy_threshold=90
    )