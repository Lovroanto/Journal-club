# file: extract_referenced_papers_balanced.py
# → Same excellent matching as before + final balanced assignment

import re
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set


def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9\s]', '', text.lower().replace('-', ' '))


def load_notion_list(notions_path: Path) -> List[Tuple[str, int]]:
    """Load notions, skip first line, return list of (notion, line_number)"""
    notions = []
    with open(notions_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines[1:], start=2):          # line 2 = first real notion
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
        entry = {}
        for line in [l.strip() for l in block.split('\n') if l.strip()]:
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


def extract_referenced_papers_balanced(
    notions_txt: str,
    grobid_tei: str,
    bib_tt: str,
    output_csv: str = "relevant_references_balanced.csv",
    fuzzy_threshold: int = 90
):
    # ------------------------------------------------------------------
    # 1. Load everything (exactly like your original script)
    # ------------------------------------------------------------------
    notions_with_line = load_notion_list(Path(notions_txt))
    normalized_notions = {normalize(n): (n, line) for n, line in notions_with_line}
    bib_db = load_bib_from_tt(Path(bib_tt))

    with open(grobid_tei, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')

    body = soup.find('body')
    if not body:
        raise ValueError("No <body> found in TEI file")

    # ------------------------------------------------------------------
    # 2. Collect *all* possible (paper ↔ notions) links
    # ------------------------------------------------------------------
    paper_to_notion_candidates: Dict[str, Set[Tuple[str, int]]] = defaultdict(set)  # bib_id → set((notion, line))
    # we will decide later which notion actually "gets" the paper

    for p in body.find_all('p'):
        text_content = p.get_text(separator=' ')
        refs_in_p = p.find_all('ref', type='bibr')
        if not refs_in_p:
            continue

        sentences = re.split(r'(?<=[.!?])\s+', text_content)
        pos = 0
        for sentence in sentences:
            if not sentence.strip():
                continue

            # ---- refs that belong to this sentence ----
            sentence_refs: List[str] = []
            for ref in refs_in_p:
                ref_text = ref.get_text(strip=True)
                ref_pos = text_content.find(ref_text, pos)
                if ref_pos != -1 and ref_pos < pos + len(sentence) + 30:   # tolerant window
                    target = ref.get('target', '')
                    if target.startswith('#'):
                        sentence_refs.append(target[1:])   # #b12 → b12

            # ---- which notions appear in this sentence? ----
            norm_sent = normalize(sentence)
            matched_notion_tuples: Set[Tuple[str, int]] = set()

            for norm_n, (orig_n, line) in normalized_notions.items():
                if (fuzz.partial_ratio(norm_n, norm_sent) >= fuzzy_threshold or
                        fuzz.token_set_ratio(norm_n, norm_sent) >= fuzzy_threshold):
                    matched_notion_tuples.add((orig_n, line))

            # ---- link every ref in the sentence to every matched notion ----
            for bib_id in sentence_refs:
                for nt in matched_notion_tuples:
                    paper_to_notion_candidates[bib_id].add(nt)

            pos += len(sentence) + 1

    # ------------------------------------------------------------------
    # 3. Fair assignment: each paper is given to the poorest notion
    # ------------------------------------------------------------------
    notion_counter = Counter()                                   # current number of papers per notion
    final_assignment: Dict[str, Tuple[str, int]] = {}            # bib_id → (chosen_notion, line)

    # Process papers in an order that helps the poorest notions first
    for bib_id, candidates in paper_to_notion_candidates.items():
        if not candidates:
            continue

        # Choose the notion that currently has the fewest papers
        chosen_notion_tuple = min(candidates,
                                  key=lambda nt: (notion_counter[nt[0]], nt[1]))   # tie-break by line number
        chosen_notion_name, line_num = chosen_notion_tuple

        final_assignment[bib_id] = chosen_notion_tuple
        notion_counter[chosen_notion_name] += 1

    # ------------------------------------------------------------------
    # 4. Write the balanced CSV (exactly the same columns as before)
    # ------------------------------------------------------------------
    rows = []
    for bib_id, (notion, line_num) in final_assignment.items():
        entry = bib_db.get(bib_id, {})
        rows.append({
            'notion': notion,
            'line_in_notions_txt': line_num,
            'bib_id': bib_id,
            'title': entry.get('title', 'UNKNOWN'),
            'authors': entry.get('authors', 'UNKNOWN'),
            'journal': entry.get('journal', 'UNKNOWN'),
            'year': entry.get('year', 'UNKNOWN')
        })

    # Nice ordering: first by notion, then by year descending
    rows.sort(key=lambda x: (x['notion'], -int(x['year'] or 0)))

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'notion', 'line_in_notions_txt', 'bib_id',
            'title', 'authors', 'journal', 'year'
        ])
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\nBalanced assignment finished!")
    print(f"   → {len(rows)} papers assigned")
    print(f"   → {len(notion_counter)} different notions received at least one paper\n")
    for notion, count in sorted(notion_counter.items(), key=lambda x: x[1]):
        print(f"   {count:2d} × {notion}")
    print(f"\n   CSV saved as: {output_csv}")


# ==================== RUN ====================
if __name__ == "__main__":
    extract_referenced_papers_balanced(
        notions_txt="notions.txt",
        grobid_tei="paper.tei.xml",
        bib_tt="extracted_notions.tt",
        output_csv="relevant_references_balanced.csv",
        fuzzy_threshold=90
    )