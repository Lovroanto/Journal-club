#!/usr/bin/env python3
"""
AssociateNotionsWithReferences.py

For each notion in notions_clean.txt, finds associated reference numbers
from main article chunks and then maps them to full references.

Outputs:
 - tosearch.txt : each notion followed by the list of full references
 - referencetosearch.txt : all unique references used, deduplicated
"""

import os
import re
from langchain_ollama import OllamaLLM

# ---------------- CONFIG ----------------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
SUMMARIES_DIR = os.path.join(BASE_DIR, "summaries")
MAIN_CHUNKS_DIR = os.path.join(SUMMARIES_DIR, "main_chunks")
NOTIONS_CLEAN_PATH = os.path.join(SUMMARIES_DIR, "notions_clean.txt")
REFERENCES_PATH = os.path.join(BASE_DIR, "main_article/references.txt")
OUTPUT_PATH = os.path.join(SUMMARIES_DIR, "tosearch.txt")
ALL_REFERENCES_PATH = os.path.join(SUMMARIES_DIR, "referencetosearch.txt")

LLM_MODEL = "llama3.1"
llm = OllamaLLM(model=LLM_MODEL)

# ---------------- Utilities ----------------
def read_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def read_main_chunks():
    files = sorted([f for f in os.listdir(MAIN_CHUNKS_DIR) if f.endswith("_summary.txt")])
    chunks = []
    for f in files:
        path = os.path.join(MAIN_CHUNKS_DIR, f)
        text = read_file(path)
        if text.strip():
            chunks.append(text)
    return chunks

# ---------------- LLM reference finder ----------------
def find_references_for_notion(notion, chunks_text):
    prompt = f"""
You are analyzing a scientific article and its summary chunks.

Notion: "{notion}"

Chunks of main text summary:
{chunks_text[:8000]}

Task:
1. Identify any reference numbers cited in the article that are associated with this notion.
2. If the notion is mentioned but no reference is cited, return "No references".
3. Output ONLY the list of reference numbers (like [1], [2,3], etc.), nothing else.
"""
    return llm.invoke(prompt).strip()

# ---------------- Map numbers to full references ----------------
def map_numbers_to_full_references(numbers_list, references_text):
    mapped_refs = []
    for num in numbers_list:
        num_clean = num.strip("[] ")
        if not num_clean.isdigit():
            continue
        # find reference starting with number
        pattern = re.compile(rf"^{num_clean}\s*[\.\)]\s*(.+?)(?=^\d+\s*[\.\)]|\Z)", re.MULTILINE | re.DOTALL)
        match = pattern.search(references_text)
        if match:
            ref_text = match.group(0).strip()
            # replace internal newlines with spaces for single-line reference
            ref_text = re.sub(r"\n", " ", ref_text)
            mapped_refs.append(ref_text)
        else:
            mapped_refs.append(f"Reference [{num_clean}] not found")
    return mapped_refs

# ---------------- Main pipeline ----------------
def main():
    notions_text = read_file(NOTIONS_CLEAN_PATH)
    if not notions_text.strip():
        print("❌ notions_clean.txt is empty.")
        return
    notions = [n.strip("-•* ").strip() for n in notions_text.splitlines() if n.strip()]

    chunks = read_main_chunks()
    if not chunks:
        print("❌ No main chunk summaries found.")
        return
    combined_chunks_text = "\n\n".join(chunks)

    references_text = read_file(REFERENCES_PATH)
    if not references_text.strip():
        print(f"❌ No references found at {REFERENCES_PATH}")
        return

    output_lines = []
    all_unique_refs = set()

    for notion in notions:
        print(f"➡️ Processing notion: {notion}")
        ref_numbers_text = find_references_for_notion(notion, combined_chunks_text)
        numbers = re.findall(r"\d+", ref_numbers_text)
        if numbers:
            full_refs = map_numbers_to_full_references(numbers, references_text)
            all_unique_refs.update(full_refs)
            output_lines.append(f"{notion}:\n")
            for r in full_refs:
                output_lines.append(f"- {r}")
            output_lines.append("\n")
        else:
            output_lines.append(f"{notion}:\n- No references found\n")

    # --- Write per-notion reference file ---
    write_file(OUTPUT_PATH, "\n".join(output_lines))
    print(f"✅ Notions mapped to references saved to {OUTPUT_PATH}")

    # --- Write deduplicated global reference file ---
    write_file(ALL_REFERENCES_PATH, "\n".join(sorted(all_unique_refs)))
    print(f"✅ All unique references saved to {ALL_REFERENCES_PATH}")
    print(f"📊 Total unique references: {len(all_unique_refs)}")

if __name__ == "__main__":
    main()
