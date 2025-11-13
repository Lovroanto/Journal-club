#!/usr/bin/env python3
"""
postreworking.py (fixed)
Checks for incorrectly split sections and merges them logically.

Now automatically detects the folder where supplementary sections were saved
(e.g., .../main_article/supplementary_sections).
"""

import os, json
from langchain_ollama import OllamaLLM

# ----------- PATH CONFIG -----------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First/main_article")
SOURCE_DIR = os.path.join(BASE_DIR, "supplementary_sections")
MERGED_DIR = os.path.join(BASE_DIR, "supplementary_merged")

os.makedirs(MERGED_DIR, exist_ok=True)

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500
LLM_MODEL = "llama3.1"

llm = OllamaLLM(model=LLM_MODEL)

# ----------- UTILITIES -----------
def read_text(path):
    with open(path, "r") as f:
        return f.read()

def sliding_chunks(text, size, overlap):
    step = size - overlap
    return [text[i:i+size] for i in range(0, max(len(text)-overlap, 1), step)]

def merge_decision(text_a, text_b):
    prompt = f"""
You are a section-continuity analyst.
Compare these two consecutive sections from a scientific paper:

BEGIN_SECTION_A (first 500 chars):
{text_a[:500]}

END_SECTION_A (last 500 chars):
{text_a[-500:]}

BEGIN_SECTION_B (first 500 chars):
{text_b[:500]}

END_SECTION_B (last 500 chars):
{text_b[-500:]}

Are these two excerpts likely to belong to the same logical section
(e.g. same figure, same subsection, or continuing paragraph)?
Answer JSON only: {{"merge_decision": "yes"|"no", "reason": "<short reason>"}}
"""
    reply = llm.invoke(prompt).strip()
    try:
        data = json.loads(reply)
    except Exception:
        data = {"merge_decision": "no", "reason": "parse_error"}
    return data.get("merge_decision", "no"), data.get("reason", "")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ----------- MAIN PROCESS -----------
def process_sections():
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"❌ SOURCE_DIR not found: {SOURCE_DIR}\n"
                                "Make sure reworkingthetext.py ran successfully and created 'supplementary_sections/'.")

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith(".txt")])
    if not files:
        raise RuntimeError(f"No .txt files found in {SOURCE_DIR}")

    print(f"Found {len(files)} section files in {SOURCE_DIR}")

    merged_sections = []
    i = 0
    while i < len(files):
        current = read_text(os.path.join(SOURCE_DIR, files[i]))
        merged_text = current
        j = i + 1
        while j < len(files):
            next_text = read_text(os.path.join(SOURCE_DIR, files[j]))
            decision, reason = merge_decision(merged_text, next_text)
            print(f"Compare {files[i]} + {files[j]} → {decision} ({reason})")
            if decision == "yes":
                merged_text += "\n" + next_text
                j += 1
            else:
                break
        merged_sections.append(merged_text)
        i = j

    # Save merged results
    for idx, text in enumerate(merged_sections, start=1):
        sec_dir = os.path.join(MERGED_DIR, f"section_{idx:03d}")
        ensure_dir(sec_dir)
        full_path = os.path.join(sec_dir, f"section_{idx:03d}_full.txt")
        with open(full_path, "w") as f:
            f.write(text)

        # Chunk if too long
        if len(text) > CHUNK_SIZE:
            chunks = sliding_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
            chunks_dir = os.path.join(sec_dir, "chunks")
            ensure_dir(chunks_dir)
            for c_idx, ch in enumerate(chunks, start=1):
                with open(os.path.join(chunks_dir, f"chunk_{c_idx:03d}.txt"), "w") as f:
                    f.write(ch)

    print(f"✅ Finished merging. {len(merged_sections)} sections saved to {MERGED_DIR}/")

# ----------- RUN -----------
if __name__ == "__main__":
    process_sections()
