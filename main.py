#!/usr/bin/env python3
"""
main_v10.py

Journal Club Presentation Pipeline - Full robust fix with nested dict/list handling

1) Chunked summaries -> summary.txt
2) Global synthesis -> plan.txt (handles list/dict and nested structures)
3) Actionable notions + references -> notions_and_refs.txt
"""

import os
import re
import json
import fitz  # PyMuPDF
from typing import List, Tuple
from langchain_ollama import OllamaLLM

# ---------------- CONFIG ----------------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
PDF_PATH = os.path.join(BASE_DIR, "main", "s41567-025-02854-4.pdf")

OUT_SUMMARY = os.path.join(BASE_DIR, "summary.txt")
OUT_PLAN = os.path.join(BASE_DIR, "plan.txt")
OUT_NOTIONS = os.path.join(BASE_DIR, "notions_and_refs.txt")

CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500
MAX_PROMPT_CHARS = 12000
LLM_MODEL = "llama3.1"

# ---------------- Utilities ----------------

def extract_pdf_pages_text(pdf_path: str) -> Tuple[List[str], str]:
    doc = fitz.open(pdf_path)
    pages = [p.get_text("text") + "\n" for p in doc]
    full = "".join(pages)
    doc.close()
    return pages, full

def find_references_section(full_text: str) -> Tuple[str, str]:
    headings = [r"(?mi)\n\s*References\s*\n", r"(?mi)\n\s*REFERENCES\s*\n", r"(?mi)\n\s*Bibliography\s*\n"]
    for pat in headings:
        m = re.search(pat, full_text)
        if m:
            idx = m.end()
            return full_text[:idx], full_text[idx:]
    cutoff = int(len(full_text) * 0.8)
    return full_text[:cutoff], full_text[cutoff:]

def sliding_chunks(text: str, size: int, overlap: int) -> List[str]:
    step = size - overlap
    return [text[i:i+size] for i in range(0, max(len(text)-overlap,1), step)]

def find_bracket_citations(text: str) -> List[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d{1,4})\]", text)})

def parse_refs_best_effort(refs_text: str) -> dict:
    mapping = {}
    lines = refs_text.splitlines()
    pattern = re.compile(r"^\s*(\d{1,4})[\.\)]\s+(.*)")
    current = None
    buffer = []
    for line in lines:
        m = pattern.match(line)
        if m:
            if current is not None:
                mapping[current] = " ".join(buffer).strip()
            current = int(m.group(1))
            buffer = [m.group(2).strip()]
        else:
            if current is not None:
                buffer.append(line.strip())
    if current and buffer:
        mapping[current] = " ".join(buffer).strip()
    return mapping

def safe_extract_json_like(text: str) -> Tuple[dict, str]:
    try:
        return json.loads(text.strip()), ""
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate), text[:start]+text[end+1:]
            except Exception:
                return {}, text
        return {}, text

def format_list_of_dicts(lst):
    """
    Recursively convert a list of dicts or nested lists into readable text.
    """
    lines = []
    for item in lst:
        if isinstance(item, dict):
            for k,v in item.items():
                if isinstance(v, list):
                    v_text = format_list_of_dicts(v)
                else:
                    v_text = str(v)
                lines.append(f"{k}: {v_text}")
        elif isinstance(item, list):
            lines.append(format_list_of_dicts(item))
        else:
            lines.append(str(item))
    return "\n".join(lines)

# ---------------- LLM setup ----------------
llm = OllamaLLM(model=LLM_MODEL)

# ---------------- Prompts ----------------
def chunk_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    return f"""
You are a precise scientific assistant preparing a journal club presentation.
Return JSON ONLY with keys:
 - "section_tag": one of ["Introduction","Background","Methods","Results","Discussion","Other"]
 - "summary": 3-6 sentences technical content
 - "technical_points": list of short technical bullets (2-6 words)
 - "experimental_details": list of parameters, equipment, sample sizes, figures, or methods
 - "citations": list of citation mentions exactly as they appear ("[12]", "Smith et al., 2020")

CHUNK_INDEX: {chunk_index}/{total_chunks}
CHUNK_TEXT (first {MAX_PROMPT_CHARS} chars):
\"\"\"{chunk_text[:MAX_PROMPT_CHARS]}\"\"\"
"""

def synthesis_prompt(chunk_summaries: List[dict]) -> str:
    return f"""
You are a senior presenter. Using provided structured chunk summaries, produce JSON with sections:
1) GLOBAL_SUMMARY: 4-8 paragraphs technical summary.
2) PRESENTATION_PLAN: slide-by-slide plan (8-12 slides) with titles + bullet points.
3) KEY_NOTIONS: 4-6 domain notions with 2-3 sentence definitions, marked "critical"/"optional".
4) INTERESTING_CITATIONS: foundational citations from chunks.

Input JSON: {json.dumps(chunk_summaries)[:MAX_PROMPT_CHARS]}
"""

def plan_to_notions_prompt(global_summary: str, plan_text: str, refs_preview: str) -> str:
    return f"""
You are preparing a retrieval list for a research assistant.

Inputs:
- GLOBAL_SUMMARY: {global_summary[:MAX_PROMPT_CHARS]}
- PRESENTATION_PLAN: {plan_text[:MAX_PROMPT_CHARS]}
- REFERENCES_SECTION: {refs_preview[:MAX_PROMPT_CHARS]}

TASK:
Produce JSON ONLY:
 - "notions": mapping notion -> short justification + search keywords
 - "citations_needed": list of objects with "token", "reason", "match" (full reference if available)
"""

# ---------------- Step 1 ----------------
def step1_chunk_and_summarize(full_body_text: str) -> List[dict]:
    print("STEP 1 — Chunking and summarizing...")
    chunks = sliding_chunks(full_body_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  -> {len(chunks)} chunks created.")
    all_chunks_info = []
    for i, ch in enumerate(chunks, start=1):
        prompt = chunk_prompt(ch, i, len(chunks))
        raw = llm.invoke(prompt)
        parsed, _ = safe_extract_json_like(raw)
        if not parsed:
            parsed = {"section_tag":"Other","summary":ch[:400]+"...","technical_points":[],"experimental_details":[],"citations":find_bracket_citations(ch)}
        all_chunks_info.append(parsed)
        with open(os.path.join(CHUNKS_DIR,f"chunk_{i:03d}.json"),"w") as f:
            json.dump(parsed,f,indent=2,ensure_ascii=False)
    return all_chunks_info

# ---------------- Step 2 ----------------
def step2_synthesize_and_plan(chunk_summaries: List[dict], refs_text: str) -> Tuple[str,str]:
    print("STEP 2 — Global synthesis and plan...")
    prompt = synthesis_prompt(chunk_summaries)
    raw = llm.invoke(prompt)
    parsed, _ = safe_extract_json_like(raw)

    global_summary = parsed.get("GLOBAL_SUMMARY", "")
    plan_text = parsed.get("PRESENTATION_PLAN", "")

    # Convert both to readable string
    if isinstance(global_summary, list):
        global_summary = format_list_of_dicts(global_summary)
    if isinstance(plan_text, list):
        plan_text = format_list_of_dicts(plan_text)

    return global_summary.strip(), plan_text.strip()

# ---------------- Step 3 ----------------
def step3_produce_notions_and_refs(global_summary: str, plan_text: str, refs_text: str) -> dict:
    print("STEP 3 — Building notions_and_refs...")
    prompt = plan_to_notions_prompt(global_summary, plan_text, refs_text[:MAX_PROMPT_CHARS])
    raw = llm.invoke(prompt)
    parsed, _ = safe_extract_json_like(raw)
    if not parsed:
        parsed = {}
    return parsed

# ---------------- Write helpers ----------------
def write_outputs_step2(global_summary: str, plan_text: str):
    with open(OUT_SUMMARY,"w") as f: f.write("=== GLOBAL SUMMARY ===\n\n"+global_summary)
    with open(OUT_PLAN,"w") as f: f.write("=== PRESENTATION PLAN ===\n\n"+plan_text)
    print(f"Saved Step 2: {OUT_SUMMARY}, {OUT_PLAN}")

def write_outputs_step3(notions_and_refs: dict, refs_map: dict):
    for entry in notions_and_refs.get("citations_needed", []):
        tok = entry.get("token","")
        m = re.match(r"^\s*\[(\d{1,4})\]\s*$", str(tok))
        if m:
            num = int(m.group(1))
            entry["match"] = refs_map.get(num)
    with open(OUT_NOTIONS,"w") as f: json.dump(notions_and_refs,f,indent=2,ensure_ascii=False)
    print(f"Saved Step 3: {OUT_NOTIONS}")

# ---------------- Runner ----------------
def main():
    print("Starting main_v10 pipeline...")
    pages, full_text = extract_pdf_pages_text(PDF_PATH)
    body_text, refs_text = find_references_section(full_text)
    refs_map = parse_refs_best_effort(refs_text)
    print(f"Document text: {len(body_text)} chars, References: {len(refs_text)} chars")

    chunks_info = step1_chunk_and_summarize(body_text)
    global_summary, plan_text = step2_synthesize_and_plan(chunks_info, refs_text)
    write_outputs_step2(global_summary, plan_text)
    notions_and_refs = step3_produce_notions_and_refs(global_summary, plan_text, refs_text)
    write_outputs_step3(notions_and_refs, refs_map)

    print("Pipeline complete. Summary, plan, and notions ready.")

if __name__ == "__main__":
    main()
