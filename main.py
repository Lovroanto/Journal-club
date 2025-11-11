#!/usr/bin/env python3
"""
main_v6.py

Three-step pipeline:
1) Deep chunked summaries -> summary.txt
2) Presentation plan from the summary -> plan.txt
3) Read the plan + original article -> produce notions_and_refs.txt
"""

import os
import re
import json
import time
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Set
from langchain_ollama import OllamaLLM

# ---------------- CONFIG ----------------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
PDF_PATH = os.path.join(BASE_DIR, "main", "s41567-025-02854-4.pdf")

OUT_SUMMARY = os.path.join(BASE_DIR, "summary.txt")
OUT_PLAN = os.path.join(BASE_DIR, "plan.txt")
OUT_NOTIONS = os.path.join(BASE_DIR, "notions_and_refs.txt")

CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Chunking parameters (characters)
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500
MAX_PROMPT_CHARS = 12000

# Ollama model name
LLM_MODEL = "llama3.1"

# ---------------- Utilities ----------------

def extract_pdf_pages_text(pdf_path: str) -> Tuple[List[str], str]:
    """Return a list of page texts and the concatenated full text."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []
    for p in doc:
        pages.append(p.get_text("text") + "\n")
    doc.close()
    full = "".join(pages)
    return pages, full

def find_references_section(full_text: str) -> Tuple[str, str]:
    """Return (body_text, references_text). Try headings and fallback to last 20%."""
    headings = [r"(?mi)\n\s*References\s*\n", r"(?mi)\n\s*REFERENCES\s*\n", r"(?mi)\n\s*Bibliography\s*\n"]
    for pat in headings:
        m = re.search(pat, full_text)
        if m:
            idx = m.end()
            return full_text[:idx], full_text[idx:]
    # fallback last 20% as references
    cutoff = int(len(full_text) * 0.8)
    return full_text[:cutoff], full_text[cutoff:]

def sliding_chunks(text: str, size: int, overlap: int) -> List[str]:
    step = size - overlap
    chunks = []
    if step <= 0:
        raise ValueError("chunk overlap must be smaller than chunk size")
    for start in range(0, max(len(text) - overlap, 1), step):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
    return chunks

def find_bracket_citations(text: str) -> List[int]:
    nums = re.findall(r"\[(\d{1,4})\]", text)
    return sorted({int(n) for n in nums})

def find_author_year_citations(text: str) -> List[str]:
    # Simple heuristic: (Author et al., 2020) or (Smith, 2020)
    matches = re.findall(r"\(([A-Z][A-Za-z\-\']{2,40}(?: et al\.)?(?:, )\d{4})\)", text)
    return sorted(set(matches))

def parse_bracket_refs(refs_text: str) -> Dict[int, str]:
    mapping = {}
    matches = list(re.finditer(r"\[(\d{1,4})\]", refs_text))
    if matches:
        for i, m in enumerate(matches):
            num = int(m.group(1))
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(refs_text)
            entry = refs_text[start:end].strip()
            entry = re.sub(r"^\s*[-\.\)]\s*", "", entry)
            entry = " ".join(entry.split())
            mapping[num] = entry
    return mapping

def parse_numbered_refs(refs_text: str) -> Dict[int, str]:
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
    if current is not None and buffer:
        mapping[current] = " ".join(buffer).strip()
    return mapping

def parse_refs_best_effort(refs_text: str) -> Dict[int, str]:
    mapping = parse_bracket_refs(refs_text)
    if mapping:
        return mapping
    mapping = parse_numbered_refs(refs_text)
    if mapping:
        return mapping
    # fallback: split by double newlines and create sequential numbers
    parts = [p.strip() for p in re.split(r"\n\s*\n", refs_text) if p.strip()]
    mapping = {i+1: " ".join(parts[i].split()) for i in range(len(parts))}
    return mapping

def safe_extract_json(text: str) -> Tuple[dict, str]:
    # Try to find first JSON object in text
    start = text.find("{")
    if start == -1:
        return {}, text
    end = text.rfind("}")
    if end == -1 or end <= start:
        return {}, text
    candidate = text[start:end+1]
    try:
        obj = json.loads(candidate)
        return obj, text[:start] + text[end+1:]
    except Exception:
        return {}, text

# ---------------- LLM setup ----------------
llm = OllamaLLM(model=LLM_MODEL)

# ---------------- Prompts ----------------
def chunk_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    # Request JSON-only output with specific fields
    return f"""
You are a precise scientific assistant preparing materials for a journal club presentation.
Return a JSON object ONLY (no commentary) with keys:
 - "section_tag": one of ["Introduction","Background","Methods","Results","Discussion","Other"]
 - "summary": 3-6 sentences focused on technical content (what was done here, key result or method detail)
 - "technical_points": list of short technical bullet phrases (2-6 words) specific to this chunk
 - "experimental_details": list of precise parameters, equipment, sample sizes, figures, or methods mentioned (if any)
 - "citations": list of citation mentions exactly as they appear in the chunk (e.g., "[12]" or "Smith et al., 2020")

CHUNK_INDEX: {chunk_index} of {total_chunks}
CHUNK_TEXT (first {MAX_PROMPT_CHARS} chars):
\"\"\"{chunk_text[:MAX_PROMPT_CHARS]}\"\"\"
"""

def synthesis_prompt(chunk_summaries: List[dict]) -> str:
    # chunk_summaries will be provided as JSON text
    return f"""
You are a senior journal-club presenter. Using the provided section-level structured summaries (JSON list),
produce a clear output with labeled sections:

1) GLOBAL_SUMMARY: a comprehensive, well-structured summary of the paper (approx 4-8 paragraphs), including
   background, main hypothesis, experimental approach, and main findings. Use technical language appropriate for researchers.

2) PRESENTATION_PLAN: a slide-by-slide plan. Target 8-12 slides total. For each slide provide a title and 2-4 bullet points describing exactly what to put on the slide (text + figures to show). Mark which slides must include background explanations.

3) KEY_NOTIONS: list 4-6 specific domain notions that the audience will need to understand. For each notion give 2-3 sentence definition and label whether it is "critical" or "optional".

4) INTERESTING_CITATIONS: from the chunk-level 'citations' fields, list citation mentions that seem foundational or required for understanding methods (e.g., method origin papers). Output these as a list of citation tokens (e.g. [12] or Smith et al., 2020).

Input (JSON): {json.dumps(chunk_summaries, indent=2)[:MAX_PROMPT_CHARS]}
"""
def plan_to_notions_prompt(global_summary: str, plan_text: str, refs_preview: str) -> str:
    return f"""
You are preparing an actionable retrieval list for a research assistant who will download background materials.

Inputs:
- GLOBAL_SUMMARY (concise; used for context):
{global_summary[:MAX_PROMPT_CHARS]}

- PRESENTATION_PLAN (what to teach, slide-by-slide):
{plan_text[:MAX_PROMPT_CHARS]}

- REFERENCES_SECTION (preview of extracted reference block):
{refs_preview[:MAX_PROMPT_CHARS]}

TASK:
Produce JSON ONLY with keys:
 - "notions": a mapping notion -> short justification (1 line) and suggested search keywords
 - "citations_needed": a list of objects with fields:
      {{"token": "<citation token as found in text e.g. [12] or Smith et al., 2020>", "reason": "<why to fetch>", "match": "<full reference text from the references section if available or null>"}}

The goal is to give Agent 2 an explicit list of what to download and why.
"""
# ---------------- Main pipeline ----------------

def step1_chunk_and_summarize(full_body_text: str) -> List[dict]:
    print("STEP 1 — Chunking and detailed chunk summaries...")
    chunks = sliding_chunks(full_body_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  -> Created {len(chunks)} chunks.")
    all_chunks_info = []
    for i, ch in enumerate(chunks, start=1):
        print(f"  - Analyzing chunk {i}/{len(chunks)}")
        prompt = chunk_prompt(ch, i, len(chunks))
        try:
            raw = llm.invoke(prompt)
        except Exception as e:
            print("LLM error, retrying once...", e)
            time.sleep(1.0)
            raw = llm.invoke(prompt)
        parsed, _ = safe_extract_json_like(raw := raw if isinstance(raw, str) else str(raw))
        # fallback if not parsed
        if not parsed:
            # try to extract bracketed JSON-like segment
            parsed_obj, _ = safe_extract_json(raw)
            parsed = parsed_obj if parsed_obj else {}
        # if still empty, create a safe fallback
        if not parsed:
            # crude fallback using heuristics
            summary = (ch[:400].strip() + "...") if ch else ""
            parsed = {
                "section_tag": "Other",
                "summary": summary,
                "technical_points": [],
                "experimental_details": [],
                "citations": find_bracket_citations(ch) + find_author_year_citations(ch)
            }
        # Normalize citations: ensure list of strings
        cit_list = parsed.get("citations", [])
        normalized_cits = []
        for c in cit_list:
            if isinstance(c, int):
                normalized_cits.append(f"[{c}]")
            else:
                normalized_cits.append(str(c).strip())
        parsed["citations"] = normalized_cits
        all_chunks_info.append(parsed)

        # write chunk debug file
        chunk_file = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}.json")
        with open(chunk_file, "w") as cf:
            json.dump(parsed, cf, indent=2, ensure_ascii=False)
    return all_chunks_info

# helper (slightly different safe json extraction)
def safe_extract_json_like(text: str) -> Tuple[dict, str]:
    # attempt straight json
    try:
        obj = json.loads(text.strip())
        return obj, ""
    except Exception:
        pass
    # find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            obj = json.loads(candidate)
            return obj, text[:start] + text[end+1:]
        except Exception:
            return {}, text
    return {}, text

def step2_synthesize_and_plan(chunk_summaries: List[dict], refs_text: str) -> Tuple[str, str, dict]:
    print("STEP 2 — Global synthesis and presentation plan...")
    # request synthesis
    prompt = synthesis_prompt(chunk_summaries)
    raw = llm.invoke(prompt)
    # crude splitting: we expect labeled sections
    raw_text = raw if isinstance(raw, str) else str(raw)
    # Save full raw to temp
    raw_path = os.path.join(BASE_DIR, "synthesis_raw.txt")
    with open(raw_path, "w") as f:
        f.write(raw_text)
    # Try to extract sections by headings
    def extract_section(label: str, text: str) -> str:
        m = re.search(rf"(?mi){label}\s*:?\s*(.+?)(?:\n[A-Z ]{{3,30}}\s*:|\Z)", text)
        if m:
            return m.group(1).strip()
        return ""
    global_summary = extract_section("GLOBAL_SUMMARY", raw_text) or raw_text[:8000]
    plan_text = extract_section("PRESENTATION_PLAN", raw_text) or ""
    # Key notions and interesting citations may be parsed too
    return global_summary.strip(), plan_text.strip(), {"raw": raw_text}

def step3_produce_notions_and_refs(global_summary: str, plan_text: str, refs_text: str) -> dict:
    print("STEP 3 — Build notions_and_refs (actionable search targets)...")
    prompt = plan_to_notions_prompt(global_summary, plan_text, refs_text[:MAX_PROMPT_CHARS])
    raw = llm.invoke(prompt)
    raw_text = raw if isinstance(raw, str) else str(raw)
    # Try to extract JSON
    parsed, _ = safe_extract_json_like(raw_text)
    if not parsed:
        # try to find JSON-like substring
        parsed_obj, _ = safe_extract_json(raw_text)
        parsed = parsed_obj if parsed_obj else {}
    # Normalize into a structured object
    result = {"notions": {}, "citations_needed": []}
    if isinstance(parsed, dict) and parsed:
        notions = parsed.get("notions", {})
        citations = parsed.get("citations_needed", parsed.get("citations", []))
        # Normalize notions
        if isinstance(notions, dict):
            for k, v in notions.items():
                result["notions"][k] = v
        elif isinstance(notions, list):
            for item in notions:
                if isinstance(item, dict):
                    key = item.get("notion") or item.get("name")
                    if key:
                        result["notions"][key] = item.get("justification", "")
                elif isinstance(item, str):
                    result["notions"][item] = ""
        # Normalize citations
        for c in citations if citations else []:
            if isinstance(c, dict):
                tok = c.get("token")
                reason = c.get("reason", "")
                match = c.get("match")
                result["citations_needed"].append({"token": tok, "reason": reason, "match": match})
            elif isinstance(c, str):
                result["citations_needed"].append({"token": c, "reason": "", "match": None})
    else:
        # fallback: try to heuristically extract notions from plan_text (lines starting with "- ")
        notions = []
        for line in plan_text.splitlines():
            m = re.match(r"^\s*[-•]\s*(.+)$", line)
            if m:
                notions.append(m.group(1).strip())
        for n in notions:
            result["notions"][n] = ""
    return result

def write_outputs(global_summary: str, plan_text: str, notions_and_refs: dict):
    with open(OUT_SUMMARY, "w") as f:
        f.write("=== GLOBAL SUMMARY ===\n\n")
        f.write(global_summary.strip() + "\n")
    with open(OUT_PLAN, "w") as f:
        f.write("=== PRESENTATION PLAN ===\n\n")
        f.write(plan_text.strip() + "\n")
    with open(OUT_NOTIONS, "w") as f:
        json.dump(notions_and_refs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_SUMMARY}, {OUT_PLAN}, {OUT_NOTIONS}")

# ---------------- Runner ----------------

def main():
    print("Starting main_v6 pipeline...")
    pages, full_text = extract_pdf_pages_text(PDF_PATH)
    body_text, refs_text = find_references_section(full_text)
    refs_map = parse_refs_best_effort(refs_text)
    print(f"Extracted document text ({len(body_text)} chars) and references preview ({len(refs_text)} chars).")
    # STEP 1
    chunks_info = step1_chunk_and_summarize(body_text)
    # STEP 2 - synthesis
    global_summary, plan_text, _meta = step2_synthesize_and_plan(chunks_info, refs_text)
    # Save intermediate files
    write_outputs(global_summary, plan_text, {})
    # STEP 3 - produce notions_and_refs (actionable)
    notions_and_refs = step3_produce_notions_and_refs(global_summary, plan_text, refs_text)
    # Try to attach resolved full reference strings from refs_map when token is [n]
    for entry in notions_and_refs.get("citations_needed", []):
        tok = entry.get("token", "")
        m = re.match(r"^\s*\[(\d{1,4})\]\s*$", str(tok))
        if m:
            num = int(m.group(1))
            entry["match"] = refs_map.get(num)
    # Final write
    write_outputs(global_summary, plan_text, notions_and_refs)
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
