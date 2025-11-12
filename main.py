#!/usr/bin/env python3
"""
main_v11_contextual.py

Journal Club Presentation Pipeline
----------------------------------
Adds two-agent summarization:
  1. Context Agent — locates where we are in the article and transitions.
  2. Summary Agent — produces a long, contextual, technical summary.

Pipeline:
  1) Context-aware chunk summaries -> chunks/*.txt
  2) Global synthesis -> summary.txt, plan.txt
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

# ---------------- Context + Summary Agents ----------------
def context_agent_prompt(prev_summary: str, current_context: str, chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    return f"""
You are a context-identification assistant reading a scientific paper in chunks.

PREVIOUS SUMMARY:
\"\"\"{prev_summary[-1500:] if prev_summary else "None"}\"\"\"

CURRENT CONTEXT: {current_context}
CHUNK {chunk_index}/{total_chunks} TEXT (truncated):
\"\"\"{chunk_text[:1500]}\"\"\"

TASK:
1. Infer where we are in the article (choose from: Abstract, Introduction, Background, Methods, Results, Discussion, Conclusion, Supplementary, References, Figure Caption, Other).
2. Write 2–3 sentences describing the *transition* from the previous section to this one.
3. Include a line at the end: NEXT_CONTEXT: <predicted_section>
"""


def summary_agent_prompt(context_text: str, chunk_text: str, current_context: str, chunk_index: int, total_chunks: int) -> str:
    return f"""
You are a scientific summarization expert preparing for a journal club presentation.

CONTEXT INFO:
\"\"\"{context_text}\"\"\"

SECTION_CONTEXT: {current_context}

Now read the following chunk ({chunk_index}/{total_chunks}) and write:
1. A detailed technical summary (150–300 words) explaining main ideas, data, and results.
2. Highlight any important methods, physical concepts, or transitions.
3. If this chunk is the references section, just write "REFERENCE SECTION — SKIPPED".
4. At the end of your answer, add one line:
   END_CONTEXT: <where we are at the end of the chunk>

CHUNK_TEXT (first {MAX_PROMPT_CHARS} chars):
\"\"\"{chunk_text[:MAX_PROMPT_CHARS]}\"\"\"
"""


def step1_chunk_and_summarize(full_body_text: str) -> List[dict]:
    print("STEP 1 — Context-aware 2-pass summarization...")
    chunks = sliding_chunks(full_body_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  -> {len(chunks)} chunks created.")

    current_context = "Unknown"
    previous_summary = ""
    all_chunks_info = []

    for i, ch in enumerate(chunks, start=1):
        print(f"  - Processing chunk {i}/{len(chunks)}")

        # 1️⃣ Context Agent
        context_prompt = context_agent_prompt(previous_summary, current_context, ch, i, len(chunks))
        context_text = llm.invoke(context_prompt).strip()
        context_file = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}_context.txt")
        with open(context_file, "w") as f:
            f.write(context_text)

        # 2️⃣ Summary Agent
        summary_prompt = summary_agent_prompt(context_text, ch, current_context, i, len(chunks))
        summary_text = llm.invoke(summary_prompt).strip()
        summary_file = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}_summary.txt")
        with open(summary_file, "w") as f:
            f.write(summary_text)

        # 3️⃣ Extract next context
        m = re.search(r"END_CONTEXT:\s*(.+)", summary_text)
        if m:
            current_context = m.group(1).strip()
        previous_summary = summary_text

        # 4️⃣ Collect info
        all_chunks_info.append({
            "chunk_index": i,
            "context": context_text,
            "summary": summary_text,
            "current_context": current_context
        })

    return all_chunks_info


# ---------------- Step 2: Global synthesis + plan ----------------
def synthesis_prompt(chunk_summaries: List[dict]) -> str:
    return f"""
You are a senior presenter. Using provided structured chunk summaries, produce JSON with sections:
1) GLOBAL_SUMMARY: 4–8 paragraphs technical summary.
2) PRESENTATION_PLAN: slide-by-slide plan (8–12 slides) with titles + bullet points.
3) KEY_NOTIONS: 4–6 domain notions with 2–3 sentence definitions, marked "critical"/"optional".
4) INTERESTING_CITATIONS: foundational citations from chunks.

Input summaries: {json.dumps(chunk_summaries)[:MAX_PROMPT_CHARS]}
"""

def step2_synthesize_and_plan(chunk_summaries: List[dict]) -> Tuple[str, str]:
    print("STEP 2 — Global synthesis and plan...")
    prompt = synthesis_prompt(chunk_summaries)
    raw = llm.invoke(prompt)
    parsed, _ = safe_extract_json_like(raw)

    global_summary = parsed.get("GLOBAL_SUMMARY", "")
    plan_text = parsed.get("PRESENTATION_PLAN", "")

    if isinstance(global_summary, list):
        global_summary = format_list_of_dicts(global_summary)
    if isinstance(plan_text, list):
        plan_text = format_list_of_dicts(plan_text)

    return global_summary.strip(), plan_text.strip()


# ---------------- Step 3: Notions + references ----------------
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

def step3_produce_notions_and_refs(global_summary: str, plan_text: str, refs_text: str) -> dict:
    print("STEP 3 — Building notions_and_refs...")
    prompt = plan_to_notions_prompt(global_summary, plan_text, refs_text)
    raw = llm.invoke(prompt)
    parsed, _ = safe_extract_json_like(raw)
    if not parsed:
        parsed = {}
    return parsed


# ---------------- Writers ----------------
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


# ---------------- Main Runner ----------------
def main():
    print("Starting main_v11_contextual pipeline...")
    pages, full_text = extract_pdf_pages_text(PDF_PATH)
    body_text, refs_text = find_references_section(full_text)
    refs_map = parse_refs_best_effort(refs_text)
    print(f"Document text: {len(body_text)} chars, References: {len(refs_text)} chars")

    chunks_info = step1_chunk_and_summarize(body_text)
    global_summary, plan_text = step2_synthesize_and_plan(chunks_info)
    write_outputs_step2(global_summary, plan_text)
    notions_and_refs = step3_produce_notions_and_refs(global_summary, plan_text, refs_text)
    write_outputs_step3(notions_and_refs, refs_map)

    print("Pipeline complete. Summary, plan, and notions ready.")


if __name__ == "__main__":
    main()
