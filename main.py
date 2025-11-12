#!/usr/bin/env python3
"""
main_v11_contextual_updated.py

Journal Club Presentation Pipeline — Context-aware 2-pass summarization, with:
 - chunk_XXX_raw.txt      : original chunk text (for comparison)
 - chunk_XXX_context.txt  : short context + detected figures + NEXT_CONTEXT
 - chunk_XXX_summary.txt  : detailed summary (150-300 words) + END_CONTEXT

Other steps unchanged from main_v10 structure.
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

def detect_figure_tokens(text: str) -> List[str]:
    # Basic regexes to find "Fig.", "Figure", "Fig", "Supplementary Fig."
    figs = set()
    for m in re.finditer(r"\b(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?)\b", text, flags=re.IGNORECASE):
        figs.add(m.group(1))
    for m in re.finditer(r"\b(Supp(?:lementary)?\.?\s*Fig(?:ure)?\.?\s*\d+)\b", text, flags=re.IGNORECASE):
        figs.add(m.group(1))
    # Normalize to Title case like "Figure 1a"
    normalized = []
    for f in figs:
        nf = re.sub(r"fig(?:ure)?\.?", "Figure", f, flags=re.IGNORECASE)
        nf = nf.replace("Supp.", "Supplementary").replace("Supp", "Supplementary")
        normalized.append(nf.strip())
    return sorted(set(normalized))

# ---------------- LLM setup ----------------
llm = OllamaLLM(model=LLM_MODEL)

# ---------------- Context + Summary Agents (two-pass) ----------------
def context_agent_prompt(prev_summary: str, current_context: str, chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    # Instruct context agent to also list figures detected
    return f"""
You are a context-identification assistant reading a scientific paper in chunks.

PREVIOUS SUMMARY (last 1500 chars):
\"\"\"{prev_summary[-1500:] if prev_summary else "None"}\"\"\"

CURRENT CONTEXT: {current_context}
CHUNK {chunk_index}/{total_chunks} TEXT (first 1500 chars):
\"\"\"{chunk_text[:1500]}\"\"\"

TASK:
1) Choose which section this chunk belongs to (one of: Abstract, Introduction, Background, Methods, Results, Discussion, Conclusion, Supplementary, References, Figure Caption, Other).
2) Write 2–3 sentences describing the transition from the previous section to this one.
3) Detect which figure(s), if any, are being discussed in this chunk (e.g. "Figure 1", "Figure 1a", "Supplementary Figure 3"). Output them in a line starting with FIGURES: and a comma-separated list (or "FIGURES: None").
4) End with a line: NEXT_CONTEXT: <predicted_section>

RETURN ONLY the transition paragraph, the FIGURES line and the NEXT_CONTEXT line.
Example ending:
... 
FIGURES: Figure 1, Figure 2a
NEXT_CONTEXT: Results
"""

def summary_agent_prompt(context_text: str, chunk_text: str, section_context: str, chunk_index: int, total_chunks: int) -> str:
    return f"""
You are a scientific summarization expert preparing for a journal club presentation.

CONTEXT INFO (short):
\"\"\"{context_text}\"\"\"

SECTION_CONTEXT: {section_context}

Now read the following chunk ({chunk_index}/{total_chunks}) and write:
1) A detailed technical summary (150–300 words) explaining main ideas, key results, and methods in this chunk.
2) If the chunk comments on any figure(s), explicitly state which figure(s) are discussed and summarize what about the figure is important (e.g., "This text comments on Figure 1: shows X, Y; main result is Z").
3) If this chunk is the References section, write exactly: REFERENCE SECTION — SKIPPED
4) Conclude the summary with a line: END_CONTEXT: <where we are at the end of the chunk>

RETURN the summary only (including the END_CONTEXT line).
CHUNK_TEXT (first {MAX_PROMPT_CHARS} chars):
\"\"\"{chunk_text[:MAX_PROMPT_CHARS]}\"\"\"
"""

def step1_chunk_and_summarize(full_body_text: str) -> List[dict]:
    print("STEP 1 — Context-aware 2-pass summarization (with raw chunk dump)...")
    chunks = sliding_chunks(full_body_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  -> {len(chunks)} chunks created.")

    current_context = "Unknown"
    previous_summary = ""
    all_chunks_info = []

    for i, ch in enumerate(chunks, start=1):
        print(f"  - Processing chunk {i}/{len(chunks)}")

        # Save raw chunk for user review
        raw_file = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}_raw.txt")
        with open(raw_file, "w") as f:
            f.write(ch)

        # 1) Context agent: determines section, transition, figures, next_context
        ctx_prompt = context_agent_prompt(previous_summary, current_context, ch, i, len(chunks))
        context_text = llm.invoke(ctx_prompt).strip()

        # try to extract FIGURES and NEXT_CONTEXT from the context_text (fail safely)
        figs_found = detect_figure_tokens(ch)
        # If the agent included FIGURES: line, prefer that; else use regex-detected
        m_fig = re.search(r"FIGURES:\s*(.+)", context_text, flags=re.IGNORECASE)
        if m_fig:
            figs_line = m_fig.group(1).strip()
            figs_list = [x.strip() for x in re.split(r",\s*", figs_line) if x.strip().lower() != "none"]
        else:
            figs_list = figs_found

        m_next = re.search(r"NEXT_CONTEXT:\s*(.+)", context_text)
        if m_next:
            predicted_next = m_next.group(1).strip()
        else:
            predicted_next = current_context  # fallback

        # Save context file (explicit, standardized header)
        context_file = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}_context.txt")
        with open(context_file, "w") as f:
            f.write(f"=== CHUNK {i}/{len(chunks)} CONTEXT ===\n")
            f.write(f"Predicted next context: {predicted_next}\n")
            f.write(f"Detected figures (agent or regex): {', '.join(figs_list) if figs_list else 'None'}\n\n")
            f.write(context_text + "\n")

        # 2) Summary agent: produce long contextual summary; must mention figures if relevant
        # Provide the context_text (first 1200 chars) and the chunk text
        short_context = context_text[:1200]
        sum_prompt = summary_agent_prompt(short_context, ch, predicted_next, i, len(chunks))
        summary_text = llm.invoke(sum_prompt).strip()

        # If the summary agent didn't explicitly mention figures but figs_list exists, add a short note
        if figs_list and not re.search(r"Figure|Fig\.|Figure\s*\d", summary_text, flags=re.IGNORECASE):
            fig_note = "\n\n[NOTE: This chunk appears to discuss the following figure(s): " + ", ".join(figs_list) + "]"
            summary_text = summary_text + fig_note

        # Save summary file
        summary_file = os.path.join(CHUNKS_DIR, f"chunk_{i:03d}_summary.txt")
        with open(summary_file, "w") as f:
            f.write(summary_text + "\n")

        # 3) Update context for next chunk
        m_end = re.search(r"END_CONTEXT:\s*(.+)", summary_text)
        if m_end:
            current_context = m_end.group(1).strip()
        else:
            # If summary didn't set END_CONTEXT, fall back to predicted_next
            current_context = predicted_next

        previous_summary = summary_text

        # 4) Collect metadata for downstream steps
        all_chunks_info.append({
            "chunk_index": i,
            "predicted_next_context": predicted_next,
            "detected_figures": figs_list,
            "context_text_snippet": context_text[:1000],
            "summary_snippet": summary_text[:1000]
        })

    return all_chunks_info

# ---------------- Step 2: Global synthesis + plan (unchanged logic from v10) ----------------
def synthesis_prompt(chunk_summaries: List[dict]) -> str:
    return f"""
You are a senior presenter. Using provided structured chunk summaries, produce JSON with sections:
1) GLOBAL_SUMMARY: 4-8 paragraphs technical summary.
2) PRESENTATION_PLAN: slide-by-slide plan (8-12 slides) with titles + bullet points.
3) KEY_NOTIONS: 4-6 domain notions with 2-3 sentence definitions, marked "critical"/"optional".
4) INTERESTING_CITATIONS: foundational citations from chunks.

Input summaries (list of chunk metadata): {json.dumps(chunk_summaries)[:MAX_PROMPT_CHARS]}
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
    prompt = plan_to_notions_prompt(global_summary, plan_text, refs_text[:MAX_PROMPT_CHARS])
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
    print("Starting main_v11_contextual_updated pipeline...")
    pages, full_text = extract_pdf_pages_text(PDF_PATH)
    body_text, refs_text = find_references_section(full_text)
    refs_map = parse_refs_best_effort(refs_text)
    print(f"Document text: {len(body_text)} chars, References: {len(refs_text)} chars")

    # Step 1: two-pass contextual chunking
    chunks_info = step1_chunk_and_summarize(body_text)

    # Step 2: synthesis & plan (feed metadata so LLM can reference which chunks discuss figures)
    global_summary, plan_text = step2_synthesize_and_plan(chunks_info)
    write_outputs_step2(global_summary, plan_text)

    # Step 3: produce notions + references from plan (plan decides which references to fetch)
    notions_and_refs = step3_produce_notions_and_refs(global_summary, plan_text, refs_text)
    write_outputs_step3(notions_and_refs, refs_map)

    print("Pipeline complete. Summary, plan, and notions ready.")

if __name__ == "__main__":
    main()
