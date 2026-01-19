#!/usr/bin/env python3
"""
chunksummary.py (updated for merged supplementary structure)

Post-processing after reworkingthetext.py + postreworking.py.

This script:
  1) Summarizes all figure captions (1–2 sentences each)
  2) Summarizes supplementary material (now from supplementary_merged/ folders)
  3) Summarizes the main text (context-aware, referencing figures and supplementary material)

It also saves:
  - Raw chunk content for inspection
  - Chunk summaries for both main and supplementary material
"""

import os
import re
import json
from langchain_ollama import OllamaLLM
from typing import List

# ---------------- CONFIG ----------------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
MAIN_DIR = os.path.join(BASE_DIR, "main_article")
OUT_DIR = os.path.join(BASE_DIR, "summaries")
os.makedirs(OUT_DIR, exist_ok=True)

LLM_MODEL = "llama3.1"
llm = OllamaLLM(model=LLM_MODEL)

# Input paths
MAIN_TEXT_PATH = os.path.join(MAIN_DIR, "body_cleaned.txt")
SUPP_MERGED_DIR = os.path.join(MAIN_DIR, "supplementary_merged")
FIGURES_DIR = os.path.join(MAIN_DIR, "figures")
FIGURE_CAPTIONS = [os.path.join(FIGURES_DIR, f) for f in sorted(os.listdir(FIGURES_DIR)) if f.endswith(".txt")]

# Output paths
FIG_SUMMARY_PATH = os.path.join(OUT_DIR, "figure_summaries.json")
SUPP_SUMMARY_PATH = os.path.join(OUT_DIR, "supplementary_summary.json")
MAIN_CHUNKS_DIR = os.path.join(OUT_DIR, "main_chunks")
os.makedirs(MAIN_CHUNKS_DIR, exist_ok=True)


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


def sliding_chunks(text: str, size: int, overlap: int) -> List[str]:
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


# ---------------- STEP 1: Figure Summaries ----------------
def summarize_figures():
    print("STEP 1 — Summarizing figures (detecting subpanels)...")
    figure_summaries = {}
    FIG_TXT_DIR = os.path.join(OUT_DIR, "figure_summaries_texts")
    os.makedirs(FIG_TXT_DIR, exist_ok=True)

    for fpath in FIGURE_CAPTIONS:
        caption = read_file(fpath)
        if not caption.strip():
            continue

        fname = os.path.basename(fpath)
        print(f"  → Processing {fname} ...")

        # --- Detect subfigures like (a), (b), (c), etc. ---
        subparts = re.split(r"(?=\(?[a-hA-H]\))", caption)
        subparts = [p.strip() for p in subparts if len(p.strip()) > 10]

        if len(subparts) > 1:
            # Multiple panels detected
            print(f"    Detected {len(subparts)} subpanels in {fname}")
            sub_summaries = {}

            for idx, subtext in enumerate(subparts, start=1):
                panel_label = re.match(r"^\(?([a-hA-H])\)?", subtext)
                label = panel_label.group(1).lower() if panel_label else f"part{idx}"

                sub_prompt = f"""Summarize this subfigure caption in 1–2 sentences.
Be specific about what panel ({label}) shows — e.g., data type, variable, or phenomenon.

SUBFIGURE ({label}):
\"\"\"{subtext[:1500]}\"\"\""""
                sub_summary = llm.invoke(sub_prompt).strip()
                sub_summaries[f"{label}"] = sub_summary

            # Global figure-level summary
            all_sub_summaries = "\n".join([f"({k}) {v}" for k, v in sub_summaries.items()])
            global_prompt = f"""Here are all subfigure summaries for a figure.
Write one concise 1–2 sentence summary describing what the entire figure (all panels) illustrates.

SUBFIGURE SUMMARIES:
{all_sub_summaries}"""
            global_summary = llm.invoke(global_prompt).strip()

            figure_summaries[fname] = {
                "type": "multi",
                "global": global_summary,
                "subpanels": sub_summaries
            }

            # Save as text files
            base = os.path.splitext(fname)[0]
            write_file(os.path.join(FIG_TXT_DIR, f"{base}_global.txt"), global_summary)
            for k, v in sub_summaries.items():
                write_file(os.path.join(FIG_TXT_DIR, f"{base}_{k}.txt"), v)

        else:
            # Single caption case
            prompt = f"""Summarize this figure caption in 1–2 sentences.
Be precise about what is being shown and what phenomenon or experiment it illustrates.

FIGURE CAPTION:
\"\"\"{caption[:2500]}\"\"\""""
            summary = llm.invoke(prompt).strip()

            figure_summaries[fname] = {"type": "single", "summary": summary}
            base = os.path.splitext(fname)[0]
            write_file(os.path.join(FIG_TXT_DIR, f"{base}.txt"), summary)

    # Save full JSON summary
    write_file(FIG_SUMMARY_PATH, json.dumps(figure_summaries, indent=2, ensure_ascii=False))
    print(f"✅ Figure summaries saved to {FIG_SUMMARY_PATH}")
    return figure_summaries


# ---------------- STEP 2: Supplementary Summaries ----------------
def summarize_supplementary():
    print("STEP 2 — Summarizing supplementary material...")

    if not os.path.exists(SUPP_MERGED_DIR):
        raise FileNotFoundError(f"❌ Supplementary merged directory not found: {SUPP_MERGED_DIR}")

    section_dirs = sorted(
        [d for d in os.listdir(SUPP_MERGED_DIR) if os.path.isdir(os.path.join(SUPP_MERGED_DIR, d))]
    )

    summaries = {}
    CHUNK_SIZE_SUPP = 4000
    CHUNK_OVERLAP_SUPP = 400
    SUPP_CHUNKS_DIR = os.path.join(OUT_DIR, "supplementary_chunks")
    os.makedirs(SUPP_CHUNKS_DIR, exist_ok=True)

    for s_idx, sec_dir in enumerate(section_dirs, start=1):
        full_txt_path = os.path.join(SUPP_MERGED_DIR, sec_dir, f"{sec_dir}_full.txt")
        section_text = read_file(full_txt_path)
        if not section_text.strip():
            continue

        section_chunks = sliding_chunks(section_text, CHUNK_SIZE_SUPP, CHUNK_OVERLAP_SUPP)
        sec_chunk_summaries = []#!/usr/bin/env python3
"""
chunksummary.py (updated for merged supplementary structure)

Post-processing after reworkingthetext.py + postreworking.py.

This script:
  1) Summarizes all figure captions (1–2 sentences each)
  2) Summarizes supplementary material (now from supplementary_merged/ folders)
  3) Summarizes the main text (context-aware, referencing figures and supplementary material)

It also saves:
  - Raw chunk content for inspection
  - Chunk summaries for both main and supplementary material
"""

import os
import re
import json
from langchain_ollama import OllamaLLM
from typing import List

# ---------------- CONFIG ----------------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
MAIN_DIR = os.path.join(BASE_DIR, "main_article")
OUT_DIR = os.path.join(BASE_DIR, "summaries")
os.makedirs(OUT_DIR, exist_ok=True)

LLM_MODEL = "llama3.1"
llm = OllamaLLM(model=LLM_MODEL)

# Input paths
MAIN_TEXT_PATH = os.path.join(MAIN_DIR, "body_cleaned.txt")
SUPP_MERGED_DIR = os.path.join(MAIN_DIR, "supplementary_merged")
FIGURES_DIR = os.path.join(MAIN_DIR, "figures")
FIGURE_CAPTIONS = [os.path.join(FIGURES_DIR, f) for f in sorted(os.listdir(FIGURES_DIR)) if f.endswith(".txt")]

# Output paths
FIG_SUMMARY_PATH = os.path.join(OUT_DIR, "figure_summaries.json")
SUPP_SUMMARY_PATH = os.path.join(OUT_DIR, "supplementary_summary.json")
MAIN_CHUNKS_DIR = os.path.join(OUT_DIR, "main_chunks")
os.makedirs(MAIN_CHUNKS_DIR, exist_ok=True)


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


def sliding_chunks(text: str, size: int, overlap: int) -> List[str]:
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


# ---------------- STEP 1: Figure Summaries ----------------
def summarize_figures():
    print("STEP 1 — Summarizing figures (detecting subpanels)...")
    figure_summaries = {}
    FIG_TXT_DIR = os.path.join(OUT_DIR, "figure_summaries_texts")
    os.makedirs(FIG_TXT_DIR, exist_ok=True)

    for fpath in FIGURE_CAPTIONS:
        caption = read_file(fpath)
        if not caption.strip():
            continue

        fname = os.path.basename(fpath)
        print(f"  → Processing {fname} ...")

        # --- Detect subfigures like (a), (b), (c), etc. ---
        subparts = re.split(r"(?=\(?[a-hA-H]\))", caption)
        subparts = [p.strip() for p in subparts if len(p.strip()) > 10]

        if len(subparts) > 1:
            # Multiple panels detected
            print(f"    Detected {len(subparts)} subpanels in {fname}")
            sub_summaries = {}

            for idx, subtext in enumerate(subparts, start=1):
                panel_label = re.match(r"^\(?([a-hA-H])\)?", subtext)
                label = panel_label.group(1).lower() if panel_label else f"part{idx}"

                sub_prompt = f"""Summarize this subfigure caption in 1–2 sentences.
Be specific about what panel ({label}) shows — e.g., data type, variable, or phenomenon.

SUBFIGURE ({label}):
\"\"\"{subtext[:1500]}\"\"\""""
                sub_summary = llm.invoke(sub_prompt).strip()
                sub_summaries[f"{label}"] = sub_summary

            # Global figure-level summary
            all_sub_summaries = "\n".join([f"({k}) {v}" for k, v in sub_summaries.items()])
            global_prompt = f"""Here are all subfigure summaries for a figure.
Write one concise 1–2 sentence summary describing what the entire figure (all panels) illustrates.

SUBFIGURE SUMMARIES:
{all_sub_summaries}"""
            global_summary = llm.invoke(global_prompt).strip()

            figure_summaries[fname] = {
                "type": "multi",
                "global": global_summary,
                "subpanels": sub_summaries
            }

            # Save as text files
            base = os.path.splitext(fname)[0]
            write_file(os.path.join(FIG_TXT_DIR, f"{base}_global.txt"), global_summary)
            for k, v in sub_summaries.items():
                write_file(os.path.join(FIG_TXT_DIR, f"{base}_{k}.txt"), v)

        else:
            # Single caption case
            prompt = f"""Summarize this figure caption in 1–2 sentences.
Be precise about what is being shown and what phenomenon or experiment it illustrates.

FIGURE CAPTION:
\"\"\"{caption[:2500]}\"\"\""""
            summary = llm.invoke(prompt).strip()

            figure_summaries[fname] = {"type": "single", "summary": summary}
            base = os.path.splitext(fname)[0]
            write_file(os.path.join(FIG_TXT_DIR, f"{base}.txt"), summary)

    # Save full JSON summary
    write_file(FIG_SUMMARY_PATH, json.dumps(figure_summaries, indent=2, ensure_ascii=False))
    print(f"✅ Figure summaries saved to {FIG_SUMMARY_PATH}")
    return figure_summaries


# ---------------- STEP 2: Supplementary Summaries ----------------
def summarize_supplementary():
    print("STEP 2 — Summarizing supplementary material...")

    if not os.path.exists(SUPP_MERGED_DIR):
        raise FileNotFoundError(f"❌ Supplementary merged directory not found: {SUPP_MERGED_DIR}")

    section_dirs = sorted(
        [d for d in os.listdir(SUPP_MERGED_DIR) if os.path.isdir(os.path.join(SUPP_MERGED_DIR, d))]
    )

    summaries = {}
    CHUNK_SIZE_SUPP = 4000
    CHUNK_OVERLAP_SUPP = 400
    SUPP_CHUNKS_DIR = os.path.join(OUT_DIR, "supplementary_chunks")
    os.makedirs(SUPP_CHUNKS_DIR, exist_ok=True)

    for s_idx, sec_dir in enumerate(section_dirs, start=1):
        full_txt_path = os.path.join(SUPP_MERGED_DIR, sec_dir, f"{sec_dir}_full.txt")
        section_text = read_file(full_txt_path)
        if not section_text.strip():
            continue

        section_chunks = sliding_chunks(section_text, CHUNK_SIZE_SUPP, CHUNK_OVERLAP_SUPP)
        sec_chunk_summaries = []
        prev_context = "START OF SECTION"

        print(f"  Summarizing supplementary {sec_dir} ({len(section_chunks)} chunks)...")

        for c_idx, chunk in enumerate(section_chunks, start=1):
            # --- Context agent ---
            context_prompt = f"""Previous chunk summary: {prev_context}

You are reading a scientific supplementary section.
In 1–2 sentences, explain where we are in the explanation or procedure.

CHUNK:
{chunk[:1500]}"""
            context_resp = llm.invoke(context_prompt)

            # --- Summary agent ---
            summary_prompt = f"""Summarize this supplementary chunk in 3–5 sentences,
focusing on what is being measured, calculated, or derived.
Also mention if this part is describing or referring to a figure (e.g. "Figure S1", "Extended Data Fig. 2") or an experimental setup.

CONTEXT: {context_resp}
CHUNK:
{chunk[:4000]}"""
            summary_resp = llm.invoke(summary_prompt).strip()
            prev_context = summary_resp
            sec_chunk_summaries.append(summary_resp)

            # Save both raw and summarized chunk for inspection
            raw_out = os.path.join(SUPP_CHUNKS_DIR, f"{sec_dir}_chunk_{c_idx:03d}_raw.txt")
            sum_out = os.path.join(SUPP_CHUNKS_DIR, f"{sec_dir}_chunk_{c_idx:03d}_summary.txt")
            write_file(raw_out, chunk)
            write_file(sum_out, summary_resp)

        # Global summary for entire section
        joined_chunks = "\n".join(sec_chunk_summaries)
        global_prompt = f"""You are a scientific summarizer.
Below are all chunk summaries from one supplementary section.
Summarize the entire section in ONE precise sentence describing what it investigates or concludes.

CHUNK SUMMARIES:
{joined_chunks}"""
        global_summary = llm.invoke(global_prompt)
        summaries[sec_dir] = global_summary.strip()

    write_file(SUPP_SUMMARY_PATH, json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"✅ Supplementary summaries saved to {SUPP_SUMMARY_PATH}")
    return summaries


# ---------------- STEP 3: Main Text Summaries ----------------
def summarize_main(figure_summaries, supp_summaries):
    print("STEP 3 — Summarizing main text (context-aware, with figure/supplementary links)...")

    text = read_file(MAIN_TEXT_PATH)
    if not text.strip():
        print("⚠️ No main text found.")
        return

    CHUNK_SIZE = 5000
    CHUNK_OVERLAP = 500
    chunks = sliding_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    prev_context = "START OF PAPER"
    figure_refs = {k.lower(): v for k, v in figure_summaries.items()}

    CONTEXT_DIR = os.path.join(OUT_DIR, "main_contexts")
    os.makedirs(CONTEXT_DIR, exist_ok=True)

    ADDITIONAL_DIR = os.path.join(OUT_DIR, "main_additional")
    os.makedirs(ADDITIONAL_DIR, exist_ok=True)

    for i, ch in enumerate(chunks, start=1):
        # --- Detect figure mentions ---
        mentioned_figures = []
        for name, info in figure_refs.items():
            base = name.replace(".txt", "").lower()

            if base in ch.lower():
                mentioned_figures.append(name)

            # check subpanels if multi-figure
            if isinstance(info, dict) and "subpanels" in info:
                for subkey, subsummary in info["subpanels"].items():
                    if f"{base}_{subkey}" in ch.lower() or f"({subkey})" in ch.lower():
                        mentioned_figures.append(f"{name} ({subkey})")

        # --- Build figure notes ---
        fig_notes_lines = []
        for name in set(mentioned_figures):
            if "(" in name and ")" in name:  # subfigure reference
                base_name = name.split()[0].lower()
                sub_label = re.search(r"\(([a-h])\)", name)
                if (
                    base_name in figure_refs
                    and isinstance(figure_refs[base_name], dict)
                    and "subpanels" in figure_refs[base_name]
                ):
                    sub_summary = figure_refs[base_name]["subpanels"].get(sub_label.group(1), "")
                    fig_notes_lines.append(f"Note: discusses {base_name} panel ({sub_label.group(1)}) → {sub_summary}")
            elif name.lower() in figure_refs:
                fig_info = figure_refs[name.lower()]
                if isinstance(fig_info, dict):
                    global_summary = fig_info.get("global", fig_info.get("summary", ""))
                    fig_notes_lines.append(f"Note: discusses {name} → {global_summary}")
                else:
                    fig_notes_lines.append(f"Note: discusses {name} → {fig_info}")
        fig_notes = "\n".join(fig_notes_lines) if fig_notes_lines else "No direct figure mentioned."

        # --- Detect supplementary link ---
        supp_note = "No clear supplementary link."
        for s_id, s_summary in supp_summaries.items():
            key_terms = s_summary.lower().split()[:8]
            if any(term in ch.lower() for term in key_terms):
                supp_note = f"Related to supplementary {s_id}: {s_summary}"
                break

        # === CONTEXT PROMPT ===
        context_prompt = f"""
You are reading a scientific article.
Previous chunk summary (for continuity):
{prev_context}

Figures mentioned:
{fig_notes}

Preliminary supplementary info:
{supp_note}

CHUNK EXCERPT:
{ch[:1500]}

Task:
Describe in 2–3 sentences where we are in the paper, relation to previous content,
and what the figures/supplementary sections provide.
"""
        context_resp = llm.invoke(context_prompt).strip()

        # --- Write context file ---
        ctx_path = os.path.join(CONTEXT_DIR, f"chunk_{i:03d}_context.txt")
        write_file(ctx_path, f"FIGURE INFO:\n{fig_notes}\n\nSUPPLEMENTARY INFO:\n{supp_note}\n\nCONTEXT:\n{context_resp}")

        # === MAIN SUMMARY PROMPT ===
        summary_prompt = f"""
Summarize this chunk in 6–10 sentences, focusing on conceptual and technical points.
Include figures and supplementary material if relevant.

FIGURE INFO:
{fig_notes}

SUPPLEMENTARY INFO:
{supp_note}

CONTEXT (from previous + inferred):
{context_resp}

CHUNK:
{ch[:4000]}
"""
        summary_resp = llm.invoke(summary_prompt).strip()
        prev_context = summary_resp

        # --- Write main summary file ---
        sum_path = os.path.join(MAIN_CHUNKS_DIR, f"chunk_{i:03d}_summary.txt")
        write_file(sum_path, summary_resp)

        # === NEW STEP: Check supplementary linkage for additional content ===
        link_prompt = f"""
You are given the summary of a main text chunk and all supplementary section summaries.

MAIN CHUNK SUMMARY:
{summary_resp}

SUPPLEMENTARY SUMMARIES:
{json.dumps(supp_summaries, indent=2, ensure_ascii=False)}

Task:
Determine which supplementary sections provide additional relevant information for this chunk.
Output a comma-separated list of section IDs (e.g., section_02, section_07), or 'None' if no sections are relevant.
"""
        linked_sections_resp = llm.invoke(link_prompt).strip()

        # --- Write to additional content file ---
        add_path = os.path.join(ADDITIONAL_DIR, f"chunk_{i:03d}_additional.txt")
        if linked_sections_resp.lower() != "none":
            write_file(add_path, f"More details in supplementary sections: {linked_sections_resp}.")
        else:
            write_file(add_path, "No additional supplementary content linked.")

        print(f"  ✓ Processed chunk {i}/{len(chunks)}")

    print(f"✅ Main text chunk contexts → {CONTEXT_DIR}")
    print(f"✅ Main text chunk summaries → {MAIN_CHUNKS_DIR}")
    print(f"✅ Main text chunk additional supplementary links → {ADDITIONAL_DIR}")


# ---------------- Main Runner ----------------
def main():
    print("Starting chunksummary.py pipeline...")
    figure_summaries = summarize_figures()
    supp_summaries = summarize_supplementary()
    summarize_main(figure_summaries, supp_summaries)
    print("Pipeline complete. ✅")


if __name__ == "__main__":
    main()

        prev_context = "START OF SECTION"

        print(f"  Summarizing supplementary {sec_dir} ({len(section_chunks)} chunks)...")

        for c_idx, chunk in enumerate(section_chunks, start=1):
            # --- Context agent ---
            context_prompt = f"""Previous chunk summary: {prev_context}

You are reading a scientific supplementary section.
In 1–2 sentences, explain where we are in the explanation or procedure.

CHUNK:
{chunk[:1500]}"""
            context_resp = llm.invoke(context_prompt)

            # --- Summary agent ---
            summary_prompt = f"""Summarize this supplementary chunk in 3–5 sentences,
focusing on what is being measured, calculated, or derived.
Also mention if this part is describing or referring to a figure (e.g. "Figure S1", "Extended Data Fig. 2") or an experimental setup.

CONTEXT: {context_resp}
CHUNK:
{chunk[:4000]}"""
            summary_resp = llm.invoke(summary_prompt).strip()
            prev_context = summary_resp
            sec_chunk_summaries.append(summary_resp)

            # Save both raw and summarized chunk for inspection
            raw_out = os.path.join(SUPP_CHUNKS_DIR, f"{sec_dir}_chunk_{c_idx:03d}_raw.txt")
            sum_out = os.path.join(SUPP_CHUNKS_DIR, f"{sec_dir}_chunk_{c_idx:03d}_summary.txt")
            write_file(raw_out, chunk)
            write_file(sum_out, summary_resp)

        # Global summary for entire section
        joined_chunks = "\n".join(sec_chunk_summaries)
        global_prompt = f"""You are a scientific summarizer.
Below are all chunk summaries from one supplementary section.
Summarize the entire section in ONE precise sentence describing what it investigates or concludes.

CHUNK SUMMARIES:
{joined_chunks}"""
        global_summary = llm.invoke(global_prompt)
        summaries[sec_dir] = global_summary.strip()

    write_file(SUPP_SUMMARY_PATH, json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"✅ Supplementary summaries saved to {SUPP_SUMMARY_PATH}")
    return summaries


# ---------------- STEP 3: Main Text Summaries ----------------
def summarize_main(figure_summaries, supp_summaries):
    print("STEP 3 — Summarizing main text (context-aware, with figure/supplementary links)...")

    text = read_file(MAIN_TEXT_PATH)
    if not text.strip():
        print("⚠️ No main text found.")
        return

    CHUNK_SIZE = 5000
    CHUNK_OVERLAP = 500
    chunks = sliding_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    prev_context = "START OF PAPER"
    figure_refs = {k.lower(): v for k, v in figure_summaries.items()}

    CONTEXT_DIR = os.path.join(OUT_DIR, "main_contexts")
    os.makedirs(CONTEXT_DIR, exist_ok=True)

    ADDITIONAL_DIR = os.path.join(OUT_DIR, "main_additional")
    os.makedirs(ADDITIONAL_DIR, exist_ok=True)

    for i, ch in enumerate(chunks, start=1):
        # --- Detect figure mentions ---
        mentioned_figures = []
        for name, info in figure_refs.items():
            base = name.replace(".txt", "").lower()

            if base in ch.lower():
                mentioned_figures.append(name)

            # check subpanels if multi-figure
            if isinstance(info, dict) and "subpanels" in info:
                for subkey, subsummary in info["subpanels"].items():
                    if f"{base}_{subkey}" in ch.lower() or f"({subkey})" in ch.lower():
                        mentioned_figures.append(f"{name} ({subkey})")

        # --- Build figure notes ---
        fig_notes_lines = []
        for name in set(mentioned_figures):
            if "(" in name and ")" in name:  # subfigure reference
                base_name = name.split()[0].lower()
                sub_label = re.search(r"\(([a-h])\)", name)
                if (
                    base_name in figure_refs
                    and isinstance(figure_refs[base_name], dict)
                    and "subpanels" in figure_refs[base_name]
                ):
                    sub_summary = figure_refs[base_name]["subpanels"].get(sub_label.group(1), "")
                    fig_notes_lines.append(f"Note: discusses {base_name} panel ({sub_label.group(1)}) → {sub_summary}")
            elif name.lower() in figure_refs:
                fig_info = figure_refs[name.lower()]
                if isinstance(fig_info, dict):
                    global_summary = fig_info.get("global", fig_info.get("summary", ""))
                    fig_notes_lines.append(f"Note: discusses {name} → {global_summary}")
                else:
                    fig_notes_lines.append(f"Note: discusses {name} → {fig_info}")
        fig_notes = "\n".join(fig_notes_lines) if fig_notes_lines else "No direct figure mentioned."

        # --- Detect supplementary link ---
        supp_note = "No clear supplementary link."
        for s_id, s_summary in supp_summaries.items():
            key_terms = s_summary.lower().split()[:8]
            if any(term in ch.lower() for term in key_terms):
                supp_note = f"Related to supplementary {s_id}: {s_summary}"
                break

        # === CONTEXT PROMPT ===
        context_prompt = f"""
You are reading a scientific article.
Previous chunk summary (for continuity):
{prev_context}

Figures mentioned:
{fig_notes}

Preliminary supplementary info:
{supp_note}

CHUNK EXCERPT:
{ch[:1500]}

Task:
Describe in 2–3 sentences where we are in the paper, relation to previous content,
and what the figures/supplementary sections provide.
"""
        context_resp = llm.invoke(context_prompt).strip()

        # --- Write context file ---
        ctx_path = os.path.join(CONTEXT_DIR, f"chunk_{i:03d}_context.txt")
        write_file(ctx_path, f"FIGURE INFO:\n{fig_notes}\n\nSUPPLEMENTARY INFO:\n{supp_note}\n\nCONTEXT:\n{context_resp}")

        # === MAIN SUMMARY PROMPT ===
        summary_prompt = f"""
Summarize this chunk in 6–10 sentences, focusing on conceptual and technical points.
Include figures and supplementary material if relevant.

FIGURE INFO:
{fig_notes}

SUPPLEMENTARY INFO:
{supp_note}

CONTEXT (from previous + inferred):
{context_resp}

CHUNK:
{ch[:4000]}
"""
        summary_resp = llm.invoke(summary_prompt).strip()
        prev_context = summary_resp

        # --- Write main summary file ---
        sum_path = os.path.join(MAIN_CHUNKS_DIR, f"chunk_{i:03d}_summary.txt")
        write_file(sum_path, summary_resp)

        # === NEW STEP: Check supplementary linkage for additional content ===
        link_prompt = f"""
You are given the summary of a main text chunk and all supplementary section summaries.

MAIN CHUNK SUMMARY:
{summary_resp}

SUPPLEMENTARY SUMMARIES:
{json.dumps(supp_summaries, indent=2, ensure_ascii=False)}

Task:
Determine which supplementary sections provide additional relevant information for this chunk.
Output a comma-separated list of section IDs (e.g., section_02, section_07), or 'None' if no sections are relevant.
"""
        linked_sections_resp = llm.invoke(link_prompt).strip()

        # --- Write to additional content file ---
        add_path = os.path.join(ADDITIONAL_DIR, f"chunk_{i:03d}_additional.txt")
        if linked_sections_resp.lower() != "none":
            write_file(add_path, f"More details in supplementary sections: {linked_sections_resp}.")
        else:
            write_file(add_path, "No additional supplementary content linked.")

        print(f"  ✓ Processed chunk {i}/{len(chunks)}")

    print(f"✅ Main text chunk contexts → {CONTEXT_DIR}")
    print(f"✅ Main text chunk summaries → {MAIN_CHUNKS_DIR}")
    print(f"✅ Main text chunk additional supplementary links → {ADDITIONAL_DIR}")


# ---------------- Main Runner ----------------
def main():
    print("Starting chunksummary.py pipeline...")
    figure_summaries = summarize_figures()
    supp_summaries = summarize_supplementary()
    summarize_main(figure_summaries, supp_summaries)
    print("Pipeline complete. ✅")


if __name__ == "__main__":
    main()
