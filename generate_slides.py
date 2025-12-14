#!/usr/bin/env python3

import os
import re
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


##############################################################
#  LOAD RAG STORES
##############################################################
def load_chroma_db(path: str):
    """
    Load an existing Chroma vectorstore from disk.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Chroma DB not found at {path}")
    return Chroma(
        persist_directory=path,
        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )


##############################################################
#  LLM CALL (Ollama)
##############################################################
def call_llm(model_name: str, prompt: str) -> str:
    """
    Call a local LLM via Ollama CLI.
    model_name: e.g. "llama3:70b"
    """
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"❌ Ollama error:\nSTDERR:\n{result.stderr}\n"
        )

    return result.stdout.strip()


##############################################################
# 1) Parse Slide Group (Extract Slides, SupMat, Figures, Context)
##############################################################
def parse_slide_group_full(path: str) -> List[Dict]:
    """
    Parse a SLIDE_GROUP text file into a list of slide dicts,
    each containing its group context + its own instruction.
    """
    text = Path(path).read_text()

    # Group-level metadata
    group_name_match = re.findall(r"SLIDE_GROUP:\s*(.*?)\|", text)
    group_name = group_name_match[0].strip() if group_name_match else "UNNAMED_GROUP"

    purpose = re.findall(r"PURPOSE:\s*(.*)", text)
    scope = re.findall(r"CONTENT_SCOPE:\s*(.*)", text)
    key_terms = re.findall(r"KEY_TERMS:\s*(.*)", text)
    transition_global = re.findall(r"KEY_TRANSITIONS:\s*(.*)", text)

    purpose = purpose[0].strip() if purpose else None
    scope = scope[0].strip() if scope else None
    key_terms = [x.strip() for x in key_terms[0].split(",")] if key_terms else []
    transition_global = transition_global[0].strip() if transition_global else None

    # Per-slide instructions
    slides = re.findall(r"Slide\s*\d+:\s*(.*)", text)

    # Figures suggested for this group
    figs = re.findall(r"Include:\s*(Figure\s*[\w\-\(\)\s]+)", text)

    # SupMat info for this group
    sup = re.findall(r"\[SupMat\]\s*(.*)", text)
    sup = sup[0].strip() if sup else None

    out: List[Dict] = []
    for idx, instruction in enumerate(slides, start=1):
        out.append({
            "slide_id": idx,
            "instruction": instruction.strip(),
            "group_name": group_name,
            "purpose": purpose,
            "content_scope": scope,
            "key_terms": key_terms,
            "figures": figs,
            "supmat_notes": sup,
            "transition_global": transition_global,
        })
    return out


##############################################################
# 2) PER-SLIDE CONTEXT BUILDER
##############################################################
def build_slide_context(slide: Dict, global_context: str, model: str) -> str:
    """
    Build a detailed textual context summary for ONE slide,
    using the global paper context + group metadata + slide instruction.
    """

    prompt = f"""
=== BUILD CONTEXT FOR SLIDE {slide['slide_id']} ===

GLOBAL CONTEXT OF PAPER (this is the ONLY paper we talk about):
{global_context}

GROUP CONTEXT FOR THIS SLIDE GROUP:
- Group name: {slide['group_name']}
- Purpose: {slide['purpose']}
- Scope: {slide['content_scope']}
- Key Terms: {slide['key_terms']}
- Suggested Figures: {slide['figures']}
- SupMat notes: {slide['supmat_notes']}

THIS SLIDE'S SPECIFIC TASK:
"{slide['instruction']}"

Write a structured context summary (10–15 sentences) that answers:
- What exactly this slide must explain in the context of the paper.
- Why this slide exists in the flow of the group.
- What the audience must understand after seeing this slide.
- Which scientific elements from the MAIN TEXT will be important.
- How the figures (if any) will anchor the explanation.
- Which details might require SupMat or Literature (but still about THIS paper).

Output as plain text. Do NOT output JSON here.
"""
    return call_llm(model, prompt)


##############################################################
# 3) RAG-AWARE SLIDE REALIZATION
##############################################################
def generate_slide_from_context(
    slide: Dict,
    context_text: str,
    global_context: str,
    rag: Dict,
    model: str
) -> str:
    """
    Turn a per-slide context summary + RAG into a single JSON slide definition.
    Anchored strictly to THIS paper (global_context).
    """

    # REQUIRED TIERS: MAIN + FIGS
    rag_main = rag["main"].similarity_search(context_text, k=6)
    rag_figs = rag["figs"].similarity_search(context_text, k=4)

    # SupMat ONLY if requested in group and DB exists
    if slide.get("supmat_notes") and rag.get("sup") is not None:
        rag_sup = rag["sup"].similarity_search(context_text, k=3)
    else:
        rag_sup = None

    # Literature ONLY for interpretation / comparison (never the main topic)
    if rag.get("lit") is not None:
        rag_lit = rag["lit"].similarity_search(context_text, k=2)
    else:
        rag_lit = None

    prompt = f"""
You are generating slide {slide['slide_id']} for a talk about ONE specific paper.

THIS PAPER (GLOBAL SUMMARY, MUST BE RESPECTED):
{global_context}

You MUST describe THIS paper and THIS experiment.
Do NOT switch to a different experiment or a different article,
even if RAG_LIT or other sources mention them.

SLIDE METADATA:
- Group name: {slide['group_name']}
- Purpose: {slide['purpose']}
- Scope: {slide['content_scope']}
- Task: {slide['instruction']}
- Key terms: {slide['key_terms']}
- Suggested figures: {slide['figures']}

### MODE SELECTION (based on Purpose):
If Purpose contains words like "introduce" or "overview":
    → Entry-level slide: define concepts, give big-picture motivation,
      avoid heavy equations and detailed zone behavior.
If Purpose contains words like "mechanism", "RIR", "self-organization", "model":
    → Mechanism slide: explain physical processes and how they work.
If Purpose contains words like "results", "regime", "zone", "linewidth", "hysteresis":
    → Results slide: focus on experimental observations and figures.
If Purpose contains words like "interpretation", "implications", "outlook":
    → Interpretation slide: summarize meaning, relate to potential applications or literature.
Otherwise:
    → Use a balanced level of detail consistent with the Scope.

CONTEXT SUMMARY FOR THIS SLIDE (conceptual skeleton):
{context_text}

RAG SOURCES (use in this strict order):

1) MAIN (from THIS paper, primary evidence):
{rag_main}

2) FIGURES (captions / info, must be referenced visually if results/mechanism slide):
{rag_figs}

3) SUPPLEMENTARY (only if needed for extra formula or detailed mechanism about THIS paper):
{rag_sup}

4) LITERATURE (only for brief comparison; NEVER let it become the main topic):
{rag_lit}

Now output ONLY a JSON object in this EXACT format.
Do NOT explain, do NOT add commentary, do NOT use markdown.

{{
  "title": "short slide title focused on THIS paper",
  "figure_used": "one of {slide['figures']} or null",
  "bullets": [
     {{"text": "one key point matching Purpose + Task, about THIS experiment", "sources": ["MAIN_x", "FIG_y"]}}
  ],
  "speech": "180-240 word narration appropriate to the Purpose. If this is an introduction slide, keep it gentle and motivational. Always talk about THIS paper, not other experiments.",
  "transition": "one sentence that logically leads to the next slide in this group"
}}
"""
    return call_llm(model, prompt)


##############################################################
# 4) FILE WRITER
##############################################################
def save_slide_output(slide_id: int, result: str, output_dir: str):
    """
    Save the raw JSON (string) for a given slide to disk.
    """
    out_path = Path(output_dir) / f"slide_{slide_id:03d}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"✓ Slide saved → {out_path}")


##############################################################
# 5) MAIN — FULL PIPELINE
##############################################################
def generate_slides(
    slide_group_file: str,
    global_context_file: str,
    db_paths: Dict[str, str],
    output_dir: str,
    model: str,
):
    """
    Orchestrate slide generation for ONE slide group file.
    - slide_group_file: path to group plan (with Slide X: ... etc.)
    - global_context_file: path to global summary of the paper
    - db_paths: dict with keys "main","figs","sup","lit"
    - output_dir: where to write slide_XXX.txt files
    - model: Ollama model string, e.g. "llama3:70b"
    """

    slides = parse_slide_group_full(slide_group_file)
    global_context = Path(global_context_file).read_text(encoding="utf-8")

    rag = {
        "main": load_chroma_db(db_paths["main"]),
        "figs": load_chroma_db(db_paths["figs"]),
        "sup":  load_chroma_db(db_paths["sup"]) if "sup" in db_paths else None,
        "lit":  load_chroma_db(db_paths["lit"]) if "lit" in db_paths else None,
    }

    for slide in slides:
        # 1) Build detailed context summary per slide
        ctx = build_slide_context(slide, global_context, model)

        # 2) Generate final slide JSON with RAG hierarchy
        final_json_str = generate_slide_from_context(slide, ctx, global_context, rag, model)

        # 3) Save JSON string as .txt (later your layout engine will parse it)
        save_slide_output(slide["slide_id"], final_json_str, output_dir)

    print("\n=== ALL SLIDES GENERATED SUCCESSFULLY ===")
