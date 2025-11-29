#!/usr/bin/env python3

import os
import json
from pathlib import Path
from typing import List, Dict, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
#  LOAD RAG STORES
# ============================================================

def load_chroma_db(path: str):
    """Loads an existing Chroma vectorstore directory."""
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Chroma DB not found at {path}")
    return Chroma(persist_directory=path, embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))


# ============================================================
#  UTILS — LLM CALL TEMPLATE  (you replace with actual Ollama call)
# ============================================================

def call_llm(model_name: str, prompt: str) -> str:
    """
    ⛔ placeholder — you plug Ollama API call here
    e.g.:
    result = subprocess.run(["ollama", "run", model_name, prompt], capture_output=True, text=True)
    return result.stdout
    """
    print(f"[DEBUG] LLM CALL → {model_name}")
    print(prompt[:200] + "...\n")
    return "<<< LLM OUTPUT PLACEHOLDER >>>"



def parse_slide_group_full(path: str) -> List[Dict]:
    """Extracts full structured context for each slide from a SLIDE_GROUP file."""

    text = Path(path).read_text()

    # --- Extract group meta blocks ---
    group_name = re.findall(r"SLIDE_GROUP:\s*(.*?)\|", text)[0].strip()
    purpose = re.findall(r"PURPOSE:\s*(.*)", text)
    scope = re.findall(r"CONTENT_SCOPE:\s*(.*)", text)
    key_terms = re.findall(r"KEY_TERMS:\s*(.*)", text)
    transition_global = re.findall(r"KEY_TRANSITIONS:\s*(.*)", text)

    purpose = purpose[0] if purpose else None
    scope = scope[0] if scope else None
    key_terms = [x.strip() for x in key_terms[0].split(",")] if key_terms else []
    transition_global = transition_global[0] if transition_global else None

    # --- Extract per-slide lines ---
    slide_lines = re.findall(r"Slide\s*\d+:\s*(.*)", text)

    # --- Extract figures suggested ---
    figs = re.findall(r"Include:\s*(Figure\s*\d+[A-Za-z\-\(\)\s]*)", text)

    # --- SupMat notes ---
    sup = re.findall(r"\[SupMat\]\s*(.*)", text)
    sup = sup[0] if sup else None

    # --- Now expand into structured list ---
    slides = []
    for idx, instruction in enumerate(slide_lines, start=1):
        slides.append({
            "slide_id": idx,
            "instruction": instruction.strip(),
            "group_name": group_name,
            "purpose": purpose,
            "content_scope": scope,
            "key_terms": key_terms,
            "figures": figs,
            "supmat_notes": sup,
            "group_transition": transition_global
        })

    return slides


# ============================================================
#  STAGE 1 — SLIDE PLANNER
# ============================================================

class SlidePlanner:

    def __init__(self, model_name, db_main, db_figs):
        self.model = model_name
        self.db_main = db_main
        self.db_figs = db_figs

    def make_plan(self, raw_instruction: str, group_scope: str) -> Dict:
        """
        Generates structured slide plan:
        - inferred title
        - required subtopics to cover
        - potential figure references
        """

        # Retrieve RAG MAIN + FIG MANDATORY
        main_chunks = self.db_main.similarity_search(raw_instruction, k=6)
        fig_chunks  = self.db_figs.similarity_search(raw_instruction, k=4)

        prompt = f"""
        You are preparing a physics presentation slide plan.

        Slide instruction:
        {raw_instruction}

        Group scope:
        {group_scope}

        Based on the retrieved material below,
        infer 3–6 subtopics that MUST be addressed on this slide:

        MAIN:
        {main_chunks}

        FIGURES:
        {fig_chunks}

        Output JSON with:
        {{
            "title": "short title",
            "required_topics": ["...", "...", "..."],
            "figure_candidates": ["Figure_2", "Figure_4", ...]
        }}
        """

        response = call_llm(self.model, prompt)
        try:
            return json.loads(response)
        except:
            return {"title": raw_instruction, "required_topics": [], "figure_candidates": []}


# ============================================================
#  STAGE 2 — SLIDE REALIZER
# ============================================================

class SlideRealizer:

    def __init__(self, model_name, db_main, db_figs, db_sup=None, db_lit=None):
        self.model = model_name
        self.db_main = db_main
        self.db_figs = db_figs
        self.db_sup  = db_sup
        self.db_lit  = db_lit

    def retrieve_ordered(self, query, required_topics, sup_allowed=True, lit_allowed=True):

        data = {
            "main": self.db_main.similarity_search(query, k=6),
            "figs": self.db_figs.similarity_search(query, k=4),
        }

        if sup_allowed and self.db_sup:
            data["sup"] = self.db_sup.similarity_search(query, k=4)

        if lit_allowed and self.db_lit:
            data["lit"] = self.db_lit.similarity_search(query, k=2)

        return data

    def build_slide(self, slide_id: int, plan: Dict, allow_sup=True, allow_lit=True) -> Dict:

        retrieved = self.retrieve_ordered(
            query=plan["title"],
            required_topics=plan["required_topics"],
            sup_allowed=allow_sup,
            lit_allowed=allow_lit
        )

        prompt = f"""
        Generate content for one scientific slide.

        Slide Title: {plan['title']}
        Required Topics: {plan['required_topics']}
        Figures available: {plan['figure_candidates']}

        Retrieved evidence:
        {retrieved}

        TASK:
        1. Generate 3–6 short bullet points using MAIN+FIG first (mandatory).
        2. If missing information → use SUP, then (optional) LIT.
        3. Produce a 130–200-word speech explaining the bullets.
        4. Reference figures clearly.
        5. Output as JSON:

        {{
            "title": "...",
            "figure": "...",
            "bullets": [
               {{"text": "...", "sources": ["MAIN_12","FIG_2"]}},
               ...
            ],
            "speech": "... 150 words ...",
            "transition": "1 sentence to next slide"
        }}
        """

        response = call_llm(self.model, prompt)
        try:
            return json.loads(response)
        except:
            return {"title": plan['title'], "bullets": [], "speech": "ERROR", "figure": None}


# ============================================================
#  SAVE RESULTS
# ============================================================

def save_slide_text(slide_num: int, slide_data: Dict, out_dir: str):
    out = Path(out_dir) / f"slide_{slide_num:03d}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write(f"SLIDE {slide_num} — {slide_data['title']}\n")
        f.write("="*60 + "\n\n")

        f.write("FIGURE: " + str(slide_data.get("figure","NONE"))+ "\n\n")

        f.write("BULLETS:\n")
        for b in slide_data.get("bullets",[]):
            f.write(f" • {b['text']}   {b.get('sources','')}\n")
        f.write("\n\nSPEECH:\n" + slide_data.get("speech","") + "\n\n")

        f.write("TRANSITION:\n" + slide_data.get("transition","") + "\n\n")

    print(f"→ saved {out}")


# ============================================================
#  MAIN ORCHESTRATOR
# ============================================================

def generate_slides(
    blueprint_text: List[str],
    db_paths: Dict,
    output_dir: str,
    model_name: str
):

    # Load RAG Stores (MAIN + FIG mandatory)
    main = load_chroma_db(db_paths["main"])
    figs = load_chroma_db(db_paths["figs"])

    sup  = load_chroma_db(db_paths["sup"]) if "sup" in db_paths else None
    lit  = load_chroma_db(db_paths["lit"]) if "lit" in db_paths else None

    planner  = SlidePlanner(model_name, main, figs)
    realizer = SlideRealizer(model_name, main, figs, sup, lit)

    for i, raw_instruction in enumerate(blueprint_text, start=1):

        plan = planner.make_plan(raw_instruction, group_scope="Global physics mechanism block")
        slide = realizer.build_slide(slide_id=i, plan=plan, allow_sup=True, allow_lit=True)

        save_slide_text(i, slide, output_dir=output_dir)

    print("\nAll slides generated ✓")


# ============================================================
# EXAMPLE USE
# ============================================================

if __name__ == "__main__":

    blueprint = [
        "Explain self-organization and atomic motion",
        "Describe recoil-induced gain and its frequency dependence",
        "Discuss the cross-over regime's influence on cavity linewidth",
        "Illustrate hysteresis in lasing due to atom number",
        "Phenomenological model behavior in zone I"
    ]

    generate_slides(
        blueprint_text=blueprint,
        db_paths={
            "main": "./chroma_main_db",
            "figs": "./chroma_figures_db",
            "sup": "./chroma_sup_db",
            "lit": "./chroma_lit_db"
        },
        output_dir="./slides_txt",
        model_name="llama3:70b"
    )
