#!/usr/bin/env python3
"""
Global.py

Generates a global summary and a presentation plan for the article.

1) Reads all chunk summaries from main text and supplementary materials.
2) Produces a global summary aimed at an audience familiar with cold atoms.
3) Produces a slide-by-slide presentation plan, highlighting key concepts, experimental setup, theory, results, and conclusions.

Outputs:
- Bugsummary.txt      (global textual summary)
- plan.txt            (presentation slide plan)
"""

import os
import json
from langchain_ollama import OllamaLLM

# ---------------- CONFIG ----------------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
SUMMARY_DIR = os.path.join(BASE_DIR, "summaries")
MAIN_CHUNKS_DIR = os.path.join(SUMMARY_DIR, "main_chunks")
SUPP_SUMMARY_PATH = os.path.join(SUMMARY_DIR, "supplementary_summary.json")
OUTPUT_DIR = SUMMARY_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

LLM_MODEL = "llama3.1"
llm = OllamaLLM(model=LLM_MODEL)

# Output files
GLOBAL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "Bugsummary.txt")
PRESENTATION_PLAN_PATH = os.path.join(OUTPUT_DIR, "plan.txt")


# ---------------- Utility functions ----------------
def read_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_main_chunk_summaries():
    summaries = []
    files = sorted([f for f in os.listdir(MAIN_CHUNKS_DIR) if f.endswith("_summary.txt")])
    for fname in files:
        path = os.path.join(MAIN_CHUNKS_DIR, fname)
        summaries.append(read_file(path))
    return summaries


def read_supplementary_summaries():
    if not os.path.exists(SUPP_SUMMARY_PATH):
        return {}
    with open(SUPP_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- STEP 1: Global Summary ----------------
def generate_global_summary(main_summaries, supp_summaries):
    print("Generating global summary...")
    combined_summary = "\n".join(main_summaries)

    supp_text = "\n".join([f"{k}: {v}" for k, v in supp_summaries.items()])

    prompt = f"""
You are summarizing a scientific article in the field of cold atoms.
The audience is familiar with cold atoms but not with this specific article.

- Main text chunk summaries:
{combined_summary}

- Supplementary section summaries:
{supp_text}

Task:
Write a coherent global summary of the article. Aim for 10–15 sentences.
Include main notions, experimental setup, theoretical concepts, results, and conclusions.
Keep it clear and structured for an audience that knows cold atoms but not this specific work.
"""
    global_summary = llm.invoke(prompt).strip()
    write_file(GLOBAL_SUMMARY_PATH, global_summary)
    print(f"✅ Global summary saved to {GLOBAL_SUMMARY_PATH}")
    return global_summary


# ---------------- STEP 2: Presentation Plan ----------------
def generate_presentation_plan(global_summary):
    print("Generating presentation plan...")

    prompt = f"""
You are a scientific presentation designer.
The audience knows cold atoms, but not this specific work.
You have the global summary of the article:

{global_summary}

Task:
1) Define a slide-by-slide presentation plan.
2) For each slide, give:
   - Slide title
   - Key concepts or points to present
   - Figures or examples if relevant
3) The plan should have a logical flow:
   a) One or two introductory concepts to explain first
   b) Main article introduction
   c) Experimental setup
   d) Theoretical background
   e) Results
   f) Conclusion and perspectives
4) Keep each slide description concise, 3–5 sentences max.

Output as a clear, numbered list of slides.
"""
    plan_text = llm.invoke(prompt).strip()
    write_file(PRESENTATION_PLAN_PATH, plan_text)
    print(f"✅ Presentation plan saved to {PRESENTATION_PLAN_PATH}")
    return plan_text


# ---------------- Main Runner ----------------
def main():
    main_summaries = read_main_chunk_summaries()
    supp_summaries = read_supplementary_summaries()

    global_summary = generate_global_summary(main_summaries, supp_summaries)
    generate_presentation_plan(global_summary)
    print("Pipeline complete. ✅")


if __name__ == "__main__":
    main()