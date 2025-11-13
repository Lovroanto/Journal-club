#!/usr/bin/env python3
"""
Global.py

Generates a global summary and a presentation plan
from the chunked main text summaries, with figure awareness and theory-experiment links.

Outputs:
  - Bugsummary.txt  : full global summary with figure/subfigure context
  - plan.txt        : structured slide plan including figures
"""

import os
import json
from langchain_ollama import OllamaLLM

# ---------------- CONFIG ----------------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
SUMMARIES_DIR = os.path.join(BASE_DIR, "summaries")
MAIN_CHUNKS_DIR = os.path.join(SUMMARIES_DIR, "main_chunks")
FIG_SUMMARY_PATH = os.path.join(SUMMARIES_DIR, "figure_summaries.json")
SUPP_SUMMARY_PATH = os.path.join(SUMMARIES_DIR, "supplementary_summary.json")

GLOBAL_SUMMARY_PATH = os.path.join(SUMMARIES_DIR, "Bugsummary.txt")
PLAN_PATH = os.path.join(SUMMARIES_DIR, "plan.txt")

LLM_MODEL = "llama3.1"
llm = OllamaLLM(model=LLM_MODEL)

# ---------------- Utility functions ----------------
def read_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_main_summaries():
    files = sorted([f for f in os.listdir(MAIN_CHUNKS_DIR) if f.endswith("_summary.txt")])
    summaries = []
    for f in files:
        path = os.path.join(MAIN_CHUNKS_DIR, f)
        text = read_file(path)
        if text.strip():
            summaries.append(text)
    return summaries


def read_supplementary_summary():
    if not os.path.exists(SUPP_SUMMARY_PATH):
        print(f"❌ Supplementary summary file not found: {SUPP_SUMMARY_PATH}")
        return {}
    with open(SUPP_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_figure_summaries():
    if not os.path.exists(FIG_SUMMARY_PATH):
        print(f"❌ Figure summary file not found: {FIG_SUMMARY_PATH}")
        return {}
    with open(FIG_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- Generate global summary ----------------
def generate_global_summary(main_summaries):
    print("Generating global summary based on chunk summaries with embedded figure references...")

    # Combine all main chunk summaries into one text
    combined_text = "\n\n".join(main_summaries)

    prompt = f"""
You are a scientific summarizer writing a comprehensive article summary for a journal club audience familiar with cold atoms.

Input:
- Main text chunk summaries:
{combined_text[:15000]}

Task:
1. Write a long, flowing global summary of the paper using the information from the chunk summaries.
2. Integrate figure and subfigure references **directly into the text** exactly as they appear in the chunk summaries.
3. Highlight connections between theory and experimental results.
4. Do not mention supplementary material.
5. Keep it readable for an audience familiar with cold atoms, but do not oversimplify.

Output: a full global summary text with figures and subfigures naturally embedded in the narrative.
"""
    global_summary = llm.invoke(prompt).strip()
    write_file(GLOBAL_SUMMARY_PATH, global_summary)
    print(f"✅ Global summary saved to {GLOBAL_SUMMARY_PATH}")
    return global_summary


# ---------------- Generate presentation plan ----------------
def generate_presentation_plan(global_summary, figure_summaries):
    print("Generating detailed presentation plan...")
    fig_info_lines = []
    for fig_name, info in figure_summaries.items():
        if info.get("type") == "multi":
            panels = ", ".join([f"{k}" for k in info["subpanels"].keys()])
            fig_info_lines.append(f"{fig_name} panels: {panels}")
        else:
            fig_info_lines.append(f"{fig_name}")
    fig_info = "\n".join(fig_info_lines)

    prompt = f"""
You are an expert in physics preparing a journal club presentation.

Audience: researchers familiar with cold atoms but not this paper.

Input:
- Global summary of the article:
{global_summary[:15000]}

- Figure summaries overview:
{fig_info[:5000]}

Task:
1. Create a slide-by-slide plan.
2. Slides should cover:
   - Introduction / key concepts for understanding the paper.
   - Theory.
   - Experimental setup.
   - Results: generate multiple slides if needed to cover all results and figures/subfigures.
   - Conclusion.
3. For each slide, suggest:
   - Title
   - 1–2 sentence description of content
   - Which figure or subfigure to show if applicable
4. Ensure the plan highlights connections between theory and experiment.
5. Make sure all major figures/subfigures are included in some slide.
6. Output a clear structured text plan for the presentation.

Output: text plan.
"""
    plan_text = llm.invoke(prompt).strip()
    write_file(PLAN_PATH, plan_text)
    print(f"✅ Presentation plan saved to {PLAN_PATH}")
    return plan_text


# ---------------- Main runner ----------------
def main():
    main_summaries = read_main_summaries()
    figure_summaries = read_figure_summaries()

    if not main_summaries:
        print("❌ No main text summaries found, aborting.")
        return

    global_summary = generate_global_summary(main_summaries)
    generate_presentation_plan(global_summary, figure_summaries)
    print("Pipeline complete. ✅")


if __name__ == "__main__":
    main()
