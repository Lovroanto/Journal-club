#!/usr/bin/env python3
"""
FinalizeAndEnrichPlan.py

Unified pipeline:
1. Split existing presentation plan into slides.
2. Enrich slides using supplementary material.
3. Use global summary to finalize each slide (integrate enrichment or create new slide if needed).
4. Save final slides individually and combine into plan_final.txt.
"""

import os
import re
import json
from langchain_ollama import OllamaLLM

# ---------------- CONFIG ----------------
BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
SUMMARIES_DIR = os.path.join(BASE_DIR, "summaries")
PLAN_PATH = os.path.join(SUMMARIES_DIR, "plan.txt")
SUPP_SUMMARY_PATH = os.path.join(SUMMARIES_DIR, "supplementary_summary.json")
GLOBAL_SUMMARY_PATH = os.path.join(SUMMARIES_DIR, "Bugsummary.txt")
SLIDES_FINAL_DIR = os.path.join(SUMMARIES_DIR, "slides_final")
FINAL_PLAN_PATH = os.path.join(SUMMARIES_DIR, "plan_final.txt")

LLM_MODEL = "llama3.1"
llm = OllamaLLM(model=LLM_MODEL)

os.makedirs(SLIDES_FINAL_DIR, exist_ok=True)

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

def read_supplementary_summary():
    if not os.path.exists(SUPP_SUMMARY_PATH):
        print(f"❌ Supplementary summary file not found: {SUPP_SUMMARY_PATH}")
        return {}
    with open(SUPP_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def split_slides(plan_text):
    parts = re.split(r"(Slide \d+:)", plan_text)
    if len(parts) < 2:
        return [plan_text]
    slides = []
    for i in range(1, len(parts), 2):
        title = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        slides.append(title + content)
    return slides

# ---------------- Step 1: Generate enrichment proposals ----------------
def generate_slide_proposals(slides, supp_summaries):
    proposals = []
    for idx, slide in enumerate(slides, start=1):
        prompt = f"""
You are an expert preparing a journal club slide.

Slide content:
{slide[:4000]}

Supplementary sections summaries:
{json.dumps(supp_summaries)[:8000]}

Task:
1. Determine if any supplementary section can enrich this slide.
2. If yes, provide a concise enrichment (1-3 sentences).
3. If the supplementary content is substantial, suggest a new slide instead with title and content.
4. If nothing relevant, state "No enrichment needed."

Output: structured text for the slide only.
"""
        resp = llm.invoke(prompt).strip()
        proposals.append(resp)
        print(f"  ✓ Generated enrichment for Slide {idx}")
    return proposals

# ---------------- Step 2: Finalize slides ----------------
def finalize_slide(original_slide, proposal, global_summary):
    prompt = f"""
You are an expert preparing a journal club slide.

Original slide content:
{original_slide[:3000]}

Proposed enrichment:
{proposal[:3000]}

Global article summary:
{global_summary[:15000]}

Task:
1. Integrate the enrichment into the slide content or create a new slide if needed.
2. Output only the final slide content (title, description, and text) without repeating old versions.
3. Keep the slide concise, clear, and suitable for presentation.
4. If a new slide is needed, provide it as an additional slide with title and content.

Output: finalized slide text ready for presentation.
"""
    final_text = llm.invoke(prompt).strip()
    return final_text

# ---------------- Runner ----------------
def main():
    plan_text = read_file(PLAN_PATH)
    supp_summaries = read_supplementary_summary()
    global_summary = read_file(GLOBAL_SUMMARY_PATH)

    if not plan_text:
        print("❌ No existing plan found.")
        return
    if not global_summary:
        print("❌ Global summary not found.")
        return

    # Split plan into slides
    slides = split_slides(plan_text)
    print(f"Split plan into {len(slides)} slides.")

    # Step 1: generate enrichment proposals
    proposals = generate_slide_proposals(slides, supp_summaries)

    # Step 2: finalize slides using global summary
    final_texts = []
    for idx, (slide, proposal) in enumerate(zip(slides, proposals), start=1):
        final_slide_text = finalize_slide(slide, proposal, global_summary)
        slide_file = os.path.join(SLIDES_FINAL_DIR, f"slide_{idx:03d}_final.txt")
        write_file(slide_file, final_slide_text)
        final_texts.append(final_slide_text)
        print(f"  ✓ Finalized Slide {idx}: {slide_file}")

    # Step 3: combine all final slides into one plan
    combined_plan = "\n\n---\n\n".join(final_texts)
    write_file(FINAL_PLAN_PATH, combined_plan)
    print(f"✅ Final combined plan saved to {FINAL_PLAN_PATH}")

if __name__ == "__main__":
    main()
