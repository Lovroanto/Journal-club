#!/usr/bin/env python3
"""
ExtractAndCleanNotionsStrict.py

Strict pipeline:
1. Extract only *essential* scientific notions per slide.
2. Keep slide-by-slide and global notions.
3. Clean to keep only notions required to understand the slide/article.
Outputs:
 - notion_slide.txt
 - notions.txt
 - notions_clean.txt
"""

import os
import re
from langchain_ollama import OllamaLLM

BASE_DIR = os.path.expanduser("~/ai_data/Journal_Club/First")
SUMMARIES_DIR = os.path.join(BASE_DIR, "summaries")

PLAN_PATH = os.path.join(SUMMARIES_DIR, "plan_final.txt")
NOTIONS_SLIDE_PATH = os.path.join(SUMMARIES_DIR, "notion_slide.txt")
NOTIONS_GLOBAL_PATH = os.path.join(SUMMARIES_DIR, "notions.txt")
NOTIONS_CLEAN_PATH = os.path.join(SUMMARIES_DIR, "notions_clean.txt")

LLM_MODEL = "llama3.1"
llm = OllamaLLM(model=LLM_MODEL)

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

def split_slides(plan_text):
    slides = re.split(r"(?=Slide\s+\d+:)", plan_text)
    return [s.strip() for s in slides if s.strip()]

# ---------------- LLM notion extractor ----------------
def extract_essential_notions(slide_text):
    prompt = f"""
You are analyzing a cold-atom physics journal club slide.

Slide text:
{slide_text[:4000]}

Task:
1. Extract ONLY the scientific notions that are *essential to understand this slide*. 
   - If a notion appears but is obvious or does not need explanation, ignore it.
2. Notion have to be very specific even inside the field of cold atoms 
3. Each notion should be concise (1–5 words), specialized physics concepts.
4. Include MAX 3 notions per slide.
5. Output ONLY the list of notions, no explanations or extra text.

Example output:
Recoil gain
Dressed cavity
Cavity QED
"""
    return llm.invoke(prompt).strip()

# ---------------- Global cleaning ----------------
def clean_global_notions(raw_text):
    prompt = f"""
You are reviewing a list of extracted scientific notions from a cold-atom physics article.

List:
{raw_text}

Task:
1. Keep ONLY notions that are *essential* to understanding the work.
2. Remove:
   - Generic terms ("system", "setup", "measurement")
   - Overly specific terms ("Gain in 88Sr atoms", "Frequency difference 50 kHz")
   - Duplicates
3. Return a clean list, one notion per line, nothing else.
"""
    return llm.invoke(prompt).strip()

# ---------------- Main pipeline ----------------
def main():
    print("🔍 Extracting strict essential notions per slide...")

    plan_text = read_file(PLAN_PATH)
    if not plan_text:
        print("❌ No plan_final.txt found.")
        return

    slides = split_slides(plan_text)
    notions_per_slide = {}
    global_notions = set()

    for i, slide in enumerate(slides, start=1):
        print(f"➡️ Processing Slide {i}/{len(slides)}...")
        notions_text = extract_essential_notions(slide)

        cleaned_lines = []
        for line in notions_text.splitlines():
            line = line.strip("-•* ").strip()
            if not line:
                continue
            # Remove generic / meta lines
            if any(k in line.lower() for k in [
                "slide", "here", "note", "new slide", "central to", "provided",
                "based on", "core", "scientific", "identified", "analysis",
                "content available", "presentation", "work", "not provided"
            ]):
                continue
            # remove too long or overly specific
            if len(line.split()) > 3 or any(substr in line.lower() for substr in [
                "gain in", "frequency difference", "maximum gain", "specific atomic", "implied by"
            ]):
                continue
            cleaned_lines.append(line)

        notions_list = sorted(set(cleaned_lines))
        notions_per_slide[f"Slide {i}"] = notions_list
        global_notions.update(notions_list)

        print(f"   → {len(notions_list)} essential notions kept for Slide {i}")

    # --- Slide-by-slide output ---
    notion_slide_text = ""
    for slide, notions in notions_per_slide.items():
        notion_slide_text += f"{slide}:\n"
        if notions:
            for n in notions:
                notion_slide_text += f"- {n}\n"
        else:
            notion_slide_text += "(No specific notions)\n"
        notion_slide_text += "\n"

    write_file(NOTIONS_SLIDE_PATH, notion_slide_text)
    write_file(NOTIONS_GLOBAL_PATH, "\n".join(sorted(global_notions)))

    print("🧹 Cleaning global notions for strict physics relevance...")
    raw_global = read_file(NOTIONS_GLOBAL_PATH)
    if raw_global.strip():
        cleaned_global = clean_global_notions(raw_global)
        write_file(NOTIONS_CLEAN_PATH, cleaned_global)
        before_count = len(raw_global.splitlines())
        after_count = len(cleaned_global.splitlines())
        print(f"✅ Cleaned notions saved: {NOTIONS_CLEAN_PATH} ({after_count}/{before_count} kept)")
    else:
        print("⚠️ notions.txt empty — skipping cleaning.")

    print(f"✅ Slide-by-slide notions saved: {NOTIONS_SLIDE_PATH}")
    print(f"✅ Global raw notions saved: {NOTIONS_GLOBAL_PATH}")


if __name__ == "__main__":
    main()
