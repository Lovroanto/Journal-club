#!/usr/bin/env python3
"""
slide_enrichment_module.py
Takes a presentation plan (plan.txt or 04_presentation_plan.txt) and enriches it
with supplementary material + global context → produces perfect final slides.
Usage:
    from slide_enrichment_module import enrich_presentation_plan
    enrich_presentation_plan(
        summaries_dir="path/to/summaries",
        plan_path="path/to/04_presentation_plan.txt", # or any plan
        output_dir="path/to/final_presentation",
        llm=llm # optional, any LangChain LLM
    )
"""
import re
import json
from pathlib import Path
from typing import Optional, Union  # ← Added Union for older Python

from langchain_core.language_models import BaseLanguageModel


# ==================== Helpers ====================
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _split_slides(plan_text: str):
    """Split plan into individual slides using 'Slide X:' or '# Slide X:'"""
    pattern = r"(?:^|\n)#?\s*Slide\s+\d+[:\.]"
    parts = re.split(pattern, plan_text)
    if len(parts) < 2:
        return [plan_text.strip()]

    slides = []
    for i in range(1, len(parts)):
        header = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        slides.append((header + "\n" + content).strip())
    return slides


def _load_supplementary(summaries_dir: Path) -> dict:
    supp_path = summaries_dir / "supplementary_sections.json"  # from chunksummary_module
    if supp_path.exists():
        try:
            return json.loads(supp_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Fallback: try old format
    old_path = summaries_dir / "supplementary_summary.json"
    if old_path.exists():
        return json.loads(old_path.read_text(encoding="utf-8"))

    print("Warning: No supplementary summary found.")
    return {}


# ==================== Core: Enrichment + Finalization ====================
def _enrich_slide(slide_text: str, supp_summaries: dict, llm: BaseLanguageModel) -> str:
    prompt = f"""
You are enhancing a journal club slide with relevant supplementary material.
Current slide:
\"\"\"{slide_text[:5000]}\"\"\"
Supplementary sections (use only if truly relevant):
{json.dumps(supp_summaries, indent=2)[:10000]}
Task:
- If any supplementary section adds important detail, method, control, or extended data → write 1–3 concise sentences to enrich the slide.
- If the supplementary content is substantial and deserves its own slide → suggest a NEW slide with title and content.
- If nothing is relevant → respond exactly: "NO_ENRICHMENT"
Output only the enrichment text or new slide (or NO_ENRICHMENT).
"""
    return llm.invoke(prompt).strip()


def _finalize_slide(original: str, enrichment: str, global_summary: str, llm: BaseLanguageModel) -> str:
    if enrichment.upper() == "NO_ENRICHMENT":
        return original

    prompt = f"""
You are finalizing a journal club slide.
Original slide:
\"\"\"{original[:5000]}\"\"\"
Proposed enrichment / new slide:
\"\"\"{enrichment[:5000]}\"\"\"
Global paper summary (for context):
\"\"\"{global_summary[:15000]}\"\"\"
Task:
- Seamlessly integrate the enrichment into the original slide, OR
- If a new slide was suggested, output it as a separate slide.
- Keep everything concise, clear, and presentation-ready.
- Preserve figure references.
Output only the final slide(s) — ready to be shown.
"""
    return llm.invoke(prompt).strip()


# ==================== Public function ====================
def enrich_presentation_plan(
    summaries_dir: str,
    plan_path: Union[str, Path],        # ← Fixed for Python <3.10
    output_dir: Union[str, Path],        # ← Fixed
    *,
    llm: Optional[BaseLanguageModel] = None,
    model: str = "llama3.1:latest"
) -> None:
    """
    Enrich an existing presentation plan with supplementary material.
    Parameters
    ----------
    summaries_dir : str
        Directory containing: supplementary_sections.json, 03_main_contextualized.txt or Bugsummary.txt
    plan_path : str or Path
        Path to the existing plan (e.g. 04_presentation_plan.txt)
    output_dir : str or Path
        Where to save: slides_final/slide_*.txt + plan_final.txt
    llm : BaseLanguageModel, optional
        Any LangChain LLM
    model : str
        Ollama model name if llm is None
    """
    base = Path(summaries_dir).expanduser().resolve()
    plan_path = Path(plan_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()

    slides_dir = out_dir / "slides_final"
    final_plan_path = out_dir / "plan_final.txt"

    # Resolve LLM
    if llm is None:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=model)
            print(f"Using Ollama: {model}")
        except ImportError:
            raise ImportError("Please install langchain-ollama: pip install langchain-ollama")

    # Load inputs
    plan_text = _read(plan_path)
    if not plan_text.strip():
        print("No plan found at", plan_path)
        return

    supp_summaries = _load_supplementary(base)

    # Prefer 03_main_contextualized.txt, fallback to Bugsummary.txt
    global_summary_path = base / "03_main_contextualized.txt"
    if not global_summary_path.exists():
        global_summary_path = base / "Bugsummary.txt"
    global_summary = _read(global_summary_path)

    print(f"Loaded plan with {len(_split_slides(plan_text))} slides → enriching with supplementary...")

    # Process each slide
    slides = _split_slides(plan_text)
    final_slides = []

    for idx, slide in enumerate(slides, start=1):
        print(f" → Enriching slide {idx}/{len(slides)}")
        enrichment = _enrich_slide(slide, supp_summaries, llm)
        finalized = _finalize_slide(slide, enrichment, global_summary, llm)

        # Save individual slide
        slide_file = slides_dir / f"slide_{idx:03d}_final.txt"
        _write(slide_file, finalized)
        final_slides.append(finalized)

    # Save combined plan
    combined = "\n\n---\n\n".join(final_slides)
    _write(final_plan_path, combined)

    print(f"\nDone! Final presentation saved:")
    print(f" Individual slides → {slides_dir}")
    print(f" Combined plan → {final_plan_path}")


# ==================== Demo ====================
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python slide_enrichment_module.py <summaries_dir> <plan_path> <output_dir>")
        sys.exit(1)
    enrich_presentation_plan(sys.argv[1], sys.argv[2], sys.argv[3])