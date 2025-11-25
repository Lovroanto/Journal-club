#!/usr/bin/env python3
"""
global_summary_module_v2.py

Advanced journal-club summary generator with:
- Smart main/supplementary boundary detection
- Three-level main summary (structure → flow → contextualized)
- Fully controllable presentation plan (slide count)
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from langchain_core.language_models import BaseLanguageModel


# ==================== I/O Helpers ====================
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""

def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ==================== Load everything ====================
def _load_main_summaries(main_dir: Path) -> List[str]:
    files = sorted(main_dir.glob("*_summary.txt"))
    return [_read(f) for f in files if _read(f)]

def _load_supplementary_global(summaries_path: Path) -> str:
    return _read(summaries_path / "supplementary_global.txt")

def _load_figure_summaries(fig_path: Path) -> Dict[str, dict]:
    if not fig_path.exists():
        return {}
    return json.loads(fig_path.read_text(encoding="utf-8"))


# ==================== Smart conclusion detection ====================
def _detect_conclusion_chunk(summaries: List[str]) -> Tuple[int, str]:
    conclusion_keywords = [
        "in conclusion", "we conclude", "to summarize", "overall", "in summary",
        "this work demonstrates", "we have shown", "our results indicate"
    ]
    for i, text in enumerate(summaries):
        lowered = text.lower()
        if any(kw in lowered for kw in conclusion_keywords):
            return i, text
    return -1, ""

# ==================== Core generation functions ====================
def _generate_structure_bullets(main_summaries: List[str], llm: BaseLanguageModel) -> Tuple[str, List[Dict]]:
    print("→ Generating structured bullet overview (001::: ...)")
    combined = "\n\n".join(f"Chunk {i+1}:\n{s}" for i, s in enumerate(main_summaries))

    prompt = f"""
Examine these main-text chunk summaries and produce a concise hierarchical bullet list of what the paper actually does.

Use this exact format:
001::: Introduction and background
002::: Theoretical model
003::: Experimental methods
...

Rules:
- One bullet per logical section
- Use three-digit numbers followed by ::: 
- Be extremely concise but never omit something essential
- If conclusion is present, include it as the last item

Chunks:
\"\"\"{combined[:18000]}\"\"\"
"""
    response = llm.invoke(prompt).strip()

    sections = []
    for line in response.splitlines():
        if m := re.match(r"(\d{3}):::\s*(.+)", line.strip()):
            sections.append({"id": m.group(1), "title": m.group(2).strip()})

    return response, sections


def _generate_flow_summary(main_summaries: List[str], llm: BaseLanguageModel) -> str:
    print("→ Generating classic flowing summary")
    combined = "\n\n".join(main_summaries)
    prompt = f"""
Write a long, natural, publication-style summary of the entire paper.
Integrate figure references naturally.
Highlight theory–experiment links.
Do not mention supplementary material.
\"\"\"{combined[:18000]}\"\"\"
"""
    return llm.invoke(prompt).strip()


def _generate_contextualized_summary(structure: str, flow: str, supp_global: str, llm: BaseLanguageModel) -> str:
    print("→ Generating final contextualized summary (the one to read aloud)")
    supp_part = f"\n\nAdditional key insights (integrate silently):\n{supp_global}" if supp_global else ""

    prompt = f"""
You are writing the definitive journal-club summary that people will actually remember.

Logical structure of the paper:
{structure}

Previous flowing summary:
\"\"\"{flow[:12000]}\"\"\"{supp_part}

Now write the final version:
- Start with the big picture and motivation
- Clearly separate introduction → methods → results → conclusion
- Weave in supplementary insights without saying "supplementary"
- End with implications
- Use engaging, precise, expert-level language
"""
    return llm.invoke(prompt).strip()


def _generate_presentation_plan(contextualized: str,
                                figure_summaries: dict,
                                desired_slides: int,
                                llm: BaseLanguageModel) -> str:
    print(f"→ Generating presentation plan (target ~{desired_slides or 'auto'} slides)")

    fig_lines = []
    for fid, data in figure_summaries.items():
        summ = data.get("summary", "(no summary)")
        label = data.get("article_label", fid.replace("fig_", ""))
        fig_lines.append(f"Figure {label}: {summ}")
    figures = "\n".join(fig_lines) if fig_lines else "No figures."

    target = "Decide the optimal number of slides yourself (typically 12–18 for 20 min)." if desired_slides == 0 else f"Target {desired_slides} slides (±20%)."

    prompt = f"""
Create a complete journal-club presentation plan (20 min talk).

Paper summary:
\"\"\"{contextualized[:18000]}\"\"\"

Figures:
\"\"\"{figures[:10000]}\"\"\"

Instructions:
{target}
- One section per slide with clear title
- Suggest exact figure/panel to display
- Use format: # Slide X: Title
- Cover: context → theory → methods → results → discussion → conclusion
- Highlight theory–experiment connections
- Include all important figures

Output only the slide plan.
"""
    return llm.invoke(prompt).strip()


# ==================== Public function ====================
def generate_global_and_plan(
    summaries_dir: str,
    output_dir: Optional[str] = None,
    *,
    llm: Optional[BaseLanguageModel] = None,
    model: str = "llama3.1:latest",
    desired_slides: int = 0          # 0 = let AI decide
) -> None:
    """
    Advanced journal-club summary + presentation generator (v2)
    """
    base = Path(summaries_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else base

    # LLM
    if llm is None:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model=model)

    main_dir = base / "main_chunks"
    main_summaries = _load_main_summaries(main_dir)
    if not main_summaries:
        print("No main summaries found!")
        return

    supp_global = _load_supplementary_global(base)
    fig_summaries = _load_figure_summaries(base / "figure_summaries.json")

    # Smart detection
    concl_idx, concl_text = _detect_conclusion_chunk(main_summaries)
    if 0 < concl_idx < len(main_summaries) - 2:
        print(f"Warning: Conclusion found early (chunk {concl_idx+1}). Consider re-splitting.")

    # Generate everything
    structure_text, sections = _generate_structure_bullets(main_summaries, llm)
    flow_summary = _generate_flow_summary(main_summaries, llm)
    final_summary = _generate_contextualized_summary(structure_text, flow_summary, supp_global, llm)
    plan = _generate_presentation_plan(final_summary, fig_summaries, desired_slides, llm)

    # Save all
    _write(out / "01_main_structure.txt", structure_text)
    _write(out / "02_main_flow_summary.txt", flow_summary)
    _write(out / "03_main_contextualized.txt", final_summary)
    _write(out / "04_presentation_plan.txt", plan)
    _write(out / "main_sections.json", json.dumps(sections, indent=2))

    print("All summaries generated!")
    print(f"   → {out}/03_main_contextualized.txt  (read this one at journal club)")
    print(f"   → {out}/04_presentation_plan.txt    (~{desired_slides or 'auto'} slides)")


# ==================== Demo ====================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python global_summary_module_v2.py <summaries_dir> [output_dir] [slides]")
        sys.exit(1)
    slides = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    generate_global_and_plan(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, desired_slides=slides)