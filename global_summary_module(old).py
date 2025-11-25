#!/usr/bin/env python3
"""
global_summary_module.py

Generate a full global summary + presentation plan from the outputs of chunksummary_module.

Usage:
    from global_summary_module import generate_global_and_plan

    generate_global_and_plan(summaries_dir="path/to/summaries", output_dir="path/to/out")
"""

import json
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseLanguageModel


# --------------------------------------------------------------------------- #
# Helper I/O
# --------------------------------------------------------------------------- #
def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Load summaries
# --------------------------------------------------------------------------- #
def _load_main_chunk_summaries(main_chunks_dir: Path):
    if not main_chunks_dir.exists():
        return []
    files = sorted(
        f for f in main_chunks_dir.glob("*_summary.txt")
        if f.is_file()
    )
    summaries = []
    for f in files:
        text = _read_text_file(f).strip()
        if text:
            summaries.append(text)
    return summaries


def _load_figure_summaries(fig_json_path: Path) -> dict:
    if not fig_json_path.exists():
        print(f"Warning: Figure summaries not found: {fig_json_path}")
        return {}
    try:
        return json.loads(fig_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading figure summaries: {e}")
        return {}


# --------------------------------------------------------------------------- #
# Core generation functions
# --------------------------------------------------------------------------- #
def _generate_global_summary(main_summaries: list[str], llm: BaseLanguageModel) -> str:
    print("Generating global paper summary...")
    combined = "\n\n".join(main_summaries)

    prompt = f"""
You are a scientific summarizer writing a comprehensive article summary for a journal club audience familiar with cold atoms.

Main text chunk summaries:
\"\"\"{combined[:18000]}\"\"\"

Task:
- Write a long, flowing, publication-quality global summary of the entire paper.
- Naturally integrate figure and subfigure references exactly as they appear in the text.
- Highlight connections between theory and experimental results.
- Do not mention supplementary material.
- Use precise but readable language for experts in ultracold atoms.

Output only the final summary text.
"""

    return llm.invoke(prompt).strip()


def _generate_presentation_plan(global_summary: str, figure_summaries: dict, llm: BaseLanguageModel) -> str:
    print("Generating journal club presentation plan...")

    # Build clean figure overview
    fig_lines = []
    for fig_id, info in figure_summaries.items():
        summary = info.get("summary", "(no summary)")
        label = info.get("article_label", fig_id.replace("fig_", ""))
        fig_lines.append(f"Figure {label}: {summary}")

    figures_overview = "\n".join(fig_lines) if fig_lines else "No figure summaries available."

    prompt = f"""
You are an expert physicist preparing a 20-minute journal club presentation on a cold-atom paper.

Global paper summary:
\"\"\"{global_summary[:18000]}\"\"\"

Figure summaries:
\"\"\"{figures_overview[:8000]}\"\"\"

Task:
Create a clear slide-by-slide presentation plan with:
- Slide title
- 2–3 bullet points or short description
- Which figure(s) or panel(s) to show (if any)

Structure:
1. Title + context
2. Key theoretical background
3. Experimental setup
4. Main results (multiple slides if needed — cover all important figures)
5. Discussion / implications
6. Conclusion

Ensure every major figure appears on at least one slide.
Highlight theory–experiment links.

Output only the structured presentation plan.
"""

    return llm.invoke(prompt).strip()


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #
def generate_global_and_plan(
    summaries_dir: str,
    output_dir: Optional[str] = None,        # ← Fixed: Optional[str]
    *,
    llm: Optional[BaseLanguageModel] = None,
    model: str = "llama3.1:latest"
) -> None:
    """
    Generate a full global summary and a journal club presentation plan.

    Parameters
    ----------
    summaries_dir : str
        Directory containing the output from chunksummary_module
    output_dir : str or None, optional
        Where to save Bugsummary.txt and plan.txt (default: inside summaries_dir)
    llm : BaseLanguageModel, optional
        Any LangChain LLM instance
    model : str
        Ollama model (only used if llm is None)
    """
    summaries_path = Path(summaries_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve() if output_dir else summaries_path

    # Resolve LLM
    if llm is None:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=model)
            print(f"Using Ollama model: {model}")
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. Run: pip install langchain-ollama"
            )
    else:
        model_name = getattr(llm, "model", "unknown")
        print(f"Using provided LLM: {llm.__class__.__name__} ({model_name})")

    main_chunks_dir = summaries_path / "main_chunks"
    fig_json = summaries_path / "figure_summaries.json"

    main_summaries = _load_main_chunk_summaries(main_chunks_dir)
    if not main_summaries:
        print("No main chunk summaries found. Did you run chunksummary_module first?")
        return

    figure_summaries = _load_figure_summaries(fig_json)

    global_summary = _generate_global_summary(main_summaries, llm)
    plan_text = _generate_presentation_plan(global_summary, figure_summaries, llm)

    _write_text_file(out_path / "Bugsummary.txt", global_summary)
    _write_text_file(out_path / "plan.txt", plan_text)

    print("Global summary and presentation plan generated!")
    print(f"   Bugsummary.txt → {out_path / 'Bugsummary.txt'}")
    print(f"   plan.txt       → {out_path / 'plan.txt'}")


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (2, 3):
        print("Usage: python global_summary_module.py <summaries_dir> [output_dir]")
        sys.exit(1)
    summaries_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None
    generate_global_and_plan(summaries_dir, output_dir)