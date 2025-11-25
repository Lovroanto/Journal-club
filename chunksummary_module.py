#!/usr/bin/env python3
"""
chunksummary_module.py
Pure module – import and call:
    from chunksummary_module import run_chunk_summary
    run_chunk_summary(input_dir="/path/to/input", output_dir="/path/to/out")
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ← NEW: LangChain base class + Optional
from langchain_core.language_models import BaseLanguageModel

# Keep Ollama only as fallback
# from langchain_ollama import OllamaLLM   # ← now imported only when needed


# --------------------------------------------------------------------------- #
# Helper: parse figure-metadata line that lives at the end of every chunk
# --------------------------------------------------------------------------- #
def _extract_figures_from_chunk(chunk_text: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Return (clean_text, {fig_id: {'article_label': ..., 'subpanels': [...]}})."""
    lines = chunk_text.splitlines(keepends=True)
    meta_start = -1
    for i in range(len(lines) - 1, max(-1, len(lines) - 20), -1):
        if "Figures used in this chunk" in lines[i]:
            meta_start = i
            break
    if meta_start == -1:
        return chunk_text.strip() + "\n", {}

    clean = "".join(lines[:meta_start]).strip() + "\n"
    meta = " ".join(l.strip() for l in lines[meta_start:] if l.strip())
    mentioned: Dict[str, Dict[str, Any]] = {}
    for m in re.finditer(r"(#fig_\d+)", meta):
        fig_raw = m.group(1)
        fig_id = fig_raw.lstrip("#")  # → fig_0, fig_1 …
        after = meta[m.end(): m.end() + 40]
        label_match = re.search(r":\s*([^\s,]+)", after)
        if not label_match:
            continue
        article_label = label_match.group(1).strip()
        panels = set(re.findall(r"[a-z]", after.lower()))
        mentioned[fig_id] = {"article_label": article_label, "subpanels": panels}
    return clean, mentioned


# --------------------------------------------------------------------------- #
# Load chunks (main or supplementary)
# --------------------------------------------------------------------------- #
def _load_chunks_with_figures(chunk_dir: Path) -> List[Tuple[str, Dict[str, Dict[str, Any]]]]:
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Directory not found: {chunk_dir}")
    files = sorted(
        chunk_dir.glob("*_chunk*.txt"),
        key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)],
    )
    result = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        clean, figs = _extract_figures_from_chunk(raw)
        result.append((clean, figs))
    return result


# --------------------------------------------------------------------------- #
# Figure caption handling
# --------------------------------------------------------------------------- #
def _load_figure_captions(figures_dir: Path) -> Dict[str, str]:
    caps: Dict[str, str] = {}
    if figures_dir.exists():
        for f in sorted(figures_dir.glob("fig_*.txt")):
            caps[f.stem] = f.read_text(encoding="utf-8")
    return caps


def _build_figure_notes(mentioned: Dict[str, Dict[str, Any]],
                        figure_summaries: Dict[str, Any]) -> str:
    if not mentioned:
        return "No figures mentioned in this chunk."

    lines = []
    for fig_id, data in mentioned.items():
        info = figure_summaries.get(fig_id, {})
        summary = info.get("summary", "(no summary available)")
        art = data["article_label"]
        lines.append(f"Figure {art}: {summary}")

    return "\n".join(lines)


def summarize_figures(figures_dir: Path, out_dir: Path, llm: BaseLanguageModel) -> Dict[str, Any]:
    """
    Create one clean, readable summary per figure.
    Detects multi-panel figures automatically and lists panels clearly.
    """
    print("STEP 1 — Summarizing figures (clean global summaries)")
    caps = _load_figure_captions(figures_dir)
    summaries: Dict[str, Any] = {}
    txt_dir = out_dir / "figure_summaries_texts"
    txt_dir.mkdir(parents=True, exist_ok=True)

    for fid, caption in caps.items():
        if not caption.strip():
            continue
        print(f" → {fid}")

        # Clean caption a bit
        caption = caption.strip()
        if len(caption) > 5000:
            caption = caption[:5000] + " [truncated]"

        # Main prompt — smart, reliable, and gives exactly what we want
        prompt = f"""
You are summarizing a scientific figure caption for quick understanding.

Caption:
\"\"\"{caption}\"\"\"

Instructions:
- Write ONE concise summary (2–4 sentences max).
- If the figure has multiple panels (a, b, c, ... or A, B, C, ...), explicitly list what each panel shows.
  Example: "Figure 5 (multi-panel): (a) shows the experimental setup; (b) presents results for condition X; (c) compares with controls."
- If it's a single-panel figure, just summarize what it shows.
- At the end, add one short sentence about the overall purpose of the figure.
- Use clear, plain language. No fluff.

Summary:"""

        response = llm.invoke(prompt).strip()

        # Store both raw and structured
        summaries[fid] = {
            "type": "global",
            "summary": response,
            "original_caption_length": len(caption)
        }

        # Pretty filename
        base = fid.replace("fig_", "Figure ")
        output_file = txt_dir / f"{base}.txt"
        output_file.write_text(response, encoding="utf-8")

        # Also save to a nice combined file later if needed
        print(f"     → {response.split('.')[0][:70]}...")

    # Save full index
    (out_dir / "figure_summaries.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summaries


# --------------------------------------------------------------------------- #
# Supplementary – short per-section + global summary
# --------------------------------------------------------------------------- #
def _is_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 150:
        return False
    if line.isupper():
        return True
    if re.match(r"^\d+\.", line) or "supplementary" in line.lower() or "method" in line.lower():
        return True
    return False


def summarize_supplementary(supp_dir: Path, out_dir: Path, llm: BaseLanguageModel,
                            figure_summaries: Dict[str, Any]) -> str:
    print("STEP 2 — Summarizing supplementary (per-section + global)")
    if not supp_dir.exists():
        print(" No supplementary folder – skipping.")
        return "No supplementary material."
    chunks = _load_chunks_with_figures(supp_dir)
    if not chunks:
        print(" No supplementary chunks found.")
        return "No supplementary material."

    chunk_out = out_dir / "supplementary_chunks"
    chunk_out.mkdir(parents=True, exist_ok=True)

    sections = []
    cur_title = "Supplementary Information"
    cur_chunk_sums: List[str] = []
    prev_context = "START OF SUPPLEMENTARY"

    for idx, (text, figs) in enumerate(chunks, 1):
        fig_notes = _build_figure_notes(figs, figure_summaries)
        ctx_prompt = (
            f"Previous context: {prev_context}\n"
            f"FIGURES: {fig_notes}\n"
            f"CHUNK START:\n{text[:1000]}\n"
            f"In 2–3 sentences say where we are and how figures contribute."
        )
        context = llm.invoke(ctx_prompt).strip()

        sum_prompt = (
            f"Summarize this supplementary chunk in 3–5 sentences.\n"
            f"EXPLICITLY reference any mentioned figures/panels (e.g. \"Figure S3 (panel b) shows …\").\n"
            f"FIGURE INFO:\n{fig_notes}\n"
            f"CONTEXT: {context}\n"
            f"CHUNK:\n{text[:4000]}"
        )
        summary = llm.invoke(sum_prompt).strip()
        (chunk_out / f"sup_chunk_{idx:03d}_summary.txt").write_text(summary, encoding="utf-8")
        cur_chunk_sums.append(summary)
        prev_context = summary

        first_line = text.splitlines()[0].strip()
        if _is_header(first_line):
            if cur_chunk_sums:
                short = llm.invoke(
                    f"Summarize this supplementary section in ONE very short sentence.\n"
                    f"{' '.join(cur_chunk_sums)}"
                ).strip()
                sections.append({"title": cur_title, "short": short})
                cur_chunk_sums = [summary]
            cur_title = first_line

    if cur_chunk_sums:
        short = llm.invoke(
            f"Summarize this supplementary section in ONE very short sentence.\n"
            f"{' '.join(cur_chunk_sums)}"
        ).strip()
        sections.append({"title": cur_title, "short": short})

    global_prompt = (
        f"Summarize the entire supplementary material in 2–4 sentences.\n"
        + "\n".join([s["short"] for s in sections])
    )
    global_sum = llm.invoke(global_prompt).strip()
    (out_dir / "supplementary_global.txt").write_text(global_sum, encoding="utf-8")
    (out_dir / "supplementary_sections.json").write_text(
        json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return global_sum


# --------------------------------------------------------------------------- #
# Main text
# --------------------------------------------------------------------------- #
def summarize_main(main_dir: Path, out_dir: Path, llm: BaseLanguageModel,
                   figure_summaries: Dict[str, Any], supp_global: str) -> None:
    print("STEP 3 — Summarizing main text")
    if not main_dir.exists():
        print(" No main folder – skipping.")
        return
    chunks = _load_chunks_with_figures(main_dir)
    if not chunks:
        print(" No main chunks.")
        return

    main_out = out_dir / "main_chunks"
    ctx_out = out_dir / "main_contexts"
    add_out = out_dir / "main_additional"
    for d in (main_out, ctx_out, add_out):
        d.mkdir(parents=True, exist_ok=True)

    prev_context = "START OF PAPER"
    sections_path = out_dir / "supplementary_sections.json"
    section_titles = []
    if sections_path.exists():
        section_titles = [s["title"] for s in json.loads(sections_path.read_text(encoding="utf-8"))]

    for idx, (text, figs) in enumerate(chunks, 1):
        fig_notes = _build_figure_notes(figs, figure_summaries)
        ctx_prompt = (
            f"Previous chunk summary: {prev_context}\n"
            f"Supplementary overview: {supp_global}\n"
            f"FIGURES IN THIS CHUNK:\n{fig_notes}\n"
            f"CHUNK EXCERPT:\n{text[:1500]}\n"
            f"In 2–3 sentences describe where we are and the role of any figures."
        )
        context = llm.invoke(ctx_prompt).strip()

        sum_prompt = (
            f"Summarize this main-text chunk in 6–10 sentences.\n"
            f"EXPLICITLY reference any mentioned figures/panels (e.g. \"Figure 4 (panel c) illustrates …\").\n"
            f"SUPPLEMENTARY OVERVIEW: {supp_global}\n"
            f"FIGURE INFO:\n{fig_notes}\n"
            f"CONTEXT: {context}\n"
            f"CHUNK:\n{text[:4000]}"
        )
        summary = llm.invoke(sum_prompt).strip()
        (main_out / f"main_chunk_{idx:03d}_summary.txt").write_text(summary, encoding="utf-8")
        (ctx_out / f"main_chunk_{idx:03d}_context.txt").write_text(
            f"FIGURES:\n{fig_notes}\n\nCONTEXT:\n{context}", encoding="utf-8"
        )

        link_prompt = (
            f"Given the main chunk summary, list any supplementary section titles that provide more detail.\n"
            f"Answer with a comma-separated list or \"None\".\n"
            f"SUMMARY:\n{summary}\n"
            f"SECTION TITLES:\n{', '.join(section_titles) if section_titles else 'None'}"
        )
        linked = llm.invoke(link_prompt).strip()
        (add_out / f"main_chunk_{idx:03d}_additional.txt").write_text(linked or "None", encoding="utf-8")
        prev_context = summary
    print(f" Processed main chunk {idx}/{len(chunks)}")


# --------------------------------------------------------------------------- #
# Public entry point – NOW FULLY FLEXIBLE
# --------------------------------------------------------------------------- #
def run_chunk_summary(
    input_dir: str,
    output_dir: str,
    *,
    llm: Optional[BaseLanguageModel] = None,
    model: str = "llama3.1:latest"   # only used if llm is None
) -> None:
    """
    Run the whole summarisation pipeline.

    Parameters
    ----------
    input_dir : str
        Path that contains ``main/``, ``supplementary/``, ``figures/`` etc.
    output_dir : str
        Where all summaries will be written (created if missing).
    llm : BaseLanguageModel, optional
        Any LangChain LLM instance (OpenAI, Groq, Anthropic, local, etc.).
        If None → falls back to Ollama with the given `model`.
    model : str, optional
        Ollama model name (default: "llama3.1:latest") used only when llm is None.
    """
    in_path = Path(input_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # === Resolve LLM ===
    if llm is None:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=model)
            print(f"Using default Ollama model: {model}")
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. Install it (`pip install langchain-ollama`) "
                "or pass your own LLM instance via the `llm` parameter."
            )
    else:
        print(f"Using user-provided LLM: {llm.__class__.__name__} ({getattr(llm, 'model', 'unknown')})")

    figures_dir = in_path / "figures"
    main_dir = in_path / "main"
    supp_dir = in_path / "supplementary"

    print("=== Starting chunk-summary pipeline ===")
    fig_summaries = summarize_figures(figures_dir, out_path, llm)
    supp_global = summarize_supplementary(supp_dir, out_path, llm, fig_summaries)
    summarize_main(main_dir, out_path, llm, fig_summaries, supp_global)
    print("=== Pipeline finished ===")
    print(f"All outputs → {out_path}")


# --------------------------------------------------------------------------- #
# Tiny demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        run_chunk_summary(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python chunksummary_module.py <input_dir> <output_dir>")