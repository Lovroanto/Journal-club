#!/usr/bin/env python3
"""
global_summary_module_v4.py
Upgraded global-summary + presentation planner.
- Internal outline reasoning step to produce coherent slide groups
- Cumulative factual bullet harvesting (chunk-by-chunk)
- Cumulative flow summary (chunk-by-chunk)
- Refined contextualized narrative (refine-chain)
- High-level group plan produced from an internal outline (hidden)
- Detailed group content generation, RAG-friendly hooks
"""

from __future__ import annotations
import json
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# langchain-ish imports: keep same abstractions you used before
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # fallback stub splitter if not available at runtime (helps linting)
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=2000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text: str):
            if not text:
                return []
            # naive split
            chunks = []
            i = 0
            L = len(text)
            while i < L:
                chunks.append(text[i:i+self.chunk_size])
                i += self.chunk_size - self.chunk_overlap
            return chunks

try:
    from langchain.chains.summarize import load_summarize_chain
except Exception:
    # create a simple stub to avoid import errors when linting; in runtime you should have langchain.
    def load_summarize_chain(*args, **kwargs):
        raise ImportError("langchain.summarize.load_summarize_chain not available; install langchain.")

try:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseLanguageModel
    from langchain.prompts import PromptTemplate
except Exception:
    # stubs so module imports; runtime requires real langchain
    class Document:
        def __init__(self, page_content: str):
            self.page_content = page_content
    class BaseLanguageModel:
        def invoke(self, prompt: str) -> str:
            raise NotImplementedError("Provide a real LLM implementing invoke().")
    class PromptTemplate:
        def __init__(self, input_variables=None, template=""):
            self.input_variables = input_variables or []
            self.template = template
        @classmethod
        def from_template(cls, t: str):
            return cls(template=t)

# Logging
logger = logging.getLogger("global_summary_module_v4")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

# ---------------- I/O helpers ----------------
def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return ""

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

# ---------------- Load helpers ----------------
def _load_main_summaries(main_dir: Path) -> List[str]:
    if not main_dir.exists():
        return []
    files = sorted([f for f in main_dir.glob("*_summary.txt")])
    return [_read(f) for f in files if _read(f)]

def _load_supplementary_global(base: Path) -> str:
    return _read(base / "supplementary_global.txt")

def _load_figure_summaries(base: Path) -> Dict[str, dict]:
    p = base / "figure_summaries.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Unable to parse figure_summaries.json")
            return {}
    return {}

# ---------------- Conclusion detection ----------------
def _detect_conclusion_chunk(summaries: List[str]) -> Tuple[int, str]:
    conclusion_keywords = [
        "in conclusion", "we conclude", "to summarize", "overall", "in summary",
        "this work demonstrates", "we have shown", "our results indicate", "conclusions"
    ]
    for i, text in enumerate(summaries):
        low = text.lower()
        if any(kw in low for kw in conclusion_keywords):
            return i, text
    return -1, ""

# ---------------- 1. CUMULATIVE BULLET HARVEST ----------------
def _harvest_factual_bullets_cumulative(
    main_summaries: List[str], llm: BaseLanguageModel
) -> str:
    """
    Cumulative bullet harvesting. At each chunk, the LLM receives previously harvested bullets
    and the next chunk, and must return ONLY new bullets (prefixed with "• ").
    """
    logger.info("1. Harvesting factual bullets (cumulative)...")
    all_bullets: List[str] = []
    # guard: limit iterations if extremely large
    for idx, chunk in enumerate(main_summaries, start=1):
        prev_text = "\n".join(all_bullets) if all_bullets else "None yet."
        prompt = f"""
You are a fact-extraction assistant.
Previously extracted bullets:
\"\"\"{prev_text[:15000]}\"\"\"

Now examine ONLY the following NEW chunk (Chunk {idx}) and output NEW bullet points
that are not present in the previously extracted bullets.

Chunk {idx}:
\"\"\"{chunk[:10000]}\"\"\"

Rules:
- Output only lines that start with "• " (bullet symbol + space).
- Each bullet should be short (max 28 words) and factual (claim, method step, result, or figure description).
- Keep figure references verbatim.
- If a fact was already captured, do NOT repeat it.
- If nothing new, output exactly: NO_NEW_BULLETS
- Do NOT output any explanations, headers, or trailing text.

Now output the new bullets or NO_NEW_BULLETS.
"""
        try:
            response = llm.invoke(prompt).strip()
        except Exception as e:
            logger.error("LLM failed on bullet harvesting chunk %d: %s", idx, e)
            # continue gracefully
            response = "NO_NEW_BULLETS"
        if "NO_NEW_BULLETS" in response.upper():
            continue
        # collect lines beginning with bullet marker
        new_lines = [ln.strip() for ln in response.splitlines() if ln.strip().startswith("• ")]
        # remove duplicates (relative to all_bullets)
        for l in new_lines:
            if l not in all_bullets:
                all_bullets.append(l)
    final_text = "\n".join(all_bullets) if all_bullets else "No factual bullets found."
    return final_text

# ---------------- 2. CUMULATIVE FLOW SUMMARY ----------------
def _generate_flow_summary_cumulative(
    main_summaries: List[str], llm: BaseLanguageModel
) -> str:
    """
    Build a single publication-style narrative incrementally, chunk by chunk.
    Each LLM step receives the running summary and the next chunk; it returns 1-2 paragraphs to append.
    """
    logger.info("2. Building faithful flow summary (cumulative)...")
    running_summary = ""
    for idx, chunk in enumerate(main_summaries, start=1):
        prompt = f"""
You are maintaining a single, continuous, publication-style summary.

Current summary so far:
\"\"\"{running_summary[:15000]}\"\"\"

Now incorporate ONLY the following NEW chunk into the narrative.
Write 1 or 2 concise paragraphs that flow naturally from the current summary.
Preserve key claims, methods, results, and figure references exactly.

Chunk {idx}:
\"\"\"{chunk[:10000]}\"\"\"

Output ONLY the new paragraph(s), no headers or numbering.
"""
        try:
            addition = llm.invoke(prompt).strip()
        except Exception as e:
            logger.error("LLM failed during flow generation at chunk %d: %s", idx, e)
            addition = ""
        # append, ensure single newline separation
        if running_summary and addition:
            running_summary = running_summary.strip() + "\n\n" + addition.strip()
        elif addition:
            running_summary = addition.strip()
    return running_summary or "No flow summary generated."

# ---------------- 3. CONTEXTUALIZED/REFINED NARRATIVE ----------------
def _generate_contextualized_summary(
    bullets: str, flow: str, supp_global: str, llm: BaseLanguageModel
) -> str:
    """
    Use a refine-style chain to produce a single, logical, fact-focused narrative.
    This is the story read-first file.
    """
    logger.info("3. Generating contextualized narrative (refine chain)...")
    # split the flow into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    flow_chunks = splitter.split_text(flow)
    docs = [Document(page_content=c) for c in flow_chunks]
    if supp_global:
        docs.append(Document(page_content=supp_global))
    # Prepare prompts for the chain
    question_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize this text focusing strictly on key facts, motivations, and logical links:
{text}
FACT-FOCUSED SUMMARY:"""
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template="""Your current fact-focused summary:
{existing_answer}

Refine it by integrating the following new text. Connect facts logically: start with the motivation/open question,
explain the theoretical/experimental approach, link methods to results, and finish with implications. Prioritize accuracy.

New text:
{text}

REFINED FACTUAL SUMMARY:"""
    )
    try:
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=False,
            input_key="input_documents",
            output_key="output_text",
        )
        result = chain.invoke({"input_documents": docs})
        return result["output_text"].strip()
    except Exception as e:
        logger.warning("Refine chain unavailable or failed: %s. Falling back to concatenated flow.", e)
        # Fallback: return the flow plus bullets condensed
        short_bullets = "\n".join(bullets.splitlines()[:20])
        fallback = f"{flow.strip()}\n\nKey facts (condensed):\n{short_bullets}"
        return fallback.strip()

# ---------------- Helper: build figures overview ----------------
def _figures_overview_text(figure_summaries: Dict[str, dict]) -> str:
    lines = []
    for fid, data in sorted(figure_summaries.items()):
        label = data.get("article_label", fid.replace("fig_", ""))
        summary = data.get("summary", "") or data.get("text", "")
        lines.append(f"Figure {label}: {summary}")
    return "\n".join(lines) if lines else "No figures."

# ---------------- 4. HIGH-LEVEL GROUP PLAN (with INTERNAL OUTLINE) ----------------
def _generate_high_level_group_plan(
    final_summary: str,
    figure_summaries: dict,
    desired_slides: int,
    llm: BaseLanguageModel,
) -> str:
    """
    Produce a concise list of slide-group headers by first asking the LLM to create an
    internal outline & reasoning (hidden). Then extract only the header lines.
    """
    logger.info("4. Generating high-level group plan (internal outline step)...")
    figures_overview = _figures_overview_text(figure_summaries)
    num_figs = len(figure_summaries or {})
    # Target slide guidance
    if desired_slides and desired_slides > 0:
        target_info = f"You must aim for approximately {desired_slides} slides (±2)."
    else:
        target_info = "Choose the optimal number of slides for a 20-minute talk (recommended 12–18)."

    prompt = f"""
You are designing a pedagogical, scientific 20-minute presentation.

STEP 1 (INTERNAL OUTLINE): BEFORE producing final headers, create a concise internal outline (this will NOT be shown)
that explains:
 - the overall story arc (1-3 sentences)
 - the purpose of each major block
 - what belongs in each block and what MUST be deferred
 - how figures (there are {num_figs}) map to blocks
 - a rough per-block slide allocation estimate

Keep the internal outline short (max 10 bullet items). This is your private reasoning.

STEP 2 (FINAL OUTPUT): Based on that internal outline, produce ONLY the final slide group headers,
one per line, exactly in this format:

===SLIDE_GROUP: Title of the Group | Slides A–B===

Rules for final headers:
 - Slide ranges must be consecutive, start at 1, and not overlap.
 - Title must be concise (3–7 words), descriptive, and follow story logic.
 - Do not create more than 8 groups.
 - Do not output any other text besides the header lines.

Paper summary (fact-checked):
\"\"\"{final_summary[:24000]}\"\"\"

Figures overview:
\"\"\"{figures_overview[:4000]}\"\"\"

{target_info}

Now produce the INTERNAL_OUTLINE followed by the FINAL headers (internal outline may be longer but will be discarded).
"""
    try:
        result = llm.invoke(prompt).strip()
    except Exception as e:
        logger.error("LLM failed when creating high-level plan: %s", e)
        return "===SLIDE_GROUP: Introduction | Slides 1–2===\n===SLIDE_GROUP: Background | Slides 3–4===\n===SLIDE_GROUP: Methods | Slides 5–7===\n===SLIDE_GROUP: Results | Slides 8–12===\n===SLIDE_GROUP: Discussion & Conclusion | Slides 13–15==="

    # Extract only header lines
    headers = [ln.strip() for ln in result.splitlines() if ln.strip().startswith("===SLIDE_GROUP:")]
    if not headers:
        # fallback simple plan
        logger.warning("No headers found in LLM output; using fallback plan.")
        headers = [
            "===SLIDE_GROUP: Introduction | Slides 1–2===",
            "===SLIDE_GROUP: Background | Slides 3–4===",
            "===SLIDE_GROUP: Methods | Slides 5–7===",
            "===SLIDE_GROUP: Results | Slides 8–11===",
            "===SLIDE_GROUP: Discussion & Conclusion | Slides 12–14===",
        ]
    return "\n".join(headers)

# ---------------- Detailed group content generation ----------------
def _generate_group_details(
    group_header: str,
    final_summary: str,
    figures_overview: str,
    factual_bullets: str,
    previous_groups: List[str],
    llm: BaseLanguageModel,
) -> str:
    """
    Produce detailed content for one group, with:
    - internal build of slide-level focus
    - integration of factual bullets (so facts are not omitted)
    """
    # Extract title and slide range for contextual cues
    m = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|\s*Slides\s*([0-9]+)\s*[-–]\s*([0-9]+)\s*===', group_header)
    if m:
        title = m.group(1).strip()
        slide_from = int(m.group(2))
        slide_to = int(m.group(3))
    else:
        # fallback parsing
        title = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', group_header).group(1).strip() if re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', group_header) else "Group"
        slide_from, slide_to = 1, 1
    prev_titles = [re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', g).group(1).strip() for g in previous_groups if re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', g)]
    prev_titles_text = ", ".join(prev_titles) if prev_titles else "None"

    # Base prompt asks the LLM to produce:
    #  - learning goals for the group (what audience must learn)
    #  - per-slide focus lines (Slide X: ...)
    #  - which figures to show and why
    base_prompt = f"""
You will generate the detailed content for this slide group (strict output only; no extra commentary).

Group header: {group_header}
Title: {title}
Slides: {slide_from} to {slide_to}

Context:
Paper summary (full):
\"\"\"{final_summary[:24000]}\"\"\"

Figures overview:
\"\"\"{figures_overview[:4000]}\"\"\"

Facts available (factual bullets; do NOT invent facts):
\"\"\"{factual_bullets[:20000]}\"\"\"

Previous groups (do not repeat; build on them): {prev_titles_text}

Output (ONLY the following, separated by blank lines):
1) Group learning goals: 1-3 short bullet lines (start each with "- ")
2) Per-slide blueprint: For each slide in this group's range, a single line "Slide N: short focus statement"
3) Figures to include: lines "Include: Figure X - reason"
4) Short notes (1-3 sentences) on transitions from previous to this group

Be concise and factual. Prefer exact figure references and bullets; if something is missing, note "MISSING_FACTS" (no other commentary).
"""
    try:
        base_content = llm.invoke(base_prompt).strip()
    except Exception as e:
        logger.error("LLM failed while generating group '%s': %s", title, e)
        base_content = f"- Group learning goals: MISSING (LLM error)\n\nSlide {slide_from}: TBD\n\nInclude: None\n\nTransition: TBD"

    # Now refine with factual bullets (to fill in any missing facts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    bullet_chunks = splitter.split_text(factual_bullets or "")
    bullet_docs = [Document(page_content=c) for c in bullet_chunks] or [Document(page_content="")]

    refine_prompt = PromptTemplate.from_template("""
Your current group content:
\"\"\"{existing_answer}\"\"\"

Integrate missing facts from the text below. Ensure each factual bullet relevant to this group's slides appears.
Do not repeat content from previous groups. Keep concise.

Facts:
\"\"\"{text}\"\"\"

Figures reminder:
\"\"\"{figures_overview}\"\"\"

Output the refined content only.
""")
    try:
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=PromptTemplate.from_template("{text}"),
            refine_prompt=refine_prompt,
        )
        # the chain expects a list of documents starting with existing answer; emulate by placing base_content first
        docs_for_chain = [Document(page_content=base_content)] + bullet_docs
        result = chain.invoke({"input_documents": docs_for_chain, "figures_overview": figures_overview})
        refined = result.get("output_text", "").strip()
        return refined or base_content
    except Exception:
        # if refine chain is not available, merge info heuristically
        logger.debug("Refine chain unavailable; merging base content and bullets heuristically.")
        merged = base_content
        if factual_bullets:
            merged += "\n\nFacts included (sample):\n" + "\n".join(factual_bullets.splitlines()[:10])
        return merged

# ---------------- 5. Split/process plan into detailed files ----------------
def _process_and_split_groups(
    plan_text: str,
    output_dir: Path,
    final_summary: str,
    figures_overview: str,
    factual_bullets: str,
    llm: BaseLanguageModel,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = [line.strip() for line in plan_text.splitlines() if line.strip().startswith("===SLIDE_GROUP:")]
    if not headers:
        logger.warning("No slide groups found in plan text.")
        return
    previous_groups: List[str] = []
    for idx, header in enumerate(headers, start=1):
        logger.info(" → Generating details for group %d: %s", idx, header)
        details = _generate_group_details(
            header,
            final_summary,
            figures_overview,
            factual_bullets,
            previous_groups,
            llm
        )
        group_content = header + "\n\n" + details.strip()
        # Build filename-safe title
        title_match = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', header)
        title = title_match.group(1).strip() if title_match else f"Group_{idx}"
        safe_title = re.sub(r'[^\w\s\-]', '', title)
        safe_title = re.sub(r'\s+', '_', safe_title).strip('_')[:120] or f"Group_{idx}"
        filename = f"{str(idx).zfill(2)}_{safe_title}.txt"
        _write(output_dir / filename, group_content + "\n")
        previous_groups.append(header)
    logger.info("Created %d detailed group files in: %s", len(headers), output_dir)

# ---------------- PUBLIC API ----------------
def generate_global_and_plan(
    summaries_dir: str,
    output_dir: Optional[str] = None,
    slides_output_dir: Optional[str] = None,
    *,
    llm: Optional[BaseLanguageModel] = None,
    model: str = "llama3.1:latest",
    desired_slides: int = 0,  # 0 = auto
) -> None:
    """
    Main entrypoint. Reads summaries and figures from <summaries_dir> and writes outputs.
    - summaries_dir structure:
        main_chunks/   (contains *_summary.txt files)
        supplementary_global.txt
        figure_summaries.json
    - output_dir: where top-level outputs are written (defaults to summaries_dir)
    - slides_output_dir: if provided, detailed group files will be written there
    """
    base = Path(summaries_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else base

    # ---- LLM ----
    if llm is None:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=model)
            logger.info("Using Ollama model: %s", model)
        except Exception as e:
            raise ImportError("No LLM provided and langchain-ollama not installed. Provide an llm or install langchain-ollama.") from e
    else:
        try:
            logger.info("Using provided LLM: %s", llm.__class__.__name__)
        except Exception:
            logger.info("Using provided LLM instance.")

    # ---- Load data ----
    main_summaries = _load_main_summaries(base / "main_chunks")
    if not main_summaries:
        logger.error("No main chunk summaries found in %s/main_chunks", base)
        return
    supp_global = _load_supplementary_global(base)
    fig_summaries = _load_figure_summaries(base)

    # ---- Smart warning if conclusion is early ----
    concl_idx, _ = _detect_conclusion_chunk(main_summaries)
    if 0 <= concl_idx < len(main_summaries) - 2:
        logger.warning("Conclusion-like language appears in chunk %d. Consider re-splitting chunks.", concl_idx + 1)

    # ---- 1. Bullets ----
    bullets_text = _harvest_factual_bullets_cumulative(main_summaries, llm)
    _write(out / "01_main_factual_bullets.txt", bullets_text)

    # ---- 2. Flow ----
    flow_text = _generate_flow_summary_cumulative(main_summaries, llm)
    _write(out / "02_main_flow_summary.txt", flow_text)

    # ---- 3. Contextualized narrative ----
    final_text = _generate_contextualized_summary(bullets_text, flow_text, supp_global, llm)
    _write(out / "03_main_contextualized.txt", final_text)

    # ---- 4. High-level group plan ----
    plan_text = _generate_high_level_group_plan(final_text, fig_summaries, desired_slides, llm)
    _write(out / "04_presentation_plan.txt", plan_text)

    # ---- 5. Detailed group files (if requested) ----
    if slides_output_dir:
        slides_path = Path(slides_output_dir).expanduser().resolve()
        figures_overview_text = _figures_overview_text(fig_summaries)
        _process_and_split_groups(plan_text, slides_path, final_text, figures_overview_text, bullets_text, llm)

    # ---- Done ----
    logger.info("=== All files generated ===")
    logger.info(" 01_main_factual_bullets.txt → factual bullets")
    logger.info(" 02_main_flow_summary.txt → incremental narrative")
    logger.info(" 03_main_contextualized.txt → story (read this!)")
    logger.info(" 04_presentation_plan.txt → high-level slide groups")
    if slides_output_dir:
        logger.info(" Detailed slide group files in: %s", slides_output_dir)

# ---------------- CLI demo ----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python global_summary_module_v4.py <summaries_dir> [output_dir] [slides] [slides_output_dir]")
        sys.exit(1)
    summaries_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    slides = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    slides_output_dir = sys.argv[4] if len(sys.argv) > 4 else None
    generate_global_and_plan(
        summaries_dir,
        output_dir=output_dir,
        slides_output_dir=slides_output_dir,
        desired_slides=slides,
    )
