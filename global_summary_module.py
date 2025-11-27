#!/usr/bin/env python3
"""
global_summary_module_v3.py
- Exhaustive bullet harvest (cumulative, chunk-by-chunk)
- Faithful flow summary (cumulative, chunk-by-chunk)
- Contextualized narrative + presentation plan
"""
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
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
def _load_supplementary_global(base: Path) -> str:
    return _read(base / "supplementary_global.txt")
def _load_figure_summaries(base: Path) -> Dict[str, dict]:
    p = base / "figure_summaries.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
# ==================== Smart conclusion detection ====================
def _detect_conclusion_chunk(summaries: List[str]) -> Tuple[int, str]:
    conclusion_keywords = [
        "in conclusion", "we conclude", "to summarize", "overall", "in summary",
        "this work demonstrates", "we have shown", "our results indicate"
    ]
    for i, text in enumerate(summaries):
        if any(kw in text.lower() for kw in conclusion_keywords):
            return i, text
    return -1, ""
# ==================== 1. CUMULATIVE BULLET HARVEST ====================
def _harvest_factual_bullets_cumulative(
    main_summaries: List[str], llm: BaseLanguageModel
) -> str:
    """
    Walk through the chunks one-by-one.
    At each step the LLM sees:
      - all bullets collected so far
      - the current chunk
    It returns **only new bullets** (no duplicates).
    """
    print("1. Harvesting factual bullets (cumulative, chunk-by-chunk)...")
    all_bullets: List[str] = []
    for idx, chunk in enumerate(main_summaries, start=1):
        prev = "\n".join(all_bullets) if all_bullets else "None yet."
        prompt = f"""
You have already extracted these bullets from earlier chunks:
\"\"\"{prev[:15000]}\"\"\"
Now examine **only** the following new chunk and add **new** bullet points that are not already present.
Chunk {idx}:
\"\"\"{chunk[:10000]}\"\"\"
Rules:
- Start every bullet with exactly "• "
- Keep figure references verbatim.
- Do **not** repeat any bullet that already exists.
- Be exhaustive: every claim, method step, result, or figure description must appear.
- If nothing new, return exactly "NO_NEW_BULLETS"
Output **only** the new bullet list (or NO_NEW_BULLETS).
"""
        response = llm.invoke(prompt).strip()
        if "NO_NEW_BULLETS" not in response.upper():
            new_lines = [ln for ln in response.splitlines() if ln.startswith("• ")]
            all_bullets.extend(new_lines)
    final_text = "\n".join(all_bullets) if all_bullets else "No factual bullets found."
    return final_text
# ==================== 2. CUMULATIVE FLOW SUMMARY ====================
def _generate_flow_summary_cumulative(
    main_summaries: List[str], llm: BaseLanguageModel
) -> str:
    """
    Build a running narrative paragraph-by-paragraph.
    At each step the LLM receives:
      - the current running summary
      - the next chunk
    It appends a coherent paragraph that integrates the new information.
    """
    print("2. Building faithful flow summary (cumulative, chunk-by-chunk)...")
    running_summary = ""
    for idx, chunk in enumerate(main_summaries, start=1):
        prompt = f"""
You are maintaining a **single, continuous, publication-style summary** of the paper.
Current summary so far:
\"\"\"{running_summary[:15000]}\"\"\"
Now incorporate **only** the following new chunk into the narrative.
Write **one or two new paragraphs** that flow naturally from what is already written.
Preserve every key claim, method, result, and figure reference exactly.
Chunk {idx}:
\"\"\"{chunk[:10000]}\"\"\"
Output **only** the new paragraphs (no preamble).
"""
        addition = llm.invoke(prompt).strip()
        running_summary = "\n\n".join([running_summary, addition]).strip()
    return running_summary or "No flow summary generated."
# ==================== 3. CONTEXTUALIZED NARRATIVE ====================
def _generate_contextualized_summary(
    bullets: str, flow: str, supp_global: str, llm: BaseLanguageModel
) -> str:
    print("3. Generating the real story — full logical narrative (using refine chain for fact-focus)...")
    # Split the flow into chunks for refinement (handles length issues)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    flow_chunks = text_splitter.split_text(flow)
    docs = [Document(page_content=chunk) for chunk in flow_chunks]
    # Add supp_global as an additional "chunk" to weave in naturally
    if supp_global:
        docs.append(Document(page_content=supp_global))
    # Initial prompt: Start with a fact-focused summary of the first chunk
    question_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize this text focusing strictly on key facts, motivation, and logical connections:
{text}
FACT-FOCUSED SUMMARY:"""
    )
    # Refine prompt: Iteratively connect facts, enforce logic, prioritize accuracy over style
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template="""Your current fact-based summary is:
{existing_answer}
Refine it by integrating this new text. Connect all facts logically: start with the open question/motivation, show theory-experiment links step by step, highlight the breakthrough with precise details, end with implications. Prioritize factual accuracy and explicit connections—do not embellish, add flair, or omit details:
{text}
REFINED FACTUAL SUMMARY:"""
    )
    # Load the refine chain (iterative, fact-focused summarization)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False, # Only return final output
        input_key="input_documents",
        output_key="output_text",
    )
    # Run the chain
    result = chain.invoke({"input_documents": docs})
    return result["output_text"].strip()
# ==================== FINAL & WORKING: TWO-STAGE PRESENTATION PLAN ====================
def _generate_presentation_plan(
    final_summary: str,
    figure_summaries: dict,
    factual_bullets: str,
    desired_slides: int,
    llm: BaseLanguageModel,
) -> str:
    print(f"4. Generating presentation plan (target: {desired_slides or 'auto'} slides)...")

    # Figures
    fig_lines = [
        f"Figure {data.get('article_label', fid.replace('fig_', ''))}: {data.get('summary', '')}"
        for fid, data in figure_summaries.items()
    ]
    figures_overview = "\n".join(fig_lines) if fig_lines else "No figures."

    target = (
        f"You must aim for approximately {desired_slides} slides (±2)."
        if desired_slides > 0
        else "Choose the optimal number of slides (12–18 is ideal for 20 min)."
    )

    # THIS PROMPT GIVES YOU THE BEST OF BOTH WORLDS
    magic_prompt = f"""
You are preparing a world-class 20-minute journal club presentation.

Here is the complete, fact-checked summary of the paper:
\"\"\"{final_summary[:24000]}\"\"\"

Available figures:
\"\"\"{figures_overview}\"\"\"

{target}

Your task: create logical, coherent groups of slides that follow the exact scientific story of the paper.

FOR EVERY GROUP YOU CREATE, OUTPUT EXACTLY THIS HEADER FIRST:
===SLIDE_GROUP: Title of the Group | Slides X–Y===

Then write the full content of that group exactly as you would present it — with the same depth, logic, and beauty as before.

Example of correct output:
===SLIDE_GROUP: Introduction and Open Question | Slides 1–3===
* What was the major challenge in continuous cavity QED?
* Why were previous approaches fundamentally limited?
• Slide 1: Title + motivation
• Slide 2: Required physical mechanism
• Slide 3: The breakthrough idea

===SLIDE_GROUP: Experimental Setup and Key Innovations | Slides 4–6===
* High-finesse ring cavity with 87Sr atoms
* Continuous loading scheme
• Slide 4: Overview diagram (Figure 1)
• Slide 5: Cooling and lattice details
• Slide 6: Cavity parameters and finesse

Rules:
- The header line must be exactly: ===SLIDE_GROUP: Title | Slides A–B===
- Title must be concise and descriptive
- Slide range must be consecutive and non-overlapping
- Inside the group: write rich, logical content exactly like your best previous versions
- You may suggest internal slide focus (Slide X: …) — this is encouraged
- Always mention relevant figures
- Be exhaustive and faithful to the science

Start now with the first group.
"""

    # Stage 1: Get perfect logical groups with machine-readable headers
    plan = llm.invoke(magic_prompt).strip()

    # Stage 2: Refine with bullets — but preserve the ===SLIDE_GROUP: headers exactly
    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200)
    bullet_chunks = splitter.split_text(factual_bullets)
    bullet_docs = [Document(page_content=c) for c in bullet_chunks]

    refine_prompt = PromptTemplate.from_template("""
CURRENT PLAN (PRESERVE ALL ===SLIDE_GROUP: ... === lines exactly as they are):
\"\"\"{existing_answer}\"\"\"

Integrate missing facts from:
\"\"\"{text}\"\"\"

You may only:
- Add content inside existing groups
- Create a new group with the exact ===SLIDE_GROUP: Title | Slides X–Y=== header if truly necessary
- Never remove or modify existing headers

Figures reminder:
\"\"\"{figures_overview}\"\"\"
""")

    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=PromptTemplate.from_template("{text}"),
        refine_prompt=refine_prompt,
    )

    result = chain.invoke({
        "input_documents": [Document(page_content=plan)] + bullet_docs,
        "figures_overview": figures_overview,
    })

    return result["output_text"].strip()
# ==================== NEW: SPLIT PLAN INTO INDIVIDUAL SLIDE FILES ====================
def _split_plan_into_slides(plan_text: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split on our magic header
    parts = re.split(r'(===SLIDE_GROUP:[^\n]*===)', plan_text)
    
    groups = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        next_header = re.search(r'===SLIDE_GROUP:', body)
        if next_header:
            body = body[:next_header.start()]
        groups.append(header + "\n" + body.strip())

    if not groups:
        # Fallback
        groups = [plan_text]

    for idx, group in enumerate(groups, 1):
        # Extract title from header
        title_match = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', group)
        title = title_match.group(1).strip() if title_match else f"Group_{idx}"
        safe_title = re.sub(r'[^\w\s\-]', '', title)
        safe_title = re.sub(r'\s+', '_', safe_title).strip('_')[:100]
        if not safe_title:
            safe_title = f"Group_{idx}"

        filename = f"{str(idx).zfill(3)}_{safe_title}.txt"
        _write(output_dir / filename, group.strip() + "\n")

    print(f"   → Created {len(groups)} perfectly split, logically coherent slide groups in: {output_dir}")
# ==================== PUBLIC API ====================
def generate_global_and_plan(
    summaries_dir: str,
    output_dir: Optional[str] = None,
    slides_output_dir: Optional[str] = None,
    *,
    llm: Optional[BaseLanguageModel] = None,
    model: str = "llama3.1:latest",
    desired_slides: int = 0, # 0 = AI decides
) -> None:
    """
    Advanced journal-club summary + presentation generator (v3)
    """
    base = Path(summaries_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else base
    # ---- LLM ----
    if llm is None:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=model)
            print(f"Using Ollama model: {model}")
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. pip install langchain-ollama"
            )
    else:
        print(f"Using provided LLM: {llm.__class__.__name__}")
    # ---- Load data ----
    main_summaries = _load_main_summaries(base / "main_chunks")
    if not main_summaries:
        print("No main chunk summaries found!")
        return
    supp_global = _load_supplementary_global(base)
    fig_summaries = _load_figure_summaries(base)
    # ---- Smart warning ----
    concl_idx, _ = _detect_conclusion_chunk(main_summaries)
    if 0 < concl_idx < len(main_summaries) - 2:
        print(f"Warning: Conclusion found early (chunk {concl_idx+1}). Consider re-splitting.")
    # ---- 1. Bullets ----
    bullets_text = _harvest_factual_bullets_cumulative(main_summaries, llm)
    _write(out / "01_main_factual_bullets.txt", bullets_text)
    # ---- 2. Flow ----
    flow_text = _generate_flow_summary_cumulative(main_summaries, llm)
    _write(out / "02_main_flow_summary.txt", flow_text)
    # ---- 3. Contextualized ----
    final_text = _generate_contextualized_summary(
        bullets_text, flow_text, supp_global, llm
    )
    _write(out / "03_main_contextualized.txt", final_text)
    # ---- 4. Plan ----
    plan_text = _generate_presentation_plan(
        final_text,
        fig_summaries,
        bullets_text, # ← pass the full bullet list
        desired_slides,
        llm
    )
    _write(out / "04_presentation_plan.txt", plan_text)
    # ---- NEW: Split plan into individual slide files (if dir provided) ----
    if slides_output_dir:
        slides_path = Path(slides_output_dir).expanduser().resolve()
        _split_plan_into_slides(plan_text, slides_path)
    # ---- Done ----
    print("\n=== All files generated ===")
    print(f" 01_main_factual_bullets.txt → every claim")
    print(f" 02_main_flow_summary.txt → faithful narrative")
    print(f" 03_main_contextualized.txt → the story (read this!)")
    print(f" 04_presentation_plan.txt → slides (~{desired_slides or 'auto'})")
    if slides_output_dir:
        print(f" Individual slide files in: {slides_output_dir}")
# ==================== Demo ====================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python global_summary_module_v3.py <summaries_dir> [output_dir] [slides] [slides_output_dir]")
        sys.exit(1)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    slides = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    slides_output_dir = sys.argv[4] if len(sys.argv) > 4 else None
    generate_global_and_plan(
        sys.argv[1],
        output_dir=output_dir,
        slides_output_dir=slides_output_dir,
        desired_slides=slides,
    )