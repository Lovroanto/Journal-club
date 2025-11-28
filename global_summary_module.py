#!/usr/bin/env python3
"""
global_summary_module_v5.py
Upgraded global-summary + presentation planner (RAG-ready hooks).

Key improvements:
- High-level plan includes PURPOSE, CONTENT_SCOPE, KEY_TRANSITIONS for each group.
- Per-group generator constrained to group's scope + cumulative previous_summary (concepts, figures, narrative so far).
- Final verification pass for duplicates, numbering, scope violations.
- Medium-density slide style by default (B).
- RAG hooks: placeholder function `rag_prefetch()` that can be filled in to fetch passages and replace main_summaries.
- Public API unchanged: generate_global_and_plan(...)

Usage:
    generate_global_and_plan(summaries_dir, output_dir=None, slides_output_dir=None, llm=..., desired_slides=0)
"""

from __future__ import annotations
import json
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# Try to import langchain utils if available; otherwise use lightweight stubs.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=2000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text: str):
            if not text:
                return []
            chunks = []
            i = 0
            L = len(text)
            while i < L:
                chunks.append(text[i:i+self.chunk_size])
                i += max(1, self.chunk_size - self.chunk_overlap)
            return chunks

try:
    from langchain.chains.summarize import load_summarize_chain
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseLanguageModel
    from langchain.prompts import PromptTemplate
except Exception:
    # Minimal stubs for static checks; in production replace with actual langchain imports/LLM
    class Document:
        def __init__(self, page_content: str):
            self.page_content = page_content
    class BaseLanguageModel:
        def invoke(self, prompt: str) -> str:
            raise NotImplementedError("Provide an LLM with an invoke(prompt) method.")
    class PromptTemplate:
        def __init__(self, input_variables=None, template=""):
            self.input_variables = input_variables or []
            self.template = template
        @classmethod
        def from_template(cls, t: str):
            return cls(template=t)
    def load_summarize_chain(*args, **kwargs):
        raise ImportError("langchain.summarize.load_summarize_chain not available in this environment.")

# Logging
logger = logging.getLogger("global_summary_module_v5")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)

# ---------- I/O ----------
def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""
    except Exception as e:
        logger.warning("Read failed for %s: %s", path, e)
        return ""

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

# ---------- Loading ----------
def _load_main_summaries(main_dir: Path) -> List[str]:
    if not main_dir.exists():
        return []
    files = sorted(main_dir.glob("*_summary.txt"))
    return [_read(f) for f in files if _read(f)]

def _load_supplementary_global(base: Path) -> str:
    return _read(base / "supplementary_global.txt")

def _load_figure_summaries(base: Path) -> Dict[str, dict]:
    p = base / "figure_summaries.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse figure_summaries.json")
            return {}
    return {}

# ---------- Conclusion detection ----------
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

# ---------- 1. Cumulative factual bullets ----------
def _harvest_factual_bullets_cumulative(
    main_summaries: List[str], llm: BaseLanguageModel
) -> str:
    logger.info("1) Harvesting factual bullets (cumulative)...")
    all_bullets: List[str] = []
    for idx, chunk in enumerate(main_summaries, start=1):
        prev = "\n".join(all_bullets) if all_bullets else "None yet."
        prompt = f"""
You are an extraction assistant. Previously extracted bullets:
\"\"\"{prev[:15000]}\"\"\"

Now examine ONLY the following NEW chunk (Chunk {idx}) and output NEW bullet points
that are not in the previous bullets.

Chunk {idx}:
\"\"\"{chunk[:10000]}\"\"\"

Rules:
- Output ONLY bullet lines that START with "• " (bullet symbol + space).
- Each bullet: concise (<= 28 words), factual (claim/method/result/figure note).
- Keep figure references verbatim.
- If nothing new, output EXACTLY: NO_NEW_BULLETS
- Do NOT output extra text.

Now output new bullets or NO_NEW_BULLETS.
"""
        try:
            resp = llm.invoke(prompt).strip()
        except Exception as e:
            logger.error("LLM error during bullet harvesting chunk %d: %s", idx, e)
            resp = "NO_NEW_BULLETS"
        if "NO_NEW_BULLETS" in resp.upper():
            continue
        new_lines = [ln.strip() for ln in resp.splitlines() if ln.strip().startswith("• ")]
        for l in new_lines:
            if l not in all_bullets:
                all_bullets.append(l)
    return "\n".join(all_bullets) if all_bullets else "No factual bullets found."

# ---------- 2. Cumulative flow summary ----------
def _generate_flow_summary_cumulative(
    main_summaries: List[str], llm: BaseLanguageModel
) -> str:
    logger.info("2) Building flow summary (cumulative)...")
    running = ""
    for idx, chunk in enumerate(main_summaries, start=1):
        prompt = f"""
You maintain a single publication-style summary. Current summary:
\"\"\"{running[:15000]}\"\"\"

Now incorporate ONLY this NEW chunk (Chunk {idx}). Write 1-2 concise paragraphs that flow from current summary.
Preserve claims, methods, figures.

Chunk:
\"\"\"{chunk[:10000]}\"\"\"

Output only the new paragraph(s).
"""
        try:
            addition = llm.invoke(prompt).strip()
        except Exception as e:
            logger.error("LLM error in flow generation chunk %d: %s", idx, e)
            addition = ""
        if running and addition:
            running = running.strip() + "\n\n" + addition.strip()
        elif addition:
            running = addition.strip()
    return running or "No flow summary generated."

# ---------- 3. Contextualized refine ----------
def _generate_contextualized_summary(
    bullets: str, flow: str, supp_global: str, llm: BaseLanguageModel
) -> str:
    logger.info("3) Generating contextualized narrative (refine chain)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    flow_chunks = splitter.split_text(flow)
    docs = [Document(page_content=c) for c in flow_chunks]
    if supp_global:
        docs.append(Document(page_content=supp_global))
    # Prepare prompts
    question_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Summarize this text focusing on key facts, motivation, and logical links:
{text}
FACT-FOCUSED SUMMARY:"""
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template="""Current fact-focused summary:
{existing_answer}

Refine by integrating this text:
{text}

Refined factual summary:"""
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
        return result.get("output_text", "").strip()
    except Exception as e:
        logger.warning("Refine chain not available or failed: %s. Falling back to flow+bullets.", e)
        fallback = flow.strip() + "\n\nKey facts (excerpt):\n" + "\n".join(bullets.splitlines()[:20])
        return fallback.strip()

# ---------- Figures overview ----------
def _figures_overview_text(figure_summaries: Dict[str, dict]) -> str:
    lines = []
    for fid, data in sorted((figure_summaries or {}).items()):
        label = data.get("article_label", fid.replace("fig_", ""))
        summary = data.get("summary", "") or data.get("text", "")
        lines.append(f"Figure {label}: {summary}")
    return "\n".join(lines) if lines else "No figures."

# ---------- Helper: parse plan headers that include metadata ----------
def _parse_plan_header_block(block_text: str) -> Dict[str, Any]:
    """
    Parse a block in the plan that includes header + metadata lines (PURPOSE / CONTENT_SCOPE / KEY_TRANSITIONS).
    Expected block structure in the plan file:
    ===SLIDE_GROUP: Title | Slides A–B===
    PURPOSE: ...
    CONTENT_SCOPE: ...
    KEY_TRANSITIONS: ...
    """
    header_line = ""
    purpose = ""
    content_scope = ""
    key_transitions = ""
    lines = block_text.strip().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("===SLIDE_GROUP:"):
            header_line = ln.strip()
        elif ln.strip().upper().startswith("PURPOSE:"):
            purpose = ln.split(":", 1)[1].strip()
        elif ln.strip().upper().startswith("CONTENT_SCOPE:"):
            content_scope = ln.split(":", 1)[1].strip()
        elif ln.strip().upper().startswith("KEY_TRANSITIONS:"):
            key_transitions = ln.split(":", 1)[1].strip()
    return {
        "header": header_line,
        "purpose": purpose,
        "content_scope": content_scope,
        "key_transitions": key_transitions
    }
def _generate_high_level_group_plan(
    final_summary: str,
    figure_summaries: dict,
    desired_slides: int,
    llm: BaseLanguageModel,
    max_groups: int = 8,
    slide_style: str = "B",
) -> str:
    """
    Produce a plan where each group includes:
      - header line: ===SLIDE_GROUP: Title | Slides A–B===
      - PURPOSE
      - CONTENT_SCOPE
      - KEY_TRANSITIONS
      - KEY_TERMS   # <<< NEW >>>

    Implementation:
      1) LLM makes INTERNAL_OUTLINE (hidden)
      2) Then emits PUBLIC METADATA PLAN (only the structured blocks)
    """

    logger.info("4) Generating high-level plan with metadata (style=%s)...", slide_style)
    figures_over = _figures_overview_text(figure_summaries)
    num_figs = len(figure_summaries or {})
    if desired_slides and desired_slides > 0:
        target_info = f"Aim for approximately {desired_slides} slides (±2)."
    else:
        target_info = "Recommend 12–18 slides for a 20-minute talk."

    internal_prompt = f"""
You are designing a pedagogical scientific presentation.

STEP 1 (INTERNAL OUTLINE — PRIVATE):
Create an internal outline with:
 - story arc (1–3 sentences)
 - 3–8 major conceptual blocks
 - purpose and constraints per block
 - what to include vs skip
 - rough slide allocation
 - figure-to-block mapping (there are {num_figs} figures)

This outline is **private reasoning**. Do NOT include it in the final plan.

STEP 2 (PUBLIC METADATA PLAN — REQUIRED OUTPUT):
Produce the final plan with the following strict structure:

For EACH group output EXACTLY:

===SLIDE_GROUP: Title | Slides A–B===
PURPOSE: one concise sentence
CONTENT_SCOPE: 1–2 short sentences describing allowed content
KEY_TRANSITIONS: how it links to previous/next group
KEY_TERMS: comma-separated list of 5–10 essential keywords   # <<< NEW >>>

Rules:
- Titles: 3–7 words
- Groups: consecutive slide ranges starting at 1, non-overlapping
- ≤ {max_groups} groups total
- No full slide text — this is a blueprint
- Content level: medium density (style {slide_style})
- KEY_TERMS define the conceptual boundary (only include terms supported by the paper)

Paper summary:
\"\"\"{final_summary[:24000]}\"\"\"

Figures overview:
\"\"\"{figures_over[:4000]}\"\"\"

Target: {target_info}

Output:
1) INTERNAL_OUTLINE
2) PUBLIC METADATA PLAN (only the blocks above)
"""

    try:
        out = llm.invoke(internal_prompt).strip()
    except Exception as e:
        logger.error("LLM failed to generate high-level plan: %s", e)
        # Fallback (now with KEY_TERMS)
        fallback_blocks = [
            ("Introduction", 1, 2,
             "Set context and question.",
             "Define motivation and basic concept.",
             "Lead to Experimental Setup.",
             "introduction, context, motivation, research question, overview"),
            ("Experimental Setup", 3, 4,
             "Explain apparatus and key variables.",
             "Hardware, geometry, cooling, parameters only.",
             "Lead to Gain Mechanism.",
             "cooling, ring cavity, Sr atoms, apparatus, parameters"),
            ("Gain Mechanism", 5, 6,
             "Explain how gain enables lasing.",
             "Conceptual mechanism; no results.",
             "Lead to Observations.",
             "gain, Raman, population inversion, mechanism, concept"),
            ("Observations", 7, 11,
             "Present core results and figures.",
             "Behaviors, figures, quantitative outcomes.",
             "Lead to Implications.",
             "results, figures, emission, linewidth, behavior"),
            ("Implications & Conclusions", 12, 14,
             "Interpret results and limitations.",
             "Outlook, implications, limitations.",
             "Close.",
             "implications, outlook, conclusions, limitations")
        ]

        blocks_text = []
        for t, a, b, p, c, k, kw in fallback_blocks:
            blocks_text.append(
                f"===SLIDE_GROUP: {t} | Slides {a}-{b}===\n"
                f"PURPOSE: {p}\nCONTENT_SCOPE: {c}\nKEY_TRANSITIONS: {k}\nKEY_TERMS: {kw}\n"
            )
        return "\n".join(blocks_text)

    # --- Extract structured blocks -------------------------------------------
    lines = out.splitlines()
    blocks = []
    cur = []

    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("===SLIDE_GROUP:"):
            if cur:
                blocks.append("\n".join(cur))
            cur = [stripped]
        elif cur and (
            stripped.upper().startswith("PURPOSE:") or
            stripped.upper().startswith("CONTENT_SCOPE:") or
            stripped.upper().startswith("KEY_TRANSITIONS:") or
            stripped.upper().startswith("KEY_TERMS:")     # <<< NEW >>>
        ):
            cur.append(stripped)
        else:
            continue

    if cur:
        blocks.append("\n".join(cur))

    if not blocks:
        logger.warning("No metadata blocks parsed; using fallback.")
        return _generate_high_level_group_plan(
            final_summary, figure_summaries, desired_slides,
            llm, max_groups=max_groups, slide_style=slide_style
        )

    # --- Slide range validation / reassignment --------------------------------
    parsed = []
    for b in blocks:
        parsed.append(_parse_plan_header_block(b))

    has_ranges = all(re.search(r'\|\s*Slides\s*[0-9]+[–-][0-9]+', block["header"])
                     for block in parsed)

    if has_ranges:
        return "\n\n".join(blocks)

    # Auto-assign consecutive ranges if LLM failed to do so
    total_slides = desired_slides if desired_slides and desired_slides > 0 else 14
    n = len(parsed)

    base = total_slides // n
    rem = total_slides - base * n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]

    assigned_blocks = []
    cur_slide = 1

    for idx, item in enumerate(parsed):
        size = sizes[idx]
        a = cur_slide
        b = cur_slide + size - 1
        cur_slide = b + 1

        header = item["header"]
        title_match = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', header)
        title = title_match.group(1).strip() if title_match else f"Group_{idx+1}"

        block_text = (
            f"===SLIDE_GROUP: {title} | Slides {a}-{b}===\n"
            f"PURPOSE: {item.get('purpose','')}\n"
            f"CONTENT_SCOPE: {item.get('content_scope','')}\n"
            f"KEY_TRANSITIONS: {item.get('key_transitions','')}\n"
            f"KEY_TERMS: {item.get('key_terms','')}\n"  # <<< NEW >>>
        )
        assigned_blocks.append(block_text)

    return "\n".join(assigned_blocks)

# ---------- Per-group generation with cumulative previous_summary ----------
def _generate_group_details(
    group_block_text: str,
    final_summary: str,
    figures_overview: str,
    factual_bullets: str,
    previous_summary: Dict[str, Any],
    llm: BaseLanguageModel,
    slide_style: str = "B",
) -> str:
    """
    group_block_text: the metadata block for the group (header + PURPOSE/CONTENT_SCOPE/KEY_TRANSITIONS)
    previous_summary: dict with keys:
        - covered_concepts: set/list of short concept strings
        - covered_figures: set/list of figure labels covered
        - narrative_so_far: short paragraph describing what was covered so far
        - last_slide: last slide number used
    Returns a concise blueprint for group slides (medium-density by default):
      - Group learning goals (3 bullets)
      - Per-slide "Slide N: Focus" lines (only idea-level)
      - Figures to include (Figure X - reason)
      - Transition note to next group
    The generator is constrained: must not repeat covered_concepts or covered_figures, must respect CONTENT_SCOPE.
    """
    # Parse header to get title & range
    header_line = ""
    for ln in group_block_text.splitlines():
        if ln.strip().startswith("===SLIDE_GROUP:"):
            header_line = ln.strip()
            break
    title_match = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|\s*Slides\s*([0-9]+)[–-]([0-9]+)\s*===', header_line)
    if title_match:
        title = title_match.group(1).strip()
        slide_from = int(title_match.group(2))
        slide_to = int(title_match.group(3))
    else:
        # fallback: try to parse a single number or default
        title = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', header_line).group(1).strip() if re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', header_line) else "Group"
        slide_from, slide_to = previous_summary.get("last_slide", 0) + 1, previous_summary.get("last_slide", 0) + 1

    # extract purpose/content_scope/transitions
    purpose = ""
    content_scope = ""
    key_transitions = ""
    for ln in group_block_text.splitlines():
        if ln.strip().upper().startswith("PURPOSE:"):
            purpose = ln.split(":",1)[1].strip()
        elif ln.strip().upper().startswith("CONTENT_SCOPE:"):
            content_scope = ln.split(":",1)[1].strip()
        elif ln.strip().upper().startswith("KEY_TRANSITIONS:"):
            key_transitions = ln.split(":",1)[1].strip()

    # Build the constrained prompt
    prev_concepts = previous_summary.get("covered_concepts", []) or []
    prev_figs = previous_summary.get("covered_figures", []) or []
    narrative_so_far = previous_summary.get("narrative_so_far", "") or ""
    prev_last_slide = previous_summary.get("last_slide", 0) or 0

    # The generator's job is to propose *which content should be included* in each slide (idea-level),
    # not to fully write slide text. That's what you asked (decide which content needs to be in each part).
    prompt = f"""
You must generate a concise blueprint for the slide group below.
DO NOT write full slide prose. Produce ideas (what belongs where) at medium density.

GROUP METADATA:
{group_block_text}

Group title: {title}
Slides: {slide_from}-{slide_to} (inclusive)

PREVIOUSLY COVERED (must NOT be repeated):
Covered concepts: {', '.join(prev_concepts) or 'None'}
Covered figures: {', '.join(prev_figs) or 'None'}
Narrative so far (short):
\"\"\"{narrative_so_far[:2000]}\"\"\"

PAPER SUMMARY (reference; do not repeat the whole thing):
\"\"\"{final_summary[:24000]}\"\"\"

FIGURES OVERVIEW:
\"\"\"{figures_overview[:4000]}\"\"\"

FACTUAL BULLETS (short list of facts; prefer exact figures/phrases):
\"\"\"{factual_bullets[:20000]}\"\"\"

CONSTRAINTS:
- Respect CONTENT_SCOPE exactly: only include items that fall under the scope.
- Do NOT repeat any covered concepts or figures.
- If a needed fact is missing for deciding placement, mark it as MISSING_FACT: <what is missing>.
- Output MUST contain ONLY the sections below, in this order, separated by blank lines:

1) Group learning goals (3 short bullets, each start with "- ")
2) Per-slide blueprint: for each slide N in the range output exactly "Slide N: <one-line focus>"
   (focus should be an idea-level sentence, 8-18 words)
3) Figures to include: lines "Include: Figure X - short reason" (only figures relevant to this group)
4) Transition to next group: one short sentence starting with "Transition:"

Keep medium-density (B): enough to indicate required content, but not full text. Keep concise.
"""
    try:
        raw = llm.invoke(prompt).strip()
    except Exception as e:
        logger.error("LLM error when generating group details for '%s': %s", title, e)
        # Fallback: create minimal blueprint using purpose and content_scope
        slides = slide_to - slide_from + 1
        group_goals = f"- {purpose}\n- Key items from scope: {content_scope}\n- Convey link to next section"
        per_slide = "\n".join([f"Slide {n}: {title} — core idea" for n in range(slide_from, slide_to+1)])
        figures_line = "Include: None"
        transition = f"Transition: {key_transitions or 'Proceed to next group.'}"
        raw = f"{group_goals}\n\n{per_slide}\n\n{figures_line}\n\n{transition}"

    # Attempt to parse LLM raw output into sections; fallback to heuristics if parsing fails.
    sections = {"goals": "", "per_slide": "", "figures": "", "transition": ""}
    # Split in double-newline blocks
    blocks = [b.strip() for b in re.split(r'\n\s*\n', raw) if b.strip()]
    # Heuristic assignment
    if blocks:
        # assign first block to goals, find the block that contains "Slide" lines for per_slide, "Include:" lines for figures, "Transition:" for transition
        for b in blocks:
            if any(re.match(r'^(Slide|Slide\s+[0-9]+:)', ln.strip()) for ln in b.splitlines()):
                sections["per_slide"] = b
            elif any(ln.strip().startswith("Include:") for ln in b.splitlines()):
                sections["figures"] = b
            elif any(ln.strip().startswith("Transition:") for ln in b.splitlines()):
                sections["transition"] = b
            elif b.startswith("- ") or b.count("\n") <= 3 and not sections["goals"]:
                sections["goals"] = b
        # Fill defaults if empty
        if not sections["goals"]:
            sections["goals"] = "- " + purpose
        if not sections["per_slide"]:
            slides = slide_to - slide_from + 1
            sections["per_slide"] = "\n".join([f"Slide {n}: {purpose} (idea-level)" for n in range(slide_from, slide_to+1)])
        if not sections["figures"]:
            sections["figures"] = "Include: None"
        if not sections["transition"]:
            sections["transition"] = f"Transition: {key_transitions or 'Proceed to next group.'}"

    # Return a canonical blueprint format
    blueprint = (
        f"{header_line}\n\n"
        f"{sections['goals']}\n\n"
        f"{sections['per_slide']}\n\n"
        f"{sections['figures']}\n\n"
        f"{sections['transition']}\n"
    )
    return blueprint

# ---------- Final verification pass ----------
def _final_plan_check_and_fix(
    plan_text: str,
    detailed_group_texts: List[str],
) -> Tuple[str, List[str]]:
    """
    Run light checks:
    - Consecutive slide numbering
    - Duplicate slide numbers across groups
    - Repeated slide titles in different groups (warn)
    - Check that per-slide counts match group ranges (simple heuristic)
    Returns (plan_text (unchanged), list_of_warnings)
    """
    warnings = []
    # Extract headers and ranges
    headers = [ln for ln in plan_text.splitlines() if ln.strip().startswith("===SLIDE_GROUP:")]
    ranges = []
    for h in headers:
        m = re.search(r'Slides\s*([0-9]+)[–-]([0-9]+)', h)
        if m:
            ranges.append((int(m.group(1)), int(m.group(2)), h))
    # Check consecutiveness
    if ranges:
        # Sort by start
        ranges_sorted = sorted(ranges, key=lambda x: x[0])
        last_end = 0
        for s, e, h in ranges_sorted:
            if s != last_end + 1:
                warnings.append(f"Non-consecutive slide range detected at header: {h} (expected start {last_end+1}).")
            last_end = e
    # Check duplicates of figures across groups? Just warn if figure labels appear multiple times in group figure lists
    fig_mentions = {}
    for text in detailed_group_texts:
        figs = re.findall(r'Include:\s*Figure\s*([0-9A-Za-z_]+)', text)
        for f in figs:
            fig_mentions.setdefault(f, 0)
            fig_mentions[f] += 1
    repeated_figs = [f for f, c in fig_mentions.items() if c > 1]
    if repeated_figs:
        warnings.append(f"Figures mentioned in multiple groups (may be fine): {', '.join(repeated_figs)}")
    # Check per-slide counts vs declared ranges
    for block_text in detailed_group_texts:
        header = next((ln for ln in block_text.splitlines() if ln.strip().startswith("===SLIDE_GROUP:")), None)
        if not header:
            continue
        m = re.search(r'Slides\s*([0-9]+)[–-]([0-9]+)', header)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        declared_count = b - a + 1
        per_slide_lines = [ln for ln in block_text.splitlines() if re.match(r'\s*Slide\s+[0-9]+:', ln)]
        # If the number of Slide lines mismatches the declared range, warn
        if len(per_slide_lines) != declared_count:
            warnings.append(f"Group '{header}' declared {declared_count} slides but has {len(per_slide_lines)} Slide lines.")
    return plan_text, warnings

# ---------- RAG hook (placeholder) ----------
def rag_prefetch(vectorstore: Any, query: str, top_k: int = 8) -> List[str]:
    """
    Placeholder RAG prefetch hook.
    - vectorstore: your vector store instance (user-provided)
    - query: high-level query (e.g., paper title or abstract)
    - top_k: number of retrieved passages to return
    Returns: list of passage strings to be used as "main_summaries" or injected into them.

    This is intentionally a stub: integrate your vectorstore retrieval here and return text passages.
    """
    # Example pseudocode (commented):
    # hits = vectorstore.similarity_search(query, k=top_k)
    # return [h.page_content for h in hits]
    logger.info("rag_prefetch called (stub). Implement vectorstore retrieval to use RAG.")
    return []

# ---------- Plan splitting and detailed generation ----------
def _process_and_split_groups(
    plan_text: str,
    output_dir: Path,
    final_summary: str,
    figures_overview: str,
    factual_bullets: str,
    llm: BaseLanguageModel,
    slide_style: str = "B",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Split plan_text into blocks (each block starts with ===SLIDE_GROUP)
    blocks = re.split(r'(?=^===SLIDE_GROUP:)', plan_text, flags=re.M)
    blocks = [b.strip() for b in blocks if b.strip()]
    if not blocks:
        logger.warning("No plan blocks found.")
        return
    previous_summary = {
        "covered_concepts": [],
        "covered_figures": [],
        "narrative_so_far": "",
        "last_slide": 0
    }
    detailed_texts = []
    for block in blocks:
        # Generate details
        detailed = _generate_group_details(
            block,
            final_summary,
            figures_overview,
            factual_bullets,
            previous_summary,
            llm,
            slide_style=slide_style
        )
        # Update previous_summary by extracting compact concepts and figures (heuristic)
        # Concepts: collect lines from goals and slide focuses (short phrases)
        goals = [ln.strip()[2:].strip() for ln in detailed.splitlines() if ln.strip().startswith("- ")]
        per_slide_lines = [ln.strip() for ln in detailed.splitlines() if re.match(r'^Slide\s+[0-9]+:', ln.strip())]
        figures_lines = [ln.strip() for ln in detailed.splitlines() if ln.strip().startswith("Include:")]
        # Extract figure labels from figures_lines
        figs = []
        for fl in figures_lines:
            m = re.findall(r'Figure\s*([0-9A-Za-z_]+)', fl)
            if m:
                figs.extend(m)
        # Append to covered_concepts minimal tokens
        for g in goals:
            # pick up short noun phrases (simple heuristic: first 6 words)
            tok = " ".join(g.split()[:6])
            if tok and tok not in previous_summary["covered_concepts"]:
                previous_summary["covered_concepts"].append(tok)
        for ps in per_slide_lines:
            # collect the focus phrases
            focus = ps.split(":",1)[1].strip() if ":" in ps else ps
            token = " ".join(focus.split()[:6])
            if token and token not in previous_summary["covered_concepts"]:
                previous_summary["covered_concepts"].append(token)
        for f in figs:
            if f not in previous_summary["covered_figures"]:
                previous_summary["covered_figures"].append(f)
        # update narrative_so_far (append short sentence)
        title_line = next((ln for ln in block.splitlines() if ln.strip().startswith("===SLIDE_GROUP:")), block.splitlines()[0])
        previous_summary["narrative_so_far"] = (previous_summary["narrative_so_far"] + " " + title_line).strip()[:1800]
        # update last slide
        m = re.search(r'Slides\s*([0-9]+)[–-]([0-9]+)', block)
        if m:
            previous_summary["last_slide"] = int(m.group(2))
        else:
            # heuristically increment
            previous_summary["last_slide"] = previous_summary.get("last_slide", 0) + len(per_slide_lines)
        # Write to file per group
        # Build safe filename
        title_match = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', block)
        title = title_match.group(1).strip() if title_match else f"Group_{previous_summary['last_slide']}"
        safe_title = re.sub(r'[^\w\s\-]', '', title)
        safe_title = re.sub(r'\s+', '_', safe_title).strip('_')[:100] or f"Group_{previous_summary['last_slide']}"
        filename = f"{str(previous_summary['last_slide']).zfill(3)}_{safe_title}.txt"
        _write(output_dir / filename, detailed + "\n")
        detailed_texts.append(detailed)
    # Final verification
    _, warnings = _final_plan_check_and_fix(plan_text, detailed_texts)
    if warnings:
        _write(output_dir / "ZZ_plan_warnings.txt", "\n".join(warnings) + "\n")
        logger.warning("Plan verification produced warnings; see ZZ_plan_warnings.txt")
    logger.info("Generated %d detailed group files in %s", len(detailed_texts), output_dir)

# ---------- PUBLIC API ----------
def generate_global_and_plan(
    summaries_dir: str,
    output_dir: Optional[str] = None,
    slides_output_dir: Optional[str] = None,
    *,
    llm: Optional[BaseLanguageModel] = None,
    model: str = "llama3.1:latest",
    desired_slides: int = 0,
    slide_style: str = "B"
) -> None:
    """
    Main entrypoint:
    - summaries_dir: path containing main_chunks/ *_summary.txt files, supplementary_global.txt, figure_summaries.json
    - output_dir: where to write the top-level files (defaults to summaries_dir)
    - slides_output_dir: if provided, write per-group detailed files there
    - llm: optional LLM instance with .invoke(prompt) -> str
    - model: model identifier (used if llm not provided and langchain_ollama available)
    - desired_slides: 0 = auto
    - slide_style: "A"|"B"|"C" (default "B")
    """
    base = Path(summaries_dir).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve() if output_dir else base

    if llm is None:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=model)
            logger.info("Using OllamaLLM: %s", model)
        except Exception as e:
            raise ImportError("No LLM provided and langchain-ollama not available. Provide llm or install langchain-ollama.") from e
    else:
        logger.info("Using provided LLM instance.")

    # Load inputs
    main_summaries = _load_main_summaries(base / "main_chunks")
    if not main_summaries:
        logger.error("No main chunk summaries found in %s/main_chunks", base)
        return
    supp_global = _load_supplementary_global(base)
    fig_summaries = _load_figure_summaries(base)

    # Intelligent warning
    concl_idx, _ = _detect_conclusion_chunk(main_summaries)
    if 0 <= concl_idx < len(main_summaries) - 2:
        logger.warning("Found conclusion-like language in chunk %d. Consider re-splitting.", concl_idx + 1)

    # 0) (Optional) RAG prefetch placeholder: user may call rag_prefetch() externally and replace main_summaries
    # main_summaries = rag_prefetch(vectorstore, query)

    # 1) Bullets
    bullets_text = _harvest_factual_bullets_cumulative(main_summaries, llm)
    _write(out / "01_main_factual_bullets.txt", bullets_text)

    # 2) Flow
    flow_text = _generate_flow_summary_cumulative(main_summaries, llm)
    _write(out / "02_main_flow_summary.txt", flow_text)

    # 3) Contextualized narrative
    final_text = _generate_contextualized_summary(bullets_text, flow_text, supp_global, llm)
    _write(out / "03_main_contextualized.txt", final_text)

    # 4) High-level plan with metadata
    plan_text = _generate_high_level_group_plan(final_text, fig_summaries, desired_slides, llm, slide_style=slide_style)
    _write(out / "04_presentation_plan.txt", plan_text)

    # 5) Detailed group files
    if slides_output_dir:
        slides_path = Path(slides_output_dir).expanduser().resolve()
        figures_over_text = _figures_overview_text(fig_summaries)
        _process_and_split_groups(plan_text, slides_path, final_text, figures_over_text, bullets_text, llm, slide_style=slide_style)

    logger.info("=== All files generated ===")
    logger.info(" 01_main_factual_bullets.txt")
    logger.info(" 02_main_flow_summary.txt")
    logger.info(" 03_main_contextualized.txt")
    logger.info(" 04_presentation_plan.txt")
    if slides_output_dir:
        logger.info(" Detailed group files in: %s", slides_output_dir)

# ---------- CLI ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python global_summary_module_v5.py <summaries_dir> [output_dir] [slides] [slides_output_dir]")
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
        slide_style="B"
    )
