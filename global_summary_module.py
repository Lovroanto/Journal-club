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
    Parse a block in the plan that includes:
      - header
      - PURPOSE
      - CONTENT_SCOPE
      - KEY_TRANSITIONS
      - KEY_TERMS   # <<< NEW >>>
    """
    header_line = ""
    purpose = ""
    content_scope = ""
    key_transitions = ""
    key_terms = ""   # <<< NEW >>>

    for ln in block_text.strip().splitlines():
        stripped = ln.strip()
        U = stripped.upper()

        if stripped.startswith("===SLIDE_GROUP:"):
            header_line = stripped
        elif U.startswith("PURPOSE:"):
            purpose = stripped.split(":", 1)[1].strip()
        elif U.startswith("CONTENT_SCOPE:"):
            content_scope = stripped.split(":", 1)[1].strip()
        elif U.startswith("KEY_TRANSITIONS:"):
            key_transitions = stripped.split(":", 1)[1].strip()
        elif U.startswith("KEY_TERMS:"):       # <<< NEW >>>
            key_terms = stripped.split(":", 1)[1].strip()

    return {
        "header": header_line,
        "purpose": purpose,
        "content_scope": content_scope,
        "key_transitions": key_transitions,
        "key_terms": key_terms,   # <<< NEW >>>
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
      - KEY_TERMS

    Model C (hybrid narrative):
      1) Motivation / why this matters
      2) Physical mechanism & theory essentials
      3) Experimental implementation
      4) Observed regimes / results
      5) Interpretation, implications, outlook

    The LLM is asked to:
      - Build an INTERNAL meta-plan (phases + dependencies)
      - Then emit PUBLIC metadata groups respecting that order
      - Make KEY_TRANSITIONS causal / dependency-based (not just “links to next”)
    """

    logger.info("4) Generating high-level plan with metadata (style=%s)...", slide_style)
    figures_over = _figures_overview_text(figure_summaries)
    num_figs = len(figure_summaries or {})
    if desired_slides and desired_slides > 0:
        target_info = f"Aim for approximately {desired_slides} slides (±2)."
    else:
        target_info = "Recommend 12–18 slides for a ~20-minute talk."

    internal_prompt = f"""
You are designing a pedagogical scientific presentation using this FACTUAL summary:

\"\"\"{final_summary[:24000]}\"\"\"

Figures:
\"\"\"{figures_over[:4000]}\"\"\"

You MUST follow a hybrid teaching model (Model C):

PHASES (INTERNAL STRUCTURE):
1) MOTIVATION
   - Why continuous lasing and this experiment matter
   - High-level impact (gate depth, sensing, simulation)
2) PHYSICAL MECHANISM
   - Key concepts: self-organization, recoil-induced processes, cross-over regime
3) EXPERIMENTAL IMPLEMENTATION
   - Atom–cavity setup, cooling, lattice, fields, detection
4) OBSERVED REGIMES & RESULTS
   - Zones (I–IV), frequency behavior, linewidths, g2, beatnote, self-regulation
5) INTERPRETATION & OUTLOOK
   - What the results mean, implications, open questions

STEP 1 — INTERNAL_OUTLINE (PRIVATE, DO NOT OUTPUT AS FINAL PLAN):
- Map the paper onto these 5 phases.
- For each phase, decide:
  * what MUST be known before it,
  * what new understanding it adds,
  * which figures belong to it,
  * a rough slide count.
- This outline is PRIVATE reasoning.

STEP 2 — PUBLIC METADATA PLAN (THIS IS THE OUTPUT):
Create a SEQUENCE of slide groups that follow the dependency order of the phases.
For EACH group output EXACTLY:

===SLIDE_GROUP: Title | Slides A–B===
PURPOSE: one concise sentence describing the narrative ROLE of this group
CONTENT_SCOPE: 1–2 short sentences describing what content is allowed here
KEY_TRANSITIONS: a causal or dependency-based sentence explaining WHY the next group follows this one
KEY_TERMS: comma-separated list of 5–10 essential keywords defining this group’s conceptual boundary

Rules:
- Titles: 3–7 words, descriptive.
- Groups: consecutive slide ranges starting at 1, non-overlapping.
- ≤ {max_groups} groups total.
- No full slide content — this is a blueprint.
- KEY_TRANSITIONS must be meaningful:
  * “Because … we now need to …”
  * “To interpret these results we must now …”
  * “Since we introduced X, the next step is to show Y …”
- KEY_TERMS must be supported by the summary, not invented.
- Try to map early groups to MOTIVATION + MECHANISM, middle to SETUP + REGIMES, late to INTERPRETATION.

Target: {target_info}

Output:
1) INTERNAL_OUTLINE (label it clearly; will be ignored later)
2) PUBLIC METADATA PLAN (only the groups in the exact format above)
"""

    try:
        out = llm.invoke(internal_prompt).strip()
    except Exception as e:
        logger.error("LLM failed to generate high-level plan: %s", e)

        fallback_blocks = [
            ("Title & Motivation", 1, 3,
             "Introduce the work and why continuous lasing matters.",
             "State achievement and high-level motivation.",
             "Leads into broader introduction of concepts and mechanisms.",
             "continuous lasing, neutral atoms, motivation, overview, quantum technologies"),
            ("Physical Mechanism", 4, 6,
             "Give the audience the key physical ideas behind continuous lasing.",
             "Explain self-organization, recoil processes, and cross-over regime.",
             "Prepares the audience to understand how the experiment implements the mechanism.",
             "self-organization, recoil, cross-over regime, gain, cavity"),
            ("Experimental Setup", 7, 9,
             "Describe how the experiment realizes the mechanism.",
             "Explain cavity, atoms, cooling, lattice, and detection.",
             "Enables understanding of the observed regimes and data.",
             "cavity, molasses, lattice, Zeeman, 689 nm, 813 nm"),
            ("Regimes & Results", 10, 14,
             "Present zones, frequency behavior, and lasing properties.",
             "Describe zones I–IV, g2, beatnote, linewidth, and stability.",
             "Leads naturally into interpretation and implications.",
             "zones, I-IV, g2, beatnote, linewidth, stability"),
            ("Interpretation & Outlook", 15, 19,
             "Interpret results and discuss implications for quantum technologies.",
             "Summarize mechanism, self-regulation, and possible applications.",
             "Closes the talk.",
             "implications, quantum gates, sensing, simulation, conclusions")
        ]

        blocks_text = []
        for t, a, b, p, c, k, kw in fallback_blocks:
            blocks_text.append(
                f"===SLIDE_GROUP: {t} | Slides {a}-{b}===\n"
                f"PURPOSE: {p}\nCONTENT_SCOPE: {c}\nKEY_TRANSITIONS: {k}\nKEY_TERMS: {kw}\n"
            )
        return "\n".join(blocks_text)

    # --- Extract only PUBLIC METADATA PLAN blocks ---------------------------
    lines = out.splitlines()
    blocks: List[str] = []
    cur: List[str] = []

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
            stripped.upper().startswith("KEY_TERMS:")
        ):
            cur.append(stripped)
        else:
            # ignore internal_outline / commentary
            continue

    if cur:
        blocks.append("\n".join(cur))

    if not blocks:
        logger.warning("No metadata blocks parsed; using fallback plan.")
        return _generate_high_level_group_plan(
            final_summary, figure_summaries, desired_slides,
            llm, max_groups=max_groups, slide_style=slide_style
        )

    # --- Validate slide ranges; if missing or broken, assign automatically ---
    parsed = [_parse_plan_header_block(b) for b in blocks]

    has_ranges = all(
        re.search(r'\|\s*Slides\s*[0-9]+[–-][0-9]+', block["header"])
        for block in parsed
    )

    if has_ranges:
        return "\n\n".join(blocks)

    # Auto-assign ranges if LLM did not provide or broke them
    total_slides = desired_slides if desired_slides and desired_slides > 0 else 19
    n = len(parsed)

    base = total_slides // n
    rem = total_slides - base * n
    sizes = [base + (1 if i < rem else 0) for _ in range(n)]

    assigned: List[str] = []
    cur_slide = 1

    for idx, item in enumerate(parsed):
        size = sizes[idx]
        a, b = cur_slide, cur_slide + size - 1
        cur_slide = b + 1

        title_match = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', item["header"])
        title = title_match.group(1).strip() if title_match else f"Group_{idx+1}"

        assigned.append(
            f"===SLIDE_GROUP: {title} | Slides {a}-{b}===\n"
            f"PURPOSE: {item.get('purpose','')}\n"
            f"CONTENT_SCOPE: {item.get('content_scope','')}\n"
            f"KEY_TRANSITIONS: {item.get('key_transitions','')}\n"
            f"KEY_TERMS: {item.get('key_terms','')}\n"
        )

    return "\n".join(assigned)

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
    Generate a concise blueprint for ONE slide group, respecting:
      - PURPOSE
      - CONTENT_SCOPE
      - KEY_TRANSITIONS
      - KEY_TERMS
      - previously covered concepts / keywords / figures

    Output format:
      ===SLIDE_GROUP: ...===
      <blank>
      <learning goals>
      <blank>
      <Slide N: focus ...>
      <blank>
      <Include: Figure ...>
      <blank>
      Transition: ...
    """

    # ---- Parse header for title & slide range ----
    header_line = ""
    for ln in group_block_text.splitlines():
        if ln.strip().startswith("===SLIDE_GROUP:"):
            header_line = ln.strip()
            break

    title_match = re.search(
        r'===SLIDE_GROUP:\s*(.*?)\s*\|\s*Slides\s*([0-9]+)[–-]([0-9]+)\s*===',
        header_line
    )
    if title_match:
        title = title_match.group(1).strip()
        slide_from = int(title_match.group(2))
        slide_to = int(title_match.group(3))
    else:
        title = (
            re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', header_line).group(1).strip()
            if re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', header_line)
            else "Group"
        )
        slide_from = previous_summary.get("last_slide", 0) + 1
        slide_to = slide_from

    # ---- Extract PURPOSE, CONTENT_SCOPE, KEY_TRANSITIONS, KEY_TERMS ----
    purpose = ""
    content_scope = ""
    key_transitions = ""
    key_terms_raw = ""
    for ln in group_block_text.splitlines():
        s = ln.strip()
        u = s.upper()
        if u.startswith("PURPOSE:"):
            purpose = s.split(":", 1)[1].strip()
        elif u.startswith("CONTENT_SCOPE:"):
            content_scope = s.split(":", 1)[1].strip()
        elif u.startswith("KEY_TRANSITIONS:"):
            key_transitions = s.split(":", 1)[1].strip()
        elif u.startswith("KEY_TERMS:"):
            key_terms_raw = s.split(":", 1)[1].strip()

    key_terms = [t.strip() for t in key_terms_raw.split(",") if t.strip()]

    # ---- Previous coverage (for non-repetition) ----
    prev_concepts = previous_summary.get("covered_concepts", []) or []
    prev_figs = previous_summary.get("covered_figures", []) or []
    prev_keywords = previous_summary.get("covered_keywords", []) or []
    narrative_so_far = previous_summary.get("narrative_so_far", "") or ""

    # ---- Build constrained prompt ----
    prompt = f"""
You are designing slides for ONE group in a scientific talk.

GROUP METADATA (authoritative):
{group_block_text}

Group title: {title}
Slides: {slide_from}-{slide_to} (inclusive)

GROUP KEY_TERMS (allowed main concepts for this group):
{", ".join(key_terms) or "None given"}

PREVIOUSLY COVERED (DO NOT REINTRODUCE as new explanations):
- Concepts (high-level phrases): {", ".join(prev_concepts) or "None"}
- Keywords: {", ".join(prev_keywords) or "None"}
- Figures: {", ".join(prev_figs) or "None"}

Narrative so far:
\"\"\"{narrative_so_far[:1500]}\"\"\"

PAPER SUMMARY (reference only):
\"\"\"{final_summary[:24000]}\"\"\"

FIGURES OVERVIEW:
\"\"\"{figures_overview[:3500]}\"\"\"

FACTUAL BULLETS (atomic facts, methods, results, figures):
\"\"\"{factual_bullets[:20000]}\"\"\"

YOUR TASK:
Create a concise blueprint for this group.

CRITICAL CONSTRAINTS:
- Respect CONTENT_SCOPE strictly.
- Use GROUP KEY_TERMS as the primary conceptual backbone.
- You may *refer* to earlier concepts, but do NOT re-explain or repeat them as new.
- Do NOT introduce new big topics that are not in KEY_TERMS or the paper summary.
- Ensure the blueprint makes sense as part of a full talk: this group should move the story forward according to KEY_TRANSITIONS.

OUTPUT FORMAT (exactly, in this order, with blank lines between sections):

1) Group learning goals (3 bullets)
   - Exactly 3 lines
   - Each line MUST start with "- "
   - Each 6–14 words long
   - Focus on what the audience should understand after this group

2) Per-slide blueprint
   - For EACH slide N from {slide_from} to {slide_to}, output exactly one line:
     Slide N: <one-line focus>
   - Each focus: 8–18 words
   - Focus = idea-level requirement, not full text
   - Use GROUP KEY_TERMS wherever possible

3) Figures to include
   - 1+ lines of the form:
     Include: Figure X - short reason
   - Only include figures that are relevant to this group’s CONTENT_SCOPE
   - If genuinely none, output exactly: Include: None

4) Transition to next group
   - Exactly one line
   - Starts with: Transition:
   - Give a causal or dependency-based link using KEY_TRANSITIONS meaningfully
   - Example style: "Transition: Because we now understand X, we can examine Y next."

Do NOT add extra sections, headings, or commentary outside this structure.
    """

    try:
        raw = llm.invoke(prompt).strip()
    except Exception as e:
        logger.error("LLM error when generating group details for '%s': %s", title, e)
        # Fallback: minimal blueprint
        slides = slide_to - slide_from + 1
        group_goals = (
            f"- {purpose}\n"
            f"- Key items from scope: {content_scope}\n"
            f"- Prepare audience for: {key_transitions or 'next section'}"
        )
        per_slide = "\n".join(
            [f"Slide {n}: {title} — core idea" for n in range(slide_from, slide_to + 1)]
        )
        figures_line = "Include: None"
        transition = f"Transition: {key_transitions or 'Proceed to next group.'}"
        raw = f"{group_goals}\n\n{per_slide}\n\n{figures_line}\n\n{transition}"

    # ---- Parse output into sections (robust but simple) ----
    sections = {"goals": "", "per_slide": "", "figures": "", "transition": ""}
    blocks = [b.strip() for b in re.split(r'\n\s*\n', raw) if b.strip()]

    if blocks:
        for b in blocks:
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
            if all(ln.startswith("- ") for ln in lines) and not sections["goals"]:
                sections["goals"] = b
            elif any(re.match(r'^Slide\s+[0-9]+:', ln) for ln in lines):
                sections["per_slide"] = b
            elif any(ln.startswith("Include:") for ln in lines):
                sections["figures"] = b
            elif any(ln.startswith("Transition:") for ln in lines):
                sections["transition"] = [ln for ln in lines if ln.startswith("Transition:")][0]

    # ---- Fill defaults if missing ----
    if not sections["goals"]:
        sections["goals"] = (
            f"- {purpose}\n"
            f"- Key items from scope: {content_scope}\n"
            f"- Prepare audience for: {key_transitions or 'next group'}"
        )

    if not sections["per_slide"]:
        sections["per_slide"] = "\n".join(
            [f"Slide {n}: {purpose} (idea-level)" for n in range(slide_from, slide_to + 1)]
        )

    if not sections["figures"]:
        sections["figures"] = "Include: None"

    if not sections["transition"]:
        sections["transition"] = f"Transition: {key_transitions or 'Proceed to next group.'}"

    # ---- Canonical blueprint ----
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
    - Duplicate slide numbers across groups (implicit via non-consecutive check)
    - Reused figures (warn)
    - Per-slide count vs declared range
    - Weak or missing KEY_TRANSITIONS (warn)
    - Per-slide focus length outside 8–18 words (warn)
    Returns (plan_text (unchanged), list_of_warnings)
    """
    warnings: List[str] = []

    # --- Slide ranges / consecutiveness ---
    headers = [ln for ln in plan_text.splitlines() if ln.strip().startswith("===SLIDE_GROUP:")]
    ranges = []
    for h in headers:
        m = re.search(r'Slides\s*([0-9]+)[–-]([0-9]+)', h)
        if m:
            ranges.append((int(m.group(1)), int(m.group(2)), h))

    if ranges:
        ranges_sorted = sorted(ranges, key=lambda x: x[0])
        last_end = 0
        for s, e, h in ranges_sorted:
            if s != last_end + 1:
                warnings.append(
                    f"Non-consecutive slide range at header: {h} (expected start {last_end + 1})."
                )
            last_end = e

    # --- Reused figures across groups ---
    fig_mentions: Dict[str, int] = {}
    for text in detailed_group_texts:
        figs = re.findall(r'Include:\s*Figure\s*([0-9A-Za-z_]+)', text)
        for f in figs:
            fig_mentions[f] = fig_mentions.get(f, 0) + 1
    repeated_figs = [f for f, c in fig_mentions.items() if c > 1]
    if repeated_figs:
        warnings.append(
            f"Figures mentioned in multiple groups (may be fine, but check): {', '.join(repeated_figs)}"
        )

    # --- KEY_TRANSITIONS quality from plan_text ---
    plan_blocks = re.split(r'(?=^===SLIDE_GROUP:)', plan_text, flags=re.M)
    plan_blocks = [b.strip() for b in plan_blocks if b.strip()]
    for pb in plan_blocks:
        meta = _parse_plan_header_block(pb)
        tr = (meta.get("key_transitions") or "").strip()
        header = meta.get("header", "").strip()
        if not tr or tr.lower() in {"none", "n/a", "no transition"}:
            warnings.append(f"Group has weak or missing KEY_TRANSITIONS: {header}")
        elif len(tr.split()) < 6:
            warnings.append(f"Group has very short KEY_TRANSITIONS, consider enriching: {header}")

    # --- Per-slide counts & length checks in detailed blueprints ---
    for block_text in detailed_group_texts:
        header = next(
            (ln for ln in block_text.splitlines() if ln.strip().startswith("===SLIDE_GROUP:")),
            None,
        )
        if not header:
            continue
        m = re.search(r'Slides\s*([0-9]+)[–-]([0-9]+)', header)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        declared_count = b - a + 1

        per_slide_lines = [
            ln.strip()
            for ln in block_text.splitlines()
            if re.match(r'^Slide\s+[0-9]+:', ln.strip())
        ]
        if len(per_slide_lines) != declared_count:
            warnings.append(
                f"Group '{header}' declared {declared_count} slides but has {len(per_slide_lines)} 'Slide N:' lines."
            )

        # Check length of each slide focus
        for ln in per_slide_lines:
            try:
                focus = ln.split(":", 1)[1].strip()
            except Exception:
                continue
            word_count = len(focus.split())
            if word_count < 6 or word_count > 22:
                warnings.append(
                    f"Slide focus length out of range ({word_count} words) in group '{header}': {ln}"
                )

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

    previous_summary: Dict[str, Any] = {
        "covered_concepts": [],
        "covered_figures": [],
        "covered_keywords": [],   # NEW: aggregated KEY_TERMS from previous groups
        "narrative_so_far": "",
        "last_slide": 0,
    }
    detailed_texts: List[str] = []

    for block in blocks:
        # Generate details for this group
        detailed = _generate_group_details(
            block,
            final_summary,
            figures_overview,
            factual_bullets,
            previous_summary,
            llm,
            slide_style=slide_style,
        )

        # ---- Update previous_summary ----
        # 1) Concepts: from goals + slide focuses
        goals = [
            ln.strip()[2:].strip()
            for ln in detailed.splitlines()
            if ln.strip().startswith("- ")
        ]
        per_slide_lines = [
            ln.strip()
            for ln in detailed.splitlines()
            if re.match(r'^Slide\s+[0-9]+:', ln.strip())
        ]
        figures_lines = [
            ln.strip()
            for ln in detailed.splitlines()
            if ln.strip().startswith("Include:")
        ]

        # Extract figure labels from figures_lines
        figs: List[str] = []
        for fl in figures_lines:
            m = re.findall(r'Figure\s*([0-9A-Za-z_]+)', fl)
            if m:
                figs.extend(m)

        # Compact concept tokens
        for g in goals:
            tok = " ".join(g.split()[:6])
            if tok and tok not in previous_summary["covered_concepts"]:
                previous_summary["covered_concepts"].append(tok)

        for ps in per_slide_lines:
            focus = ps.split(":", 1)[1].strip() if ":" in ps else ps
            token = " ".join(focus.split()[:6])
            if token and token not in previous_summary["covered_concepts"]:
                previous_summary["covered_concepts"].append(token)

        # 2) Figures
        for f in figs:
            if f not in previous_summary["covered_figures"]:
                previous_summary["covered_figures"].append(f)

        # 3) Keywords from plan block (KEY_TERMS line)
        for ln in block.splitlines():
            s = ln.strip()
            if s.upper().startswith("KEY_TERMS:"):
                kw_raw = s.split(":", 1)[1].strip()
                kws = [t.strip() for t in kw_raw.split(",") if t.strip()]
                for k in kws:
                    if k not in previous_summary["covered_keywords"]:
                        previous_summary["covered_keywords"].append(k)

        # 4) Narrative so far: append header line
        title_line = next(
            (ln for ln in block.splitlines() if ln.strip().startswith("===SLIDE_GROUP:")),
            block.splitlines()[0],
        )
        previous_summary["narrative_so_far"] = (
            previous_summary["narrative_so_far"] + " " + title_line
        ).strip()[:1800]

        # 5) Last slide number
        m = re.search(r'Slides\s*([0-9]+)[–-]([0-9]+)', block)
        if m:
            previous_summary["last_slide"] = int(m.group(2))
        else:
            previous_summary["last_slide"] = previous_summary.get("last_slide", 0) + len(
                per_slide_lines
            )

        # ---- Write per-group blueprint ----
        title_match = re.search(r'===SLIDE_GROUP:\s*(.*?)\s*\|', block)
        title = (
            title_match.group(1).strip()
            if title_match
            else f"Group_{previous_summary['last_slide']}"
        )
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
