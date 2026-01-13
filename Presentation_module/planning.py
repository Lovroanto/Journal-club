from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# =========================
# Data models
# =========================

@dataclass
class SlidePlan:
    slide_id: int
    blueprint_text: str
    group_name: str
    group_purpose: Optional[str]
    group_scope: Optional[str]
    group_transition: Optional[str]
    figure_used: Optional[str] = None
    supmat_notes: Optional[str] = None


@dataclass
class SlideGroupPlan:
    group_name: str
    purpose: Optional[str]
    scope: Optional[str]
    key_terms: List[str]
    transition: Optional[str]
    supmat_notes: Optional[str]
    available_figures: List[str]
    slides: List[SlidePlan]


# =========================
# Helpers
# =========================

def _extract_single_line(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip() if m.lastindex else m.group(0).strip()


def _extract_all(pattern: str, text: str) -> List[str]:
    return [m.strip() for m in re.findall(pattern, text, flags=re.MULTILINE)]


def _infer_figure_from_blueprint(slide: SlidePlan, available_figures: List[str]) -> Optional[str]:
    text = slide.blueprint_text.lower()
    for fig in available_figures:
        if fig.lower() in text:
            return fig
    return None

# =========================
# Semantic utilities
# =========================

def _semantic_score(slide: SlidePlan, figure: str) -> int:
    """
    Very conservative relevance score.
    We only want to match when it is genuinely obvious.
    """
    slide_text = (
        slide.blueprint_text.lower()
        + " "
        + (slide.group_scope or "").lower()
        + " "
        + (slide.group_purpose or "").lower()
    )

    score = 0
    for token in re.findall(r"\w+", figure.lower()):
        if len(token) > 4 and token in slide_text:
            score += 1
    return score


def _infer_insert_position(fig: str, slides: List[SlidePlan]) -> int:
    """
    Decide where a forced figure slide should go.
    Conservative and narrative-preserving.
    """
    f = fig.lower()

    # Experimental setup → early, but not first
    if any(k in f for k in ["setup", "experimental", "apparatus"]):
        return min(1, len(slides))

    # Mechanisms / gain / dynamics → after concept intro
    if any(k in f for k in ["mechanism", "gain", "dynamics", "loss"]):
        return max(1, len(slides) // 2)

    # Otherwise → late
    return len(slides)


# =========================
# Core planning logic
# =========================

def refine_plan_with_figures(group_plan: SlideGroupPlan) -> None:
    """
    Architectural rule set:

    1. Slides define the narrative
    2. Figures are attached ONLY if they naturally support a slide
    3. If a slide explicitly names a figure, it MUST use it
    4. Unused figures get their OWN slide
    5. Only forced figure slides are figure-centered
    """

    slides = group_plan.slides
    unused_figures = list(group_plan.available_figures)

    # ---- Phase 1: explicit figure binding (hard rule) ----
    for slide in slides:
        if slide.figure_used:
            continue

        inferred = _infer_figure_from_blueprint(slide, unused_figures)
        if inferred:
            slide.figure_used = inferred
            unused_figures.remove(inferred)

    # ---- Phase 2: conservative semantic assignment ----
    for slide in slides:
        if slide.figure_used:
            continue

        best_fig = None
        best_score = 0

        for fig in unused_figures:
            s = _semantic_score(slide, fig)
            if s > best_score:
                best_score = s
                best_fig = fig

        # Only assign if the match is clearly meaningful
        if best_fig and best_score >= 2:
            slide.figure_used = best_fig
            unused_figures.remove(best_fig)

    # ---- Phase 3: create forced figure slides for leftovers ----
    if unused_figures:
        next_slide_id = max(s.slide_id for s in slides) + 1 if slides else 1

        for fig in unused_figures:
            forced_slide = SlidePlan(
                slide_id=next_slide_id,
                blueprint_text=(
                    f"Explain {fig}: what it shows, why it was measured, "
                    f"and how it supports the broader narrative of this section."
                ),
                group_name=group_plan.group_name,
                group_purpose=group_plan.purpose,
                group_scope=group_plan.scope,
                group_transition=group_plan.transition,
                figure_used=fig,
                supmat_notes=group_plan.supmat_notes,
            )

            insert_at = _infer_insert_position(fig, slides)
            slides.insert(insert_at, forced_slide)
            next_slide_id += 1

    # ---- Phase 4: re-number slides cleanly ----
    for i, slide in enumerate(slides, start=1):
        slide.slide_id = i


# =========================
# Parser
# =========================

def parse_slide_group_file(path: str | Path) -> SlideGroupPlan:
    text = Path(path).read_text(encoding="utf-8")

    group_name_match = re.search(
        r"===\s*SLIDE_GROUP:\s*(.*?)\s*\|", text, flags=re.IGNORECASE
    )
    group_name = group_name_match.group(1).strip() if group_name_match else "UNNAMED_GROUP"

    purpose = _extract_single_line(r"^\s*PURPOSE:\s*(.*)$", text)
    scope = _extract_single_line(r"^\s*CONTENT_SCOPE:\s*(.*)$", text)

    key_terms_raw = _extract_single_line(r"^\s*KEY_TERMS:\s*(.*)$", text)
    key_terms = [k.strip() for k in key_terms_raw.split(",")] if key_terms_raw else []

    transition = _extract_single_line(r"^\s*Transition:\s*(.*)$", text)
    supmat_notes = _extract_single_line(r"^\s*\[SupMat\]\s*(.*)$", text)

    slide_lines = re.findall(r"Slide\s+(\d+):\s*(.*)", text, flags=re.IGNORECASE)

    slides = [
        SlidePlan(
            slide_id=int(num),
            blueprint_text=txt.strip(),
            group_name=group_name,
            group_purpose=purpose,
            group_scope=scope,
            group_transition=transition,
            supmat_notes=supmat_notes,
        )
        for num, txt in slide_lines
    ]

    available_figures = _extract_all(r"Include:\s*(.*)", text)

    group_plan = SlideGroupPlan(
        group_name=group_name,
        purpose=purpose,
        scope=scope,
        key_terms=key_terms,
        transition=transition,
        supmat_notes=supmat_notes,
        available_figures=available_figures,
        slides=slides,
    )

    # 🔑 Core architectural step
    refine_plan_with_figures(group_plan)

    return group_plan
