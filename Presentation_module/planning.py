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


# =========================
# Figure-aware planning
# =========================

def _semantic_score(slide: SlidePlan, figure: str) -> int:
    slide_text = (
        slide.blueprint_text.lower()
        + " "
        + (slide.group_scope or "").lower()
        + " "
        + (slide.group_purpose or "").lower()
    )
    score = 0
    for token in re.findall(r"\w+", figure.lower()):
        if len(token) > 3 and token in slide_text:
            score += 1
    return score


def _rewrite_blueprint_for_figure(slide: SlidePlan) -> None:
    """
    Rewrite slide blueprint so the figure becomes the core explanatory object.
    """
    if not slide.figure_used:
        return

    fig = slide.figure_used
    slide.blueprint_text = (
        f"Explain {fig} and how it supports the following point: "
        f"{slide.blueprint_text}"
    )


def _infer_insert_position(fig: str, slides: List[SlidePlan]) -> int:
    """
    Decide where a new slide explaining a figure should go.
    """
    f = fig.lower()
    if any(k in f for k in ["setup", "experimental", "apparatus"]):
        return 1
    if any(k in f for k in ["mechanism", "gain", "dynamics"]):
        return len(slides) // 2
    return len(slides)


def refine_plan_with_figures(group_plan: SlideGroupPlan) -> None:
    """
    Core architectural step:
    - Assign figures
    - Rewrite slide intents to center figures
    - Create slides for unused figures
    """

    slides = group_plan.slides
    available_figures = list(group_plan.available_figures)

    used_figures: List[str] = []

    # ---- Step 1: assign figures to existing slides ----
    for slide in slides:
        best_fig = None
        best_score = 0
        for fig in available_figures:
            s = _semantic_score(slide, fig)
            if s > best_score:
                best_score = s
                best_fig = fig

        if best_fig and best_score > 0:
            slide.figure_used = best_fig
            used_figures.append(best_fig)
            available_figures.remove(best_fig)

    # ---- Step 2: rewrite blueprints to center figures ----
    for slide in slides:
        _rewrite_blueprint_for_figure(slide)

    # ---- Step 3: create slides for unused figures ----
    next_slide_id = max(s.slide_id for s in slides) + 1 if slides else 1

    for fig in available_figures:
        new_slide = SlidePlan(
            slide_id=next_slide_id,
            blueprint_text=(
                f"Explain {fig} and its role in achieving "
                f"{group_plan.purpose or 'the goals of this section'}."
            ),
            group_name=group_plan.group_name,
            group_purpose=group_plan.purpose,
            group_scope=group_plan.scope,
            group_transition=group_plan.transition,
            figure_used=fig,
            supmat_notes=group_plan.supmat_notes,
        )

        insert_at = _infer_insert_position(fig, slides)
        slides.insert(insert_at, new_slide)
        next_slide_id += 1

    # ---- Step 4: re-number slides deterministically ----
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

    # 🔑 THIS IS THE FIX
    refine_plan_with_figures(group_plan)

    return group_plan
