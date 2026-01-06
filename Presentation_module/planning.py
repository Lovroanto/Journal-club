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
    blueprint_text: str         # the "Slide X: ..." line
    group_name: str             # e.g. "Introduction to Continuous Lasing"
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
    available_figures: List[str]      # group-level figures
    slides: List[SlidePlan]


# =========================
# Helpers
# =========================

def _extract_single_line(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None

    if m.lastindex and m.group(1) is not None:
        return m.group(1).strip()

    return m.group(0).strip()


def _extract_all(pattern: str, text: str) -> List[str]:
    return [m.strip() for m in re.findall(pattern, text, flags=re.MULTILINE)]


# =========================
# Figure assignment logic
# =========================

def assign_figures_to_slides(
    slides: List[SlidePlan],
    available_figures: List[str],
) -> None:
    """
    Deterministically assign figures to slides.

    Rules:
    - Each figure used at most once
    - Respect pre-assigned figures
    - Assign based on semantic overlap between slide intent and figure description
    """

    unused_figures = available_figures.copy()

    def semantic_score(slide: SlidePlan, figure: str) -> int:
        slide_text = (
            slide.blueprint_text.lower()
            + " "
            + (slide.group_scope or "").lower()
            + " "
            + (slide.group_purpose or "").lower()
        )

        fig_text = figure.lower()

        score = 0
        for token in re.findall(r"\w+", fig_text):
            if len(token) > 3 and token in slide_text:
                score += 1
        return score

    for slide in slides:
        # Keep LLM-selected figure if present
        if slide.figure_used:
            if slide.figure_used in unused_figures:
                unused_figures.remove(slide.figure_used)
            continue

        if not unused_figures:
            break

        best_figure = None
        best_score = 0

        for fig in unused_figures:
            s = semantic_score(slide, fig)
            if s > best_score:
                best_score = s
                best_figure = fig

        # Assign only if there's a meaningful match
        if best_figure and best_score > 0:
            slide.figure_used = best_figure
            unused_figures.remove(best_figure)


# =========================
# Parser
# =========================

def parse_slide_group_file(path: str | Path) -> SlideGroupPlan:
    text = Path(path).read_text(encoding="utf-8")

    # ---- Group name ----
    group_name_match = re.search(
        r"===\s*SLIDE_GROUP:\s*(.*?)\s*\|", text, flags=re.IGNORECASE
    )
    group_name = (
        group_name_match.group(1).strip()
        if group_name_match
        else "UNNAMED_GROUP"
    )

    # ---- Purpose ----
    purpose = _extract_single_line(r"^\s*PURPOSE:\s*(.*)$", text)
    if purpose is None:
        purpose = _extract_single_line(r"^\s*-\s*(.*)$", text)

    # ---- Scope ----
    scope = _extract_single_line(r"^\s*CONTENT_SCOPE:\s*(.*)$", text)
    if scope is None:
        scope = _extract_single_line(
            r"^\s*-\s*Key items from scope:\s*(.*)$", text
        )

    # ---- Key terms ----
    key_terms_raw = _extract_single_line(r"^\s*KEY_TERMS:\s*(.*)$", text)
    key_terms: List[str] = []
    if key_terms_raw:
        key_terms = [k.strip() for k in key_terms_raw.split(",") if k.strip()]

    # ---- Transition & SupMat ----
    transition = _extract_single_line(r"^\s*Transition:\s*(.*)$", text)
    supmat_notes = _extract_single_line(r"^\s*\[SupMat\]\s*(.*)$", text)

    # ---- Slides ----
    slide_lines = re.findall(
        r"Slide\s+(\d+):\s*(.*)", text, flags=re.IGNORECASE
    )

    slides: List[SlidePlan] = []
    for slide_num_str, blueprint_text in slide_lines:
        slides.append(
            SlidePlan(
                slide_id=int(slide_num_str),
                blueprint_text=blueprint_text.strip(),
                group_name=group_name,
                group_purpose=purpose,
                group_scope=scope,
                group_transition=transition,
                figure_used=None,          # intentionally empty
                supmat_notes=supmat_notes,
            )
        )

    # ---- Figures (group-level only) ----
    available_figures = _extract_all(
        r"Include:\s*(.*)", text
    )

    # ---- Planning-time figure assignment ----
    assign_figures_to_slides(slides, available_figures)

    return SlideGroupPlan(
        group_name=group_name,
        purpose=purpose,
        scope=scope,
        key_terms=key_terms,
        transition=transition,
        supmat_notes=supmat_notes,
        available_figures=available_figures,
        slides=slides,
    )
