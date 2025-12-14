# Presentation_module/planning.py

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SlidePlan:
    slide_id: int
    blueprint_text: str         # the "Slide X: ..." line
    group_name: str             # e.g. "Introduction to Continuous Lasing"
    group_purpose: Optional[str]
    group_scope: Optional[str]
    group_transition: Optional[str]
    suggested_figures: List[str] = field(default_factory=list)
    supmat_notes: Optional[str] = None


@dataclass
class SlideGroupPlan:
    group_name: str
    purpose: Optional[str]
    scope: Optional[str]
    key_terms: List[str]
    transition: Optional[str]
    supmat_notes: Optional[str]
    slides: List[SlidePlan]


def _extract_single_line(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None

    # If there is at least one capturing group, use the first group.
    # Otherwise use the full match.
    if m.lastindex and m.group(1) is not None:
        return m.group(1).strip()

    return m.group(0).strip()


def _extract_all(pattern: str, text: str) -> List[str]:
    return [m.strip() for m in re.findall(pattern, text, flags=re.MULTILINE)]


def parse_slide_group_file(path: str | Path) -> SlideGroupPlan:
    text = Path(path).read_text(encoding="utf-8")

    group_name_match = re.search(
        r"===\s*SLIDE_GROUP:\s*(.*?)\s*\|", text, flags=re.IGNORECASE
    )
    group_name = group_name_match.group(1).strip() if group_name_match else "UNNAMED_GROUP"

    # Purpose: prefer explicit label, else use the first "-" bullet line
    purpose = _extract_single_line(r"^\s*PURPOSE:\s*(.*)$", text)
    if purpose is None:
        purpose = _extract_single_line(r"^\s*-\s*(.*)$", text)

    # Scope: prefer explicit label, else use the "Key items from scope" line
    scope = _extract_single_line(r"^\s*CONTENT_SCOPE:\s*(.*)$", text)
    if scope is None:
        scope = _extract_single_line(r"^\s*-\s*Key items from scope:\s*(.*)$", text)

    key_terms_raw = _extract_single_line(r"^\s*KEY_TERMS:\s*(.*)$", text)
    key_terms: List[str] = []
    if key_terms_raw:
        key_terms = [k.strip() for k in key_terms_raw.split(",") if k.strip()]

    transition = _extract_single_line(r"^\s*Transition:\s*(.*)$", text)
    supmat_notes = _extract_single_line(r"^\s*\[SupMat\]\s*(.*)$", text)

    slide_lines = re.findall(
        r"Slide\s+(\d+):\s*(.*)", text, flags=re.IGNORECASE
    )

    figure_lines = re.findall(
        r"Include:\s*(.*)", text, flags=re.IGNORECASE
    )

    slides: List[SlidePlan] = []
    for slide_num_str, blueprint_text in slide_lines:
        slide_id = int(slide_num_str)
        slides.append(
            SlidePlan(
                slide_id=slide_id,
                blueprint_text=blueprint_text.strip(),
                group_name=group_name,
                group_purpose=purpose,
                group_scope=scope,
                group_transition=transition,
                suggested_figures=figure_lines,
                supmat_notes=supmat_notes,
            )
        )

    return SlideGroupPlan(
        group_name=group_name,
        purpose=purpose,
        scope=scope,
        key_terms=key_terms,
        transition=transition,
        supmat_notes=supmat_notes,
        slides=slides,
    )

