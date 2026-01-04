# Presentation_module/assignment.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal, Union

from .llm import call_llm
from .planning import SlideGroupPlan
from .bullets import BulletPoint


Exclusivity = Literal["strong", "medium", "weak"]


@dataclass
class BulletCandidate:
    bullet_id: str
    slide_uid: str
    priority: float              # 0..1
    exclusivity: Exclusivity     # "strong" | "medium" | "weak"
    argument: str


def _extract_first_json_array(raw: str) -> list:
    """
    Robustly parse the first JSON array found in raw LLM output.
    - Handles text before/after JSON.
    - Handles trailing corruption by trimming the tail until valid.
    """
    raw = raw.strip()

    # 1) Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) Extract substring from first '[' to last ']'
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array bracketed by [ ... ] found in model output.")

    candidate = raw[start : end + 1]

    # 3) Try parse candidate
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 4) Tail-trim recovery: often the model breaks near the end
    # Scan backwards for a closing ']' and attempt parse at each possible cut.
    # Limit scans to last ~50k chars to avoid huge slowdowns.
    scan_from = max(0, len(candidate) - 50000)
    for cut in range(len(candidate) - 1, scan_from, -1):
        if candidate[cut] != "]":
            continue
        snippet = candidate[: cut + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue

    # If we got here, we couldn't recover
    raise ValueError(
        "Could not parse a valid JSON array from model output (even after trimming). "
        "Enable debug_save=True to inspect candidates_raw.txt."
    )

def propose_bullet_candidates(
    group_plan: SlideGroupPlan,
    bullets: List[BulletPoint],
    model: str,
    group_id: Optional[str] = None,
    debug_save: bool = False,
    debug_dir: Optional[Union[str, Path]] = None,
    min_per_slide: int = 6,
    max_per_slide: int = 12,
) -> List[BulletCandidate]:
    """
    For ONE slide group:
    - Ask LLM to propose bullet candidates PER SLIDE
    - AND assign at most ONE figure per slide (from available figures)
    """

    if group_id is None:
        group_id = group_plan.group_name.replace(" ", "_")

    slide_blueprints = [
        {
            "slide_id": s.slide_id,
            "slide_uid": f"{group_id}::Slide_{s.slide_id}",
            "blueprint": s.blueprint_text,
        }
        for s in group_plan.slides
    ]

    bullet_list = [{"bullet_id": b.bullet_id, "text": b.text} for b in bullets]

    available_figures = group_plan.available_figures or []

    prompt = f"""
You are planning a scientific presentation about ONE paper.
We are currently working on ONE SLIDE GROUP.

GROUP_ID: {group_id}
GROUP_NAME: {group_plan.group_name}
GROUP_PURPOSE: {group_plan.purpose}
GROUP_SCOPE: {group_plan.scope}
GROUP_TRANSITION: {group_plan.transition}
SUPMAT_NOTES: {group_plan.supmat_notes}

SLIDES (use slide_uid exactly as given):
{json.dumps(slide_blueprints, indent=2)}

AVAILABLE FIGURES (each figure can be used at most once):
{json.dumps(available_figures, indent=2)}

NUMBERED BULLETS (use bullet_id exactly as given):
{json.dumps(bullet_list, indent=2)}

TASK:
For EACH slide:
1) Select between {min_per_slide} and {max_per_slide} bullet points that could support the slide.
2) Optionally assign ONE figure from AVAILABLE FIGURES that best supports the slide.
   - A figure may be used by at most ONE slide.
   - If no figure is appropriate, set figure_used to null.

Rules for bullets:
- Same bullet_id may appear on multiple slides with different priorities.
- Priority scale:
    0.85–1.00 = essential
    0.60–0.85 = good support
    0.35–0.60 = weak but plausible
- Do NOT output priorities below 0.35.
- exclusivity:
    strong = tightly bound to this slide
    medium = movable
    weak = optional
- argument must be specific and technical.

OUTPUT FORMAT:
Return ONLY valid JSON. No commentary, no markdown.

[
  {{
    "slide_uid": "{group_id}::Slide_1",
    "figure_used": "Figure X description" | null,
    "candidates": [
      {{
        "bullet_id": "BP_001",
        "priority": 0.82,
        "exclusivity": "medium",
        "argument": "..."
      }}
    ]
  }}
]
"""

    raw = call_llm(model, prompt)

    if debug_save and debug_dir is not None:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "candidates_raw.txt").write_text(raw, encoding="utf-8")

    data = _extract_first_json_array(raw)

    if debug_save and debug_dir is not None:
        d = Path(debug_dir)
        (d / "candidates.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    out: List[BulletCandidate] = []

    # NEW: map slide_uid → figure_used
    figure_by_slide: dict[str, Optional[str]] = {}

    for slide_block in data:
        slide_uid = str(slide_block.get("slide_uid", "")).strip()
        if not slide_uid:
            continue

        fig = slide_block.get("figure_used", None)
        if isinstance(fig, str):
            fig = fig.strip()
        else:
            fig = None

        figure_by_slide[slide_uid] = fig

        cand_list = slide_block.get("candidates", [])
        if not isinstance(cand_list, list):
            continue

        for item in cand_list:
            bullet_id = str(item.get("bullet_id", "")).strip()
            if not bullet_id:
                continue

            argument = str(item.get("argument", "")).strip() or "MISSING_ARGUMENT"

            try:
                priority = float(item.get("priority", 0.5))
            except (TypeError, ValueError):
                priority = 0.5

            priority = max(0.35, min(priority, 1.0))

            exclusivity = str(item.get("exclusivity", "medium")).lower()
            if exclusivity not in ("strong", "medium", "weak"):
                exclusivity = "medium"

            out.append(
                BulletCandidate(
                    bullet_id=bullet_id,
                    slide_uid=slide_uid,
                    priority=priority,
                    exclusivity=exclusivity,  # type: ignore
                    argument=argument,
                )
            )

    # 🔗 Attach figure_used back to SlidePlan objects
    for s in group_plan.slides:
        uid = f"{group_id}::Slide_{s.slide_id}"
        s.figure_used = figure_by_slide.get(uid)

    return out

