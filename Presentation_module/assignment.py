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
    Handles cases where model prints text before/after JSON.
    """
    raw = raw.strip()

    # 1) Try direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) Try substring from first '[' to last ']'
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed

    raise ValueError(f"Could not extract a JSON array from model output:\n{raw}")


def propose_bullet_candidates(
    group_plan: SlideGroupPlan,
    bullets: List[BulletPoint],
    model: str,
    group_id: Optional[str] = None,
    debug_save: bool = False,
    debug_dir: Optional[Union[str, Path]] = None,
) -> List[BulletCandidate]:
    """
    For ONE slide group:
    - Ask LLM to propose *candidate* placements bullet -> slide_uid (0..N per bullet).
    - Each candidate includes priority (0..1) and exclusivity (strong/medium/weak).
    - Keeps everything in memory as BulletCandidate objects.
    - Optionally writes debug files (raw + parsed + skipped) if debug_save=True.
    """

    if group_id is None:
        # Fallback: use group name, but file-stem is better if you pass it in
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

NUMBERED BULLETS (use bullet_id exactly as given):
{json.dumps(bullet_list, indent=2)}

TASK:
Propose CANDIDATE placements of bullets into slides for THIS group.

Rules:
- You may output ZERO candidates for a bullet if it is not useful in this group.
- Prefer 0 or 1 candidate per bullet; allow 2 only if truly necessary.
- Do NOT output "Not assigned". If you don't assign it, just omit it.
- Do NOT repeat the same bullet across many slides.

For each candidate you MUST provide ALL keys:
- bullet_id
- slide_uid
- priority: number from 0.0 to 1.0 (how strongly it belongs on that slide)
- exclusivity: one of ["strong","medium","weak"]
    strong = essential/only makes sense here
    medium = good fit, but could go elsewhere
    weak = optional/context; could be moved/removed
- argument: SPECIFIC reason tied to the slide blueprint (not generic)

OUTPUT FORMAT:
Return ONLY a JSON array. The output MUST begin with '[' and end with ']'.
No commentary, no markdown.

[
  {{
    "bullet_id": "BP_001",
    "slide_uid": "{group_id}::Slide_1",
    "priority": 0.82,
    "exclusivity": "strong",
    "argument": "..."
  }}
]
"""

    raw = call_llm(model, prompt)

    # Optional debug writing: raw output
    if debug_save and debug_dir is not None:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "candidates_raw.txt").write_text(raw, encoding="utf-8")

    data = _extract_first_json_array(raw)

    # Optional debug writing: parsed JSON
    if debug_save and debug_dir is not None:
        d = Path(debug_dir)
        (d / "candidates.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    out: List[BulletCandidate] = []
    skipped: List[dict] = []

    for item in data:
        # Required fields
        bullet_id = str(item.get("bullet_id", "")).strip()
        slide_uid = str(item.get("slide_uid", "")).strip()

        if not bullet_id or not slide_uid:
            skipped.append(item)
            continue

        # Argument (accept common alternative key "reason")
        argument = str(item.get("argument", "")).strip()
        if not argument:
            argument = str(item.get("reason", "")).strip()
        if not argument:
            argument = "MISSING_ARGUMENT_FROM_MODEL"

        # Priority (robust parsing + clamp)
        try:
            priority = float(item.get("priority", 0.5))
        except (TypeError, ValueError):
            priority = 0.5

        if priority < 0.0:
            priority = 0.0
        elif priority > 1.0:
            priority = 1.0

        # Exclusivity (robust parsing)
        exclusivity = str(item.get("exclusivity", "medium")).strip().lower()
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

    # Optional debug writing: skipped items
    if debug_save and debug_dir is not None and skipped:
        d = Path(debug_dir)
        (d / "candidates_skipped_items.json").write_text(
            json.dumps(skipped, indent=2),
            encoding="utf-8",
        )

    return out
