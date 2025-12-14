# Presentation_module/pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .planning import parse_slide_group_file, SlideGroupPlan, SlidePlan
from .bullets import number_bullet_points, BulletPoint
from .assignment import BulletCandidate, propose_bullet_candidates
from .models import PlanningResult, SlideBundle
from .slide_context import build_slide_context  # add near imports at top



def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_planning_pipeline(
    slide_group_file: str | Path,
    bullet_summary_file: str | Path,
    model: str,
    output_dir: str | Path | None = None,
    run_name: str = "run",
    debug_save: bool = False,
    debug_filename: str = "planning_debug.json",
) -> PlanningResult:
    """
    PLANNING STAGE (NO RAG, NO FINAL SLIDES)

    For ONE slide group:
      1) Parse slide-group plan -> SlideGroupPlan + SlidePlan objects
      2) Parse + number article bullets -> BulletPoint objects
      3) Ask LLM for *candidate* bullet placements -> BulletCandidate objects
      4) Attach candidates to slides (in memory)
      5) Optionally write ONE debug JSON file

    Returns:
        PlanningResult (pure in-memory representation)
    """

    slide_group_file = Path(slide_group_file)
    bullet_summary_file = Path(bullet_summary_file)

    # --------------------------------------------------
    # 1) Parse slide-group plan
    # --------------------------------------------------
    group_plan: SlideGroupPlan = parse_slide_group_file(slide_group_file)

    # --------------------------------------------------
    # 2) Parse & number bullets
    # --------------------------------------------------
    raw_bullets = bullet_summary_file.read_text(encoding="utf-8")
    numbered_bullets: List[BulletPoint] = number_bullet_points(raw_bullets)

    # Use run_name as a stable group_id for slide_uid construction
    group_id = run_name

    # --------------------------------------------------
    # 3) Candidate generation (LLM)
    # --------------------------------------------------
    candidates: List[BulletCandidate] = propose_bullet_candidates(
        group_plan=group_plan,
        bullets=numbered_bullets,
        model=model,
        group_id=group_id,
        debug_save=False,   # all debug handled here, not in assignment.py
        debug_dir=None,
    )

    # --------------------------------------------------
    # 4) Build slide bundles (plan + candidates)
    # --------------------------------------------------
    slideplan_by_uid: Dict[str, SlidePlan] = {
        f"{group_id}::Slide_{s.slide_id}": s
        for s in group_plan.slides
    }

    # Build slide bundles with slide_context
    slides: Dict[str, SlideBundle] = {}

    # Sort slide_uids by slide_id to get previous/next continuity inside the group
    ordered = sorted(
        slideplan_by_uid.items(),
        key=lambda kv: kv[1].slide_id
    )

    prev_blueprint = None
    for uid, sp in ordered:
        ctx = build_slide_context(
            group_plan=group_plan,
            slide_plan=sp,
            model=model,
            prev_slide_blueprint=prev_blueprint,
        )
        slides[uid] = SlideBundle(slide_uid=uid, slide_plan=sp, slide_context=ctx)
        prev_blueprint = sp.blueprint_text

    # Attach candidates to slides
    for c in candidates:
        if c.slide_uid in slides:
            slides[c.slide_uid].candidates.append(c)

    # --------------------------------------------------
    # 5) Build PlanningResult (in memory)
    # --------------------------------------------------
    result = PlanningResult(
        group_id=group_id,
        group_plan=group_plan,
        numbered_bullets=numbered_bullets,
        candidates=candidates,
        slides=slides,
    )

    # --------------------------------------------------
    # 6) Optional: ONE debug JSON
    # --------------------------------------------------
    if debug_save and output_dir is not None:
        out_base = Path(output_dir) / run_name
        _ensure_dir(out_base)

        debug_payload = {
            "group_id": result.group_id,
            "group_name": result.group_plan.group_name,
            "purpose": result.group_plan.purpose,
            "scope": result.group_plan.scope,
            "transition": result.group_plan.transition,
            "supmat_notes": result.group_plan.supmat_notes,
            "slides": [
                {
                    "slide_uid": uid,
                    "slide_id": bundle.slide_plan.slide_id,
                    "plan": bundle.slide_plan.blueprint_text,
                    "slide_context": bundle.slide_context,
                    "suggested_figures": bundle.slide_plan.suggested_figures,
                    "candidates": [
                        {
                            "bullet_id": c.bullet_id,
                            "priority": c.priority,
                            "exclusivity": c.exclusivity,
                            "argument": c.argument,
                        }
                        for c in bundle.candidates
                    ],
                }
                for uid, bundle in sorted(result.slides.items())
            ],
            "numbered_bullets": [
                {"bullet_id": b.bullet_id, "text": b.text}
                for b in result.numbered_bullets
            ],
        }

        debug_path = out_base / debug_filename
        debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")

    return result
