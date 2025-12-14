# Presentation_module/presentation_pipeline.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

from .pipeline import run_planning_pipeline
from .presentation_models import PresentationPlan


def run_presentation_planning(
    slide_groups_dir: Union[str, Path],
    bullet_summary_file: Union[str, Path],
    model: str,
    output_dir: Optional[Union[str, Path]] = None,
    debug_save: bool = False,
    debug_filename: str = "presentation_planning_debug.json",
) -> PresentationPlan:
    """
    Runs planning for ALL slide-group txt files in a folder.

    - Each .txt file in slide_groups_dir is treated as one slide-group plan.
    - Produces a list of PlanningResult objects bundled into PresentationPlan.
    - Optionally writes ONE debug JSON for the whole presentation.
    """

    slide_groups_dir = Path(slide_groups_dir)
    bullet_summary_file = Path(bullet_summary_file)

    # Pick up all .txt files; sort by filename so 001_, 002_... define order
    group_files = sorted(slide_groups_dir.glob("*.txt"))

    groups: List = []
    for f in group_files:
        # Use the filename stem as run_name => stable group_id and slide_uid prefix
        run_name = f.stem

        res = run_planning_pipeline(
            slide_group_file=f,
            bullet_summary_file=bullet_summary_file,
            model=model,
            output_dir=None,        # avoid per-group files
            run_name=run_name,
            debug_save=False,       # centralized debug below
        )
        groups.append(res)

    plan = PresentationPlan(groups=groups)

    # Optional: ONE debug file for the whole presentation
    if debug_save and output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_path = out_dir / debug_filename

        payload = {
            "slide_groups_dir": str(slide_groups_dir),
            "bullet_summary_file": str(bullet_summary_file),
            "num_groups": len(plan.groups),
            "groups": [
                {
                    "group_id": g.group_id,
                    "group_name": g.group_plan.group_name,
                    "purpose": g.group_plan.purpose,
                    "scope": g.group_plan.scope,
                    "transition": g.group_plan.transition,
                    "num_slides": len(g.slides),
                    "slides": [
                        {
                            "slide_uid": uid,
                            "slide_id": b.slide_plan.slide_id,
                            "plan": b.slide_plan.blueprint_text,
                            "slide_context": b.slide_context,
                            "suggested_figures": b.slide_plan.suggested_figures,
                            "num_candidates": len(b.candidates),
                            "candidates": [
                                {
                                    "bullet_id": c.bullet_id,
                                    "priority": c.priority,
                                    "exclusivity": c.exclusivity,
                                    "argument": c.argument,
                                }
                                for c in b.candidates
                            ],
                        }
                        for uid, b in sorted(g.slides.items())
                    ],
                }
                for g in plan.groups
            ],
        }

        debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return plan
