from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from .pipeline import run_planning_pipeline
from .presentation_models import PresentationPlan
from .models import PlanningResult, SlideBundle
from .planning import SlideGroupPlan, SlidePlan
from .bullets import BulletPoint
from .assignment import BulletCandidate


def run_presentation_planning(
    slide_groups_dir: Union[str, Path],
    bullet_summary_file: Union[str, Path],
    model: str,
    output_dir: Optional[Union[str, Path]] = None,
    debug_save: bool = False,
    debug_filename: str = "presentation_planning_debug.json",
    continue_on_error: bool = True,
    error_log_filename: str = "planning_errors.json",
) -> PresentationPlan:
    """
    Runs planning for ALL slide-group .txt files in a folder.
    """

    slide_groups_dir = Path(slide_groups_dir)
    bullet_summary_file = Path(bullet_summary_file)

    group_files = sorted(slide_groups_dir.glob("*.txt"))

    groups: List[PlanningResult] = []
    errors: List[Dict[str, Any]] = []

    for f in group_files:
        run_name = f.stem
        try:
            res = run_planning_pipeline(
                slide_group_file=f,
                bullet_summary_file=bullet_summary_file,
                model=model,
                output_dir=None,
                run_name=run_name,
                debug_save=False,
            )
            groups.append(res)

        except Exception as e:
            err = {
                "group_file": str(f),
                "run_name": run_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            errors.append(err)

            if not continue_on_error:
                raise

    plan = PresentationPlan(groups=groups)

    # Write error log if needed
    if output_dir is not None and errors:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / error_log_filename).write_text(
            json.dumps(errors, indent=2),
            encoding="utf-8",
        )

    # Optional: ONE debug file for the whole presentation
    if debug_save and output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_path = out_dir / debug_filename

        payload = {
            "slide_groups_dir": str(slide_groups_dir),
            "bullet_summary_file": str(bullet_summary_file),
            "num_groups_total": len(group_files),
            "num_groups_success": len(plan.groups),
            "num_groups_failed": len(errors),
            "errors": errors,
            "groups": [
                {
                    "group_id": g.group_id,
                    "group_name": g.group_plan.group_name,
                    "purpose": g.group_plan.purpose,
                    "scope": g.group_plan.scope,
                    "transition": g.group_plan.transition,
                    "supmat_notes": g.group_plan.supmat_notes,
                    "available_figures": g.group_plan.available_figures,
                    "num_slides": len(g.slides),
                    "slides": [
                        {
                            "slide_uid": uid,
                            "slide_id": b.slide_plan.slide_id,
                            "plan": b.slide_plan.blueprint_text,
                            "slide_context": b.slide_context,
                            "figure_used": b.slide_plan.figure_used,
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

        debug_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    return plan


def load_presentation_plan_from_debug(
    debug_json_path: Union[str, Path]
) -> PresentationPlan:
    """
    Load PresentationPlan back from debug JSON (skip LLM calls).
    """

    p = Path(debug_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    groups = []

    for g in data["groups"]:
        group_plan = SlideGroupPlan(
            group_name=g["group_name"],
            purpose=g.get("purpose"),
            scope=g.get("scope"),
            key_terms=[],
            transition=g.get("transition"),
            supmat_notes=g.get("supmat_notes"),
            available_figures=g.get("available_figures", []),
            slides=[],
        )

        slides_dict = {}

        for s in g["slides"]:
            sp = SlidePlan(
                slide_id=int(s["slide_id"]),
                blueprint_text=s["plan"],
                group_name=g["group_name"],
                group_purpose=g.get("purpose"),
                group_scope=g.get("scope"),
                group_transition=g.get("transition"),
                figure_used=s.get("figure_used"),
                supmat_notes=g.get("supmat_notes"),
            )

            group_plan.slides.append(sp)

            bundle = SlideBundle(
                slide_uid=s["slide_uid"],
                slide_plan=sp,
                slide_context=s.get("slide_context", ""),
                candidates=[],
            )

            for c in s.get("candidates", []):
                bundle.candidates.append(
                    BulletCandidate(
                        bullet_id=c["bullet_id"],
                        slide_uid=s["slide_uid"],
                        priority=float(c["priority"]),
                        exclusivity=str(c["exclusivity"]),
                        argument=str(c["argument"]),
                    )
                )

            slides_dict[s["slide_uid"]] = bundle

        numbered_bullets = [
            BulletPoint(bullet_id=b["bullet_id"], text=b["text"])
            for b in g.get("numbered_bullets", data.get("numbered_bullets", []))
        ]

        all_cands = []
        for b in slides_dict.values():
            all_cands.extend(b.candidates)

        groups.append(
            PlanningResult(
                group_id=g["group_id"],
                group_plan=group_plan,
                numbered_bullets=numbered_bullets,
                candidates=all_cands,
                slides=slides_dict,
            )
        )

    return PresentationPlan(groups=groups)
