# Presentation_module/bullet_resolution.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from .assignment import BulletCandidate
from .presentation_models import PresentationPlan


UsageType = Literal["primary", "reminder", "drop"]


@dataclass
class BulletUsage:
    bullet_id: str
    bullet_text: str
    slide_uid: str              # where it appears
    usage: UsageType            # primary / reminder / drop
    priority: float
    exclusivity: str
    argument: str
    primary_slide_uid: str      # where it is introduced


@dataclass
class BulletResolutionResult:
    """
    Output of global resolution.
    - primary_by_bullet: the chosen "introduction" slide per bullet
    - usage_by_slide: for each slide, list of BulletUsage entries
    - report: counts/diagnostics
    """
    primary_by_bullet: Dict[str, str]
    usage_by_slide: Dict[str, List[BulletUsage]]
    report: Dict[str, int]


_EXCL_WEIGHT = {"strong": 1.20, "medium": 1.00, "weak": 0.85}


def _slide_order_map(presentation: PresentationPlan) -> Dict[str, int]:
    """
    Determine global slide order from PresentationPlan.groups order and slide_id.
    Assumes groups are already in presentation order.
    """
    order: Dict[str, int] = {}
    idx = 0
    for g in presentation.groups:
        # Sort slides by numeric slide_id within group for stable ordering
        slides_sorted = sorted(g.slides.items(), key=lambda kv: kv[1].slide_plan.slide_id)
        for slide_uid, _bundle in slides_sorted:
            order[slide_uid] = idx
            idx += 1
    return order


def resolve_bullets_globally(
    presentation: PresentationPlan,
    *,
    low_cutoff: float = 0.55,
    early_bonus: float = 0.06,
    debug_save: bool = False,
    debug_path: Optional[Union[str, Path]] = None,
) -> BulletResolutionResult:
    """
    Global pass across the whole presentation.

    Policy:
    - For each bullet_id, pick ONE primary slide (where it is introduced).
      score = priority * exclusivity_weight + early_bonus_scaled
    - Other occurrences:
        - if priority < low_cutoff -> drop
        - else -> reminder (used in narration as "as mentioned earlier...")
    - Allows a bullet to be referenced multiple times, but distinguishes first intro vs later recall.

    Args:
        low_cutoff: below this => drop (weak match noise)
        early_bonus: small preference for earlier introduction (0..~0.1)
        debug_save: optionally writes one JSON summary
        debug_path: where to write the debug JSON

    Returns:
        BulletResolutionResult
    """
    slide_order = _slide_order_map(presentation)
    n_slides = max(1, len(slide_order))

    # Build bullet_text map (assumes same bullet list used everywhere)
    bullet_text: Dict[str, str] = {}
    for g in presentation.groups:
        for b in g.numbered_bullets:
            bullet_text.setdefault(b.bullet_id, b.text)

    # Collect all occurrences by bullet_id
    by_bullet: Dict[str, List[BulletCandidate]] = {}
    for c in presentation.all_candidates:
        by_bullet.setdefault(c.bullet_id, []).append(c)

    # Prepare usage containers
    usage_by_slide: Dict[str, List[BulletUsage]] = {uid: [] for uid in slide_order.keys()}
    primary_by_bullet: Dict[str, str] = {}

    count_primary = 0
    count_reminder = 0
    count_drop = 0
    count_bullets_with_dupes = 0

    def intro_score(c: BulletCandidate) -> float:
        w = _EXCL_WEIGHT.get(str(c.exclusivity).lower().strip(), 1.00)
        idx = slide_order.get(c.slide_uid, n_slides)
        # earlier slides get slightly larger bonus
        bonus = early_bonus * (1.0 - (idx / max(1, n_slides - 1)))
        return float(c.priority) * w + bonus

    for bullet_id, occs in by_bullet.items():
        if not occs:
            continue
        if len(occs) > 1:
            count_bullets_with_dupes += 1

        # Choose primary as max intro_score, tie-break by higher priority then earlier slide
        occs_sorted = sorted(
            occs,
            key=lambda c: (intro_score(c), float(c.priority), -slide_order.get(c.slide_uid, 10**9)),
            reverse=True,
        )
        primary = occs_sorted[0]
        primary_uid = primary.slide_uid
        primary_by_bullet[bullet_id] = primary_uid

        # Now classify each occurrence
        for c in occs:
            if c.slide_uid not in usage_by_slide:
                # Slide not known in ordering map => skip safely
                continue

            if c.slide_uid == primary_uid:
                usage = "primary"
                count_primary += 1
            else:
                if float(c.priority) < low_cutoff:
                    usage = "drop"
                    count_drop += 1
                else:
                    usage = "reminder"
                    count_reminder += 1

            usage_by_slide[c.slide_uid].append(
                BulletUsage(
                    bullet_id=bullet_id,
                    bullet_text=bullet_text.get(bullet_id, ""),
                    slide_uid=c.slide_uid,
                    usage=usage,  # type: ignore
                    priority=float(c.priority),
                    exclusivity=str(c.exclusivity),
                    argument=str(c.argument),
                    primary_slide_uid=primary_uid,
                )
            )

    # Sort usage lists per slide: primary first, then reminders (high→low), drops last
    usage_rank = {"primary": 0, "reminder": 1, "drop": 2}
    for uid in usage_by_slide:
        usage_by_slide[uid].sort(
            key=lambda u: (usage_rank[u.usage], -u.priority)
        )

    report = {
        "num_slides": len(slide_order),
        "num_bullets_seen": len(by_bullet),
        "bullets_with_duplicates": count_bullets_with_dupes,
        "primary_assignments": count_primary,
        "reminders": count_reminder,
        "drops": count_drop,
    }

    result = BulletResolutionResult(
        primary_by_bullet=primary_by_bullet,
        usage_by_slide=usage_by_slide,
        report=report,
    )

    if debug_save and debug_path is not None:
        p = Path(debug_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "report": report,
            "primary_by_bullet": primary_by_bullet,
            "usage_by_slide": {
                uid: [
                    {
                        "bullet_id": u.bullet_id,
                        "usage": u.usage,
                        "priority": u.priority,
                        "exclusivity": u.exclusivity,
                        "primary_slide_uid": u.primary_slide_uid,
                        "bullet_text": u.bullet_text,
                        "argument": u.argument,
                    }
                    for u in items
                ]
                for uid, items in usage_by_slide.items()
            },
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return result
