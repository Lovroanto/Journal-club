from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple


EXCLUSIVITY_WEIGHT = {
    "strong": 1.20,
    "medium": 1.00,
    "weak": 0.85,
}


@dataclass
class ResolvedPlacement:
    bullet_id: str
    slide_uid: str
    score: float
    priority: float
    exclusivity: str
    argument: str


def resolve_global_unique(
    candidates: List[dict],
) -> Tuple[Dict[str, ResolvedPlacement], List[str]]:
    """
    Enforce: each bullet_id used at most once.
    Select best candidate by (priority * exclusivity_weight).

    Returns:
        resolved: bullet_id -> ResolvedPlacement
        unused: list of bullet_ids that had no candidates at all
    """
    by_bullet: Dict[str, List[dict]] = {}
    for c in candidates:
        by_bullet.setdefault(c["bullet_id"], []).append(c)

    resolved: Dict[str, ResolvedPlacement] = {}
    unused: List[str] = []

    for bullet_id, opts in by_bullet.items():
        if not opts:
            unused.append(bullet_id)
            continue

        best = None
        best_score = -1.0
        for o in opts:
            excl = str(o.get("exclusivity", "medium")).lower().strip()
            w = EXCLUSIVITY_WEIGHT.get(excl, 1.00)
            pr = float(o.get("priority", 0.0))
            score = pr * w
            if score > best_score:
                best_score = score
                best = o

        if best is None:
            unused.append(bullet_id)
            continue

        resolved[bullet_id] = ResolvedPlacement(
            bullet_id=bullet_id,
            slide_uid=str(best["slide_uid"]),
            score=best_score,
            priority=float(best.get("priority", 0.0)),
            exclusivity=str(best.get("exclusivity", "medium")),
            argument=str(best.get("argument", "")),
        )

    return resolved, unused
