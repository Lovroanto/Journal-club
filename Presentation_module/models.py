# Presentation_module/models.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from .planning import SlideGroupPlan, SlidePlan
from .bullets import BulletPoint
from .assignment import BulletCandidate


@dataclass
class SlideBundle:
    """
    One slide, with its plan (blueprint) + candidate bullet matches.
    """
    slide_uid: str
    slide_plan: SlidePlan
    candidates: List[BulletCandidate] = field(default_factory=list)


@dataclass
class PlanningResult:
    """
    One slide-group run result: plans + bullets + candidates, all in memory.
    """
    group_id: str
    group_plan: SlideGroupPlan
    numbered_bullets: List[BulletPoint]
    candidates: List[BulletCandidate]
    slides: Dict[str, SlideBundle]  # slide_uid -> SlideBundle
