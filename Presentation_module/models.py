# Presentation_module/models.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from .planning import SlideGroupPlan, SlidePlan
from .bullets import BulletPoint
from .assignment import BulletCandidate


@dataclass
class SlideBundle:
    slide_uid: str
    slide_plan: SlidePlan
    slide_context: str = ""  # NEW: why this slide exists + how it connects
    candidates: List[BulletCandidate] = field(default_factory=list)


@dataclass
class PlanningResult:
    group_id: str
    group_plan: SlideGroupPlan
    numbered_bullets: List[BulletPoint]
    candidates: List[BulletCandidate]
    slides: Dict[str, SlideBundle]
