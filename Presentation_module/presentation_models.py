# Presentation_module/presentation_models.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from .models import PlanningResult, SlideBundle
from .assignment import BulletCandidate


@dataclass
class PresentationPlan:
    """All groups planned, but not globally resolved yet."""
    groups: List[PlanningResult]

    @property
    def all_slides(self) -> Dict[str, SlideBundle]:
        out: Dict[str, SlideBundle] = {}
        for g in self.groups:
            out.update(g.slides)
        return out

    @property
    def all_candidates(self) -> List[BulletCandidate]:
        return [c for g in self.groups for c in g.candidates]
