# Presentation_module/slide_selector.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Tuple

from .presentation_models import PresentationPlan


@dataclass(frozen=True)
class SlideRef:
    group_id: str
    slide_id: int

    def __str__(self) -> str:
        return f"{self.group_id}::Slide_{self.slide_id}"


def list_groups(presentation: PresentationPlan) -> List[str]:
    """Return group_ids in presentation order."""
    return [g.group_id for g in presentation.groups]


def list_slides_in_group(presentation: PresentationPlan, group_id: str) -> List[str]:
    """Return slide_uids for a group_id (sorted by slide_id)."""
    group = next((g for g in presentation.groups if g.group_id == group_id), None)
    if group is None:
        raise KeyError(f"Unknown group_id: {group_id}")
    return [uid for uid, _ in sorted(group.slides.items(), key=lambda x: x[1].slide_plan.slide_id)]


def resolve_slide_uid(
    presentation: PresentationPlan,
    *,
    slide_uid: Optional[str] = None,
    group_id: Optional[str] = None,
    slide_id: Optional[int] = None,
    group_index: Optional[int] = None,
    group_name_contains: Optional[str] = None,
) -> str:
    """
    Resolve a human-friendly reference to a concrete slide_uid.

    Supported:
    - slide_uid="003_Intro::Slide_2"
    - group_id="003_Intro", slide_id=2
    - group_index=0, slide_id=2
    - group_name_contains="Continuous Lasing", slide_id=2
    """
    if slide_uid:
        if slide_uid not in presentation.all_slides:
            raise KeyError(f"slide_uid not found: {slide_uid}")
        return slide_uid

    if slide_id is None:
        raise ValueError("slide_id is required unless slide_uid is provided.")

    # 1) group_id exact
    if group_id is not None:
        uid = f"{group_id}::Slide_{slide_id}"
        if uid not in presentation.all_slides:
            raise KeyError(f"Resolved uid not found: {uid}")
        return uid

    # 2) group_index
    if group_index is not None:
        if group_index < 0 or group_index >= len(presentation.groups):
            raise IndexError(f"group_index out of range: {group_index}")
        gid = presentation.groups[group_index].group_id
        uid = f"{gid}::Slide_{slide_id}"
        if uid not in presentation.all_slides:
            raise KeyError(f"Resolved uid not found: {uid}")
        return uid

    # 3) group_name_contains (substring match)
    if group_name_contains is not None:
        matches = [
            g.group_id for g in presentation.groups
            if group_name_contains.lower() in g.group_plan.group_name.lower()
        ]
        if not matches:
            raise KeyError(f"No group_name contains: {group_name_contains}")
        if len(matches) > 1:
            raise KeyError(f"Multiple groups match '{group_name_contains}': {matches}")
        gid = matches[0]
        uid = f"{gid}::Slide_{slide_id}"
        if uid not in presentation.all_slides:
            raise KeyError(f"Resolved uid not found: {uid}")
        return uid

    # 4) fallback: find any group with this slide_id (must be unique)
    candidates = [uid for uid in presentation.all_slides.keys() if uid.endswith(f"::Slide_{slide_id}")]
    if not candidates:
        raise KeyError(f"No slide found with slide_id={slide_id}")
    if len(candidates) > 1:
        raise KeyError(f"Ambiguous slide_id={slide_id}. Matches: {sorted(candidates)}")
    return candidates[0]
