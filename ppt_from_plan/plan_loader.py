from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class SlidePlan:
    slide_uid: str
    title: str
    figure_used: Optional[str]
    bullets: List[str]
    speaker_notes: str
    transition: str
    reminders: List[str]


def load_plan(path: str) -> List[SlidePlan]:
    """
    Expected JSON structure (top-level):
      {
        ...metadata...,
        "slides": [
          {
            "slide_uid": "...",
            "title": "...",
            "figure_used": null or "...",
            "on_slide_bullets": ["...", ...],
            "speaker_notes": "...",
            "transition": "...",
            "reminders": [...]
          }, ...
        ]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if "slides" not in raw or not isinstance(raw["slides"], list):
        raise ValueError("Plan JSON must contain a top-level 'slides' list.")

    out: List[SlidePlan] = []
    for s in raw["slides"]:
        if not isinstance(s, dict):
            continue

        out.append(
            SlidePlan(
                slide_uid=str(s.get("slide_uid", "")),
                title=str(s.get("title", "")),
                figure_used=s.get("figure_used", None),
                bullets=list(s.get("on_slide_bullets", [])) or [],
                speaker_notes=str(s.get("speaker_notes", "") or ""),
                transition=str(s.get("transition", "") or ""),
                reminders=list(s.get("reminders", [])) or [],
            )
        )

    return out
