# Presentation_module/bullets.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re


@dataclass
class BulletPoint:
    bullet_id: str   # e.g. "BP_001"
    text: str        # the original bullet text


def parse_bullet_points(raw_text: str) -> List[str]:
    """
    Parse a big string containing bullet points into a list of clean bullet texts.

    We assume bullets look like:
        • text...
        - text...
        * text...

    Empty lines are ignored.
    """
    lines = raw_text.splitlines()
    bullets: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Remove common bullet symbols
        stripped = re.sub(r"^[•\-\*\u2022]+\s*", "", stripped)

        if stripped:
            bullets.append(stripped)

    return bullets


def number_bullet_points(raw_text: str) -> List[BulletPoint]:
    """
    Parse and number all bullet points from the article summary.

    Returns a list of BulletPoint with bullet_id = "BP_001", "BP_002", ...
    """
    bullet_texts = parse_bullet_points(raw_text)
    numbered: List[BulletPoint] = []

    for idx, txt in enumerate(bullet_texts, start=1):
        bullet_id = f"BP_{idx:03d}"
        numbered.append(BulletPoint(bullet_id=bullet_id, text=txt))

    return numbered
