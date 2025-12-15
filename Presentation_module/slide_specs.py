# Presentation_module/slide_specs.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict

SourceType = Literal["MAIN", "FIG", "SUP", "LIT"]

@dataclass
class EvidenceChunk:
    source: SourceType
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class SlideSpec:
    slide_uid: str
    title: str
    figure_used: Optional[str]
    on_slide_bullets: List[str]
    speaker_notes: str
    transition: str
    reminders: List[str] = field(default_factory=list)
    evidence: List[EvidenceChunk] = field(default_factory=list)
