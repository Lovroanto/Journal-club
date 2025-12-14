# Presentation_module/slide_context.py

from __future__ import annotations
from typing import Optional

from .llm import call_llm
from .planning import SlidePlan, SlideGroupPlan


def build_slide_context(
    group_plan: SlideGroupPlan,
    slide_plan: SlidePlan,
    model: str,
    prev_slide_blueprint: Optional[str] = None,
) -> str:
    """
    Generate a short slide-level context (planning layer, no RAG):
    - why the slide exists in the talk
    - what audience should learn
    - how it connects from previous and to next (if applicable)
    """
    prev_txt = prev_slide_blueprint if prev_slide_blueprint else "None (this is the first slide in the group)"

    prompt = f"""
You are preparing a scientific presentation plan (NOT writing the final slide).

GROUP:
- Name: {group_plan.group_name}
- Purpose: {group_plan.purpose}
- Scope: {group_plan.scope}
- Transition (group-level): {group_plan.transition}

PREVIOUS SLIDE (blueprint text):
{prev_txt}

CURRENT SLIDE (blueprint text):
{slide_plan.blueprint_text}

Task:
Write a concise slide-context note (4–6 sentences) that includes:
1) Why this slide exists at this point in the talk
2) What the audience should understand after this slide
3) How it connects to the previous slide (or how it starts the group if first)
4) What it sets up for the next slide

Output plain text only. No JSON. No bullet lists.
"""
    return call_llm(model, prompt).strip()
