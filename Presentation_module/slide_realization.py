# Presentation_module/slide_realization.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from .llm import call_llm
from .presentation_models import PresentationPlan
from .bullet_resolution import BulletResolutionResult
from .rag_loader import RAGStores
from .slide_specs import SlideSpec, EvidenceChunk


def _docs_to_text(docs) -> List[str]:
    # LangChain docs: doc.page_content + optional metadata
    out = []
    for d in docs:
        try:
            out.append(d.page_content)
        except Exception:
            out.append(str(d))
    return out


def retrieve_evidence_for_slide(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    primary_bullets: List[str],
    reminder_bullets: List[str],
    rag: RAGStores,
    k_main: int = 6,
    k_figs: int = 4,
    k_sup: int = 3,
    k_lit: int = 2,
    use_figs: bool = True,
    use_sup: bool = False,
    use_lit: bool = False,
    include_reminders_in_query: bool = False,
) -> List[EvidenceChunk]:
    """
    Retrieve evidence chunks for one slide.
    """
    parts = [
        f"SLIDE_UID: {slide_uid}",
        f"PLAN: {plan_text}",
        f"CONTEXT: {slide_context}",
        "PRIMARY BULLETS:",
        *[f"- {b}" for b in primary_bullets],
    ]

    # Reminders are OPTIONAL for retrieval
    if include_reminders_in_query and reminder_bullets:
        parts += ["REMINDERS:"] + [f"- {b}" for b in reminder_bullets]

    query = "\n".join(parts)

    evidence: List[EvidenceChunk] = []

    main_docs = rag.main.similarity_search(query, k=k_main)
    for t in _docs_to_text(main_docs):
        evidence.append(EvidenceChunk(source="MAIN", text=t))

    if use_figs:
        fig_docs = rag.figs.similarity_search(query, k=k_figs)
        for t in _docs_to_text(fig_docs):
            evidence.append(EvidenceChunk(source="FIG", text=t))

    if use_sup and rag.sup is not None:
        sup_docs = rag.sup.similarity_search(query, k=k_sup)
        for t in _docs_to_text(sup_docs):
            evidence.append(EvidenceChunk(source="SUP", text=t))

    if use_lit and rag.lit is not None:
        lit_docs = rag.lit.similarity_search(query, k=k_lit)
        for t in _docs_to_text(lit_docs):
            evidence.append(EvidenceChunk(source="LIT", text=t))

    return evidence


def generate_slide_spec(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    suggested_figures: List[str],
    primary_bullets: List[str],
    reminder_bullets: List[str],
    evidence: List[EvidenceChunk],
    model: str,
) -> SlideSpec:
    """
    Speech-first -> slide spec generation.
    Output must be strict JSON, parsed into SlideSpec.
    """
    ev_txt = "\n\n".join([f"[{e.source}]\n{e.text}" for e in evidence])

    prompt = f"""
You are generating ONE slide for a scientific talk about ONE specific paper.

SLIDE_UID: {slide_uid}

SLIDE PLAN:
{plan_text}

SLIDE CONTEXT (role in presentation):
{slide_context}

SUGGESTED FIGURES:
{json.dumps(suggested_figures, indent=2)}

PRIMARY BULLETS (must be covered in the SPEECH):
{json.dumps(primary_bullets, indent=2)}

REMINDERS (may be referenced briefly as "as mentioned earlier"):
{json.dumps(reminder_bullets, indent=2)}

EVIDENCE (from RAG, use MAIN first, FIG if relevant, SUP if needed, LIT only briefly):
{ev_txt}

TASK:
1) Write SPEAKER_NOTES first (150–220 words). Must cover all PRIMARY BULLETS.
   If reminders are used, phrase them as recall.
   If you use a figure, explicitly refer to it ("In Figure ...").

2) Then create the SLIDE CONTENT as a compression of the speech:
   - Title (short)
   - 3–5 slide bullets (short, not full sentences)
   - figure_used: either one of suggested_figures or null
   - Transition: one sentence leading to next slide

OUTPUT ONLY VALID JSON (no markdown, no commentary):
{{
  "title": "...",
  "figure_used": null,
  "on_slide_bullets": ["...", "..."],
  "speaker_notes": "...",
  "transition": "...",
  "reminders": ["..."] 
}}
"""

    raw = call_llm(model, prompt).strip()

    data = json.loads(raw)

    return SlideSpec(
        slide_uid=slide_uid,
        title=str(data.get("title", "")).strip(),
        figure_used=data.get("figure_used", None),
        on_slide_bullets=[str(x).strip() for x in data.get("on_slide_bullets", [])],
        speaker_notes=str(data.get("speaker_notes", "")).strip(),
        transition=str(data.get("transition", "")).strip(),
        reminders=[str(x).strip() for x in data.get("reminders", [])],
        evidence=evidence,
    )


def realize_presentation_slides(
    presentation: PresentationPlan,
    resolved: BulletResolutionResult,
    rag: RAGStores,
    model: str,
    output_dir: Optional[Union[str, Path]] = None,
    debug_save: bool = False,
    only_slide_uids: Optional[List[str]] = None,
) -> Dict[str, SlideSpec]:
    """
    Generate SlideSpec for every slide (or subset).
    only_slide_uids: list of slide_uid strings to generate; if None => generate all.
    """

    slide_specs: Dict[str, SlideSpec] = {}

    all_items = sorted(presentation.all_slides.items())
    if only_slide_uids is not None:
        only_set = set(only_slide_uids)
        all_items = [(uid, b) for (uid, b) in all_items if uid in only_set]

    for slide_uid, bundle in all_items:
        usages = resolved.usage_by_slide.get(slide_uid, [])
        primary = [u for u in usages if u.usage == "primary"]
        reminders = [u for u in usages if u.usage == "reminder"]

        primary_texts = [u.bullet_text for u in primary if u.bullet_text]
        reminder_texts = [u.bullet_text for u in reminders if u.bullet_text]

        use_figs = bool(bundle.slide_plan.suggested_figures)
        # conservative defaults; you can add heuristics later
        use_sup = False
        use_lit = False

        evidence = retrieve_evidence_for_slide(
            slide_uid=slide_uid,
            plan_text=bundle.slide_plan.blueprint_text,
            slide_context=bundle.slide_context,
            primary_bullets=primary_texts,
            reminder_bullets=reminder_texts,
            rag=rag,
            use_figs=use_figs,
            use_sup=use_sup,
            use_lit=use_lit,
        )

        spec = generate_slide_spec(
            slide_uid=slide_uid,
            plan_text=bundle.slide_plan.blueprint_text,
            slide_context=bundle.slide_context,
            suggested_figures=bundle.slide_plan.suggested_figures,
            primary_bullets=primary_texts,
            reminder_bullets=reminder_texts,
            evidence=evidence,
            model=model,
        )

        slide_specs[slide_uid] = spec

    if debug_save and output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "slide_specs_debug.json").write_text(
            json.dumps({k: asdict(v) for k, v in slide_specs.items()}, indent=2),
            encoding="utf-8",
        )

    return slide_specs
