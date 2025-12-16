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
    out: List[str] = []
    for d in docs:
        try:
            out.append(d.page_content)
        except Exception:
            out.append(str(d))
    return out

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[1] if "\n" in s else ""
        # remove trailing fence
        if "```" in s:
            s = s.rsplit("```", 1)[0]
    return s.strip()


def distill_evidence_for_slide(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    primary_bullets: List[str],
    reminder_bullets: List[str],
    evidence: List[EvidenceChunk],
    model: str,
    debug_path: Optional[Path] = None,
) -> str:
    """
    Compress raw evidence into a short list of grounded facts for THIS slide.
    Output: plain text (NOT json), ~6-10 bullet facts with [MAIN]/[FIG]/[SUP]/[LIT] tags.
    """
    ev_txt = "\n\n".join([f"[{e.source}]\n{e.text}" for e in evidence])

    prompt = f"""
You are an evidence distiller for ONE slide of a scientific talk.

SLIDE_UID: {slide_uid}
PLAN: {plan_text}
CONTEXT (role): {slide_context}

PRIMARY BULLETS (what must be supported):
{json.dumps(primary_bullets, indent=2)}

TASK:
From the evidence below, extract 6–10 short FACTS that directly help explain this slide.
Rules:
- Each fact must start with a source tag like [MAIN] or [FIG] or [SUP] or [LIT].
- Facts must be specific and relevant to PLAN + PRIMARY BULLETS (avoid experimental setup unless required).
- Do NOT summarize the whole paper. Do NOT mention "zones I/II/III/IV" unless the plan is about regimes/results.
- Keep each fact to one sentence.

EVIDENCE:
{ev_txt}

OUTPUT:
Return ONLY the list of facts as plain text lines.
"""

    raw = call_llm(model, prompt).strip()

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw, encoding="utf-8")

    return raw

def _extract_first_json_object(raw: str) -> dict:
    """
    Robustly extract the first JSON object from an LLM response.

    Handles:
    - text before/after JSON
    - ```json fences
    - literal newlines/tabs inside JSON strings (sanitized)
    - partial trailing corruption (tail-trim)
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("LLM returned empty output.")

    raw = _strip_code_fences(raw)

    # 1) direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) take substring from first '{' to last '}'
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output:\n{raw[:500]}")

    candidate = raw[start : end + 1]

    def _escape_in_strings(s: str) -> str:
        out = []
        in_string = False
        escape = False
        for ch in s:
            if not in_string:
                out.append(ch)
                if ch == '"':
                    in_string = True
                continue

            if escape:
                out.append(ch)
                escape = False
                continue

            if ch == "\\":
                out.append(ch)
                escape = True
                continue

            if ch == '"':
                out.append(ch)
                in_string = False
                continue

            if ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        return "".join(out)

    # 2a) parse candidate (try raw, then sanitized)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    candidate2 = _escape_in_strings(candidate)
    try:
        parsed = json.loads(candidate2)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3) tail-trim recovery: try progressively shorter endings
    scan_from = max(0, len(candidate2) - 50000)
    for cut in range(len(candidate2) - 1, scan_from, -1):
        if candidate2[cut] != "}":
            continue
        snippet = candidate2[: cut + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON object from model output (even after trimming).")



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
        f"PLAN: {plan_text}",
        "PRIMARY BULLETS (key claims to support):",
        *[f"- {b}" for b in primary_bullets],
    ]
    # Add only first 1–2 sentences of context (if you want)
    if slide_context:
        short_ctx = " ".join(slide_context.split()[:40])
        parts.append(f"CONTEXT (short): {short_ctx}")

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
    debug_raw_path: Optional[Path] = None,
) -> SlideSpec:
    """
    Speech-first -> slide spec generation.
    Output must be strict JSON, parsed into SlideSpec.
    """
    ev_txt = "\n\n".join([f"[{e.source}]\n{e.text}" for e in evidence])

    prompt = f"""
SYSTEM CONTRACT (MUST FOLLOW):
- You are generating EXACTLY ONE slide: {slide_uid}
- Do NOT generate multiple slides.
- Do NOT write "Slide 1", "Slide 2", "Here are the slides", or any multi-slide structure.
- Output MUST be a SINGLE JSON object and NOTHING ELSE.
- JSON MUST include the field "slide_uid" and it MUST equal "{slide_uid}".

INPUTS:
SLIDE_UID: {slide_uid}
PLAN: {plan_text}
CONTEXT: {slide_context}
SUGGESTED FIGURES: {json.dumps(suggested_figures, indent=2)}
PRIMARY BULLETS (must be covered): {json.dumps(primary_bullets, indent=2)}
REMINDERS (optional brief recall): {json.dumps(reminder_bullets, indent=2)}

EVIDENCE (use MAIN first; FIG if relevant; keep it grounded):
{ev_txt}

OUTPUT (STRICT JSON ONLY):
{{
  "slide_uid": "{slide_uid}",
  "title": "short title",
  "figure_used": null,
  "on_slide_bullets": ["...", "...", "..."],
  "speaker_notes": "150-220 words, covers all PRIMARY BULLETS, may mention REMINDERS as recall.",
  "transition": "one sentence leading to the next slide",
  "reminders": ["..."]
}}
"""


    last_err: Optional[Exception] = None
    raw = ""

    for attempt in range(2):
        raw = call_llm(model, prompt).strip()

        # ✅ ALWAYS save raw if debug_raw_path is provided
        if debug_raw_path is not None:
            debug_raw_path.parent.mkdir(parents=True, exist_ok=True)
            # save attempt-specific raw outputs
            attempt_path = debug_raw_path.with_name(debug_raw_path.stem + f"_attempt{attempt+1}.txt")
            attempt_path.write_text(raw, encoding="utf-8")

        try:
            data = _extract_first_json_object(raw)

            # Hard validation: single-slide contract
            returned_uid = str(data.get("slide_uid", "")).strip()
            if returned_uid != slide_uid:
                # do not fail: we know the target slide_uid deterministically
                # keep a warning in the raw debug output instead
                data["slide_uid"] = slide_uid

            # Reject multi-slide behavior
            raw_lower = raw.lower()
            bad_markers = ["here are the requested slides", "slide 1:", "slide 2:", "slide 3:"]
            if any(m in raw_lower for m in bad_markers):
                raise ValueError("Model produced multi-slide text (bad markers detected).")

            break  # ✅ success

        except Exception as e:
            last_err = e
            # Second attempt: stricter prompt
            prompt = (
                prompt
                + "\n\nFINAL WARNING: Output ONLY the single JSON object. No other text."
            )

    else:
        # both attempts failed
        raise last_err  # type: ignore

    # Normalize/guard fields
    title = str(data.get("title", "")).strip()
    figure_used = data.get("figure_used", None)
    on_slide_bullets = [str(x).strip() for x in data.get("on_slide_bullets", [])]
    speaker_notes = str(data.get("speaker_notes", "")).strip()
    transition = str(data.get("transition", "")).strip()
    reminders = [str(x).strip() for x in data.get("reminders", [])]

    return SlideSpec(
        slide_uid=slide_uid,
        title=title,
        figure_used=figure_used,
        on_slide_bullets=on_slide_bullets,
        speaker_notes=speaker_notes,
        transition=transition,
        reminders=reminders,
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
            include_reminders_in_query=False,
        )

        debug_raw_path = None
        if debug_save and output_dir is not None:
            debug_raw_path = Path(output_dir) / "slide_realization_raw" / f"{slide_uid.replace('::','__')}.txt"

        spec = generate_slide_spec(
            slide_uid=slide_uid,
            plan_text=bundle.slide_plan.blueprint_text,
            slide_context=bundle.slide_context,
            suggested_figures=bundle.slide_plan.suggested_figures,
            primary_bullets=primary_texts,
            reminder_bullets=reminder_texts,
            evidence=evidence,
            model=model,
            debug_raw_path=debug_raw_path,
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
