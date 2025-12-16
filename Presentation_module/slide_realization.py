# Presentation_module/slide_realization.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from .llm import call_llm
from .presentation_models import PresentationPlan
from .bullet_resolution import BulletResolutionResult
from .rag_loader import RAGStores
from .slide_specs import SlideSpec, EvidenceChunk


# ----------------------------
# Helpers
# ----------------------------

def _docs_to_text(docs) -> List[str]:
    out: List[str] = []
    for d in docs:
        try:
            out.append(d.page_content)
        except Exception:
            out.append(str(d))
    return out


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if "```" in s:
            s = s.rsplit("```", 1)[0]
    return s.strip()


def _extract_first_json_object(raw: str) -> dict:
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

    # 2) substring from first '{' to last '}'
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

    # 2a) parse candidate (raw, then sanitized)
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

    # 3) tail-trim recovery
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


def _safe_get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _infer_priority_for_bullet_on_slide(bundle, bullet_id: Optional[str], bullet_text: str) -> float:
    """
    Try to infer priority from bundle.candidates.
    - Prefer matching by bullet_id
    - fallback: conservative default
    """
    if not hasattr(bundle, "candidates") or bundle.candidates is None:
        return 0.6

    if bullet_id:
        ps = [
            float(getattr(c, "priority", 0.0))
            for c in bundle.candidates
            if getattr(c, "bullet_id", None) == bullet_id
        ]
        if ps:
            return float(max(ps))

    return 0.6


def _build_bullet_requirements(
    bundle,
    primary_usages: List[Any],
    must_cutoff: float = 0.75,
    should_cutoff: float = 0.55,
) -> List[dict]:
    """
    Convert "primary" resolved bullets into MUST/SHOULD/OPTIONAL requirements.
    Uses candidate priority if possible.
    """
    reqs: List[dict] = []

    for u in primary_usages:
        bullet_text = (_safe_get(u, "bullet_text", "") or "").strip()
        if not bullet_text:
            continue

        bullet_id = _safe_get(u, "bullet_id", None)
        pr = _infer_priority_for_bullet_on_slide(bundle, bullet_id, bullet_text)

        if pr >= must_cutoff:
            level = "MUST"
        elif pr >= should_cutoff:
            level = "SHOULD"
        else:
            level = "OPTIONAL"

        reqs.append(
            {
                "bullet_id": bullet_id,
                "text": bullet_text,
                "priority": round(float(pr), 3),
                "level": level,
            }
        )

    level_order = {"MUST": 0, "SHOULD": 1, "OPTIONAL": 2}
    reqs.sort(key=lambda r: (level_order.get(r["level"], 9), -r["priority"]))
    return reqs


def _normalize_tagged_lines(
    text: str,
    allowed=("MAIN", "FIG", "SUP", "LIT"),
    default_tag="MAIN",
    max_lines: int = 10,
) -> str:
    """
    CRITICAL FIX: do not drop everything if model forgot tags.
    - If tagged lines exist, keep them.
    - Else salvage bullet lines and prefix with [MAIN].
    - Else salvage sentences and prefix with [MAIN].
    """
    text = (text or "").strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 1) keep allowed tagged lines
    tagged: List[str] = []
    for ln in lines:
        for tag in allowed:
            if ln.startswith(f"[{tag}]"):
                tagged.append(ln)
                break
    if tagged:
        return "\n".join(tagged[:max_lines]).strip()

    # 2) salvage bullet-ish lines
    bullets: List[str] = []
    for ln in lines:
        s = ln.lstrip("-*•").strip()
        if not s or len(s) < 6:
            continue
        low = s.lower()
        if low.startswith("here are") or low.startswith("summary") or low.startswith("the following"):
            continue
        bullets.append(f"[{default_tag}] {s}")
    if bullets:
        return "\n".join(bullets[:max_lines]).strip()

    # 3) salvage sentences from paragraph
    joined = " ".join(lines)
    parts: List[str] = []
    buf = ""
    for ch in joined:
        buf += ch
        if ch in ".!?":
            s = buf.strip()
            buf = ""
            if len(s) >= 20:
                parts.append(s)
    if buf.strip():
        parts.append(buf.strip())

    parts = parts[:max_lines]
    parts = [f"[{default_tag}] {p}" for p in parts if p]
    return "\n".join(parts).strip()


# ----------------------------
# Evidence retrieval + distillation
# ----------------------------

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
    parts = [
        f"PLAN: {plan_text}",
        "PRIMARY BULLETS (key claims to support):",
        *[f"- {b}" for b in primary_bullets],
    ]

    if slide_context:
        short_ctx = " ".join(slide_context.split()[:40])
        parts.append(f"CONTEXT (short): {short_ctx}")

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
    ev_txt = "\n\n".join([f"[{e.source}]\n{e.text}" for e in evidence])

    prompt = f"""
You are an evidence distiller for ONE slide of a scientific talk.

SLIDE_UID: {slide_uid}
PLAN: {plan_text}
CONTEXT (role): {slide_context}

PRIMARY BULLETS (what must be supported):
{json.dumps(primary_bullets, indent=2)}

TASK:
From the evidence below, extract 6–12 short FACTS that directly help explain this slide.

Rules:
- Prefer facts that explain the PLAN and the PRIMARY BULLETS.
- Each fact should ideally start with [MAIN]/[FIG]/[SUP]/[LIT] but if you forget, still output bullet lines.
- Avoid experimental setup unless the PLAN explicitly requires it.
- Do NOT summarize the whole paper.
- Keep each fact to one sentence.

EVIDENCE:
{ev_txt}

OUTPUT:
Return ONLY the list of facts as plain text lines.
"""

    raw = call_llm(model, prompt).strip()
    raw = _normalize_tagged_lines(raw, max_lines=12)

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw, encoding="utf-8")

    return raw


# ----------------------------
# Keypoints + Speech
# ----------------------------

def select_key_points_for_slide(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    bullet_requirements: List[dict],
    distilled_facts: str,
    model: str,
    debug_path: Optional[Path] = None,
) -> str:
    prompt = f"""
You are selecting KEY POINTS for ONE slide.

SLIDE_UID: {slide_uid}
PLAN: {plan_text}
CONTEXT: {slide_context}

BULLET REQUIREMENTS:
{json.dumps(bullet_requirements, indent=2)}

DISTILLED FACTS:
{distilled_facts}

TASK:
Pick 5–8 KEY POINTS this slide MUST communicate.
Rules:
- Must cover all items with level MUST.
- SHOULD items should be included if they fit.
- Each key point should be grounded in the distilled facts.
- Output as plain text bullet lines. Tags like [MAIN]/[FIG]/[SUP]/[LIT] are preferred but not mandatory.

OUTPUT:
Return ONLY the list of key points as plain text lines.
"""
    raw = call_llm(model, prompt).strip()
    raw = _normalize_tagged_lines(raw, max_lines=10)

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw, encoding="utf-8")

    return raw


def generate_speaker_notes(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    bullet_requirements: List[dict],
    key_points: str,
    model: str,
    debug_path: Optional[Path] = None,
) -> str:
    prompt = f"""
You are writing SPEAKER NOTES for EXACTLY ONE slide.

SLIDE_UID: {slide_uid}
PLAN: {plan_text}
CONTEXT: {slide_context}

BULLET REQUIREMENTS:
{json.dumps(bullet_requirements, indent=2)}

KEY POINTS (grounded):
{key_points}

TASK:
Write 150–220 words speaker notes.

Rules:
- Must explicitly cover all level MUST requirements.
- SHOULD items should be included if they fit naturally.
- Do not invent facts not supported by KEY POINTS.
- No headings, no "Slide 1/2", just a single paragraph.

OUTPUT:
Return ONLY the speaker notes text.
"""
    raw = call_llm(model, prompt).strip()

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw, encoding="utf-8")

    return raw


# ----------------------------
# Slide spec generation (3-step)
# ----------------------------

def generate_slide_spec(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    suggested_figures: List[str],
    bullet_requirements: List[dict],
    reminder_bullets: List[str],
    distilled_facts: str,
    evidence: List[EvidenceChunk],
    model: str,
    debug_raw_path: Optional[Path] = None,
    debug_keypoints_path: Optional[Path] = None,
    debug_speech_path: Optional[Path] = None,
) -> SlideSpec:
    # 1) Key points
    key_points = select_key_points_for_slide(
        slide_uid=slide_uid,
        plan_text=plan_text,
        slide_context=slide_context,
        bullet_requirements=bullet_requirements,
        distilled_facts=distilled_facts,
        model=model,
        debug_path=debug_keypoints_path,
    )

    # 2) Speaker notes
    speaker_notes = generate_speaker_notes(
        slide_uid=slide_uid,
        plan_text=plan_text,
        slide_context=slide_context,
        bullet_requirements=bullet_requirements,
        key_points=key_points,
        model=model,
        debug_path=debug_speech_path,
    )

    # 3) Compress to slide JSON (ONLY slide text, do not regenerate speech)
    prompt = f"""
SYSTEM CONTRACT:
- Generate EXACTLY ONE slide: {slide_uid}
- Output MUST be a SINGLE JSON object and NOTHING ELSE.
- JSON MUST include "slide_uid" = "{slide_uid}"

SLIDE_UID: {slide_uid}
PLAN: {plan_text}
SUGGESTED FIGURES: {json.dumps(suggested_figures, indent=2)}

SPEAKER NOTES (DO NOT rewrite; only compress into slide bullets):
{speaker_notes}

KEY POINTS:
{key_points}

TASK:
Create slide content as a compression of the speaker notes.

Rules:
- 3–5 on-slide bullets, short phrases (no long sentences).
- figure_used must be one of suggested_figures or null.
- transition: 1 sentence leading to the next slide.
- reminders: can be empty list.

OUTPUT JSON:
{{
  "slide_uid": "{slide_uid}",
  "title": "short title",
  "figure_used": null,
  "on_slide_bullets": ["...", "...", "..."],
  "transition": "...",
  "reminders": []
}}
"""

    raw = call_llm(model, prompt).strip()

    if debug_raw_path is not None:
        debug_raw_path.parent.mkdir(parents=True, exist_ok=True)
        debug_raw_path.write_text(raw, encoding="utf-8")

    data = _extract_first_json_object(raw)
    data["slide_uid"] = slide_uid  # force correct

    title = str(data.get("title", "")).strip()
    figure_used = data.get("figure_used", None)
    on_slide_bullets = [str(x).strip() for x in data.get("on_slide_bullets", [])]
    transition = str(data.get("transition", "")).strip()
    reminders = [str(x).strip() for x in data.get("reminders", [])]

    return SlideSpec(
        slide_uid=slide_uid,
        title=title,
        figure_used=figure_used,
        on_slide_bullets=on_slide_bullets,
        speaker_notes=speaker_notes,   # keep the richer speech we generated
        transition=transition,
        reminders=reminders,
        evidence=evidence,
    )


# ----------------------------
# Presentation realization (all slides or subset)
# ----------------------------

def realize_presentation_slides(
    presentation: PresentationPlan,
    resolved: BulletResolutionResult,
    rag: RAGStores,
    model: str,
    output_dir: Optional[Union[str, Path]] = None,
    debug_save: bool = False,
    only_slide_uids: Optional[List[str]] = None,
) -> Dict[str, SlideSpec]:

    slide_specs: Dict[str, SlideSpec] = {}

    all_items = sorted(presentation.all_slides.items())
    if only_slide_uids is not None:
        only_set = set(only_slide_uids)
        all_items = [(uid, b) for (uid, b) in all_items if uid in only_set]

    for slide_uid, bundle in all_items:
        usages = resolved.usage_by_slide.get(slide_uid, [])
        primary_usages = [u for u in usages if _safe_get(u, "usage", "") == "primary"]
        reminder_usages = [u for u in usages if _safe_get(u, "usage", "") == "reminder"]

        primary_texts = [(_safe_get(u, "bullet_text", "") or "").strip() for u in primary_usages]
        primary_texts = [t for t in primary_texts if t]
        reminder_texts = [(_safe_get(u, "bullet_text", "") or "").strip() for u in reminder_usages]
        reminder_texts = [t for t in reminder_texts if t]

        bullet_requirements = _build_bullet_requirements(bundle, primary_usages)

        # Debug: requirements JSON
        if debug_save and output_dir is not None:
            req_path = Path(output_dir) / "slide_realization_requirements" / f"{slide_uid.replace('::','__')}.json"
            req_path.parent.mkdir(parents=True, exist_ok=True)
            req_path.write_text(json.dumps(bullet_requirements, indent=2), encoding="utf-8")

        use_figs = bool(getattr(bundle.slide_plan, "suggested_figures", []))
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

        distilled_debug = None
        if debug_save and output_dir is not None:
            distilled_debug = Path(output_dir) / "slide_realization_distilled" / f"{slide_uid.replace('::','__')}.txt"

        distilled_facts = distill_evidence_for_slide(
            slide_uid=slide_uid,
            plan_text=bundle.slide_plan.blueprint_text,
            slide_context=bundle.slide_context,
            primary_bullets=primary_texts,
            reminder_bullets=reminder_texts,
            evidence=evidence,
            model=model,
            debug_path=distilled_debug,
        )

        debug_raw_path = None
        debug_keypoints_path = None
        debug_speech_path = None
        if debug_save and output_dir is not None:
            base = Path(output_dir)
            debug_raw_path = base / "slide_realization_raw" / f"{slide_uid.replace('::','__')}.txt"
            debug_keypoints_path = base / "slide_realization_keypoints" / f"{slide_uid.replace('::','__')}.txt"
            debug_speech_path = base / "slide_realization_speech" / f"{slide_uid.replace('::','__')}.txt"

        spec = generate_slide_spec(
            slide_uid=slide_uid,
            plan_text=bundle.slide_plan.blueprint_text,
            slide_context=bundle.slide_context,
            suggested_figures=bundle.slide_plan.suggested_figures,
            bullet_requirements=bullet_requirements,
            reminder_bullets=reminder_texts,
            distilled_facts=distilled_facts,
            evidence=evidence,
            model=model,
            debug_raw_path=debug_raw_path,
            debug_keypoints_path=debug_keypoints_path,
            debug_speech_path=debug_speech_path,
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
