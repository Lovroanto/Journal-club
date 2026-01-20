# Presentation_module/slide_realization.py

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

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

def _rewrite_plan_for_figure(plan: str, figure: str) -> str:
    return (
        f"Explain {figure} clearly and use it to support the following message:\n"
        f"{plan}"
    )


def _make_figure_only_slide_uid(base_uid: str, fig_idx: int) -> str:
    return f"{base_uid}::Figure_{fig_idx+1}"


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

def _get_slide_group(pr):
    """
    Return the SlideGroupPlan from a PlanningResult, regardless of field name.
    This makes slide_realization compatible with older/newer caches.
    """
    if hasattr(pr, "slide_group"):
        return pr.slide_group
    if hasattr(pr, "group_plan"):
        return pr.group_plan
    if hasattr(pr, "slide_group_plan"):
        return pr.slide_group_plan
    raise AttributeError(
        f"PlanningResult has no slide group attribute. Available fields: {dir(pr)}"
    )

def _build_bullet_text_lookup(presentation: PresentationPlan) -> Dict[str, str]:
    """
    bullet_id -> bullet_text from PresentationPlan (most reliable, if IDs align)
    """
    out: Dict[str, str] = {}
    for g in getattr(presentation, "groups", []):
        for b in getattr(g, "numbered_bullets", []) or []:
            bid = str(getattr(b, "bullet_id", "")).strip()
            txt = str(getattr(b, "text", "")).strip()
            if bid and txt:
                out[bid] = txt
    return out


def _build_candidate_argument_lookup(bundle) -> Dict[str, str]:
    """
    bullet_id -> candidate.argument (very useful fallback when bullet text is missing)
    """
    out: Dict[str, str] = {}
    for c in list(getattr(bundle, "candidates", []) or []):
        bid = str(getattr(c, "bullet_id", "") or "").strip()
        arg = str(getattr(c, "argument", "") or "").strip()
        if bid and arg:
            out[bid] = arg
    return out


def _safe_get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tokenize_for_overlap(s: str) -> set:
    s = (s or "").lower()
    toks = re.findall(r"[a-z0-9]+", s)
    return {t for t in toks if len(t) >= 4}


def _looks_like_results_slide(plan_text: str, slide_context: str) -> bool:
    s = (plan_text + " " + slide_context).lower()
    explicit = ["results", "result", "data", "measurement", "measured", "observed", "scan", "spectrum"]
    regime = ["regime", "regimes", "zones", "zone i", "zone ii", "zone iii", "zone iv", "hysteresis", "frequency pinning"]

    if any(k in s for k in explicit):
        return True

    hits = 0
    for k in regime + ["linewidth", "threshold", "bistability"]:
        if k in s:
            hits += 1

    return hits >= 2


def _infer_priority_for_bullet_on_slide(bundle, bullet_id: Optional[str], bullet_text: str) -> float:
    cands = list(getattr(bundle, "candidates", []) or [])
    base: float = 0.70 if not cands else 0.65

    if bullet_id and cands:
        ps = [float(getattr(c, "priority", base)) for c in cands if getattr(c, "bullet_id", None) == bullet_id]
        if ps:
            base = max(base, float(max(ps)))

    plan = ""
    ctx = ""
    try:
        plan = str(getattr(getattr(bundle, "slide_plan", None), "blueprint_text", "") or "")
    except Exception:
        plan = ""
    try:
        ctx = str(getattr(bundle, "slide_context", "") or "")
    except Exception:
        ctx = ""

    overlap = _tokenize_for_overlap(bullet_text) & _tokenize_for_overlap(plan + " " + ctx)
    if overlap:
        bump = min(0.15, 0.03 * len(overlap))
        base = min(0.95, base + bump)

    return float(max(0.0, min(1.0, base)))


def _build_bullet_requirements(
    bundle,
    primary_usages: List[Any],
    must_cutoff: float = 0.75,
    should_cutoff: float = 0.55,
) -> List[dict]:
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


def _fallback_requirements_from_candidates(
    bundle,
    bullet_text_by_id: Dict[str, str],
    max_items: int = 4,
) -> List[dict]:
    cands = list(getattr(bundle, "candidates", []) or [])
    if not cands:
        return []

    cands.sort(key=lambda c: float(getattr(c, "priority", 0.0)), reverse=True)
    cands = cands[:max_items]

    reqs: List[dict] = []
    for c in cands:
        bullet_id = getattr(c, "bullet_id", None)
        pr = float(getattr(c, "priority", 0.6))

        bullet_text = ""
        if bullet_id:
            bullet_text = (bullet_text_by_id.get(str(bullet_id).strip(), "") or "").strip()

        level = "MUST" if pr >= 0.80 else "SHOULD"
        reqs.append(
            {
                "bullet_id": bullet_id,
                "text": bullet_text,
                "priority": round(pr, 3),
                "level": level,
            }
        )

    level_order = {"MUST": 0, "SHOULD": 1, "OPTIONAL": 2}
    reqs.sort(key=lambda r: (level_order.get(r["level"], 9), -r["priority"]))
    return reqs


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _dedupe_evidence(evidence: List[EvidenceChunk]) -> List[EvidenceChunk]:
    seen: set = set()
    out: List[EvidenceChunk] = []
    for e in evidence:
        txt = _norm_text(getattr(e, "text", "") or "")
        if not txt:
            continue
        key = txt.lower()
        h = hashlib.sha1(key[:4000].encode("utf-8", errors="ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(e)
    return out


def _normalize_tagged_lines(raw: str, allowed_tags: Tuple[str, ...]) -> str:
    raw = _strip_code_fences(raw or "")
    lines = [l.strip() for l in raw.splitlines()]
    lines = [l for l in lines if l]

    tag_re = re.compile(r"^\[(%s)\]\s*" % "|".join(re.escape(t) for t in allowed_tags), flags=re.IGNORECASE)
    num_re = re.compile(r"^\s*(?:[-*]|\d+[\).\]]|•)\s*")

    kept: List[str] = []
    for l in lines:
        l2 = num_re.sub("", l).strip()
        if tag_re.match(l2):
            kept.append(l2)

    if not kept and lines:
        return "\n".join(lines)

    return "\n".join(kept)


def _validate_slide_spec_json(data: dict, slide_uid: str, suggested_figures: List[str]) -> None:
    if not isinstance(data, dict):
        raise ValueError("SlideSpec JSON is not an object")

    if data.get("slide_uid") != slide_uid:
        raise ValueError(f"slide_uid mismatch: {data.get('slide_uid')} != {slide_uid}")

    title = str(data.get("title", "")).strip()
    if not title:
        raise ValueError("Missing/empty title")

    fig = data.get("figure_used", None)
    if fig is not None and fig not in suggested_figures:
        raise ValueError("figure_used must be one of suggested_figures or null")

    bullets = data.get("on_slide_bullets", None)
    if not isinstance(bullets, list):
        raise ValueError("on_slide_bullets must be a list")

    bullets2 = [str(b).strip() for b in bullets if str(b).strip()]
    if not (5 <= len(bullets2) <= 7):
        raise ValueError("on_slide_bullets must contain 5–7 non-empty items")

    transition = str(data.get("transition", "")).strip()
    if not transition:
        raise ValueError("Missing/empty transition")

    reminders = data.get("reminders", [])
    if not isinstance(reminders, list):
        raise ValueError("reminders must be a list")

    for r in reminders:
        if not isinstance(r, (str, int, float)):
            raise ValueError("reminders items must be strings")
        if not str(r).strip():
            raise ValueError("reminders must not contain empty items")


def _count_evidence_sources(evidence: List[EvidenceChunk]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in evidence or []:
        src = str(getattr(e, "source", "") or "").strip().upper() or "UNKNOWN"
        counts[src] = counts.get(src, 0) + 1
    return counts


def _slice_evidence_by_source(evidence: List[EvidenceChunk], source: str) -> List[EvidenceChunk]:
    src = (source or "").upper().strip()
    out: List[EvidenceChunk] = []
    for e in evidence or []:
        if str(getattr(e, "source", "") or "").upper().strip() == src:
            out.append(e)
    return out


def _evidence_text(evidence: List[EvidenceChunk]) -> str:
    return "\n\n".join([f"[{e.source}]\n{e.text}" for e in (evidence or [])])


def _merge_fact_banks(banks: List[str]) -> str:
    """
    Merge multiple line-lists into one, lightly de-dup by normalized text (ignoring tag).
    Preserves tags.
    """
    seen = set()
    out_lines: List[str] = []

    for bank in banks:
        for line in (bank or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("[") or "]" not in line:
                continue
            tag = line[: line.find("]") + 1]
            body = _norm_text(line[line.find("]") + 1 :]).lower()
            key = hashlib.sha1(body[:2000].encode("utf-8", errors="ignore")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out_lines.append(f"{tag} {line[line.find(']')+1:].strip()}".strip())

    return "\n".join(out_lines)


def _format_requirements_for_query(bullet_requirements: Optional[List[dict]]) -> List[str]:
    req_lines: List[str] = []
    for r in (bullet_requirements or []):
        txt = str(r.get("text", "") or "").strip()
        lvl = str(r.get("level", "") or "").strip()
        if txt:
            req_lines.append(f"- ({lvl}) {txt}")
    return req_lines


# ----------------------------
# Evidence retrieval + distillation
# ----------------------------

def retrieve_evidence_for_slide(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    primary_bullets: List[str],
    reminder_bullets: List[str],
    bullet_requirements: Optional[List[dict]],
    candidate_arguments: Optional[List[str]],
    rag: RAGStores,
    k_main: int = 18,
    k_figs: int = 10,
    k_sup: int = 10,
    k_lit: int = 8,
    use_figs: bool = True,
    use_sup: bool = False,
    use_lit: bool = False,
    include_reminders_in_query: bool = False,
) -> List[EvidenceChunk]:
    """
    High-recall retrieval.

    MAIN is queried in multiple ways:
    - plan/context only
    - bullets only
    - plan/context + bullets together (strong)
    - requirements/candidate arguments (extra intent signal)

    This matches the design goal:
    retrieve as much as possible now; selection happens later.
    """

    req_lines = _format_requirements_for_query(bullet_requirements)
    cand_lines = [f"- {a}" for a in (candidate_arguments or []) if str(a or "").strip()]

    plan_ctx_block = "\n".join([p for p in [
        f"PLAN: {plan_text}",
        f"CONTEXT: {slide_context}" if slide_context else "",
    ] if p]).strip()

    bullets_block = "\n".join(
        ["PRIMARY BULLETS:"] + [f"- {b}" for b in primary_bullets if str(b or "").strip()]
    ).strip()

    if include_reminders_in_query and reminder_bullets:
        bullets_block += "\n" + "\n".join(["REMINDERS:"] + [f"- {b}" for b in reminder_bullets])

    intent_block = "\n".join(
        (["BULLET REQUIREMENTS:"] + req_lines + ["CANDIDATE ARGUMENTS:"] + cand_lines)
        if (req_lines or cand_lines)
        else []
    ).strip()

    # Queries: separate + combined
    q_plan = plan_ctx_block
    q_bullets = bullets_block
    q_combo = "\n\n".join([x for x in [plan_ctx_block, bullets_block] if x]).strip()
    q_intent = "\n\n".join([x for x in [plan_ctx_block, bullets_block, intent_block] if x]).strip()

    evidence: List[EvidenceChunk] = []

    # ---- MAIN: multiple passes, higher recall
    main_docs = []
    if q_plan:
        main_docs += rag.main.similarity_search(q_plan, k=max(4, k_main // 3))
    if q_bullets:
        main_docs += rag.main.similarity_search(q_bullets, k=max(4, k_main // 3))
    if q_combo:
        main_docs += rag.main.similarity_search(q_combo, k=k_main)
    if q_intent and q_intent != q_combo:
        main_docs += rag.main.similarity_search(q_intent, k=max(4, k_main // 2))

    for t in _docs_to_text(main_docs):
        evidence.append(EvidenceChunk(source="MAIN", text=t))

    # ---- FIG: use combo + bullets
    if use_figs:
        fig_docs = []
        if q_combo:
            fig_docs += rag.figs.similarity_search(q_combo, k=k_figs)
        elif q_bullets:
            fig_docs += rag.figs.similarity_search(q_bullets, k=k_figs)
        for t in _docs_to_text(fig_docs):
            evidence.append(EvidenceChunk(source="FIG", text=t))

    # ---- SUP: use intent + combo (if enabled)
    if use_sup and rag.sup is not None:
        sup_docs = []
        if q_intent:
            sup_docs += rag.sup.similarity_search(q_intent, k=k_sup)
        elif q_combo:
            sup_docs += rag.sup.similarity_search(q_combo, k=k_sup)
        for t in _docs_to_text(sup_docs):
            evidence.append(EvidenceChunk(source="SUP", text=t))

    # ---- LIT: use plan + combo (if enabled)
    if use_lit and rag.lit is not None:
        lit_docs = []
        if q_plan:
            lit_docs += rag.lit.similarity_search(q_plan, k=max(3, k_lit // 2))
        if q_combo:
            lit_docs += rag.lit.similarity_search(q_combo, k=k_lit)
        for t in _docs_to_text(lit_docs):
            evidence.append(EvidenceChunk(source="LIT", text=t))

    return _dedupe_evidence(evidence)


def _distill_facts_for_source(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    primary_bullets: List[str],
    reminder_bullets: List[str],
    evidence: List[EvidenceChunk],
    model: str,
    target_source: str,
    n_min: int,
    n_max: int,
    is_results: bool,
    debug_path: Optional[Path] = None,
) -> str:
    src = target_source.upper().strip()
    ev_txt = _evidence_text(evidence)

    prompt = f"""
You are an evidence distiller for ONE slide of a scientific talk.

SLIDE_UID: {slide_uid}
PLAN: {plan_text}
CONTEXT (role): {slide_context}

PRIMARY BULLETS (claims to support):
{json.dumps(primary_bullets, indent=2)}

REMINDERS (optional; lower priority):
{json.dumps(reminder_bullets, indent=2)}

TASK:
Extract {n_min}–{n_max} short FACTS from the evidence ONLY for source [{src}] that directly help this slide.

Rules:
- Every line MUST start with [{src}] exactly.
- Facts must be relevant to PLAN + PRIMARY BULLETS.
- Prefer mechanism/causal phrasing if PLAN is "how/why/facilitates/enables".
- STRICT: Do NOT list experimental setup details (lasers, lattice beams, molasses, etc.)
  unless they directly explain the PLAN mechanism (continuous lasing via cavity QED / pinning / gain / inversion).
- STRICT: {"You MAY mention zones/regimes if relevant." if is_results else "Do NOT mention zones I/II/III/IV or regime labels unless the PLAN explicitly says results/regimes."}
- Keep each fact to one sentence.
- No numbering, no preamble.

EVIDENCE:
{ev_txt}

OUTPUT:
Return ONLY the list of facts, one per line.
"""
    raw = call_llm(model, prompt).strip()
    raw2 = _normalize_tagged_lines(raw, (src,))
    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw2, encoding="utf-8")
    return raw2


def distill_evidence_for_slide(
    slide_uid: str,
    plan_text: str,
    slide_context: str,
    primary_bullets: List[str],
    reminder_bullets: Optional[List[str]] = None,
    evidence: Optional[List[EvidenceChunk]] = None,
    model: str = "llama3.1",
    debug_path: Optional[Path] = None,
) -> str:
    evidence = evidence or []
    reminder_bullets = reminder_bullets or []

    is_results = _looks_like_results_slide(plan_text, slide_context)
    counts = _count_evidence_sources(evidence)

    banks: List[str] = []

    if counts.get("MAIN", 0) > 0:
        banks.append(
            _distill_facts_for_source(
                slide_uid, plan_text, slide_context, primary_bullets, reminder_bullets,
                _slice_evidence_by_source(evidence, "MAIN"),
                model=model, target_source="MAIN",
                n_min=8, n_max=14,
                is_results=is_results,
                debug_path=(debug_path.parent / (debug_path.stem + "__MAIN.txt")) if debug_path else None,
            )
        )

    if counts.get("FIG", 0) > 0:
        banks.append(
            _distill_facts_for_source(
                slide_uid, plan_text, slide_context, primary_bullets, reminder_bullets,
                _slice_evidence_by_source(evidence, "FIG"),
                model=model, target_source="FIG",
                n_min=3, n_max=6,
                is_results=is_results,
                debug_path=(debug_path.parent / (debug_path.stem + "__FIG.txt")) if debug_path else None,
            )
        )

    if counts.get("SUP", 0) > 0:
        banks.append(
            _distill_facts_for_source(
                slide_uid, plan_text, slide_context, primary_bullets, reminder_bullets,
                _slice_evidence_by_source(evidence, "SUP"),
                model=model, target_source="SUP",
                n_min=3, n_max=6,
                is_results=is_results,
                debug_path=(debug_path.parent / (debug_path.stem + "__SUP.txt")) if debug_path else None,
            )
        )

    if counts.get("LIT", 0) > 0:
        banks.append(
            _distill_facts_for_source(
                slide_uid, plan_text, slide_context, primary_bullets, reminder_bullets,
                _slice_evidence_by_source(evidence, "LIT"),
                model=model, target_source="LIT",
                n_min=2, n_max=5,
                is_results=is_results,
                debug_path=(debug_path.parent / (debug_path.stem + "__LIT.txt")) if debug_path else None,
            )
        )

    merged = _merge_fact_banks(banks)

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(merged, encoding="utf-8")

    return merged


# ----------------------------
# Content planning (key points) + speech
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
    is_results = _looks_like_results_slide(plan_text, slide_context)

    prompt = f"""
You are planning the CONTENT of ONE slide.

SLIDE_UID: {slide_uid}
PLAN: {plan_text}
CONTEXT: {slide_context}

BULLET REQUIREMENTS (MUST items are mandatory):
{json.dumps(bullet_requirements, indent=2)}

DISTILLED FACTS (fact bank; MAIN is primary, others are supporting):
{distilled_facts}

TASK:
Pick 6–10 KEY POINTS that this slide should communicate.

Rules:
- Each key point must be grounded in the facts above and start with [MAIN]/[FIG]/[SUP]/[LIT].
- Prefer [MAIN] key points when overlapping information exists; use [FIG]/[SUP]/[LIT] only for added nuance/details.
- Cover ALL items with level MUST (explicitly).
- If PLAN is mechanism/how/why/facilitates/enables: key points must explain a causal chain:
  (why cavity helps → how gain/inversion arises → what limits/defines it → why this enables precision).
- {"You MAY include zones/regimes." if is_results else "Do NOT include zones/regimes unless PLAN explicitly says results/regimes."}
- Avoid wasting points on setup unless required by PLAN.
- No numbering, no preamble.

OUTPUT:
Return ONLY the list of key points, one per line.
"""

    raw = call_llm(model, prompt).strip()
    raw2 = _normalize_tagged_lines(raw, ("MAIN", "FIG", "SUP", "LIT"))

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw2, encoding="utf-8")
    return raw2


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
Write 160–240 words speaker notes.

Rules:
- Must explicitly cover all level MUST requirements.
- SHOULD requirements should be included if they fit naturally.
- Stay aligned with PLAN + CONTEXT; do not drift into unrelated results.
- Do not invent facts beyond key points.
- No headings, no "Slide 1/2", just a single paragraph.

OUTPUT:
Return ONLY the speaker notes paragraph.
"""
    raw = call_llm(model, prompt).strip()
    raw2 = _strip_code_fences(raw).strip()

    if debug_path is not None:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw2, encoding="utf-8")
    return raw2


# ----------------------------
# Slide spec generation (JSON)
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
    max_json_attempts: int = 3,
) -> SlideSpec:

    key_points = select_key_points_for_slide(
        slide_uid=slide_uid,
        plan_text=plan_text,
        slide_context=slide_context,
        bullet_requirements=bullet_requirements,
        distilled_facts=distilled_facts,
        model=model,
        debug_path=debug_keypoints_path,
    )

    speaker_notes = generate_speaker_notes(
        slide_uid=slide_uid,
        plan_text=plan_text,
        slide_context=slide_context,
        bullet_requirements=bullet_requirements,
        key_points=key_points,
        model=model,
        debug_path=debug_speech_path,
    )

    prompt = f"""
SYSTEM CONTRACT (MUST FOLLOW):
- You are generating EXACTLY ONE slide: {slide_uid}
- Output MUST be a SINGLE JSON object and NOTHING ELSE.
- JSON MUST include the field "slide_uid" and it MUST equal "{slide_uid}".

SLIDE_UID: {slide_uid}
PLAN: {plan_text}
CONTEXT: {slide_context}

SUGGESTED FIGURES: {json.dumps(suggested_figures, indent=2)}

SPEAKER NOTES (ground truth):
{speaker_notes}

KEY POINTS (grounded):
{key_points}

REMINDERS (optional):
{json.dumps(reminder_bullets, indent=2)}

TASK:
Create slide content as a compression of the speaker notes.

Rules:
- Produce 5–7 on-slide bullets (short phrases, no full sentences).
- At least 2 bullets must directly express the PLAN (cavity QED → continuous lasing → high precision).
- At least 1 bullet must reflect the highest-priority MUST requirement.
- Prefer causal/mechanism phrasing if PLAN says "how/why/facilitates/enables".
- figure_used must be one of suggested_figures or null.
- transition: 1 sentence leading to next slide.
- reminders: keep as short list (can be empty).

OUTPUT (STRICT JSON ONLY):
{{
  "slide_uid": "{slide_uid}",
  "title": "short title",
  "figure_used": null,
  "on_slide_bullets": ["...", "...", "...", "...", "..."],
  "transition": "...",
  "reminders": []
}}
"""

    last_raw = ""
    last_err: Optional[Exception] = None

    for attempt in range(1, max_json_attempts + 1):
        raw = call_llm(model, prompt).strip()
        last_raw = raw

        if debug_raw_path is not None:
            debug_raw_path.parent.mkdir(parents=True, exist_ok=True)
            debug_raw_path.write_text(raw, encoding="utf-8")

        try:
            data = _extract_first_json_object(raw)
            data["slide_uid"] = slide_uid
            _validate_slide_spec_json(data, slide_uid=slide_uid, suggested_figures=suggested_figures)

            title = str(data.get("title", "")).strip()
            figure_used = data.get("figure_used", None)
            on_slide_bullets = [str(x).strip() for x in data.get("on_slide_bullets", []) if str(x).strip()]
            transition = str(data.get("transition", "")).strip()
            reminders = [str(x).strip() for x in data.get("reminders", []) if str(x).strip()]

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
        except Exception as e:
            last_err = e
            continue

    raise ValueError(
        f"Failed to generate valid SlideSpec JSON for {slide_uid} after {max_json_attempts} attempts.\n"
        f"Last error: {last_err}\n"
        f"Last raw output (first 800 chars):\n{(last_raw or '')[:800]}"
    )


# ----------------------------
# Presentation realization
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
    bullet_text_by_id: Dict[str, str] = _build_bullet_text_lookup(presentation)

    # Track used figures PER GROUP NAME
    used_figures: Dict[str, set] = {}

    all_items = sorted(presentation.all_slides.items())
    if only_slide_uids is not None:
        only_set = set(only_slide_uids)
        all_items = [(uid, b) for (uid, b) in all_items if uid in only_set]

    # =====================================================
    # FIRST PASS: narrative-driven slides
    # =====================================================

    # FIRST PASS: narrative-driven slides
    total_slides = len(all_items)
    for idx, (slide_uid, bundle) in enumerate(all_items, start=1):
        print(f"[4/5] Realizing slide {idx}/{total_slides}: {slide_uid}")
        group_name = bundle.slide_plan.group_name
        used_figures.setdefault(group_name, set())

        plan_text = bundle.slide_plan.blueprint_text
        slide_context = bundle.slide_context

        usages = resolved.usage_by_slide.get(slide_uid, [])
        primary_usages = [u for u in usages if _safe_get(u, "usage", "") == "primary"]
        reminder_usages = [u for u in usages if _safe_get(u, "usage", "") == "reminder"]

        primary_texts = [
            (_safe_get(u, "bullet_text", "") or "").strip()
            for u in primary_usages
            if (_safe_get(u, "bullet_text", "") or "").strip()
        ]
        reminder_texts = [
            (_safe_get(u, "bullet_text", "") or "").strip()
            for u in reminder_usages
            if (_safe_get(u, "bullet_text", "") or "").strip()
        ]

        bullet_requirements = _build_bullet_requirements(bundle, primary_usages)
        candidate_arguments = list(_build_candidate_argument_lookup(bundle).values())

        # ✅ DIRECT lookup: groups ARE SlideGroupPlan
        slide_group = next(
            (_get_slide_group(pr) for pr in presentation.groups
            if _get_slide_group(pr).group_name == group_name),
            None,
        )

        if slide_group is None:
            continue  # meta / warnings group

        # --- Planned figure enforcement (CRITICAL) ---
        planned_fig = getattr(bundle.slide_plan, "figure_used", None)
        planned_fig = planned_fig.strip() if isinstance(planned_fig, str) else planned_fig

        # If planning already dedicated this slide to a figure, FORCE that figure.
        if planned_fig:
            available_figures = [planned_fig]
        else:
            available_figures = [
                f for f in (slide_group.available_figures or [])
                if f not in used_figures[group_name]
            ]

        evidence = retrieve_evidence_for_slide(
            slide_uid=slide_uid,
            plan_text=plan_text,
            slide_context=slide_context,
            primary_bullets=primary_texts,
            reminder_bullets=reminder_texts,
            bullet_requirements=bullet_requirements,
            candidate_arguments=candidate_arguments,
            rag=rag,
            use_figs=bool(available_figures),
            use_sup=True,
            use_lit=True,
        )

        distilled_facts = distill_evidence_for_slide(
            slide_uid=slide_uid,
            plan_text=plan_text,
            slide_context=slide_context,
            primary_bullets=primary_texts,
            reminder_bullets=reminder_texts,
            evidence=evidence,
            model=model,
        )

        spec = generate_slide_spec(
            slide_uid=slide_uid,
            plan_text=plan_text,
            slide_context=slide_context,
            suggested_figures=available_figures,
            bullet_requirements=bullet_requirements,
            reminder_bullets=reminder_texts,
            distilled_facts=distilled_facts,
            evidence=evidence,
            model=model,
        )

        slide_specs[slide_uid] = spec

        # If the slide was planned as figure-centered, do not allow null output.
        if planned_fig and not spec.figure_used:
            spec.figure_used = planned_fig

        if spec.figure_used:
            used_figures[group_name].add(spec.figure_used)

    # =====================================================
    # SECOND PASS: forced figure explanation slides
    # =====================================================

    for pr in presentation.groups:
        slide_group = _get_slide_group(pr)
        group_name = slide_group.group_name
        used_figures.setdefault(group_name, set())

        unused_figures = [
            f for f in (slide_group.available_figures or [])
            if f not in used_figures[group_name]
        ]

        if not unused_figures:
            continue

        group_slide_uids = [
            uid for uid, bundle in presentation.all_slides.items()
            if bundle.slide_plan.group_name == group_name and uid in slide_specs
        ]

        if not group_slide_uids:
            continue

        base_uid = max(group_slide_uids)

        for i, fig in enumerate(unused_figures):
            new_uid = _make_figure_only_slide_uid(base_uid, i)
            print(f"[4/5] Realizing forced-figure slide: {new_uid}  (figure: {fig})")
            plan_text = (
                "Explain the scientific insight conveyed by this figure and "
                "why it matters in the context of the overall story."
            )
            slide_context = (
                "This slide exists to explain a figure that did not naturally "
                "fit into the narrative flow of earlier slides."
            )

            evidence = retrieve_evidence_for_slide(
                slide_uid=new_uid,
                plan_text=plan_text,
                slide_context=slide_context,
                primary_bullets=[],
                reminder_bullets=[],
                bullet_requirements=None,
                candidate_arguments=[],
                rag=rag,
                use_figs=True,
                use_sup=False,
                use_lit=False,
            )

            distilled_facts = distill_evidence_for_slide(
                slide_uid=new_uid,
                plan_text=plan_text,
                slide_context=slide_context,
                primary_bullets=[],
                reminder_bullets=[],
                evidence=evidence,
                model=model,
            )

            spec = generate_slide_spec(
                slide_uid=new_uid,
                plan_text=plan_text,
                slide_context=slide_context,
                suggested_figures=[fig],
                bullet_requirements=[],
                reminder_bullets=[],
                distilled_facts=distilled_facts,
                evidence=evidence,
                model=model,
            )

            slide_specs[new_uid] = spec
            used_figures[group_name].add(fig)

    return slide_specs


