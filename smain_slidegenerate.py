from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

from Presentation_module.presentation_pipeline import (
    run_presentation_planning,
    load_presentation_plan_from_debug,
)
from Presentation_module.bullet_resolution import (
    resolve_bullets_globally,
    load_bullet_resolution_from_debug,
)
from Presentation_module.rag_loader import load_rag_stores
from Presentation_module.slide_realization import realize_presentation_slides
from Presentation_module.slide_selector import list_groups


# ----------------------------
# Configuration
# ----------------------------
PROJECT_ROOT = Path("/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary")

SLIDE_GROUPS_DIR = PROJECT_ROOT / "slides"
BULLET_SUMMARY_FILE = PROJECT_ROOT / "01_main_factual_bullets.txt"
OUTPUT_DIR = PROJECT_ROOT / "slides_final_run"

# Your local model name (as you used before)
MODEL = "llama3.1"

# RAG store paths
RAG_PATHS = {
    "main": "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/main/chroma_db",
    "figs": "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/figures/chroma_db",
    "sup":  "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/supplementary/chroma_db",
    "lit":  "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/RAG/literature_rag/chroma_db",
}

# Debug / caching toggles
DEBUG_SAVE = True

# If caches exist, load them. If not, compute and create them.
USE_CACHES_IF_AVAILABLE = True

# Optional checks (no LLM)
RUN_SIMPLE_REPETITION_REPORT = True


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")

def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def slide_uid_sort_key(uid: str) -> Tuple[str, int]:
    # uid format: "003_GroupName::Slide_2"
    # Robust fallback if parsing fails
    try:
        group, slide = uid.split("::")
        slide_num = int(slide.replace("Slide_", ""))
        return (group, slide_num)
    except Exception:
        return (uid, 0)

def render_speaker_script(slides: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# Speaker Script\n")
    for s in slides:
        lines.append(f"## {s['slide_uid']} — {s.get('title','')}".strip())
        lines.append("")
        lines.append("**On-slide bullets**")
        for b in s.get("on_slide_bullets", []):
            lines.append(f"- {b}")
        lines.append("")
        if s.get("speaker_notes"):
            lines.append("**Speaker notes**")
            lines.append(s["speaker_notes"].strip())
            lines.append("")
        if s.get("transition"):
            lines.append("**Transition**")
            lines.append(s["transition"].strip())
            lines.append("")
        lines.append("---\n")
    return "\n".join(lines)

def simple_repetition_report(slides: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Very lightweight repetition detector:
    - exact duplicate bullets across slides
    - exact duplicate sentences in speaker notes across slides (normalized)
    """
    import re
    def norm(x: str) -> str:
        x = x.strip().lower()
        x = re.sub(r"\s+", " ", x)
        return x

    bullet_map: Dict[str, List[str]] = {}
    for s in slides:
        for b in s.get("on_slide_bullets", []):
            bullet_map.setdefault(norm(b), []).append(s["slide_uid"])

    dup_bullets = [
        {"bullet": k, "slides": v}
        for k, v in bullet_map.items()
        if len(v) >= 2 and k
    ]

    # sentence duplicates (rough)
    sent_map: Dict[str, List[str]] = {}
    for s in slides:
        notes = s.get("speaker_notes") or ""
        # split on ., !, ?, but keep it simple
        parts = re.split(r"[.!?]\s+", notes)
        for sent in parts:
            n = norm(sent)
            if len(n) < 40:  # ignore very short phrases
                continue
            sent_map.setdefault(n, []).append(s["slide_uid"])

    dup_sents = [
        {"sentence": k, "slides": v}
        for k, v in sent_map.items()
        if len(v) >= 2 and k
    ]

    return {
        "duplicate_bullets_exact": sorted(dup_bullets, key=lambda x: (-len(x["slides"]), x["bullet"])),
        "duplicate_sentences_exact": sorted(dup_sents, key=lambda x: (-len(x["slides"]), x["sentence"])),
    }


# ----------------------------
# Main pipeline
# ----------------------------
def main() -> None:
    ensure_dir(OUTPUT_DIR)

    planning_cache = OUTPUT_DIR / "presentation_planning_debug.json"
    resolution_cache = OUTPUT_DIR / "bullet_resolution_debug.json"

    # 1) Planning
    if USE_CACHES_IF_AVAILABLE and planning_cache.exists():
        print(f"[1/5] Loading planning cache: {planning_cache}")
        presentation = load_presentation_plan_from_debug(planning_cache)
    else:
        print("[1/5] Running presentation planning...")
        presentation = run_presentation_planning(
            slide_groups_dir=SLIDE_GROUPS_DIR,
            bullet_summary_file=BULLET_SUMMARY_FILE,
            model=MODEL,
            output_dir=OUTPUT_DIR if DEBUG_SAVE else None,
            debug_save=DEBUG_SAVE,
            debug_filename=planning_cache.name,
        )

    print(f"Planned {len(presentation.groups)} slide groups.")
    print(f"Total slides: {len(presentation.all_slides)}")
    print(f"Total candidates: {len(presentation.all_candidates)}")
    print("Group order:")
    for i, gid in enumerate(list_groups(presentation)):
        print(f"  [{i}] {gid}")

    # 2) Bullet resolution
    if USE_CACHES_IF_AVAILABLE and resolution_cache.exists():
        print(f"[2/5] Loading resolution cache: {resolution_cache}")
        resolved = load_bullet_resolution_from_debug(resolution_cache)
    else:
        print("[2/5] Running global bullet resolution...")
        resolved = resolve_bullets_globally(
            presentation,
            low_cutoff=0.55,
            early_bonus=0.06,
            debug_save=DEBUG_SAVE,
            debug_path=resolution_cache if DEBUG_SAVE else None,
        )

    print("\n=== GLOBAL RESOLUTION REPORT ===")
    for k, v in getattr(resolved, "report", {}).items():
        print(f"{k}: {v}")

    # 3) Load RAG
    print("[3/5] Loading RAG stores...")
    rag = load_rag_stores(RAG_PATHS)

    # 4) Realize ALL slides
    print("[4/5] Realizing all slides...")
    slide_specs = realize_presentation_slides(
        presentation=presentation,
        resolved=resolved,
        rag=rag,
        model=MODEL,
        output_dir=OUTPUT_DIR,
        debug_save=True,
        only_slide_uids=None,  # <-- all slides
    )

    print(f"Generated SlideSpecs: {len(slide_specs)}")

    # 5) Assemble final outputs (JSON + speaker script)
    print("[5/5] Writing assembled outputs...")

    # Convert SlideSpec objects to serializable dicts
    slides_out: List[Dict[str, Any]] = []
    for uid in sorted(slide_specs.keys(), key=slide_uid_sort_key):
        spec = slide_specs[uid]
        slides_out.append({
            "slide_uid": getattr(spec, "slide_uid", uid),
            "title": getattr(spec, "title", ""),
            "figure_used": getattr(spec, "figure_used", None),
            "on_slide_bullets": list(getattr(spec, "on_slide_bullets", []) or []),
            "speaker_notes": getattr(spec, "speaker_notes", ""),
            "transition": getattr(spec, "transition", ""),
            "reminders": list(getattr(spec, "reminders", []) or []),
        })

    final_obj = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "project_root": str(PROJECT_ROOT),
        "n_groups": len(presentation.groups),
        "n_slides": len(slides_out),
        "slides": slides_out,
    }

    final_json_path = OUTPUT_DIR / "final_presentation.json"
    speaker_md_path = OUTPUT_DIR / "speaker_script.md"

    write_json(final_json_path, final_obj)
    write_text(speaker_md_path, render_speaker_script(slides_out))

    print(f"Wrote: {final_json_path}")
    print(f"Wrote: {speaker_md_path}")

    if RUN_SIMPLE_REPETITION_REPORT:
        report = simple_repetition_report(slides_out)
        report_path = OUTPUT_DIR / "coherence_report.json"
        write_json(report_path, report)
        print(f"Wrote: {report_path}")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()
