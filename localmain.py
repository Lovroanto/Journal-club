from pathlib import Path

from Presentation_module.presentation_pipeline import (
    run_presentation_planning,
    load_presentation_plan_from_debug,
)
from Presentation_module.bullet_resolution import resolve_bullets_globally

# ----------------------------
# Project paths
# ----------------------------
project_root = Path("/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary")

slide_groups_dir = project_root / "slides"                 # folder of group .txt files
bullet_summary_file = project_root / "01_main_factual_bullets.txt"
output_dir = project_root / "slides_newversion"            # debug outputs folder

# ----------------------------
# Debug toggles
# ----------------------------
DEBUG_SAVE = True

# If True: load cached JSON, skip all LLM calls (fast)
# If False: run full planning (slow) and write cache JSON
SKIP_PLANNING = True

CACHED_PRESENTATION_DEBUG = output_dir / "presentation_planning_debug.json"

# ----------------------------
# 1) Build / Load presentation plan
# ----------------------------
if SKIP_PLANNING:
    if not CACHED_PRESENTATION_DEBUG.exists():
        raise FileNotFoundError(
            f"Cache file not found:\n  {CACHED_PRESENTATION_DEBUG}\n\n"
            "Run once with SKIP_PLANNING=False to generate it."
        )
    presentation = load_presentation_plan_from_debug(CACHED_PRESENTATION_DEBUG)
else:
    presentation = run_presentation_planning(
        slide_groups_dir=slide_groups_dir,
        bullet_summary_file=bullet_summary_file,
        model="llama3.1",
        output_dir=output_dir if DEBUG_SAVE else None,
        debug_save=DEBUG_SAVE,
        debug_filename="presentation_planning_debug.json",
    )

print(f"\nPlanned {len(presentation.groups)} slide groups.")
print(f"Total slides: {len(presentation.all_slides)}")
print(f"Total candidates: {len(presentation.all_candidates)}")

# ----------------------------
# 2) Global bullet resolution (primary / reminder / drop)
# ----------------------------
resolved = resolve_bullets_globally(
    presentation,
    low_cutoff=0.55,
    early_bonus=0.06,
    debug_save=DEBUG_SAVE,
    debug_path=output_dir / "bullet_resolution_debug.json" if DEBUG_SAVE else None,
)

print("\n=== GLOBAL RESOLUTION REPORT ===")
for k, v in resolved.report.items():
    print(f"{k}: {v}")

# ----------------------------
# 3) Per-slide summary (compact)
# ----------------------------
print("\n=== PER-SLIDE SUMMARY ===")

for uid, bundle in sorted(presentation.all_slides.items()):
    usages = resolved.usage_by_slide.get(uid, [])
    prim = [u for u in usages if u.usage == "primary"]
    rem = [u for u in usages if u.usage == "reminder"]
    drp = [u for u in usages if u.usage == "drop"]

    print(f"\n--- {uid} ---")
    print(f"Plan: {bundle.slide_plan.blueprint_text}")
    print(f"Context: {bundle.slide_context}")
    print(f"Candidates proposed: {len(bundle.candidates)}")
    print(f"Resolved -> Primary: {len(prim)} | Reminders: {len(rem)} | Dropped: {len(drp)}")

    # Show a few examples
    for u in prim[:5]:
        print(f"  [PRIMARY] {u.bullet_id} (p={u.priority:.2f}, excl={u.exclusivity})")
        # Uncomment if you want more verbosity:
        # print(f"    text: {u.bullet_text}")
        # print(f"    arg:  {u.argument}")

    for u in rem[:3]:
        print(f"  [REMIND ] {u.bullet_id} (p={u.priority:.2f}) intro={u.primary_slide_uid}")

print("\nDone.")
