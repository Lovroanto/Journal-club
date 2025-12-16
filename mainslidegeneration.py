from pathlib import Path

from Presentation_module.presentation_pipeline import load_presentation_plan_from_debug
from Presentation_module.bullet_resolution import load_bullet_resolution_from_debug
from Presentation_module.rag_loader import load_rag_stores
from Presentation_module.slide_realization import realize_presentation_slides
from Presentation_module.slide_selector import resolve_slide_uid, list_groups

# ----------------------------
# Paths
# ----------------------------
project_root = Path("/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary")
output_dir = project_root / "slides_newversion"

presentation_cache = output_dir / "presentation_planning_debug.json"
resolution_cache = output_dir / "bullet_resolution_debug.json"

# ----------------------------
# Load cached planning + resolution
# ----------------------------
presentation = load_presentation_plan_from_debug(presentation_cache)
resolved = load_bullet_resolution_from_debug(resolution_cache)

print(f"\nLoaded presentation: {len(presentation.groups)} groups, {len(presentation.all_slides)} slides")

# Optional: show group ids (useful once)
print("\nGroup IDs in order:")
for i, gid in enumerate(list_groups(presentation)):
    print(f"  [{i}] {gid}")

# ----------------------------
# Load RAG stores
# ----------------------------
rag_paths = {
    "main": "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/main/chroma_db",
    "figs": "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/figures/chroma_db",
    "sup":  "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/supplementary/chroma_db",
    "lit":  "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/RAG/literature_rag/chroma_db",
}
rag = load_rag_stores(rag_paths)

# ----------------------------
# Choose ONE slide to debug (pick ONE approach)
# ----------------------------

# Approach A: explicit slide_uid (exact)
REQUESTED_SLIDE_UID = None  # e.g. "003_Introduction_to_Continuous_Lasing::Slide_2"

# Approach B: group_id + slide_id (recommended once you know group_id)
REQUESTED_GROUP_ID = None   # e.g. "003_Introduction_to_Continuous_Lasing"
REQUESTED_SLIDE_ID = 2      # slide number inside that group

# Approach C: group_name substring + slide_id (human-friendly)
REQUESTED_GROUP_NAME_CONTAINS = "Continuous Lasing"  # or None
# slide_id reused from REQUESTED_SLIDE_ID

# Approach D: group index + slide_id (fastest if groups are ordered 001,002,003...)
REQUESTED_GROUP_INDEX = None  # e.g. 2 for the 3rd group (0-based)

# Resolve to a real slide_uid
TARGET_SLIDE_UID = resolve_slide_uid(
    presentation,
    slide_uid=REQUESTED_SLIDE_UID,
    group_id=REQUESTED_GROUP_ID,
    slide_id=REQUESTED_SLIDE_ID,
    group_index=REQUESTED_GROUP_INDEX,
    group_name_contains=REQUESTED_GROUP_NAME_CONTAINS,
)

print("\nResolved TARGET_SLIDE_UID:", TARGET_SLIDE_UID)

# ----------------------------
# Generate slide spec for ONLY that slide
# ----------------------------
slide_specs = realize_presentation_slides(
    presentation=presentation,
    resolved=resolved,
    rag=rag,
    model="llama3.1",
    output_dir=output_dir,
    debug_save=True,
    only_slide_uids=[TARGET_SLIDE_UID],
)

if TARGET_SLIDE_UID not in slide_specs:
    raise KeyError(
        f"SlideSpec not generated for {TARGET_SLIDE_UID}. "
        f"Generated keys: {list(slide_specs.keys())}"
    )

spec = slide_specs[TARGET_SLIDE_UID]

# ----------------------------
# Print result
# ----------------------------
print("\n=== GENERATED SLIDE SPEC ===")
print("UID:", spec.slide_uid)
print("Title:", spec.title)
print("Figure:", spec.figure_used)
print("Bullets:")
for b in spec.on_slide_bullets:
    print(" -", b)
print("\nSpeaker notes:\n", spec.speaker_notes)
print("\nTransition:", spec.transition)
