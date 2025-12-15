from pathlib import Path

from Presentation_module.presentation_pipeline import load_presentation_plan_from_debug
from Presentation_module.bullet_resolution import load_bullet_resolution_from_debug
from Presentation_module.rag_loader import load_rag_stores
from Presentation_module.slide_realization import realize_presentation_slides

project_root = Path("/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary")
output_dir = project_root / "slides_newversion"

# ---- caches (skip expensive steps) ----
presentation_cache = output_dir / "presentation_planning_debug.json"
resolution_cache = output_dir / "bullet_resolution_debug.json"

presentation = load_presentation_plan_from_debug(presentation_cache)
resolved = load_bullet_resolution_from_debug(resolution_cache)

# ---- RAG paths (your corrected paths) ----
rag_paths = {
    "main": "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/main/chroma_db",
    "figs": "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/figures/chroma_db",
    "sup":  "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/supplementary/chroma_db",
    "lit":  "/home/lbarisic/ai_data/Journal_Club/First/Newversion/Decomposition/RAG/literature_rag/chroma_db",
}

rag = load_rag_stores(rag_paths)

# ---- choose one slide to debug ----
TARGET_SLIDE_UID = "intro_continuous_lasing::Slide_2"

slide_specs = realize_presentation_slides(
    presentation=presentation,
    resolved=resolved,
    rag=rag,
    model="llama3.1",
    output_dir=output_dir,
    debug_save=True,
    only_slide_uids=[TARGET_SLIDE_UID],
)

spec = slide_specs[TARGET_SLIDE_UID]
print("\n=== GENERATED SLIDE SPEC ===")
print("UID:", spec.slide_uid)
print("Title:", spec.title)
print("Figure:", spec.figure_used)
print("Bullets:")
for b in spec.on_slide_bullets:
    print(" -", b)
print("\nSpeaker notes:\n", spec.speaker_notes)
print("\nTransition:", spec.transition)
