from pathlib import Path
from Presentation_module.pipeline import run_planning_pipeline

# ----------------------------
# Project paths
# ----------------------------
project_root = Path("/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary")

slide_group_file = project_root / "slides/003_Introduction_to_Continuous_Lasing.txt"
bullet_summary_file = project_root / "01_main_factual_bullets.txt"

output_dir = project_root / "slides_newversion"

# ----------------------------
# Debug toggle
# ----------------------------
DEBUG_SAVE = True  # set False to keep everything only in memory (no output files)

# ----------------------------
# Run planning pipeline
# ----------------------------
per_slide_bullets = run_planning_pipeline(
    slide_group_file=slide_group_file,
    bullet_summary_file=bullet_summary_file,
    model="llama3.1",
    output_dir=output_dir,
    run_name="intro_continuous_lasing",
    debug_save=DEBUG_SAVE,
)

# ----------------------------
# Console debug output
# ----------------------------
for slide_uid, bullets in sorted(per_slide_bullets.items()):
    print(f"\n=== {slide_uid} ({len(bullets)} bullets) ===")
    for bp in bullets:
        print(f"{bp.bullet_id}: {bp.text}")
