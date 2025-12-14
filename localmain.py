from pathlib import Path
from Presentation_module.presentation_pipeline import run_presentation_planning

project_root = Path("/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary")

slide_groups_dir = project_root / "slides"                 # folder of group .txt files
bullet_summary_file = project_root / "01_main_factual_bullets.txt"
output_dir = project_root / "slides_newversion"            # where to write ONE debug file

DEBUG_SAVE = True

presentation = run_presentation_planning(
    slide_groups_dir=slide_groups_dir,
    bullet_summary_file=bullet_summary_file,
    model="llama3.1",
    output_dir=output_dir,
    debug_save=DEBUG_SAVE,
)

print(f"\nPlanned {len(presentation.groups)} slide groups.")
print(f"Total slides: {len(presentation.all_slides)}")
print(f"Total candidates: {len(presentation.all_candidates)}")

# Optional: quick per-slide print
for uid, bundle in sorted(presentation.all_slides.items()):
    print(f"\n--- {uid} ---")
    print(f"Plan: {bundle.slide_plan.blueprint_text}")
    print(f"Context: {bundle.slide_context}")
    print(f"Candidates: {len(bundle.candidates)}")
