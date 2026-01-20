ppt_from_plan - Generate PowerPoint slides from a plan JSON using python-pptx
==========================================================================

Goal
----
This module converts a "plan JSON" (Johnson-document-like structure) into a .pptx.
It assumes Ollama is already running locally and can be queried to infer logical
connections (arrows) between bullet points.

Default Ollama model: qwen2.5:7b

What it does (current stage)
----------------------------
- Creates one PowerPoint slide per plan entry.
- Adds slide title.
- Adds each bullet as its own rounded rectangle box (so arrows can connect them).
- If figure_used is present (non-null), it reserves space on the right side and
  draws a "figure placeholder" box with the caption inside.
- Optionally calls Ollama to infer relationships between bullet points and draws
  arrows between bullet boxes.

What it does NOT do (yet)
-------------------------
- It does not extract images from the paper PDF.
- It does not insert real figure images; it only reserves space (placeholder).

JSON plan format expected
------------------------
Top-level JSON must include a "slides" list:
{
  "slides": [
    {
      "slide_uid": "...",
      "title": "...",
      "figure_used": null OR "Figure ...",
      "on_slide_bullets": ["...", "..."],
      "speaker_notes": "...",
      "transition": "...",
      "reminders": [...]
    },
    ...
  ]
}

Module structure
----------------
ppt_from_plan/
  __init__.py          - exports build_ppt_from_plan
  main.py              - main API function + CLI entrypoint
  plan_loader.py       - loads plan JSON into SlidePlan objects
  ollama_client.py     - minimal REST client for Ollama /api/generate
  llm_inference.py     - asks LLM to infer edges between bullet points (JSON schema)
  chunking.py          - chunk speaker notes if needed
  slide_builder.py     - python-pptx rendering (title, bullets, figure placeholder, arrows)

Usage (Python)
--------------
from ppt_from_plan import build_ppt_from_plan

build_ppt_from_plan(
    plan_path="final_presentation.json",
    out_path="deck.pptx",
    model="qwen2.5:7b",               # default
    ollama_base_url="http://localhost:11434",
    enable_llm=True,
    enable_arrows=True,
    reserve_figure_space=True,
)

Usage (CLI)
-----------
python -m ppt_from_plan.main --plan final_presentation.json --out deck.pptx

Options:
  --model <name>            choose Ollama model (default qwen2.5:7b)
  --ollama <url>            Ollama base URL (default http://localhost:11434)
  --template <path>         optional PPTX template
  --no-llm                  disable LLM calls
  --no-arrows               do not infer/draw arrows
  --no-figure-space         ignore figure_used and do not reserve space

Dependencies
------------
pip install python-pptx requests

Notes / Gotchas
---------------
- Arrowheads on connectors can be version-dependent in python-pptx/PowerPoint.
  If arrowheads don't render reliably, the next step is to implement line + triangle
  arrowheads manually.
- The LLM inference is deliberately conservative: it caps to <=3 edges per slide to
  avoid clutter.

Next extension (PDF figure extraction)
--------------------------------------
Later, add a new module that:
- parses the article PDF
- extracts figures (images)
- maps figure identifiers to file paths
- replaces the placeholder with slide.shapes.add_picture(...)
