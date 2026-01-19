assignment.py:

This code segment from assignment.py focuses on proposing and assigning bullet points (from a paper) to slides in a slide group for scientific presentations, using an LLM for intelligent candidate generation. It includes robust handling of LLM outputs and optional figure assignments.

Imports and Types: Brings in standard libs (json, dataclasses, Path, typing) and local modules (llm, planning, bullets). Defines Exclusivity as a Literal type for bullet binding strength ("strong", "medium", "weak").
BulletCandidate Class: Dataclass to represent a proposed bullet-slide assignment, with fields: bullet_id, slide_uid, priority (0-1 float), exclusivity (from type), and argument (explanation string).
_extract_first_json_array Function: Helper to parse raw LLM string into a JSON list. Operations: Strip input, try direct parse; if fails, extract []-bounded substring; if still fails, backward-scan and trim tail for valid JSON. Raises ValueError on failure.
propose_bullet_candidates Function: Main logic for one slide group. Inputs: SlideGroupPlan, list of BulletPoints, LLM model, options (debug, min/max bullets per slide). Operations: Prepare data (slide blueprints, bullet list, figures); build detailed prompt instructing LLM to select 6-12 bullets per slide with priority/exclusivity/argument, and optionally one unique figure; call LLM; parse response with helper; validate/clean candidates; create BulletCandidate list; attach figures to group_plan slides in-place. Returns list of candidates.

bullet_resolution.py:

This file handles the global resolution of bullet assignments across the entire presentation, turning raw LLM-proposed candidates (from assignment.py) into final decisions: where each bullet is introduced (primary) and how it's used elsewhere (reminder or drop).

UsageType: Literal type ("primary", "reminder", "drop") for how a bullet appears on a slide.
BulletUsage Dataclass: Final usage record per bullet per slide – includes bullet_id/text, slide_uid, usage type, priority, exclusivity, argument, and the primary_slide_uid (where it's first introduced).
BulletResolutionResult Dataclass: Output container with:
primary_by_bullet: dict mapping bullet_id → primary slide_uid
usage_by_slide: dict mapping slide_uid → list of BulletUsage
report: diagnostic counts

_slide_order_map Function: Builds a global chronological order of all slides (by group order + slide_id) → dict slide_uid → index.
load_bullet_resolution_from_debug Function: Utility to reload a previously saved resolution JSON from disk into a BulletResolutionResult (for debugging/inspection).
resolve_bullets_globally Function (main logic):
Input: full PresentationPlan (with all_candidates collected from previous step) + optional params (low_cutoff=0.55, early_bonus=0.06, debug_save).
Operations:
Builds slide order and bullet text map.
Groups all candidates by bullet_id.
For each bullet:
Computes intro_score = priority × exclusivity_weight + small early-slide bonus.
Picks the highest-scoring candidate as primary (introduced there).
For other occurrences: if priority < low_cutoff → drop; else → reminder.

Fills usage_by_slide with BulletUsage entries.
Sorts usages per slide: primaries first, then reminders (high to low priority), drops last.
Generates report with counts.
Optionally saves full resolution to JSON for debug.

Returns BulletResolutionResult.


Role: Cleans up overlapping proposals → enforces "introduce once, recall later" policy, removes weak matches, prefers strong/early placements. Output ready for final slide rendering.

bullets.py:

This small module is responsible for extracting and structuring bullet points from raw text (typically a paper summary or abstract).
Type alias:

    SourceType
    Literal: "MAIN", "FIG", "SUP", or "LIT" — indicates where a piece of evidence comes from.

Classes:

    EvidenceChunk
    One retrieved supporting text chunk.
    Holds: source (SourceType), text (the chunk), metadata (dict, default empty).
    SlideSpec
    Complete final specification for one rendered slide.
    Holds:
        slide_uid
        title
        figure_used (optional)
        on_slide_bullets (list of bullet texts shown on slide)
        speaker_notes (full narration text)
        transition (text to next slide)
        reminders (list, default empty)
        evidence (list of EvidenceChunk, default empty)

BulletPoint Dataclass: Simple container for a single bullet with:
bullet_id: Unique identifier like "BP_001"
text: The cleaned bullet content

parse_bullet_points Function:
Takes raw multi-line text, splits into lines, removes empty lines, strips common bullet symbols (•, -, *, etc.) using regex, and returns a plain list of cleaned bullet strings.
number_bullet_points Function:
Main function – calls parse_bullet_points, then assigns sequential numbered IDs ("BP_001", "BP_002", ...) to each bullet and returns a list of BulletPoint objects.

Role in the system: Provides the initial structured list of key statements/facts from the paper that will later be assigned to slides (used as input in assignment.py and beyond). Very lightweight preprocessing step.

global_resolver.py:

This small module provides a strict global resolver for bullet assignments, enforcing a hard rule: each bullet_id can appear on at most ONE slide across the entire presentation.

EXCLUSIVITY_WEIGHT: Constant dict mapping exclusivity levels to score multipliers (strong: 1.20, medium: 1.00, weak: 0.85).
ResolvedPlacement Dataclass: Represents the final chosen placement for a bullet:
bullet_id, slide_uid, score (computed), priority, exclusivity, argument.

resolve_global_unique Function (main and only logic):
Input: List of candidate dicts (each with at least bullet_id, slide_uid, priority, exclusivity, argument).
Operations:
Groups candidates by bullet_id.
For each bullet, computes score = priority × exclusivity_weight.
Picks the single highest-scoring candidate as the winner.
All other candidates for that bullet are discarded.
Bullets with no candidates are tracked as unused.

Returns:
Dict: bullet_id → ResolvedPlacement (the chosen one)
List: bullet_ids that had zero candidates (unused)

Role in the system: Alternative or optional resolver compared to bullet_resolution.py. While bullet_resolution.py allows reminders/repeats with primary + recall logic, this one enforces strict uniqueness — no bullet repetition at all. Likely used when a simpler, non-redundant presentation is desired.

llm.py:

This tiny module provides the interface to a local LLM using Ollama.

call_llm Function (only function):
Inputs: model_name (e.g., "llama3:70b"), prompt (full prompt string), optional temperature.
Operations:
Builds the command ollama run <model_name>.
If temperature is given, prepends a line temperature=X to the prompt (simple hack for models that support it).
Runs the command via subprocess.run, feeding the prompt via stdin, capturing stdout.
Checks for errors (non-zero return code) → raises RuntimeError with stderr.
Returns the raw model output as a stripped string.



Role in the system: All LLM calls in the presentation module (e.g., in assignment.py for proposing bullet candidates) go through this function. It keeps the system fully local/offline, relying on Ollama-installed models. No API keys, no cloud dependency.

models.py

Defines two simple container classes to organize data:

SlideBundle: Packs everything for one slide – uid, plan, context string, and its list of candidates.
PlanningResult: Holds results for one whole slide group – group_id, group_plan, numbered bullets, all candidates, and a dict of SlideBundles (one per slide).

Role: Just data holders to bundle and pass information cleanly between stages (especially after planning and assignment for one group).

pipeline.py:

The Presentation_module is a pipeline that automatically turns a scientific paper into a structured PowerPoint-style presentation plan using a local LLM (via Ollama).
Step-by-step flow:

Input:
A text file with a summary of the paper (list of bullet points).
One or more "slide group" plan files (human-written or pre-generated outline describing groups of slides, their purpose, scope, and rough blueprints).

Per slide group (handled by pipeline.py):
Load the slide group plan (planning.py).
Extract and number all key facts from the paper summary as BulletPoints (bullets.py → "BP_001", "BP_002", ...).
Use LLM to propose which bullets fit each slide (6–12 per slide), with priority, exclusivity, argument, and optionally assign one unique figure per slide (assignment.py).
Generate explanatory context text for each slide (slide_context.py).
Bundle everything into a PlanningResult containing the group plan, bullets, candidates, and per-slide bundles.

Global resolution (across all groups – done separately):
Option A (bullet_resolution.py – default/recommended):
For each bullet, pick ONE primary slide (best score = priority × exclusivity + small early bonus).
On other slides: keep as "reminder" if priority decent, else drop.
Allows controlled repetition ("as mentioned earlier...").

Option B (global_resolver.py – alternative):
Strict rule: each bullet appears on exactly ONE slide (no reminders).


LLM integration (llm.py):
All LLM calls go through a simple local Ollama wrapper.


Final output:
A complete presentation plan with:

Which bullets go on which slides
Primary vs reminder usage
Assigned figures
Context/transition text
Ready for export to actual slides (e.g., PPT or Markdown)

Key strengths: Fully local, robust LLM parsing, handles bullet repetition intelligently, modular and debuggable.

planning.py:

Classes:

SlidePlan
Represents one single slide in a group.
Holds: slide_id (int), blueprint_text (the "Slide X: ..." description), shared group info (name, purpose, scope, transition, supmat_notes), and figure_used (None initially, filled later).
SlideGroupPlan
Represents a whole group of related slides.
Holds: group_name, purpose, scope, key_terms (list), transition, supmat_notes, available_figures (list of figure descriptions for the group), and slides (list of SlidePlan objects).

Functions:

_extract_single_line (private helper)
Finds and returns the text from the first line matching a regex pattern (captures group 1 if exists, otherwise the whole match).
_extract_all (private helper)
Finds all lines matching a regex pattern and returns a list of cleaned matches.
parse_slide_group_file (main function)
Reads a text file, parses it with regex, extracts group name, purpose, scope, key terms, transition, supmat notes, available figures, and all "Slide X:" lines, then builds and returns a complete SlideGroupPlan object.

presentation_models.py:

Class:

PresentationPlan
Represents the entire presentation after processing all slide groups, but before global bullet resolution.
Holds:
groups: List of PlanningResult (one per slide group – each contains its own plan, bullets, candidates, and slide bundles).
Properties (computed):
all_slides → Returns a single dict mapping every slide_uid across all groups to its SlideBundle.
all_candidates → Returns a flat list of allBulletCandidate objects from every group.


Role: Acts as the top-level container for the full presentation. Makes it easy to access all slides and all proposed candidates globally – perfect input for the global resolution step (like in bullet_resolution.py).

presentation_pipeline.py:

Class:

    PresentationPlan
    Represents the entire presentation after processing all slide groups, but before global bullet resolution.
    Holds:
        groups: List of PlanningResult (one per slide group – each contains its own plan, bullets, candidates, and slide bundles).

    Properties (computed):
        all_slides → Returns a single dict mapping every slide_uid across all groups to its SlideBundle.
        all_candidates → Returns a flat list of allBulletCandidate objects from every group.

presentation_pipeline.py:

Functions:

run_presentation_planning (main function)
Runs the full planning pipeline for ALL slide-group .txt files in a folder.
Does:
Finds and sorts all .txt files in slide_groups_dir.
For each file, calls run_planning_pipeline (from pipeline.py) to process one group.
Collects successful results into a list of PlanningResult.
Catches errors, logs them (optionally continues or stops).
Builds and returns a PresentationPlan containing all processed groups.
Optionally saves: one big debug JSON for the whole presentation + an error log.

load_presentation_plan_from_debug
Loads a full PresentationPlan back from a debug JSON file (the one saved by run_presentation_planning).
Reconstructs all objects (SlideGroupPlan, SlidePlan, SlideBundle, BulletCandidate, etc.) without running any LLM calls — useful for testing or resuming.

Role: This is the top-level entry point to generate a complete presentation plan from multiple slide groups.

rag_loader.py:

Class:

RAGStores
Simple container that holds loaded vector stores for Retrieval-Augmented Generation (RAG).
Fields:
main: Chroma database for main paper content (required)
figs: Chroma database for figures/captions (required)
sup: Optional Chroma for supplementary material
lit: Optional Chroma for literature/references


Functions:

load_chroma_db
Loads a single persisted Chroma vector database from a folder path.
Uses HuggingFaceEmbeddings (specified model) for embeddings.
Raises error if the folder doesn't exist.
load_rag_stores (main function)
Loads all available RAG databases based on a dict of paths.
Always loads "main" and "figs".
Optionally loads "sup" and/or "lit" if paths are provided.
Returns a single RAGStores object with all loaded databases.
Default embedding model: "BAAI/bge-small-en-v1.5".

Role: Provides easy loading of pre-built Chroma vector stores (created elsewhere) so other parts of the system can perform similarity search/retrieval on paper text, figures, supplementary material, etc. – useful for future RAG-enhanced prompts (not used in the current planning pipeline, but ready for extensions).

slide_context.py:

Function:

build_slide_context
Generates a short natural-language context description for a single slide using the LLM.
Inputs:
group_plan: The SlideGroupPlan (provides group name, purpose, scope, transition)
slide_plan: The current SlidePlan (provides blueprint_text)
model: LLM model name
prev_slide_blueprint: Optional text of the previous slide's blueprint (or None if first)
What it does:
Builds a detailed prompt with group info, previous slide, and current slide blueprint.
Asks the LLM to write 4–6 sentences explaining:
Why the slide exists here
What the audience should learn
Connection from previous slide (or group start)
What it sets up for the next slide

Calls call_llm and returns the plain text response (stripped).


Role: Adds explanatory "narrative context" to each slide (stored in SlideBundle.slide_context) to help later stages (e.g., final slide writing or narration) understand flow and purpose. Used in pipeline.py when building slide bundles.

slide realization.py

build_slide_context (slide_context.py)
Calls: call_llm (from llm.py)
→ Uses the LLM to generate a 4–6 sentence context explanation for one slide based on group info, previous slide blueprint, and current blueprint.
run_presentation_planning (presentation_pipeline.py)
Calls: run_planning_pipeline (from pipeline.py) for each group file
→ Loops over all .txt files in a directory, processes each slide group via the per-group pipeline, collects results, handles errors, and returns a full PresentationPlan.
run_planning_pipeline (pipeline.py)
Calls:
parse_slide_group_file (from planning.py)
number_bullet_points (from bullets.py)
propose_bullet_candidates (from assignment.py)
build_slide_context (from slide_context.py) → which itself calls call_llm
→ Orchestrates the complete processing of one slide group: load plan, number bullets, generate LLM candidates, build slide contexts, attach candidates, return PlanningResult.

slide_selector.py:

Class:

SlideRef (frozen dataclass)
Immutable reference to a specific slide.
Fields: group_id (str), slide_id (int).
Has __str__ method that returns the full slide_uid string like "003_Intro::Slide_2".
No function calls inside.

Functions:

list_groups
Returns a list of all group_ids from the PresentationPlan, preserving the original order of groups.
Calls: nothing else.
list_slides_in_group
Given a group_id, returns a list of all slide_uids in that group, sorted by slide_id.
Calls:
Uses sorted() on group.slides.items() with a key based on slide_plan.slide_id.
Uses built-in next() to find the matching group in presentation.groups.
No external or module functions called.

resolve_slide_uid (main function)
Flexible resolver: turns various human-friendly ways of referring to a slide into the exact slide_uid.
Supports: direct slide_uid, group_id + slide_id, group_index + slide_id, group_name_contains (substring) + slide_id, or just slide_id (if unique).
Raises clear KeyError/ValueError if not found or ambiguous.
Calls:
Uses presentation.all_slides (property from PresentationPlan that collects all SlideBundle objects).
Uses presentation.groups list to access group data when needed.
Built-in functions: sorted(), list comprehensions, next().
No other module functions called.


Role: Pure utility — helps safely and conveniently identify slides without hardcoding full uids. Used when you need to pick a specific slide for final content generation, narration, or debugging.

slide_specs.py:

Type alias:

SourceType
Literal: "MAIN", "FIG", "SUP", or "LIT" — indicates where a piece of evidence comes from.

Classes:

EvidenceChunk
One retrieved supporting text chunk.
Holds: source (SourceType), text (the chunk), metadata (dict, default empty).
SlideSpec
Complete final specification for one rendered slide.
Holds:
slide_uid
title
figure_used (optional)
on_slide_bullets (list of bullet texts shown on slide)
speaker_notes (full narration text)
transition (text to next slide)
reminders (list, default empty)
evidence (list of EvidenceChunk, default empty)