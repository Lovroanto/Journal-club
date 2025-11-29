#!/usr/bin/env python3
"""
notion_extractor_module.py

Goal:
- Read a presentation plan (or any long structured text)
- Chunk it
- Extract highly specialized notions (concepts unique to the paper)
- Use a refine chain to accumulate notions without duplicates
- Output: one notion per line (flat list)
"""

import json
from pathlib import Path
from typing import Optional, List, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel


# ============================================================
# INTERNAL: Extract notions with a refine chain
# ============================================================
def _extract_notions_refine_chain(
    plan_text: str,
    llm: BaseLanguageModel
) -> str:
    """
    🔥 Updated scientific notion extractor (improved over original version)

    Extracts domain-specific concepts from the contextual summary or full summary.
    Now returns enriched terms in the form:

        <notion> — <short meaning (≤8 words)>

    Designed for downstream RAG / Literature lookup.
    """

    # --------------- Chunking ---------------
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.chains.summarize import load_summarize_chain
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = [Document(page_content=c) for c in splitter.split_text(plan_text)]

    # --------------- NEW Prompts ---------------

    question_prompt = PromptTemplate.from_template(
        """
You are extracting core scientific notions from a research summary.

You must return **ONLY** the key concepts that are central to understanding the work.
These include:

✓ physical / quantum mechanisms
✓ theoretical constructs or regimes
✓ experimental subsystems / apparatus
✓ named phenomena or emergent behavior
✓ measurement tools or spectroscopic signals
✓ important parameters (if experimentally decisive)

⚠ KEEP terms even if mentioned once *if they are domain-distinctive*.

EXCLUDE completely:

✗ generic words (feedback, atoms, photons, cooling)
✗ figure references, slide labels
✗ broad phrases without scientific identity
✗ consequences or motivations instead of mechanisms

OUTPUT FORMAT (strict):
- <notion> — <short meaning (≤8 words)>

Example GOOD output:
- recoil-induced resonance — momentum grating driven gain
- cross-over regime — cavity linewidth ≈ gain linewidth
- self-regulated atomic loss — stabilizes dressed cavity detuning

Text to extract from:
\"\"\"{text}\"\"\"

Return ONLY bullet lines in the exact format.
"""
    )

    refine_prompt = PromptTemplate.from_template(
        """
Current extracted notions:
\"\"\"{existing_answer}\"\"\"

New text:
\"\"\"{text}\"\"\"

TASK:
• Append only new scientific notions, do NOT remove earlier ones.
• Same bullet formatting:
    - notion — meaning (≤8 words)
• New notions may be kept even if only one appearance
    if scientifically distinctive or named.

🚫 Do NOT add duplicates, paraphrases, merged concepts, or sentences.

Return the updated notion list in the same bullet style.
"""
    )

    # --------------- Refine extraction ---------------
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text",
    )

    result = chain.invoke({"input_documents": docs})
    return result["output_text"].strip()


# ============================================================
# PUBLIC API
# ============================================================
def extract_specialized_notions(
    input_plan_path: str,
    output_path: str,
    *,
    llm: Optional[BaseLanguageModel] = None,
    model: str = "llama3.1:latest",
    num_repeats: int = 3,          # number of extraction repetitions
    min_occurrences: int = 2,      # minimum times a notion must appear to be kept
) -> None:
    """
    Extract specialized notions from a presentation plan using a refine chain.

    Args:
        input_plan_path: Path to the presentation plan text file.
        output_path: Output path for the list of notions.
        llm: Optional external LLM object.
        model: Name of Ollama model to use if llm is not provided.
        num_repeats: How many times to repeat the extraction.
        min_occurrences: Minimum number of times a notion must appear to be accepted.
    """

    from collections import Counter

    input_path = Path(input_plan_path).expanduser().resolve()
    out_path = Path(output_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Plan file not found: {input_path}")

    # ---- Load LLM ----
    if llm is None:
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=model, temperature=0.0, top_p=1.0, max_tokens=1024)
            print(f"Using Ollama model: {model}")
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. Install with: pip install langchain-ollama"
            )
    else:
        print(f"Using provided LLM: {llm.__class__.__name__}")

    # ---- Read plan ----
    plan_text = input_path.read_text(encoding="utf-8")
    print(f"Loaded plan ({len(plan_text)} chars)")

    # ---- Repeat extraction ----
    all_extracted = []
    for i in range(num_repeats):
        print(f"Extraction pass {i+1}/{num_repeats}...")
        raw_output = _extract_notions_refine_chain(plan_text, llm)
        for line in raw_output.splitlines():
            line = line.strip("-•* \t\n")
            if line:
                all_extracted.append(line)

    # ---- Count occurrences and filter ----
    counter = Counter(all_extracted)
    filtered = [item for item in counter if counter[item] >= min_occurrences]

    # ---- Preserve original order (first appearance in all_extracted) ----
    seen: Set[str] = set()
    unique_filtered = []
    for item in all_extracted:
        if item in filtered and item not in seen:
            seen.add(item)
            unique_filtered.append(item)

    # ---- Write output ----
    out_path.write_text("\n".join(unique_filtered), encoding="utf-8")
    print(f"Done. Extracted {len(unique_filtered)} specialized notions (min_occurrences={min_occurrences}).")
    print(f"Saved to: {out_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python notion_extractor_module.py <input_plan> <output_notions> [model] [num_repeats] [min_occurrences]")
        sys.exit(1)

    model = sys.argv[3] if len(sys.argv) > 3 else "llama3.1:latest"
    num_repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    min_occurrences = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    extract_specialized_notions(
        sys.argv[1],
        sys.argv[2],
        model=model,
        num_repeats=num_repeats,
        min_occurrences=min_occurrences
    )
