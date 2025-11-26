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
    """Run a refine chain to extract specialized notions."""

    # ---- Split into chunks ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    chunks = splitter.split_text(plan_text)
    docs = [Document(page_content=c) for c in chunks]

    # ---- Prompts ----
    question_prompt = PromptTemplate.from_template(
        """
You are extracting the *core scientific concepts* from a plan of presentation.
Use the logic of the sentences to understand which notions are central to the presentation. 
Follow these rules strictly:

1. Extract only concepts that are explicitly part of:
   - the main goal or motivation
   - key physical processes
   - experimental setup components
   - theoretical models or mechanisms
   - phenomena central to the paper’s results
2. Ignore:
   - generic physics terms
   - side phenomena
   - figure labels, section headers, or meta-text

Format:
- One concept per line starting with "- "
- Each line must be a short noun phrase, no sentences
- DO NOT add, merge, or rewrite concepts

Examples:

Good:
- Recoil-induced resonance (RIR)
- High-finesse ring cavity

Bad:
- "Slide 4 illustrates…"           # meta-text
- "Zone I, II, III, IV"             # figure labels
- "This allows for better cooling"  # consequence

Text:
\"\"\"{text}\"\"\"

Return ONLY the bullet list following the rules.
"""
    )

    refine_prompt = PromptTemplate.from_template(
        """
Current extracted core notions:
\"\"\"{existing_answer}\"\"\"

New text chunk to analyze:
\"\"\"{text}\"\"\"

Your task:
- Identify ONLY new *core scientific notions* that are central to the article.
- A core notion is any concept that is part of:
  * the logic of the sentences points to it
  * main goal or motivation
  * key physical process
  * experimental setup component
  * theoretical model or mechanism
  * phenomenon essential for understanding the claims
- Ignore:
  * generic terms
  * secondary or side effects
  * consequences
  * figure labels, slide numbers, section headers
  * anything mentioned only once that is not central

Rules:
1. Each bullet must contain ONE concept only.
2. Use a short noun phrase; do NOT write full sentences.
3. DO NOT reword, merge, or invent concepts.
4. DO NOT repeat any notion already in {existing_answer}.
5. Focus on the sentence logic — pick notions that the text emphasizes as central.
6. Keep the same bullet format as the existing list.

Return the UPDATED bullet list including both previous notions and new valid ones.

"""
    )

    # ---- Refine chain ----
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text",
    )

    result = chain.invoke({
        "input_documents": docs
    })

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
