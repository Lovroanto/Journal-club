from __future__ import annotations

from typing import Any, Dict, List

from .chunking import chunk_text
from .ollama_client import OllamaClient


# Schema for "edges" the LLM should return
EDGES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "integer"},
                    "to": {"type": "integer"},
                    "type": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["from", "to", "type"],
            },
        }
    },
    "required": ["edges"],
}

SYSTEM = (
    "You infer logical relationships between bullet points to draw arrows on slides. "
    "Return ONLY valid JSON matching the schema. "
    "Use bullet indices (0..n-1). "
    "Keep the number of edges small (<=3) and avoid cycles."
)


def infer_edges(
    client: OllamaClient,
    *,
    model: str,
    title: str,
    bullets: List[str],
    speaker_notes: str,
    max_prompt_chars: int = 12000,
) -> List[Dict[str, Any]]:
    """
    Returns a list of edges like:
      {"from": 0, "to": 2, "type": "leads_to", "label": "thus"}

    If speaker_notes is long, it is chunked; edges are merged and deduplicated.
    """
    if not bullets:
        return []

    bullets_block = "\n".join([f"{i}. {b}" for i, b in enumerate(bullets)])

    base_prompt = (
        f"Slide title: {title}\n\n"
        f"Bullets:\n{bullets_block}\n\n"
        "Speaker notes:\n"
    )

    chunks = chunk_text(speaker_notes or "", max_chars=6000, overlap=300) or [""]

    all_edges: List[Dict[str, Any]] = []

    for chunk in chunks:
        prompt = base_prompt + chunk
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]

        result = client.generate_json(
            model=model,
            system=SYSTEM,
            prompt=prompt,
            schema=EDGES_SCHEMA,
        )
        edges = result.get("edges", [])
        if isinstance(edges, list):
            all_edges.extend([e for e in edges if isinstance(e, dict)])

    # Deduplicate
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for e in all_edges:
        try:
            frm = int(e.get("from"))
            to = int(e.get("to"))
        except Exception:
            continue
        typ = str(e.get("type", "")).strip()
        label = str(e.get("label", "")).strip()

        if typ == "" or frm == to:
            continue

        key = (frm, to, typ, label)
        if key not in seen:
            seen.add(key)
            deduped.append({"from": frm, "to": to, "type": typ, "label": label})

    # Keep it simple: cap to 3 edges
    return deduped[:3]
