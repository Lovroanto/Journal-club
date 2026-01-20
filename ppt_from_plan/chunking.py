from __future__ import annotations
from typing import List


def chunk_text(text: str, max_chars: int = 6000, overlap: int = 300) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Intended mostly for speaker notes (bullets are usually short).
    """
    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        if j >= n:
            break
        i = max(0, j - overlap)

    return chunks
