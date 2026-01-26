from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from lxml import etree as ET

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _tail_words(text: str, n: int = 12) -> str:
    words = re.findall(r"\w+", _norm(text))
    return " ".join(words[-n:]) if words else ""

def _extract_label(formula_el) -> str:
    lab = formula_el.find(".//tei:label", namespaces=NS)
    return _norm("".join(lab.itertext())) if lab is not None else ""

def extract_formulas_with_anchors(tei_path: Path, anchor_tail_words: int = 12) -> List[Dict]:
    tree = ET.parse(str(tei_path))
    root = tree.getroot()
    body = root.find(".//tei:body", namespaces=NS)
    if body is None:
        return []

    last_p_text = ""
    out: List[Dict] = []

    for el in body.iter():
        tag = ET.QName(el.tag).localname

        if tag == "p":
            last_p_text = _norm("".join(el.itertext()))
            continue

        if tag == "formula":
            xml_id = (
                el.get("{http://www.w3.org/XML/1998/namespace}id")
                or el.get("xml:id")
                or el.get("id")
                or "formula_unknown"
            )
            label = _extract_label(el)

            raw_formula = _norm("".join(el.itertext()))
            # remove label text from formula text if it appears
            if label and label in raw_formula:
                raw_formula = _norm(raw_formula.replace(label, ""))

            anchor_tail = _tail_words(last_p_text, n=anchor_tail_words)

            # optional: skip tiny junk formulas
            if len(raw_formula) < 10:
                continue

            out.append({
                "xml_id": xml_id,
                "label": label,               # keep "(1)"
                "formula_text": raw_formula,  # textual content
                "anchor_tail": anchor_tail,   # last words of preceding paragraph
            })

    return out

def _parse_sentence_lines(sentence_txt: str) -> List[Dict]:
    lines = [l.rstrip("\n") for l in sentence_txt.splitlines() if l.strip()]
    items = []
    pat = re.compile(r"^SENTENCE\s+(\d+)\s+([^:]*?)\s*:::\s*(.*?)\s*////\s*$")
    for line in lines:
        m = pat.match(line)
        if not m:
            continue
        items.append({
            "num": int(m.group(1)),
            "meta": m.group(2).strip(),
            "text": m.group(3).strip(),
        })
    return items

def _best_match_index(items: List[Dict], anchor_tail: str) -> Optional[int]:
    if not anchor_tail:
        return None
    low_anchor = anchor_tail.lower()

    # 1) direct containment
    for i, it in enumerate(items):
        if low_anchor in it["text"].lower():
            return i

    # 2) fuzzy match against sentence tails
    best_i = None
    best_score = 0.0
    for i, it in enumerate(items):
        tail = _tail_words(it["text"], n=14)
        if not tail:
            continue
        score = SequenceMatcher(None, low_anchor, tail.lower()).ratio()
        if score > best_score:
            best_score = score
            best_i = i

    return best_i if best_score >= 0.55 else None

def insert_formulas_and_write_context(
    sentence_file: Path,
    tei_path: Path,
    output_sentence_file: Path,
    output_formula_context_file: Path,
    *,
    anchor_tail_words: int = 12,
    context_window: int = 2,
) -> None:
    """
    - Reads sentence_file (TEI sentences)
    - Extracts <formula> blocks from tei_path
    - Inserts formulas after matched sentence
    - Writes:
        1) output_sentence_file (renumbered stream with FORMULA lines)
        2) output_formula_context_file (each formula + 2 sentences before/after)
    """
    sentence_txt = sentence_file.read_text(encoding="utf-8")
    items = _parse_sentence_lines(sentence_txt)

    formulas = extract_formulas_with_anchors(tei_path, anchor_tail_words=anchor_tail_words)

    # index -> list of formulas
    inserts: Dict[int, List[Dict]] = {}
    for f in formulas:
        idx = _best_match_index(items, f["anchor_tail"])
        if idx is None:
            idx = len(items) - 1 if items else 0
        inserts.setdefault(idx, []).append(f)

    # Build combined stream (with tags so we can write context easily)
    combined = []  # each entry: {"type": "sent"/"formula", ...}
    for i, it in enumerate(items):
        combined.append({"type": "sent", "meta": it["meta"], "text": it["text"]})

        for f in inserts.get(i, []):
            label = f["label"]
            xml_id = f["xml_id"]
            formula_text = f["formula_text"]

            prefix = f"Formula {label}: " if label else "Formula: "
            combined.append({
                "type": "formula",
                "meta": f"FORMULA;{xml_id}",
                "text": _norm(prefix + formula_text),
                "xml_id": xml_id,
                "label": label,
            })

    # 1) Write updated sentence stream (renumbered)
    out_lines = []
    for k, entry in enumerate(combined, start=1):
        out_lines.append(f"SENTENCE {k:03d} {entry['meta']} ::: {entry['text']} ////")
    output_sentence_file.parent.mkdir(parents=True, exist_ok=True)
    output_sentence_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # 2) Write FORMULA_OF_ARTICLE.xml as TEI (formulas only + context)
    # output_formula_context_file should now be something like .../FORMULA_OF_ARTICLE.xml

    tei = ET.Element("TEI", nsmap={None: NS["tei"]})
    text_el = ET.SubElement(tei, "text")
    body_el = ET.SubElement(text_el, "body")

    # Precompute indices of sentence entries in combined
    sent_indices = [idx for idx, e in enumerate(combined) if e["type"] == "sent"]

    def nearest_sentence_pos(pos: int) -> int:
        before = [j for j, ci in enumerate(sent_indices) if ci < pos]
        return before[-1] if before else 0

    for pos, entry in enumerate(combined):
        if entry["type"] != "formula":
            continue

        s_pos = nearest_sentence_pos(pos)
        start_s = max(0, s_pos - context_window)
        end_s = min(len(sent_indices) - 1, s_pos + context_window)

        # One div per formula
        div = ET.SubElement(
            body_el,
            "div",
            attrib={
                "type": "formula_context",
                "{http://www.w3.org/XML/1998/namespace}id": entry.get("xml_id", "formula_unknown"),
            },
        )

        # context before
        for j in range(start_s, s_pos):
            ci = sent_indices[j]
            p = ET.SubElement(div, "p", attrib={"type": "context", "n": str(j - s_pos)})
            p.text = combined[ci]["text"]

        # formula element
        f_el = ET.SubElement(
            div,
            "formula",
            attrib={"{http://www.w3.org/XML/1998/namespace}id": entry.get("xml_id", "formula_unknown")},
        )
        if entry.get("label"):
            lab = ET.SubElement(f_el, "label")
            lab.text = entry["label"]
        # Store “pretty” text (without the "Formula (x):" prefix if you want)
        txt = ET.SubElement(f_el, "p", attrib={"type": "formula_text"})
        txt.text = entry["text"]

        # context after
        for j in range(s_pos + 1, end_s + 1):
            ci = sent_indices[j]
            p = ET.SubElement(div, "p", attrib={"type": "context", "n": str(j - s_pos)})
            p.text = combined[ci]["text"]

    output_formula_context_file.parent.mkdir(parents=True, exist_ok=True)
    output_formula_context_file.write_text(
        ET.tostring(tei, pretty_print=True, encoding="unicode"),
        encoding="utf-8",
    )
# ============================================================