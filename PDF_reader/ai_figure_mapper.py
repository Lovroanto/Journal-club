# PDF_reader/ai_figure_mapper.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from bs4 import BeautifulSoup
from lxml import etree as ET

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NSMAP = {None: TEI_NS}

def extract_figures_from_tei(tei_xml: str) -> List[Dict]:
    soup = BeautifulSoup(tei_xml, "lxml-xml")
    out = []
    for i, fig in enumerate(soup.find_all("figure"), start=1):
        xmlid = fig.get("xml:id") or fig.get("id") or f"fig_{i}"
        figdesc = fig.find("figDesc")
        caption = figdesc.get_text(" ", strip=True) if figdesc else ""
        head = fig.find("head")
        title = head.get_text(" ", strip=True) if head else ""
        out.append({"xml_id": xmlid, "title": title, "caption": caption})
    return out

def write_mapping_tei(mapping: Dict[str, Any], out_path: Path):
    tei = ET.Element(ET.QName(TEI_NS, "TEI"), nsmap=NSMAP)
    text_el = ET.SubElement(tei, ET.QName(TEI_NS, "text"))
    body_el = ET.SubElement(text_el, ET.QName(TEI_NS, "body"))
    div_el = ET.SubElement(body_el, ET.QName(TEI_NS, "div"), attrib={"type": "figure_image_map"})

    for fig in mapping["figures"]:
        fdiv = ET.SubElement(
            div_el,
            ET.QName(TEI_NS, "div"),
            attrib={"type": "figure", ET.QName(XML_NS, "id"): fig["xml_id"]},
        )
        if fig.get("caption"):
            p = ET.SubElement(fdiv, ET.QName(TEI_NS, "p"), attrib={"type": "caption"})
            p.text = fig["caption"]

        if fig.get("matched_image"):
            p = ET.SubElement(fdiv, ET.QName(TEI_NS, "p"), attrib={"type": "matched_image"})
            p.text = fig["matched_image"]

        score = str(fig.get("score", ""))
        if score:
            p = ET.SubElement(fdiv, ET.QName(TEI_NS, "p"), attrib={"type": "score"})
            p.text = score

    out_path.write_text(ET.tostring(tei, pretty_print=True, encoding="unicode"), encoding="utf-8")

def export_matched_images(mapping_json: Path, images_dir: Path, out_dir: Path) -> None:
    import json
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(mapping_json.read_text(encoding="utf-8"))

    for fig in data.get("figures", []):
        matched = fig.get("matched_image")
        fig_id = fig.get("id") or fig.get("xml_id") or fig.get("n") or "FIG"
        if not matched:
            continue

        src = images_dir / matched
        if src.exists():
            # make filename safe-ish
            dst = out_dir / f"{fig_id}_{src.name}"
            shutil.copy2(src, dst)

def describe_image_with_ollama(image_path: Path, vision_model: str) -> str:
    """
    Requires local ollama with a vision model (e.g., llava).
    If you don’t have that, you can stub this out.
    """
    import ollama
    prompt = (
        "Describe this scientific figure briefly.\n"
        "Include: plot/table/schematic, axes labels if visible, key keywords."
    )
    resp = ollama.chat(
        model=vision_model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [str(image_path)]
        }],
    )
    return (resp.get("message", {}).get("content") or "").strip()

def choose_best_match_llm(
    figure_caption: str,
    image_candidates: List[Dict[str, str]],
    llm: Any,
) -> Dict[str, Any]:
    """
    image_candidates: [{ "file": "...", "desc": "..."}]
    Returns: {"file": "...", "score": 0-1, "reason": "..."} (reason optional)
    """
    # keep prompt compact
    candidates_text = "\n".join(
        [f"- {c['file']}: {c['desc'][:250]}" for c in image_candidates]
    )
    prompt = (
        "You match a TEI figure caption to the correct extracted PDF image.\n"
        "Return ONLY JSON: {\"file\": \"...\", \"score\": 0-1}\n\n"
        f"CAPTION:\n{figure_caption}\n\n"
        f"IMAGES:\n{candidates_text}\n"
    )
    raw = llm(prompt)
    if not isinstance(raw, str):
        raw = getattr(raw, "text", str(raw))

    # best-effort JSON parse
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        js = json.loads(raw[start:end+1])
        return js if isinstance(js, dict) else {}
    except Exception:
        return {}

def run_ai_figure_mapping(
    tei_xml: str,
    images_dir: Path,
    out_json: Path,
    out_tei: Optional[Path] = None,
    llm: Any = None,
    vision_model: str = "llava:latest",
    max_images: int = 80,
) -> Dict[str, Any]:
    figures = extract_figures_from_tei(tei_xml)

    image_files = sorted([p for p in images_dir.glob("*") if p.is_file()])[:max_images]

    # 1) describe all images (cacheable later)
    candidates = []
    for img in image_files:
        desc = describe_image_with_ollama(img, vision_model=vision_model)
        candidates.append({"file": img.name, "desc": desc})

    # 2) match each figure caption to best image
    results = {"figures": []}
    for fig in figures:
        if not fig["caption"]:
            results["figures"].append({**fig, "matched_image": None, "score": 0.0})
            continue
        if llm is None:
            # no matching without LLM
            results["figures"].append({**fig, "matched_image": None, "score": 0.0})
            continue

        pick = choose_best_match_llm(fig["caption"], candidates, llm=llm)
        matched = pick.get("file")
        score = pick.get("score", 0.0)

        results["figures"].append({**fig, "matched_image": matched, "score": score})

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if out_tei is not None:
        write_mapping_tei(results, out_tei)

    return results
