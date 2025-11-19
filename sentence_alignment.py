# sentence_alignment.py — FINAL VERSION — WORKS WITH MULTI-LINE SENTENCES
from pathlib import Path
from difflib import SequenceMatcher
import re
from typing import List, Dict

def _load_multiline_sentences(file_path: Path) -> List[dict]:
    """
    Robustly reads files where sentences can span multiple lines.
    Example:
        SENTENCE 003 ::: However, investigations of these phenomena
        typically occur in a discontinuous manner due to the need to reload atomic
        ensembles. ////
    → is correctly parsed as one sentence.
    """
    content = file_path.read_text(encoding="utf-8")

    # Replace all line breaks with space, but keep //// as delimiter
    # We temporarily protect //// so they don't get eaten
    content = content.replace("////", "END_SENTENCE")  # unique marker
    content = re.sub(r'\s+', ' ', content)  # all whitespace → single space
    content = content.replace("END_SENTENCE", "////")

    sentences = []
    parts = re.split(r'(////)', content)  # split on //// but keep delimiter

    current_text = ""
    current_prefix = None

    for i in range(0, len(parts) - 1, 2):
        block = parts[i].strip()
        delimiter = parts[i + 1]  # should be "////"

        if not block and not current_text:
            continue

        # Look for SENTENCE header in this block
        header_match = re.search(r'SENTENCE\s+(\d+)(?:\s+.*)?:::', block)
        if header_match:
            # Save previous sentence if exists
            if current_text.strip() and current_prefix is not None:
                num = int(current_prefix)
                text = re.sub(r'\s+', ' ', current_text.strip())
                sentences.append({"num": num, "text": text, "provenance": ""})
            # Start new sentence
            current_prefix = header_match.group(1)
            current_text = block[header_match.end():].strip()
        else:
            # Continuation of previous sentence
            if current_text:
                current_text += " " + block

        if delimiter == "////" and current_text.strip() and current_prefix is not None:
            num = int(current_prefix)
            provenance_match = re.search(r'SENTENCE\s+\d+\s+(.*?):::', parts[i])
            provenance = provenance_match.group(1).strip() if provenance_match else ""
            text = re.sub(r'\s+', ' ', current_text.strip())
            sentences.append({
                "num": num,
                "text": text,
                "provenance": provenance
            })
            current_text = ""
            current_prefix = None

    return sentences

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def align_clean_to_grobid(
    clean_txt_path: Path,
    grobid_txt_path: Path,
    min_score_to_accept: float = 0.25
) -> List[Dict]:
    clean_sents = _load_multiline_sentences(clean_txt_path)
    grobid_sents = _load_multiline_sentences(grobid_txt_path)

    # Sort by number
    clean_sents.sort(key=lambda x: x["num"])
    grobid_sents.sort(key=lambda x: x["num"])

    results = []

    for clean in clean_sents:
        best_score = 0.0
        best_g = None

        for g in grobid_sents:
            score = similarity(clean["text"], g["text"])
            if score > best_score:
                best_score = score
                best_g = g

        if best_score >= min_score_to_accept and best_g:
            result = {
                "clean_num": clean["num"],
                "clean_text": clean["text"],
                "grobid_num": best_g["num"],
                "grobid_provenance": best_g["provenance"] or "(unknown)",
                "grobid_text": best_g["text"],
                "score": round(best_score, 4),
                "status": "match"
            }
        else:
            result = {
                "clean_num": clean["num"],
                "clean_text": clean["text"],
                "grobid_num": None,
                "grobid_provenance": None,
                "grobid_text": None,
                "score": round(best_score, 4),
                "status": "not_found"
            }
        results.append(result)

    return results

def save_alignment_report(alignment_results: List[Dict], output_path: Path, threshold_used: float) -> None:
    sorted_res = sorted(alignment_results, key=lambda x: x["clean_num"])

    lines = [
        "CLEAN → GROBID ALIGNMENT — MULTI-LINE SUPPORT FIXED",
        f"Total clean sentences found: {len(sorted_res)}",
        f"Threshold used: {threshold_used:.1%}",
        "="*120,
    ]

    for r in sorted_res:
        if r["status"] == "not_found":
            lines.append(f"CLEAN {r['clean_num']:03d} → NOT FOUND (best={r['score']:.3f})")
        else:
            prov = r["grobid_provenance"]
            strength = "STRONG" if r['score'] >= 0.75 else "WEAK" if r['score'] >= 0.4 else "VERY WEAK"
            lines.append(
                f"CLEAN {r['clean_num']:03d} → GROBID {r['grobid_num']:03d} {prov}  "
                f"(score: {r['score']:.3f} ← {strength})"
            )

    lines += [
        "="*120,
        f"Final count: {len(sorted_res)} clean sentences processed → NO SKIPS",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Multi-line robust alignment saved → {output_path}")