# sentence_alignment.py
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import List, Tuple, Dict, Optional

@dataclass
class Sentence:
    num: int
    provenance: str   # e.g. "DIV01PARA04" or empty for clean
    text: str

def _parse_sentences(file_path: Path) -> List[Sentence]:
    """Internal: parse your exact format (SENTENCE xxx [DIV..PARA..] ::: text ////)"""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    sentences = []
    for line in lines:
        line = line.strip()
        if not line or "////" not in line or ":::" not in line:
            continue

        prefix, rest = line.split(":::", 1)
        text = rest.split("////")[0].strip()
        text = re.sub(r'\s+', ' ', text)  # normalize whitespace

        parts = prefix.strip().split()
        if len(parts) < 2 or parts[0] != "SENTENCE":
            continue
        try:
            num = int(parts[1])
        except ValueError:
            continue

        provenance = " ".join(parts[2:]).strip() if len(parts) > 2 else ""

        sentences.append(Sentence(num=num, provenance=provenance, text=text))
    return sentences

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def align_clean_to_grobid(
    clean_txt_path: Path,
    grobid_txt_path: Path,
    min_score_to_accept: float = 0.35   # ← ONLY below 35% → NOT FOUND
) -> List[Dict]:
    """
    Aligns EVERY clean sentence to the SINGLE most similar GROBID sentence,
    no matter how low the score is (as long as ≥ min_score_to_accept).
    """
    clean_sents = _parse_sentences(clean_txt_path)
    grobid_sents = _parse_sentences(grobid_txt_path)

    # Sort both lists by sentence number (just in case)
    clean_sents.sort(key=lambda x: x.num)
    grobid_sents.sort(key=lambda x: x.num)

    results = []

    for clean in clean_sents:
        best_grobid = None
        best_score = -1.0
        best_num = None

        # Search through ALL GROBID sentences → find the absolute best match
        for g in grobid_sents:
            score = similarity(clean.text, g.text)
            if score > best_score:
                best_score = score
                best_grobid = g
                best_num = g.num

        # Final decision: accept even very low scores
        if best_score >= min_score_to_accept:
            prov = best_grobid.provenance if best_grobid.provenance else "(no location)"
            status = "match" if best_score >= 0.75 else "weak_match"
            result = {
                "clean_num": clean.num,
                "clean_text": clean.text,
                "grobid_num": best_num,
                "grobid_provenance": prov,
                "grobid_text": best_grobid.text,
                "score": round(best_score, 4),
                "status": status
            }
        else:
            result = {
                "clean_num": clean.num,
                "clean_text": clean.text,
                "grobid_num": None,
                "grobid_provenance": None,
                "grobid_text": None,
                "score": round(best_score, 4),
                "status": "not_found"
            }
        results.append(result)

    return results

# Optional: quick helper to save pretty report
def save_alignment_report(alignment_results: List[Dict], output_path: Path) -> None:
    lines = [
        "CLEAN → GROBID SENTENCE ALIGNMENT (EVERY CLEAN SENTENCE MAPPED – BEST POSSIBLE MATCH)",
        f"Total clean sentences: {len(alignment_results)}",
        f"Threshold: matches accepted down to {0.35:.0%} similarity",
        "=" * 110,
    ]

    for r in sorted(alignment_results, key=lambda x: x["clean_num"]):
        if r["status"] == "not_found":
            lines.append(
                f"CLEAN {r['clean_num']:03d} → NOT FOUND (best was {r['score']:.3f})"
            )
        else:
            prov = r["grobid_provenance"]
            score_color = "STRONG" if r['score'] >= 0.75 else "WEAK"
            lines.append(
                f"CLEAN {r['clean_num']:03d} → GROBID {r['grobid_num']:03d} {prov}  "
                f"(score: {r['score']:.3f} ← {score_color})"
            )

    matched = sum(1 for r in alignment_results if r["status"] != "not_found")
    lines += [
        "=" * 110,
        f"FINAL SUMMARY:",
        f"  Total clean sentences   : {len(alignment_results)}",
        f"  Mapped (≥35%)           : {matched}",
        f"  Truly not found (<35%)  : {len(alignment_results) - matched}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Full forced alignment report → {output_path}")