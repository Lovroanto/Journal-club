# sentence_alignment.py — FINAL VERSION — SHOWS METADATA FROM BOTH SIDES
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import List, Dict


@dataclass
class Sentence:
    num: int                  # sentence number
    is_clean: bool            # True = clean file, False = GROBID file
    metadata: str             # ";FIG001", "DIV08PARA01", or empty
    text: str                 # actual sentence text


def parse_sentences(file_path: Path, is_clean: bool) -> List[Sentence]:
    """Parse both clean.txt and tei_corrected.txt — handles ALL your formats."""
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("////", "|||END|||")
    content = re.sub(r'\s+', ' ', content)
    content = content.replace("|||END|||", " ////")

    sentences = []
    parts = re.split(r'( ////)', content)
    i = 0
    while i < len(parts) - 1:
        block = parts[i].strip()
        if not block:
            i += 2
            continue

        match = re.search(r'SENTENCE\s*(\d+)\s*(.*?)\s*:::', block)
        if not match:
            i += 2
            continue

        num = int(match.group(1))
        metadata = match.group(2).strip()
        text = block[match.end():].strip()

        # Collect multi-line continuation
        i += 2
        while i < len(parts) and parts[i].strip() and not parts[i].strip().startswith("SENTENCE"):
            text += " " + parts[i].strip()
            i += 2

        sentences.append(Sentence(
            num=num,
            is_clean=is_clean,
            metadata=metadata,
            text=re.sub(r'\s+', ' ', text).strip()
        ))
    return sentences


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def align_clean_to_grobid(
    clean_txt_path: Path,
    grobid_txt_path: Path,
    min_score_to_accept: float = 0.25
) -> List[Dict]:
    """Main function — returns results with clean metadata included."""
    clean_sents = parse_sentences(clean_txt_path, is_clean=True)
    grobid_sents = parse_sentences(grobid_txt_path, is_clean=False)

    clean_sents.sort(key=lambda x: x.num)
    grobid_sents.sort(key=lambda x: x.num)

    results = []
    for clean in clean_sents:
        best_score = 0.0
        best_match = None

        for grobid in grobid_sents:
            score = similarity(clean.text, grobid.text)
            if score > best_score:
                best_score = score
                best_match = grobid

        # Always include clean metadata
        clean_meta = clean.metadata or "(none)"

        if best_score >= min_score_to_accept and best_match:
            results.append({
                "clean_num": clean.num,
                "clean_meta": clean_meta,                      # ← NOW INCLUDED
                "clean_text": clean.text,
                "grobid_num": best_match.num,
                "grobid_provenance": best_match.metadata or "(unknown)",
                "grobid_text": best_match.text,
                "score": round(best_score, 4),
                "status": "match"
            })
        else:
            results.append({
                "clean_num": clean.num,
                "clean_meta": clean_meta,                      # ← AND HERE
                "clean_text": clean.text,
                "grobid_num": None,
                "grobid_provenance": None,
                "grobid_text": None,
                "score": round(best_score, 4),
                "status": "not_found"
            })
    return results


def save_alignment_report(
    alignment_results: List[Dict],
    output_path: Path,
    threshold_used: float
) -> None:
    """Shows metadata from BOTH clean and GROBID — exactly what you wanted."""
    sorted_res = sorted(alignment_results, key=lambda x: x["clean_num"])
    lines = [
        "CLEAN → GROBID ALIGNMENT — FINAL VERSION (BOTH METADATA SHOWN)",
        f"Total clean sentences found: {len(sorted_res)}",
        f"Threshold used: {threshold_used:.1%}",
        "="*150,
    ]

    for r in sorted_res:
        clean_num = r["clean_num"]
        clean_meta = r["clean_meta"]
        clean_meta_str = f" {clean_meta}" if clean_meta and clean_meta != "(none)" else ""

        if r["status"] == "not_found":
            lines.append(
                f"CLEAN {clean_num:03d}{clean_meta_str} → NOT FOUND (best score {r['score']:.3f})"
            )
        else:
            grobid_meta = r["grobid_provenance"]
            strength = "STRONG" if r['score'] >= 0.75 else "WEAK" if r['score'] >= 0.4 else "VERY WEAK"
            lines.append(
                f"CLEAN {clean_num:03d}{clean_meta_str} → "
                f"GROBID {r['grobid_num']:03d} {grobid_meta} "
                f"(score: {r['score']:.3f} ← {strength})"
            )

    lines += [
        "="*150,
        f"Processed {len(sorted_res)} clean sentences — ALL FIGURES + METADATA INCLUDED",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final alignment report saved → {output_path.resolve()}")


# Optional: test directly
if __name__ == "__main__":
    align_clean_to_grobid(
        clean_txt_path=Path("clean.txt"),
        grobid_txt_path=Path("grobid.txt"),
        min_score_to_accept=0.25
    )