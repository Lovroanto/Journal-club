# sentence_alignment.py — FINAL VERSION — BIDIRECTIONAL + CONFLICT RESOLVED (FIXED INDENTATION)
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import List, Dict
from collections import defaultdict


@dataclass
class Sentence:
    num: int
    is_clean: bool
    metadata: str
    text: str
    link: int = 0          # GROBID number it's linked to (0 = not linked)
    link_score: float = 0.0  # score of the link (0.0 = rejected or not found)


def parse_sentences(file_path: Path, is_clean: bool) -> List[Sentence]:
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

        match = re.search(r'SENTENCE\s*(\d+)\s*([^:]*?)\s*:::', block)
        if not match:
            i += 2
            continue

        num = int(match.group(1))
        raw_metadata = match.group(2).strip()
        metadata = raw_metadata if raw_metadata and not raw_metadata.startswith('#') else ""
        text = block[match.end():].strip()

        i += 2
        while i < len(parts) and parts[i].strip() and not re.match(r'SENTENCE\s*\d+', parts[i].strip()):
            text += " " + parts[i].strip()
            i += 2

        text = re.sub(r'^#\s*Original label:[^|]*\|?\s*', '', text).strip()

        sentences.append(Sentence(
            num=num,
            is_clean=is_clean,
            metadata=metadata,
            text=re.sub(r'\s+', ' ', text).strip(),
            link=0,
            link_score=0.0
        ))
    return sentences


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def align_clean_to_grobid(
    clean_txt_path: Path,
    grobid_txt_path: Path,
    min_score_to_accept: float = 0.25
) -> List[Dict]:
    clean_sents = parse_sentences(clean_txt_path, is_clean=True)
    grobid_sents = parse_sentences(grobid_txt_path, is_clean=False)

    clean_dict = {s.num: s for s in clean_sents}
    grobid_dict = {s.num: s for s in grobid_sents}

    # Step 1: Each clean finds its best GROBID
    for clean in clean_sents:
        best_score = 0.0
        best_grobid = None
        for grobid in grobid_sents:
            score = similarity(clean.text, grobid.text)
            if score > best_score:
                best_score = score
                best_grobid = grobid

        if best_score >= min_score_to_accept and best_grobid:
            clean.link = best_grobid.num
            clean.link_score = round(best_score, 4)

    # Step 2: Resolve conflicts — only the best clean wins each GROBID
    grobid_claims = defaultdict(list)  # grobid_num → list of (score, clean_sentence)

    for clean in clean_sents:
        if clean.link != 0:
            grobid_claims[clean.link].append((clean.link_score, clean))

    for grobid_num, claimants in grobid_claims.items():
        if len(claimants) > 1:
            claimants.sort(key=lambda x: x[0], reverse=True)
            winner_score, winner = claimants[0]
            # Winner keeps the link
            winner.link = grobid_num
            winner.link_score = winner_score

            # Losers: keep the grobid num, but score = 0.0 → means "tried but lost"
            for score, loser in claimants[1:]:
                loser.link = grobid_num
                loser.link_score = 0.0

            # Set bidirectional link for GROBID
            grobid_dict[grobid_num].link = winner.num
            grobid_dict[grobid_num].link_score = winner_score
        else:
            score, winner = claimants[0]
            grobid_dict[grobid_num].link = winner.num
            grobid_dict[grobid_num].link_score = score

    # Step 3: Build final results
    results = []
    for clean in clean_sents:
        if clean.link == 0 or clean.link_score == 0.0:
            tried_grobid = clean.link if clean.link != 0 else None
            results.append({
                "clean_num": clean.num,
                "clean_meta": clean.metadata or "(none)",
                "clean_text": clean.text,
                "grobid_num": tried_grobid,
                "grobid_provenance": grobid_dict[tried_grobid].metadata or "(unknown)" if tried_grobid else None,
                "score": 0.0,
                "status": "rejected" if tried_grobid else "not_found"
            })
        else:
            results.append({
                "clean_num": clean.num,
                "clean_meta": clean.metadata or "(none)",
                "clean_text": clean.text,
                "grobid_num": clean.link,
                "grobid_provenance": grobid_dict[clean.link].metadata or "(unknown)",
                "grobid_text": grobid_dict[clean.link].text,
                "score": clean.link_score,
                "status": "match"
            })
    return results


def save_alignment_report(
    alignment_results: List[Dict],
    output_path: Path,
    threshold_used: float
) -> None:
    sorted_res = sorted(alignment_results, key=lambda x: x["clean_num"])
    lines = [
        "CLEAN → GROBID BIDIRECTIONAL ALIGNMENT — CONFLICT-RESOLVED",
        f"Total clean sentences: {len(sorted_res)}",
        f"Threshold used: {threshold_used:.1%}",
        "="*160,
    ]

    for r in sorted_res:
        clean_num = r["clean_num"]
        clean_meta = r["clean_meta"]
        clean_meta_str = f" {clean_meta}" if clean_meta and clean_meta != "(none)" else ""

        if r["status"] in ["not_found", "rejected"]:
            if r["grobid_num"]:
                lines.append(f"CLEAN {clean_num:03d}{clean_meta_str} → Did not find (tried GROBID {r['grobid_num']:03d})")
            else:
                lines.append(f"CLEAN {clean_num:03d}{clean_meta_str} → Did not find")
        else:
            strength = "STRONG" if r['score'] >= 0.75 else "WEAK" if r['score'] >= 0.4 else "VERY WEAK"
            lines.append(
                f"CLEAN {clean_num:03d}{clean_meta_str} → "
                f"GROBID {r['grobid_num']:03d} {r['grobid_provenance']} "
                f"(score: {r['score']:.3f} ← {strength})"
            )

    lines += [
        "="*160,
        "BIDIRECTIONAL ALIGNMENT COMPLETE — NO DUPLICATES — CONFLICTS RESOLVED",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final report saved → {output_path.resolve()}")


if __name__ == "__main__":
    align_clean_to_grobid(
        clean_txt_path=Path("clean.txt"),
        grobid_txt_path=Path("grobid.txt"),
        min_score_to_accept=0.25
    )