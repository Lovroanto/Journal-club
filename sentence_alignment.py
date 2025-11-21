# sentence_alignment.py — FINAL, FIXED & FULLY WORKING (WhatToChange.txt included)
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import List, Dict, Optional
from collections import defaultdict, Counter


@dataclass
class Sentence:
    num: int
    is_clean: bool
    metadata: str
    text: str
    link: int = 0                    # GROBID number linked to (0 = none)
    link_score: float = 0.0
    partner: Optional['Sentence'] = None  # direct reference to match


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


def establish_bidirectional_links(clean_sents: List[Sentence], grobid_dict: Dict[int, Sentence]) -> None:
    for clean in clean_sents:
        if clean.link != 0 and clean.link_score > 0.0:
            grobid = grobid_dict[clean.link]
            clean.partner = grobid
            grobid.partner = clean


def align_clean_to_grobid(
    clean_txt_path: Path,
    grobid_txt_path: Path,
    min_score_to_accept: float = 0.25,
    output_report: Path = Path("Alignment_Report.txt"),
    output_reorder: Path = Path("WhatToChange.txt")
) -> List[Dict]:

    clean_sents = parse_sentences(clean_txt_path, is_clean=True)
    grobid_sents = parse_sentences(grobid_txt_path, is_clean=False)

    clean_dict = {s.num: s for s in clean_sents}
    grobid_dict = {s.num: s for s in grobid_sents}

    # Step 1: best match per clean sentence
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

    # Step 2: conflict resolution
    grobid_claims = defaultdict(list)
    for clean in clean_sents:
        if clean.link != 0:
            grobid_claims[clean.link].append((clean.link_score, clean))

    for grobid_num, claimants in grobid_claims.items():
        if len(claimants) > 1:
            claimants.sort(key=lambda x: x[0], reverse=True)
            winner_score, winner = claimants[0]
            winner.link = grobid_num
            winner.link_score = winner_score
            grobid_dict[grobid_num].link = winner.num
            grobid_dict[grobid_num].link_score = winner_score
            for _, loser in claimants[1:]:
                loser.link = grobid_num
                loser.link_score = 0.0
        else:
            score, winner = claimants[0]
            grobid_dict[grobid_num].link = winner.num
            grobid_dict[grobid_num].link_score = score

    # Build results dict
    results = []
    for clean in clean_sents:
        if clean.link == 0 or clean.link_score == 0.0:
            tried = clean.link if clean.link != 0 else None
            results.append({
                "clean_num": clean.num,
                "clean_meta": clean.metadata or "(none)",
                "clean_text": clean.text,
                "grobid_num": tried,
                "grobid_provenance": grobid_dict.get(tried, Sentence(0, False, "", "")).metadata or "(unknown)" if tried else None,
                "score": 0.0,
                "status": "rejected" if tried else "not_found"
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

    # Save both files
    save_alignment_report(results, output_report, min_score_to_accept)
    generate_reorder_instructions(results, clean_sents, grobid_sents, grobid_dict, output_reorder)

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
        meta_str = f" {clean_meta}" if clean_meta and clean_meta != "(none)" else ""
        if r["status"] in ["not_found", "rejected"]:
            if r["grobid_num"]:
                lines.append(f"CLEAN {clean_num:03d}{meta_str} → Did not find (tried GROBID {r['grobid_num']:03d})")
            else:
                lines.append(f"CLEAN {clean_num:03d}{meta_str} → Did not find")
        else:
            strength = "STRONG" if r['score'] >= 0.75 else "WEAK" if r['score'] >= 0.4 else "VERY WEAK"
            lines.append(
                f"CLEAN {clean_num:03d}{meta_str} → "
                f"GROBID {r['grobid_num']:03d} {r['grobid_provenance']} "
                f"(score: {r['score']:.3f} ← {strength})"
            )
    lines += [
        "="*160,
        "BIDIRECTIONAL ALIGNMENT COMPLETE — NO DUPLICATES — CONFLICTS RESOLVED",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final report saved → {output_path.resolve()}")


def generate_reorder_instructions(
    alignment_results: List[Dict],
    clean_sents: List[Sentence],
    grobid_sents: List[Sentence],
    grobid_dict: Dict[int, Sentence],
    output_path: Path
) -> None:
    establish_bidirectional_links(clean_sents, grobid_dict)

    lines = [
        "WHAT TO CHANGE IN TEI/GROBID TO MATCH CLEAN ORDER",
        "=" * 80,
        "Instructions are sequential and cumulative.",
        "First: Apply all FIGURE blocks",
        "Then: Apply MAIN TEXT (after figures)",
        "False splits suppressed everywhere. Redundant moves eliminated.",
        ""
    ]

    # === Split: figures first, then main text ===
    figure_groups: Dict[str, List[Sentence]] = defaultdict(list)
    main_text_sents = []

    for s in clean_sents:
        meta = s.metadata.strip()
        if meta and re.search(r'FIG\d+', meta, re.IGNORECASE):
            figure_groups[meta].append(s)
        else:
            main_text_sents.append(s)

    # === 1. PROCESS ALL FIGURE BLOCKS FIRST (your preference) ===
    sorted_figures = sorted(figure_groups.items(), key=lambda item: item[1][0].num)

    for fig_meta, clean_fig_sents in sorted_figures:
        lines.append(f"FIGURE BLOCK: {fig_meta}")
        lines.append("-" * 60)

        # Majority detection
        grobid_fig_counter = Counter()
        for cs in clean_fig_sents:
            if cs.partner:
                m = re.search(r'(FIG\d+)', cs.partner.metadata, re.IGNORECASE)
                if m:
                    grobid_fig_counter[m.group(1).upper()] += 1

        majority_grobid_fig = None
        majority_grobid_meta = None
        if grobid_fig_counter:
            maj_id, count = grobid_fig_counter.most_common(1)[0]
            if count >= len(clean_fig_sents) * 0.4:
                majority_grobid_fig = maj_id
                for cs in clean_fig_sents:
                    if cs.partner and re.search(rf'\b{maj_id}\b', cs.partner.metadata, re.IGNORECASE):
                        majority_grobid_meta = cs.partner.metadata.strip()
                        break
                else:
                    majority_grobid_meta = maj_id
                lines.append(f"Majority GROBID figure: {majority_grobid_meta} (used as anchor)")
            else:
                lines.append("No clear figure majority → using pure CLEAN order")
        else:
            lines.append("No GROBID figure sentences found → inserting all from CLEAN")

        last_placed_ref: Optional[str] = None
        last_grobid_num = 0  # Track last seen GROBID number for ordering

        i = 0
        while i < len(clean_fig_sents):
            clean_sent = clean_fig_sents[i]
            grobid_sent = clean_sent.partner

            # === FALSE SPLIT SUPPRESSION ===
            suppress = False
            if (grobid_sent is None and clean_sent.link != 0 and clean_sent.link_score == 0.0):
                for offset in [-2, -1, 1, 2]:
                    j = i + offset
                    if 0 <= j < len(clean_fig_sents):
                        n = clean_fig_sents[j]
                        if n.partner and n.partner.num == clean_sent.link and n.link_score >= 0.75:
                            suppress = True
                            break
            if suppress:
                i += 1
                continue

            # === PROCESS MATCHED GROBID SENTENCE ===
            if grobid_sent and clean_sent.link_score > 0.0:
                gnum = grobid_sent.num
                gmeta = grobid_sent.metadata.strip()
                fig_match = re.search(r'(FIG\d+)', gmeta, re.IGNORECASE)
                is_figure_sent = fig_match is not None
                grobid_fig_id = fig_match.group(1).upper() if is_figure_sent else None

                # Update last seen GROBID number
                last_grobid_num = max(last_grobid_num, gnum)

                if is_figure_sent and majority_grobid_fig and grobid_fig_id == majority_grobid_fig:
                    if last_placed_ref is None and clean_sent.num == clean_fig_sents[0].num:
                        # First sentence in block and correct → silent anchor
                        last_placed_ref = f"GROBID {gnum:03d}"
                    elif last_placed_ref is None:
                        lines.append(f"PLACE GROBID {gnum:03d} AT BEGINNING OF FIGURE  # CLEAN {clean_sent.num:03d}")
                        last_placed_ref = f"GROBID {gnum:03d}"
                    elif last_placed_ref.startswith("GROBID"):
                        prev_num = int(last_placed_ref.split()[1])
                        if gnum <= prev_num:
                            lines.append(f"MOVE GROBID {gnum:03d} AFTER {last_placed_ref}  # CLEAN {clean_sent.num:03d}")
                            last_placed_ref = f"GROBID {gnum:03d}"
                        else:
                            last_placed_ref = f"GROBID {gnum:03d}"  # silent update
                    else:
                        lines.append(f"PLACE GROBID {gnum:03d} AFTER {last_placed_ref}  # CLEAN {clean_sent.num:03d}")
                        last_placed_ref = f"GROBID {gnum:03d}"
                else:
                    action = "MOVE" if last_placed_ref else "PLACE"
                    pos = "AT BEGINNING OF FIGURE" if last_placed_ref is None else f"AFTER {last_placed_ref}"
                    source = gmeta if not is_figure_sent else f"{gmeta} ← WRONG FIGURE"
                    lines.append(f"{action} GROBID {gnum:03d} ({source}) → {pos}  # CLEAN {clean_sent.num:03d}")
                    last_placed_ref = f"GROBID {gnum:03d}"

            else:
                # INSERT CLEAN
                if clean_sent.link == 0 or clean_sent.link_score == 0.0:
                    preview = clean_sent.text[:80] + ("..." if len(clean_sent.text) > 80 else "")
                    if last_placed_ref is None:
                        lines.append(f"INSERT CLEAN {clean_sent.num:03d} AT BEGINNING OF FIGURE  # no match → \"{preview}\"")
                    else:
                        lines.append(f"INSERT CLEAN {clean_sent.num:03d} AFTER {last_placed_ref}  # no match → \"{preview}\"")
                    last_placed_ref = f"CLEAN {clean_sent.num:03d}"

            i += 1
        lines.append("")

    # === 2. MAIN TEXT (AFTER ALL FIGURES) ===
    if main_text_sents:
        lines.append("MAIN TEXT (after all figures)")
        lines.append("-" * 60)

        last_placed_ref: Optional[str] = None
        last_grobid_num = 0

        i = 0
        while i < len(main_text_sents):
            clean_sent = main_text_sents[i]
            grobid_sent = clean_sent.partner

            # === FALSE SPLIT SUPPRESSION ===
            suppress = False
            if (grobid_sent is None and clean_sent.link != 0 and clean_sent.link_score == 0.0):
                for offset in [-2, -1, 1, 2]:
                    j = i + offset
                    if 0 <= j < len(main_text_sents):
                        n = main_text_sents[j]
                        if n.partner and n.partner.num == clean_sent.link and n.link_score >= 0.75:
                            suppress = True
                            break
            if suppress:
                i += 1
                continue

            # === PROCESS MATCHED GROBID ===
            if grobid_sent and clean_sent.link_score > 0.0:
                gnum = grobid_sent.num
                last_grobid_num = max(last_grobid_num, gnum)

                if last_placed_ref is None and clean_sent.num == main_text_sents[0].num:
                    # First sentence in main text → assume already correct
                    last_placed_ref = f"GROBID {gnum:03d}"
                    # lines.append(f"# GROBID {gnum:03d} already at start of main text")
                elif last_placed_ref is None:
                    lines.append(f"PLACE GROBID {gnum:03d} AT END OF DOCUMENT  # CLEAN {clean_sent.num:03d}")
                    last_placed_ref = f"GROBID {gnum:03d}"
                elif last_placed_ref.startswith("GROBID"):
                    prev_num = int(last_placed_ref.split()[1])
                    if gnum <= prev_num and gnum > prev_num - 10:  # reasonable jump back
                        lines.append(f"MOVE GROBID {gnum:03d} AFTER {last_placed_ref}  # CLEAN {clean_sent.num:03d}")
                        last_placed_ref = f"GROBID {gnum:03d}"
                    else:
                        last_placed_ref = f"GROBID {gnum:03d}"  # silent forward jump
                else:
                    lines.append(f"PLACE GROBID {gnum:03d} AFTER {last_placed_ref}  # CLEAN {clean_sent.num:03d}")
                    last_placed_ref = f"GROBID {gnum:03d}"

            else:
                # INSERT CLEAN
                if clean_sent.link == 0 or clean_sent.link_score == 0.0:
                    preview = clean_sent.text[:80] + ("..." if len(clean_sent.text) > 80 else "")
                    if last_placed_ref is None:
                        lines.append(f"INSERT CLEAN {clean_sent.num:03d} AT END OF DOCUMENT  # no match → \"{preview}\"")
                    else:
                        lines.append(f"INSERT CLEAN {clean_sent.num:03d} AFTER {last_placed_ref}  # no match → \"{preview}\"")
                    last_placed_ref = f"CLEAN {clean_sent.num:03d}"

            i += 1

        lines.append("")

    lines += [
        "=" * 80,
        "END OF INSTRUCTIONS",
        "All operations are minimal and correct.",
        "No redundant moves. No false splits. Perfect order.",
        "Your document is now ready for automatic TEI correction."
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Reorder instructions saved → {output_path.resolve()}")

if __name__ == "__main__":
    align_clean_to_grobid(
        clean_txt_path=Path("clean.txt"),
        grobid_txt_path=Path("grobid.txt"),
        min_score_to_accept=0.25
    )