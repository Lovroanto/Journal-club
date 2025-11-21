#!/usr/bin/env python3
"""
apply_tei_changes.py

Apply the edits described in WhatToChange.txt to a TEI XML file,
using the extracted GROBID sentences (tei_corrected.txt) and
the cleaned sentences (sentences_corrected.txt).

Behavior:
- MOVE / PLACE operations move a GROBID sentence from its original location
  to a new location (figure figDesc or after another sentence).
- INSERT CLEAN operations insert a cleaned sentence (from sentences_corrected.txt)
  after the referenced sentence (GROBID or previously-inserted CLEAN).
- When a sentence is moved, its original text is removed from the original <p>.
- Inserted sentences are put into their own <p> elements.
"""

import re
import argparse
from pathlib import Path
from collections import OrderedDict, defaultdict
from bs4 import BeautifulSoup, Tag, NavigableString
import copy
import sys

# ---------------------------
# Helpers: parsing sentences
# ---------------------------

def normalize_space(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def parse_tei_extracted_sentences(tei_txt_path: Path):
    """
    Parse lines like:
      SENTENCE 231 DIV08PARA01 ::: The lasing is not continuous ... ////
      SENTENCE236;FIG001 ::: Characterization ... ////
    Returns ordered list and mapping:
      ordered_ids -> list of sentence ids (in file order)
      sents[id] = {"id": 231, "is_fig": True/False, "fig_label": "FIG001" or None,
                    "div": "DIV08" or None, "para": "PARA01" or None,
                    "text": "...", "raw": original_line}
    """
    text = tei_txt_path.read_text(encoding='utf-8')
    # Ensure consistent separators
    # We'll split on the 'SENTENCE' marker but preserve the first one
    parts = re.split(r'(?=SENTENCE\s*\d|SENTENCE\s*\d+;)', text)
    sents = OrderedDict()
    ordered = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Attempt to match patterns:
        # 1) SENTENCE 231 DIV08PARA01 ::: text ////
        # 2) SENTENCE236;FIG001 ::: text ////
        m1 = re.match(r'^SENTENCE\s*(\d+)\s+([A-Z0-9;_,-]+)?\s*:::\s*(.*?)\/\/\/\/\s*$', p, re.S)
        if not m1:
            # try variant with semicolon for figures: SENTENCE236;FIG001 ::: ...
            m2 = re.match(r'^SENTENCE\s*(\d+)\s*;\s*(FIG\d+)\s*:::\s*(.*?)\/\/\/\/\s*$', p, re.S | re.I)
            if m2:
                sid = int(m2.group(1))
                fig = m2.group(2).upper()
                txt = normalize_space(m2.group(3))
                sents[sid] = {
                    "id": sid,
                    "is_fig": True,
                    "fig_label": fig,
                    "div": None,
                    "para": None,
                    "text": txt,
                    "raw": p
                }
                ordered.append(sid)
                continue
            # fallback: try to split by ':::' and '////'
            m3 = re.match(r'^SENTENCE\s*(\d+).*?:::\s*(.*?)\/\/\/\/\s*$', p, re.S)
            if m3:
                sid = int(m3.group(1))
                txt = normalize_space(m3.group(2))
                # attempt to parse DIV/FIG info inside the part between id and ::: if present
                head = p[:p.find(':::')].strip()
                # look for DIVxxPARAyy
                div_m = re.search(r'(DIV\d{2,}\s*PARA\d{2,})', head, re.I)
                fig_m = re.search(r'FIG\d+', head, re.I)
                fig_label = fig_m.group(0).upper() if fig_m else None
                div_label = None
                para_label = None
                if div_m:
                    # may be like DIV08PARA01 or DIV08 PARA01
                    dv = div_m.group(1).replace(' ', '')
                    # split DIV08PARA01 into DIV08 and PARA01
                    dv_match = re.match(r'(DIV\d+)(?:PARA(\d+))?', dv, re.I)
                    if dv_match:
                        div_label = dv_match.group(1).upper()
                        if dv_match.group(2):
                            para_label = f"PARA{int(dv_match.group(2)):02d}"
                sents[sid] = {
                    "id": sid,
                    "is_fig": bool(fig_label),
                    "fig_label": fig_label,
                    "div": div_label,
                    "para": para_label,
                    "text": txt,
                    "raw": p
                }
                ordered.append(sid)
                continue
            # if still not parsed, skip but warn
            print(f"[WARN] Could not parse TEI-extracted sentence line (skipped):\n  {p[:120]}...", file=sys.stderr)
            continue
        # m1 matched
        sid = int(m1.group(1))
        head = m1.group(2) or ""
        txt = normalize_space(m1.group(3) or "")
        fig_label = None
        div_label = None
        para_label = None
        fm = re.search(r'(FIG\d+)', head, re.I)
        if fm:
            fig_label = fm.group(1).upper()
        dv = re.search(r'(DIV\d+)', head, re.I)
        pa = re.search(r'(PARA\d+)', head, re.I)
        if dv:
            div_label = dv.group(1).upper()
        if pa:
            para_label = pa.group(1).upper()
        sents[sid] = {
            "id": sid,
            "is_fig": bool(fig_label),
            "fig_label": fig_label,
            "div": div_label,
            "para": para_label,
            "text": txt,
            "raw": p
        }
        ordered.append(sid)
    return ordered, sents

def parse_clean_sentences(clean_txt_path: Path):
    """
    Parse cleaned sentences. Accept lines like:
      SENTENCE 196 ::: text ////
    or simple "SENTENCE196 :::" variants. Return dict id->text.
    """
    text = clean_txt_path.read_text(encoding='utf-8')
    parts = re.split(r'(?=SENTENCE\s*\d)', text)
    sents = {}
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r'^SENTENCE\s*(\d+).*?:::\s*(.*?)\/\/\/\/\s*$', p, re.S)
        if not m:
            # try simpler: SENTENCE196 ::: text ////
            m2 = re.match(r'^SENTENCE\s*(\d+)\s*:::\s*(.*?)$', p, re.S)
            if m2:
                sid = int(m2.group(1))
                txt = normalize_space(m2.group(2))
                sents[sid] = txt
                continue
            # try pattern with semicolon and FIG (unlikely for clean, but safe)
            m3 = re.match(r'^SENTENCE\s*(\d+).*?:::\s*(.*?)\/\/\/\/\s*$', p, re.S)
            if m3:
                sid = int(m3.group(1))
                txt = normalize_space(m3.group(2))
                sents[sid] = txt
                continue
            print(f"[WARN] Could not parse CLEAN sentence block (skipped): {p[:120]}...", file=sys.stderr)
            continue
        sid = int(m.group(1))
        txt = normalize_space(m.group(2))
        sents[sid] = txt
    return sents

# ---------------------------
# Parse instructions
# ---------------------------

def parse_instructions(instr_path: Path):
    """
    Parse WhatToChange.txt into a list of instruction groups:
    - figure_blocks: list of (fig_label_from_header_or_none, operations_list)
      where fig_label_from_header is like ';FIG001' or ';FIG002' (as in the file)
    - main_ops: operations for main text
    Each operation is a dict describing the op:
       { "op": "place"|"move"|"insert",
         "what": "grobid"|"clean",
         "id": 37,
         "after": ("grobid", 37) or ("clean", 203) or ("fig_begin", None) etc,
         "raw": original_line }
    """
    raw = instr_path.read_text(encoding='utf-8')
    lines = [l.rstrip() for l in raw.splitlines()]
    i = 0
    n = len(lines)
    figure_blocks = []  # list of (header_label, ops)
    main_ops = []
    current_block_label = None
    current_ops = None
    in_fig_section = False
    # Look for 'FIGURE BLOCK:' and 'MAIN TEXT'
    while i < n:
        line = lines[i].strip()
        if line.upper().startswith("FIGURE BLOCK:"):
            # start new figure block
            # header line may look like: FIGURE BLOCK: ;FIG001
            header = line[len("FIGURE BLOCK:"):].strip()
            header = header if header else None
            current_block_label = header  # e.g., ';FIG001' or ';FIG002'
            current_ops = []
            figure_blocks.append((current_block_label, current_ops))
            # advance
            i += 1
            # skip separator lines until operations
            while i < n and lines[i].strip().startswith("-"):
                i += 1
            continue
        if line.upper().startswith("MAIN TEXT"):
            # gather main text operations until end marker
            i += 1
            # skip separator
            while i < n and lines[i].strip().startswith("-"):
                i += 1
            # everything until a big ===== or end constitutes main ops
            while i < n and not lines[i].strip().startswith("="):
                l = lines[i].strip()
                if l:
                    op = parse_instruction_line(l)
                    if op:
                        main_ops.append(op)
                i += 1
            continue
        # otherwise, parse op lines inside current figure block if any
        if current_ops is not None:
            if line == "" or line.startswith("="):
                # block end or skip
                current_ops = None
                current_block_label = None
                i += 1
                continue
            # parse op if line looks like an operation
            if line:
                op = parse_instruction_line(line)
                if op:
                    # annotate block label if the op is figure-anchored (we keep header label separately)
                    current_ops.append(op)
            i += 1
            continue
        i += 1

    return figure_blocks, main_ops

def parse_instruction_line(line: str):
    """
    Parse single instruction line examples into an operation dict.

    Examples:
      PLACE GROBID 037 (DIV01PARA07) → AT BEGINNING OF FIGURE  # CLEAN 198
      MOVE GROBID 038 (DIV01PARA08) → AFTER GROBID 037  # CLEAN 199
      INSERT CLEAN 203 AFTER GROBID 042  # no match → "Statistical errors..."
      PLACE GROBID 020 AFTER CLEAN 019  # CLEAN 020
      INSERT CLEAN 236 AT BEGINNING OF FIGURE  # no match → "Pulsed lasing."
      PLACE GROBID 037 (DIV01PARA07) → AT BEGINNING OF FIGURE  # CLEAN 198
    """
    raw = line
    # remove inline comments after '#'
    line = line.split('#', 1)[0].strip()
    if not line:
        return None

    # Normalize long arrow
    line = line.replace('→', '->').replace('→', '->')

    # Patterns
    # INSERT CLEAN 203 AFTER GROBID 042
    m = re.match(r'INSERT\s+CLEAN\s+(\d+)\s+(?:AFTER|FOLLOWING)\s+(GROBID|CLEAN)\s*(\d+)', line, re.I)
    if m:
        cid = int(m.group(1))
        tgt_type = m.group(2).lower()
        tgt_id = int(m.group(3))
        return {"op":"insert", "what":"clean", "id":cid, "after":(tgt_type, tgt_id), "raw": raw}

    # INSERT CLEAN 236 AT BEGINNING OF FIGURE
    m = re.match(r'INSERT\s+CLEAN\s+(\d+)\s+AT\s+BEGINNING\s+OF\s+FIGURE', line, re.I)
    if m:
        cid = int(m.group(1))
        return {"op":"insert", "what":"clean", "id":cid, "after":("fig_begin", None), "raw": raw}

    # PLACE GROBID 037 (....) -> AT BEGINNING OF FIGURE
    m = re.match(r'PLACE\s+GROBID\s+(\d+).*?AT\s+BEGINNING\s+OF\s+FIGURE', line, re.I)
    if m:
        gid = int(m.group(1))
        return {"op":"place", "what":"grobid", "id":gid, "after":("fig_begin", None), "raw": raw}

    # PLACE GROBID 020 AFTER CLEAN 019
    m = re.match(r'PLACE\s+GROBID\s+(\d+).*?AFTER\s+(GROBID|CLEAN)\s*(\d+)', line, re.I)
    if m:
        gid = int(m.group(1))
        tgt_type = m.group(2).lower()
        tgt_id = int(m.group(3))
        return {"op":"place", "what":"grobid", "id":gid, "after":(tgt_type, tgt_id), "raw":raw}

    # MOVE GROBID 038 ... AFTER GROBID 037
    m = re.match(r'MOVE\s+GROBID\s+(\d+).*?AFTER\s+(GROBID|CLEAN)\s*(\d+)', line, re.I)
    if m:
        gid = int(m.group(1))
        tgt_type = m.group(2).lower()
        tgt_id = int(m.group(3))
        return {"op":"move", "what":"grobid", "id":gid, "after":(tgt_type, tgt_id), "raw":raw}

    # MOVE GROBID 038 (...) -> AFTER CLEAN 203
    m = re.match(r'MOVE\s+GROBID\s+(\d+).*?AFTER\s+CLEAN\s*(\d+)', line, re.I)
    if m:
        gid = int(m.group(1))
        tgt_id = int(m.group(2))
        return {"op":"move", "what":"grobid", "id":gid, "after":("clean", tgt_id), "raw":raw}

    # PLACE GROBID ... AFTER GROBID ...
    m = re.match(r'PLACE\s+GROBID\s+(\d+).*?AFTER\s+GROBID\s*(\d+)', line, re.I)
    if m:
        gid = int(m.group(1))
        tgt_id = int(m.group(2))
        return {"op":"place", "what":"grobid", "id":gid, "after":("grobid", tgt_id), "raw":raw}

    # Generic fallback: '... AFTER GROBID 037' or '... AFTER CLEAN 191'
    m = re.match(r'.*AFTER\s+(GROBID|CLEAN)\s*(\d+)', line, re.I)
    if m:
        tgt_type = m.group(1).lower()
        tgt_id = int(m.group(2))
        # find id mentioned at start
        m2 = re.match(r'^(MOVE|PLACE|INSERT)\s+(GROBID|CLEAN)\s*(\d+)', line, re.I)
        if m2:
            op = m2.group(1).lower()
            what = m2.group(2).lower()
            sid = int(m2.group(3))
            return {"op":op, "what":what, "id":sid, "after":(tgt_type, tgt_id), "raw":raw}

    # No match
    print(f"[WARN] Could not parse instruction line: {line}", file=sys.stderr)
    return None

# ---------------------------
# TEI XML helpers
# ---------------------------

def find_all_paragraphs(soup: BeautifulSoup):
    """Return list of all <p> tags in document order"""
    return soup.find_all('p')

def find_figures(soup: BeautifulSoup):
    """Return list of <figure> tags in document order"""
    return soup.find_all('figure')

def create_p_tag(soup: BeautifulSoup, text: str):
    p = soup.new_tag('p')
    # insert text preserving as plain text
    p.append(NavigableString(text))
    return p

# ---------------------------
# Main application logic
# ---------------------------

def apply_instructions_to_tei(
    raw_tei_path: Path,
    tei_sents_path: Path,
    clean_sents_path: Path,
    instructions_path: Path,
    output_path: Path
):
    # Parse inputs
    print("Parsing extracted TEI sentences...")
    ordered_ids, tei_sents = parse_tei_extracted_sentences(tei_sents_path)
    print(f"  Parsed {len(ordered_ids)} extracted TEI sentences.")

    print("Parsing clean sentences (for INSERT CLEAN)...")
    clean_map = parse_clean_sentences(clean_sents_path)
    print(f"  Parsed {len(clean_map)} clean sentences.")

    print("Parsing instruction file...")
    figure_blocks, main_ops = parse_instructions(instructions_path)
    print(f"  Found {len(figure_blocks)} figure blocks and {len(main_ops)} main text operations.")

    # Load TEI XML
    print(f"Loading raw TEI XML: {raw_tei_path}")
    soup = BeautifulSoup(raw_tei_path.read_text(encoding='utf-8'), "lxml-xml")

    # Build TEI paragraph -> text normalization mapping and sentence lookup
    p_list = find_all_paragraphs(soup)
    print(f"Document contains {len(p_list)} <p> elements (initial).")

    # Build mapping p -> text normalized and mapping of sentence id -> p element if found
    p_texts = {}
    for p in p_list:
        txt = normalize_space(" ".join([t for t in p.strings]))
        p_texts[p] = txt

    sent_to_p = {}   # id -> p element (the p it came from)
    # We'll try to locate each extracted TEI sentence in the paragraphs
    for sid, info in tei_sents.items():
        target = normalize_space(info['text'])
        found = False
        for p, pt in p_texts.items():
            if target and target in pt:
                # assign mapping; if multiple matches exist we pick the first
                sent_to_p[sid] = p
                found = True
                break
        if not found:
            # not found inside any p => warn; will handle later by creating new p when inserting/moving
            # print a debug warning but not fatal
            print(f"[INFO] GROBID sentence {sid} not found in any <p> (will create/move as needed).", file=sys.stderr)

    # Build figure mapping
    figures = find_figures(soup)
    print(f"Found {len(figures)} <figure> elements in TEI.")
    # Map instruction figure blocks in order to TEI figure elements in order
    # If counts mismatch we still map min(len(blocks), len(figures)) in order.
    figure_block_labels = [lbl for lbl, _ops in figure_blocks]
    fig_id_map = {}   # maps header label (e.g., ';FIG001' or ';FIG002') -> actual <figure> Tag
    for idx, (lbl, ops) in enumerate(figure_blocks):
        if idx < len(figures):
            fig_tag = figures[idx]
            fig_id_map[lbl] = fig_tag
            # print mapping
            fi = fig_tag.get('xml:id') or fig_tag.get('id') or f"FIG_TAG_{idx}"
            print(f"Mapping instruction block {lbl} -> TEI figure xml:id='{fi}'")
        else:
            # no corresponding figure element available
            fig_id_map[lbl] = None
            print(f"[WARN] No TEI <figure> available to map for instruction block {lbl}", file=sys.stderr)

    # Track created/in-use mapping for inserted clean sentences
    inserted_clean_to_p = {}   # clean_id -> p_tag created

    # Track moved grobid mapping (where a grobid sentence now lives) - will update as we move
    current_grobid_location = { sid: sent_to_p.get(sid, None) for sid in tei_sents.keys() }

    # Utility: create paragraph after a given tag
    def insert_after_tag(reference_tag: Tag, new_tag: Tag):
        # If ref has .next_sibling, insert before that; otherwise append to parent
        if reference_tag is None:
            # no anchor, append to body end
            body = soup.find('body')
            if body is None:
                raise RuntimeError("No <body> in TEI")
            body.append(new_tag)
            return new_tag
        # Insert after reference_tag
        reference_tag.insert_after(new_tag)
        return new_tag

    # Utility: remove a sentence text from its original paragraph (sent is a string)
    def remove_sentence_text_from_paragraph(p_tag: Tag, sentence_text: str):
        if p_tag is None:
            return
        orig = normalize_space(" ".join([t for t in p_tag.strings]))
        if sentence_text in orig:
            new_text = orig.replace(sentence_text, "").strip()
            new_text = normalize_space(new_text)
            if new_text:
                # replace contents
                p_tag.clear()
                p_tag.append(NavigableString(new_text))
            else:
                # remove paragraph entirely
                p_tag.decompose()
        else:
            # not exact substring; try a loose replace by sentences boundary heuristic
            # We'll attempt to remove the longest common substring around words of sentence_text
            # But to avoid damaging content, we simply log the fact and leave original p intact
            print(f"[WARN] Could not remove exact occurrence of sentence in paragraph; leaving original <p>. Sentence preview: {sentence_text[:80]}", file=sys.stderr)

    # Helper to ensure we can resolve anchor (grobid or clean)
    def resolve_anchor(anchor_tuple):
        """
        anchor_tuple: ("grobid" or "clean" or "fig_begin", id or None)
        Returns: a Tag after which we should insert (i.e., anchor p), or None meaning append to body end.
        If anchor is fig_begin, returns the figDesc tag for that figure (so insertion should be AFTER figDesc).
        """
        typ, aid = anchor_tuple
        if typ == "fig_begin":
            # The caller must supply the current figure context; this will be handled where necessary.
            return ("fig_begin", None)
        if typ == "grobid":
            if aid in current_grobid_location and current_grobid_location[aid] is not None:
                return current_grobid_location[aid]
            else:
                # maybe the grobid was not found; warn
                print(f"[WARN] Anchor GROBID {aid} not found (yet). We'll append to body end as fallback.", file=sys.stderr)
                return None
        if typ == "clean":
            if aid in inserted_clean_to_p:
                return inserted_clean_to_p[aid]
            else:
                # Might be referenced before insertion — but instructions are sequential and should not do that.
                print(f"[WARN] Anchor CLEAN {aid} not found among inserted cleans. Fallback: append to body.", file=sys.stderr)
                return None
        return None

    # ---------------------------
    # Process FIGURE BLOCKS (in order)
    # ---------------------------
    for block_label, ops in figure_blocks:
        print(f"\n--- Processing figure block {block_label} (ops: {len(ops)}) ---")
        fig_tag = fig_id_map.get(block_label)
        if fig_tag is None:
            print(f"[WARN] No TEI figure mapped for block {block_label}; operations will append to body.", file=sys.stderr)

        # find the figDesc inside the figure to place "AT BEGINNING OF FIGURE" inserts
        fig_desc_tag = None
        if fig_tag is not None:
            fig_desc_tag = fig_tag.find('figDesc')
            # If there's no figDesc, we'll create one at top of figure
            if fig_desc_tag is None:
                # create figDesc
                fig_desc_tag = soup.new_tag('figDesc')
                # insert as first child
                fig_tag.insert(0, fig_desc_tag)

        # Keep a local order list for the caption we are assembling (list of p tags in order)
        caption_order = []  # list of Tags (p) in final order for this fig

        # NOTE: instructions are sequential: when they say AFTER GROBID 037 they assume 037
        # is already placed in caption_order or exists in document; so we must respect that.
        for op in ops:
            op_raw = op.get('raw')
            if op['op'] in ('place', 'move') and op['what'] == 'grobid':
                gid = op['id']
                # Get source text for gid
                ginfo = tei_sents.get(gid)
                if not ginfo:
                    print(f"[WARN] GROBID {gid} referenced but not present in tei_extracted list.", file=sys.stderr)
                    continue
                sent_text = normalize_space(ginfo['text'])
                # If this grobid currently has a p element, remove its text from original paragraph
                orig_p = current_grobid_location.get(gid)
                if orig_p is not None:
                    # create a new p tag for placement with the exact sentence text
                    new_p = create_p_tag(soup, sent_text)
                    # remove the sentence text from original paragraph
                    remove_sentence_text_from_paragraph(orig_p, sent_text)
                    # We'll insert new_p to target location below
                    # remove original mapping (we will set to new location)
                    current_grobid_location[gid] = None
                else:
                    # original not found: create new p
                    new_p = create_p_tag(soup, sent_text)

                # Determine anchor for insertion
                after = op['after']
                if after[0] == 'fig_begin':
                    # Insert at beginning of figure (i.e., directly after figDesc)
                    if fig_desc_tag is not None:
                        insert_after_tag(fig_desc_tag, new_p)
                        caption_order.append(new_p)
                        current_grobid_location[gid] = new_p
                    else:
                        # fallback: append to figure or body
                        if fig_tag is not None:
                            fig_tag.append(new_p)
                            caption_order.append(new_p)
                            current_grobid_location[gid] = new_p
                        else:
                            # append to body
                            body = soup.find('body')
                            body.append(new_p)
                            caption_order.append(new_p)
                            current_grobid_location[gid] = new_p
                else:
                    # after target may be GROBID or CLEAN
                    anchor_tag = resolve_anchor(after)
                    if isinstance(anchor_tag, tuple) and anchor_tag[0] == "fig_begin":
                        # Should not happen here, but fallback
                        if fig_desc_tag is not None:
                            insert_after_tag(fig_desc_tag, new_p)
                            caption_order.append(new_p)
                            current_grobid_location[gid] = new_p
                        else:
                            body = soup.find('body')
                            body.append(new_p)
                            caption_order.append(new_p)
                            current_grobid_location[gid] = new_p
                    else:
                        # typical case anchor_tag is a Tag or None
                        if anchor_tag is None:
                            # append to end of figure or body
                            if fig_tag is not None:
                                fig_tag.append(new_p)
                                caption_order.append(new_p)
                                current_grobid_location[gid] = new_p
                            else:
                                body = soup.find('body')
                                body.append(new_p)
                                caption_order.append(new_p)
                                current_grobid_location[gid] = new_p
                        else:
                            # insert after the anchor_tag
                            insert_after_tag(anchor_tag, new_p)
                            caption_order.append(new_p)
                            current_grobid_location[gid] = new_p

            elif op['op'] == 'insert' and op['what'] == 'clean':
                cid = op['id']
                ctext = clean_map.get(cid)
                if not ctext:
                    print(f"[WARN] CLEAN {cid} referenced but not found in clean sentences file.", file=sys.stderr)
                    ctext = f"[MISSING CLEAN {cid}]"
                new_p = create_p_tag(soup, ctext)
                after = op['after']
                if after[0] == 'fig_begin':
                    if fig_desc_tag is not None:
                        insert_after_tag(fig_desc_tag, new_p)
                        inserted_clean_to_p[cid] = new_p
                        caption_order.append(new_p)
                    else:
                        if fig_tag is not None:
                            fig_tag.append(new_p)
                            inserted_clean_to_p[cid] = new_p
                            caption_order.append(new_p)
                        else:
                            body = soup.find('body')
                            body.append(new_p)
                            inserted_clean_to_p[cid] = new_p
                            caption_order.append(new_p)
                else:
                    anchor_tag = resolve_anchor(after)
                    if anchor_tag is None:
                        # append to end of figure or body
                        if fig_tag is not None:
                            fig_tag.append(new_p)
                            inserted_clean_to_p[cid] = new_p
                            caption_order.append(new_p)
                        else:
                            body = soup.find('body')
                            body.append(new_p)
                            inserted_clean_to_p[cid] = new_p
                            caption_order.append(new_p)
                    else:
                        insert_after_tag(anchor_tag, new_p)
                        inserted_clean_to_p[cid] = new_p
                        caption_order.append(new_p)
            else:
                # Other operations or unsupported ones are ignored for figure phase
                print(f"[INFO] Skipping unsupported or unexpected operation in figure block: {op}", file=sys.stderr)

        # After processing all ops for this figure block, ensure the actual <figDesc> contains the concatenation
        # of caption_order (we place the ordered paragraphs INSIDE the figure, after figDesc if it exists)
        # However, we've already inserted each new p into the figure in order; so figDesc replacement is not needed.
        # Optionally: if figDesc had original text that should be replaced, we can replace figDesc contents with caption text
        # We'll replace figDesc text with concatenated text from caption_order for clarity (but keep the inserted <p> elements too).
        if fig_tag is not None and fig_desc_tag is not None:
            # Build a single string: join paragraph texts with space
            caption_texts = []
            for ptag in caption_order:
                if isinstance(ptag, Tag):
                    caption_texts.append(normalize_space(" ".join([t for t in ptag.strings])))
            if caption_texts:
                # Replace figDesc content with single text node combining sentence texts
                combined = " ".join(caption_texts)
                fig_desc_tag.clear()
                fig_desc_tag.append(NavigableString(combined))
                # Optionally remove the separate <p> nodes we added into the figure (to avoid duplication),
                # but that may remove original structure. We will keep both figDesc (as combined) and the individual <p>.
                # If you prefer to only keep <p> elements inside figure and remove figDesc, implement that here.
        print(f"[OK] Processed figure block {block_label}.")

    # ---------------------------
    # Process MAIN TEXT operations sequentially
    # ---------------------------
    print("\n--- Processing main text operations ---")
    for op in main_ops:
        r = op
        if r['op'] in ('place', 'move') and r['what'] == 'grobid':
            gid = r['id']
            info = tei_sents.get(gid)
            if not info:
                print(f"[WARN] GROBID {gid} not present in extraction list.", file=sys.stderr)
                continue
            stext = normalize_space(info['text'])
            orig_p = current_grobid_location.get(gid)
            if orig_p is not None:
                # remove from original paragraph
                remove_sentence_text_from_paragraph(orig_p, stext)
                # create new p
                new_p = create_p_tag(soup, stext)
            else:
                new_p = create_p_tag(soup, stext)

            # find anchor and insert after
            anchor = r['after']
            anchor_tag = resolve_anchor(anchor)
            if anchor_tag is None:
                # append to end of body
                body = soup.find('body')
                if body is None:
                    raise RuntimeError("No <body> in TEI while processing main ops.")
                body.append(new_p)
                current_grobid_location[gid] = new_p
            else:
                insert_after_tag(anchor_tag, new_p)
                current_grobid_location[gid] = new_p

        elif r['op'] == 'insert' and r['what'] == 'clean':
            cid = r['id']
            ctext = clean_map.get(cid, f"[MISSING CLEAN {cid}]")
            new_p = create_p_tag(soup, ctext)
            anchor = r['after']
            anchor_tag = resolve_anchor(anchor)
            if anchor_tag is None:
                body = soup.find('body')
                body.append(new_p)
                inserted_clean_to_p[cid] = new_p
            else:
                insert_after_tag(anchor_tag, new_p)
                inserted_clean_to_p[cid] = new_p
        else:
            print(f"[INFO] Skipping unsupported main operation: {r}", file=sys.stderr)

    # ---------------------------
    # Final: write out modified TEI
    # ---------------------------
    out_text = soup.prettify()
    output_path.write_text(out_text, encoding='utf-8')
    print(f"\nAll done. Corrected TEI written to: {output_path}")

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply WhatToChange instructions to TEI XML")
    parser.add_argument("--raw-tei", required=True, type=Path, help="Input raw TEI XML file")
    parser.add_argument("--tei-sents", required=True, type=Path, help="Extracted TEI sentences file (tei_corrected.txt)")
    parser.add_argument("--clean-sents", required=True, type=Path, help="Clean sentences file (sentences_corrected.txt)")
    parser.add_argument("--instructions", required=True, type=Path, help="WhatToChange.txt file with instructions")
    parser.add_argument("--out", required=True, type=Path, help="Output corrected TEI XML path")
    args = parser.parse_args()

    apply_instructions_to_tei(
        raw_tei_path=args.raw_tei,
        tei_sents_path=args.tei_sents,
        clean_sents_path=args.clean_sents,
        instructions_path=args.instructions,
        output_path=args.out
    )


if __name__ == "__main__":
    main()
