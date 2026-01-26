# figure_cropper.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _build_text_mask(thr_inv: np.ndarray, page_w: int, page_h: int) -> np.ndarray:
    """
    Build a mask of likely text lines using strong horizontal morphology.
    This is intentionally biased toward captions/paragraphs (long horizontal runs).
    """
    # Merge letters into long horizontal lines
    # (wide kernel to connect words into lines)
    k_w = max(40, int(page_w * 0.08))   # e.g. ~8% of page width
    k_h = max(2, int(page_h * 0.004))   # thin
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
    text_lines = cv2.morphologyEx(thr_inv, cv2.MORPH_DILATE, horiz, iterations=2)

    # Clean up: remove small specks
    small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_lines = cv2.morphologyEx(text_lines, cv2.MORPH_OPEN, small, iterations=1)

    # Convert detected lines into a mask (filter by geometry)
    contours, _ = cv2.findContours(text_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thr_inv)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Heuristics for "text block":
        # - wide relative to page
        # - not too tall
        # - contains enough ink
        if w < page_w * 0.25:
            continue
        if h > page_h * 0.20:  # very tall blocks probably not just text
            continue

        # draw into mask
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)

    return mask


def _trim_caption_from_bottom(
    crop_img: np.ndarray,
    crop_thr_inv: np.ndarray,
    crop_text_mask: np.ndarray,
    *,
    max_bottom_text_frac: float = 0.18,
    strip_px: int = 24,
    min_keep_h: int = 120
) -> Tuple[np.ndarray, int]:
    """
    If the bottom part of the crop is very text-heavy, trim it off iteratively.
    Returns (trimmed_img, trimmed_pixels_from_bottom)
    """
    h, w = crop_img.shape[:2]
    trimmed = 0

    # Ensure strip size scales on small crops
    strip_px = _clamp(strip_px, 12, max(12, h // 8))

    while h - trimmed > min_keep_h:
        y0 = h - trimmed - strip_px
        y1 = h - trimmed
        if y0 < 0:
            break

        strip_mask = crop_text_mask[y0:y1, :]
        text_frac = float(cv2.countNonZero(strip_mask)) / float(strip_mask.size)

        # if bottom strip is mostly text, trim it
        if text_frac > max_bottom_text_frac:
            trimmed += strip_px
            continue
        break

    if trimmed > 0:
        new_h = h - trimmed
        crop_img = crop_img[:new_h, :]
    return crop_img, trimmed


def crop_figures_from_rendered_pages(
    rendered_pages_dir: Path,
    out_dir: Path,
    *,
    min_area_ratio: float = 0.03,     # ignore tiny blocks
    max_area_ratio: float = 0.90,     # ignore full-page blocks
    max_text_frac_in_crop: float = 0.22,  # reject crops that are mostly text
    pad_px: int = 8,                  # padding around bbox
    max_candidates_per_page: int = 8, # allow a few more now that we filter better
    debug_save_masks: bool = False,   # if True, writes masks for inspection
) -> Dict:
    """
    Finds large rectangular regions that look like figures on each rendered page and crops them.

    Improvements vs old version:
      - Detect caption/paragraph-like text blocks and suppress them.
      - If a candidate bbox includes a caption underneath, trim text-heavy bottom strips.
    """
    rendered_pages_dir = Path(rendered_pages_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict] = []
    crop_count = 0

    page_files = sorted(rendered_pages_dir.glob("*.png"))
    for p_i, page_path in enumerate(page_files, start=1):
        img = cv2.imread(str(page_path))
        if img is None:
            continue

        H, W = img.shape[:2]
        page_area = float(H * W)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1) Binarize: ink becomes white (255), background black (0)
        thr = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 9
        )

        # 2) Build a mask of likely text-lines (captions/paragraphs)
        text_mask = _build_text_mask(thr, W, H)

        # 3) Remove text from thr to focus on graphical regions
        thr_wo_text = cv2.bitwise_and(thr, cv2.bitwise_not(text_mask))

        # 4) Connect remaining components into larger blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
        closed = cv2.morphologyEx(thr_wo_text, cv2.MORPH_CLOSE, kernel, iterations=2)

        if debug_save_masks:
            dbg_dir = out_dir / "_debug_masks"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dbg_dir / f"p{p_i:04d}_thr.png"), thr)
            cv2.imwrite(str(dbg_dir / f"p{p_i:04d}_textmask.png"), text_mask)
            cv2.imwrite(str(dbg_dir / f"p{p_i:04d}_closed.png"), closed)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[int, int, int, int, float]] = []
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = float(ww * hh)
            ratio = area / page_area

            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue

            # ignore very thin strips
            if ww < W * 0.15 or hh < H * 0.10:
                continue

            # reject regions that are mostly text according to text_mask
            tm = text_mask[y:y+hh, x:x+ww]
            text_frac = float(cv2.countNonZero(tm)) / float(tm.size)
            if text_frac > max_text_frac_in_crop:
                continue

            candidates.append((x, y, ww, hh, area))

        # sort biggest first; keep more since we filter heavily now
        candidates.sort(key=lambda t: t[4], reverse=True)
        candidates = candidates[:max_candidates_per_page]

        for j, (x, y, ww, hh, _) in enumerate(candidates, start=1):
            # add padding
            x0 = _clamp(int(x - pad_px), 0, W - 1)
            y0 = _clamp(int(y - pad_px), 0, H - 1)
            x1 = _clamp(int(x + ww + pad_px), 0, W)
            y1 = _clamp(int(y + hh + pad_px), 0, H)

            crop_img = img[y0:y1, x0:x1]
            crop_text_mask = text_mask[y0:y1, x0:x1]
            crop_thr = thr[y0:y1, x0:x1]

            # If caption attached underneath, trim it
            crop_img_trimmed, trimmed_px = _trim_caption_from_bottom(
                crop_img,
                crop_thr,
                crop_text_mask,
                max_bottom_text_frac=max_text_frac_in_crop,
                strip_px=max(18, int(H * 0.02)),
                min_keep_h=max(120, int(H * 0.10)),
            )

            crop_name = f"crop_p{p_i:04d}_{j:02d}.png"
            crop_path = out_dir / crop_name
            cv2.imwrite(str(crop_path), crop_img_trimmed)

            crop_count += 1
            items.append({
                "page": p_i,
                "crop": crop_name,
                "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "trimmed_bottom_px": int(trimmed_px),
                "source_page": page_path.name,
            })

    return {
        "crops_dir": str(out_dir),
        "count": crop_count,
        "items": items,
    }
