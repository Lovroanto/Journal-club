from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor

EMU_PER_INCH = 914400

def emu_to_inches(x_emu: int) -> float:
    return x_emu / EMU_PER_INCH

def get_slide_size_in(prs) -> tuple[float, float]:
    return emu_to_inches(prs.slide_width), emu_to_inches(prs.slide_height)


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float


def _add_title(slide, title: str, x: float, y: float, w: float, h: float) -> None:
    tx = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tx.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)


def _add_speaker_notes(slide, speaker_notes: str) -> None:
    # Keep notes in the slide notes pane (not visible on slide)
    try:
        notes = slide.notes_slide.notes_text_frame
        notes.clear()
        notes.text = speaker_notes or ""
    except Exception:
        pass


def _add_figure_placeholder(slide, caption: str, x: float, y: float, w: float, h: float) -> None:
    shp = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor(245, 245, 245)
    shp.line.color.rgb = RGBColor(180, 180, 180)

    tf = shp.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = caption or "Figure"
    p.font.size = Pt(14)
    p.font.italic = True


def _add_bullet_box(slide, text: str, box: Box):
    shp = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(box.x), Inches(box.y), Inches(box.w), Inches(box.h),
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor(250, 250, 250)
    shp.line.color.rgb = RGBColor(120, 120, 120)

    tf = shp.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(0, 0, 0)
    return shp


def _center(box: Box) -> Tuple[float, float]:
    return (box.x + box.w / 2.0, box.y + box.h / 2.0)


def _add_arrow(slide, a: Box, b: Box) -> None:
    ax, ay = _center(a)
    bx, by = _center(b)

    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(ax), Inches(ay), Inches(bx), Inches(by),
    )
    conn.line.color.rgb = RGBColor(60, 60, 60)
    conn.line.width = Pt(2)

    # Arrowheads in python-pptx can be version-dependent; try and ignore if unsupported.
    try:
        conn.line.end_arrowhead = True  # may not exist depending on python-pptx version
    except Exception:
        pass


def add_slide_from_plan(
    prs: Presentation,
    *,
    title: str,
    bullets: List[str],
    figure_used: Optional[str],
    edges: List[Dict],
    speaker_notes: str = "",
):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # Get real slide size in inches and derive percent-based layout
    slide_w, slide_h = get_slide_size_in(prs)

    # Percent-based margins and regions
    margin_x = 0.05 * slide_w
    margin_top = 0.05 * slide_h
    margin_bottom = 0.05 * slide_h
    gap_x = 0.03 * slide_w

    # Title region (~14% of height)
    title_h = 0.14 * slide_h
    title_y = margin_top
    content_y = title_y + title_h
    content_h = slide_h - content_y - margin_bottom

    left_x = margin_x
    usable_w = slide_w - 2 * margin_x

    _add_title(slide, title, left_x, title_y, usable_w, title_h)
    _add_speaker_notes(slide, speaker_notes)

    has_fig = bool(figure_used)
    if has_fig:
        fig_w = 0.40 * usable_w
        bullets_w = usable_w - fig_w - gap_x
        bullets_x = left_x
        fig_x = bullets_x + bullets_w + gap_x
    else:
        bullets_x = left_x
        bullets_w = usable_w
        fig_x = fig_w = 0.0

    if has_fig:
        _add_figure_placeholder(slide, str(figure_used), fig_x, content_y, fig_w, content_h)

    n = len(bullets)
    if n == 0:
        return slide

    # Simple rule: if many bullets and enough width, use 2 columns
    cols = 2 if (n >= 6 and bullets_w >= 7.0) else 1
    col_gap = 0.02 * usable_w
    col_w = (bullets_w - col_gap * (cols - 1)) / cols

    rows = (n + cols - 1) // cols
    row_gap = 0.02 * slide_h
    box_h = min(1.1, (content_h - row_gap * (rows - 1)) / max(1, rows))

    boxes: List[Box] = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            x = bullets_x + c * (col_w + col_gap)
            y = content_y + r * (box_h + row_gap)
            b = Box(x=x, y=y, w=col_w, h=box_h)
            boxes.append(b)
            _add_bullet_box(slide, bullets[idx], b)
            idx += 1

    for e in edges or []:
        try:
            i = int(e.get("from", -1))
            j = int(e.get("to", -1))
        except Exception:
            continue
        if 0 <= i < len(boxes) and 0 <= j < len(boxes) and i != j:
            _add_arrow(slide, boxes[i], boxes[j])

    return slide
