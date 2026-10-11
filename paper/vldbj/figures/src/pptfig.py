"""Shared drawing helpers for the PowerPoint figures of the VLDBJ paper.

Figures 3 and 4 are drawn on a 12.2-inch-wide slide and included at
\\textwidth (495pt), a scale of 0.564.  Font sizes are chosen for the printed
size required by The VLDB Journal (similar to the 10pt running text):

    TITLE 16pt -> 9.0pt printed,  TEXT 15pt -> 8.5pt,  SMALL 12.5pt -> 7.0pt.

Sub- and superscripts are set at SUBSCALE times the base size because the
renderers shrink baseline-shifted text (LibreOffice to 58%), which keeps them
at about 5.5-6pt printed (as LaTeX's own first-level scripts of 8pt text).

Export (LibreOffice must see the system fonts Times New Roman and STIX Two Math):

    export FONTCONFIG_FILE=/opt/homebrew/etc/fonts/fonts.conf FONTCONFIG_PATH=/opt/homebrew/etc/fonts
    soffice --headless --convert-to pdf <file>.pptx
    pdfcrop --margins 1 <file>.pdf ../<file>.pdf
"""
import re

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

TITLE, TEXT, SMALL = 16, 15, 12.5
SUBSCALE = 1.15

TEXT_FONT = 'Times New Roman'
MATH_FONT = 'Times New Roman'         # math letters in italic Times, as in Fig. 3
SYM_FONT = 'STIX Two Math'            # symbols missing in Times, accented letters
SYMBOLS = set('∈ℝ⊙ℎ')
MATH_ITALIC = {'h': 'ℎ', 'x': '𝑥', 'z': '𝑧', 'y': '𝑦'}
ITALIC = re.compile(r'[A-Za-zα-ωℓ]')

BLACK = RGBColor(0, 0, 0)
WHITE = RGBColor(255, 255, 255)
GRAY_PANEL = RGBColor(0xF2, 0xF2, 0xF2)
GRAY_BOX = RGBColor(0xD0, 0xD0, 0xD0)
GRAY_LINE = RGBColor(0xB0, 0xB0, 0xB0)
PEACH = RGBColor(0xFB, 0xE0, 0xD3)
YELLOW = RGBColor(0xFF, 0xF2, 0xCC)
BLUE_PANEL = RGBColor(0xDE, 0xEB, 0xF7)
BLUE_BOX = RGBColor(0xC5, 0xDC, 0xF2)
GREEN_PANEL = RGBColor(0xE2, 0xF0, 0xD9)
TOKEN_BLUE = RGBColor(0x1F, 0x9F, 0xD8)
TOKEN_GREEN = RGBColor(0x3E, 0x9B, 0x2A)
TOKEN_LIGHT = RGBColor(0xA9, 0xD1, 0x8E)
PURPLE = RGBColor(0x4B, 0x1E, 0x4F)
BRANCH = RGBColor(0x1F, 0x4E, 0x96)   # arrows of the new level branch

prs = None
shapes = None


def new_slide(width_in, height_in):
    global prs, shapes
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(width_in), Inches(height_in)
    shapes = prs.slides.add_slide(prs.slide_layouts[6]).shapes
    return prs


def save(path):
    prs.save(path)
    print('wrote', path)


# --------------------------------------------------------------------------
# Text with a small math markup: $...$ is math; inside it ASCII letters and
# lowercase Greek are italic, _x/_{..} and ^x/^{..} are sub/superscripts,
# 𝟏 is the bold all-ones vector, \b{..} bold upright, \r{..} upright.  **...** outside math is bold.
# --------------------------------------------------------------------------
def _run(par, text, size, bold=False, italic=False, font=TEXT_FONT, baseline=0, color=BLACK,
         underline=False):
    run = par.add_run()
    run.text = text
    f = run.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.underline = underline
    f.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ('a:latin', 'a:ea', 'a:cs'):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set('typeface', font)
    if baseline:
        rpr.set('baseline', str(baseline))


def _math(par, src, size, color, baseline=0, bold=False):
    i = 0
    while i < len(src):
        ch = src[i]
        if ch in '_^':
            shift = -15000 if ch == '_' else 28000
            i += 1
            if src[i] == '{':
                depth, j = 1, i + 1
                while depth:
                    depth += {'{': 1, '}': -1}.get(src[j], 0)
                    j += 1
                inner, i = src[i + 1:j - 1], j
            else:
                inner, i = src[i], i + 1
                while i < len(src) and '̀' <= src[i] <= 'ͯ':
                    inner += src[i]
                    i += 1
            _math(par, inner, size * SUBSCALE if not baseline else size, color, shift, bold)
            continue
        if ch == '𝟏':
            _run(par, '1', size, bold=True, font=MATH_FONT, baseline=baseline, color=color)
            i += 1
            continue
        if ch == '\\':                 # \b{X}: bold upright (\mathbf), \r{x}: upright (\text)
            j = src.index('}', i)
            _run(par, src[i + 3:j], size, bold=src[i + 1] == 'b', font=MATH_FONT, baseline=baseline,
                 color=color)
            i = j + 1
            continue
        j = i + 1
        while j < len(src) and '̀' <= src[j] <= 'ͯ':
            j += 1
        tok = src[i:j]
        # Accented letters come from the math font (mark anchors place the
        # accent over the letter), except the overline U+0305, which Times
        # sets above the ascender of h.
        accented = j > i + 1 and '̅' not in tok
        if accented:
            tok = MATH_ITALIC.get(ch, ch) + tok[1:]
        font = SYM_FONT if (ch in SYMBOLS or ord(ch) > 0xFFFF or accented) else MATH_FONT
        italic = bool(ITALIC.match(ch)) and font == MATH_FONT
        _run(par, tok, size, bold=bold, italic=italic, font=font, baseline=baseline, color=color)
        i = j


def write(tf, lines, size=TEXT, color=BLACK, align=PP_ALIGN.CENTER, spacing=1.0):
    """lines: list of strings in the markup above (one paragraph each)."""
    tf.clear()
    for k, line in enumerate(lines):
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        par.alignment = align
        par.line_spacing = spacing
        for seg in re.split(r'(\$[^$]*\$)', line):
            if seg.startswith('$'):
                _math(par, seg[1:-1], size, color)
            elif seg:
                for part in re.split(r'(\*\*[^*]+\*\*|__[^_]+__)', seg):
                    if part.startswith('**'):
                        _run(par, part[2:-2], size, bold=True, color=color)
                    elif part.startswith('__'):
                        _run(par, part[2:-2], size, bold=True, color=color, underline=True)
                    elif part:
                        _run(par, part, size, color=color)


def _plain(shape):
    """No theme style: explicit line/fill only, no shadow (LibreOffice draws the
    theme's effectRef shadow otherwise)."""
    style = shape._element.find(qn('p:style'))
    if style is not None:
        shape._element.remove(style)


def _frame(tf, anchor, margin):
    tf.word_wrap = False               # every line break is explicit
    tf.vertical_anchor = anchor
    for side in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, side, Inches(margin))


def box(x, y, w, h, lines=(), fill=WHITE, line=BLACK, lw=1.5, dash=None, size=TEXT,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12, anchor=MSO_ANCHOR.MIDDLE,
        align=PP_ALIGN.CENTER, spacing=1.0, margin=0.04, color=BLACK):
    s = shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = min(radius / min(w, h), 0.5)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
        if dash:
            ln = s.line._get_or_add_ln()
            etree.SubElement(ln, qn('a:prstDash')).set('val', 'dash' if dash is True else dash)
    _plain(s)
    _frame(s.text_frame, anchor, margin)
    if lines:
        write(s.text_frame, list(lines), size=size, align=align, spacing=spacing, color=color)
    return s


def label(x, y, w, h, lines, size=TEXT, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, color=BLACK,
          spacing=1.0):
    tb = shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    _frame(tb.text_frame, anchor, 0)
    write(tb.text_frame, list(lines), size=size, align=align, color=color, spacing=spacing)
    return tb


def path(points, color=BLACK, lw=1.5, arrow=True, dash=None, fill=None, closed=False):
    """Polyline through points (inches), with an arrowhead at the end."""
    fb = shapes.build_freeform(Inches(points[0][0]), Inches(points[0][1]), scale=1.0)
    fb.add_line_segments([(Inches(px), Inches(py)) for px, py in points[1:]], close=closed)
    s = fb.convert_to_shape()
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if color is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = color
        s.line.width = Pt(lw)
    _plain(s)
    ln = s.line._get_or_add_ln()
    if dash:
        etree.SubElement(ln, qn('a:prstDash')).set('val', dash)
    if arrow:
        tail = etree.SubElement(ln, qn('a:tailEnd'))
        tail.set('type', 'triangle')
        tail.set('w', 'med')
        tail.set('len', 'med')
    return s


def tokens(x, y, n, color, w=0.2, h=0.13, gap=0.05, lw=0.75):
    for k in range(n):
        box(x + k * (w + gap), y, w, h, fill=color, line=BLACK, lw=lw, shape=MSO_SHAPE.RECTANGLE)
    return x + n * (w + gap) - gap


def curve(x, y, w, h, ys, color=BLACK, lw=1.0):
    """Polyline through normalized ys in [0, 1] inside the box (x, y, w, h)."""
    n = len(ys)
    pts = [(x + w * k / (n - 1), y + h * (1 - v)) for k, v in enumerate(ys)]
    return path(pts, color=color, lw=lw, arrow=False)


def brace(x0, x1, y, depth=0.08, up=True, color=BLACK, lw=1.0):
    """Horizontal curly brace from x0 to x1 at height y, opening down (up=True) or up."""
    s = -1 if up else 1
    m = (x0 + x1) / 2
    pts = [(x0, y), (x0 + depth * 0.6, y + s * depth * 0.8), (m - depth * 0.6, y + s * depth * 0.8),
           (m, y + s * depth * 1.6), (m + depth * 0.6, y + s * depth * 0.8),
           (x1 - depth * 0.6, y + s * depth * 0.8), (x1, y)]
    return path(pts, color=color, lw=lw, arrow=False)


def block_arrow(x, y, w, h, color=BLACK):
    """Outlined right block arrow (the ⇨ of Fig. 3)."""
    s = shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = WHITE
    s.line.color.rgb = color
    s.line.width = Pt(1.5)
    _plain(s)
    return s


__all__ = [n for n in dir() if not n.startswith('__')]
