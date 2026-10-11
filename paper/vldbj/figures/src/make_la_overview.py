"""Build the PhaseFormer-LA overview (Fig. 4) as a PowerPoint slide.

The style follows the PhaseFormer overview of Fig. 3 (figures/overview.pdf):
Times New Roman text with bold titles, rounded boxes with black outlines,
dashed panels with light fills, green/blue token bars and black arrows.
Colors: components of the conference PhaseFormer keep their Fig. 3 colors
(embedding peach, routing gray, predictor yellow); the new components of
PhaseFormer-LA sit in light-blue panels.

    python make_la_overview.py            # writes la_overview.pptx next to this file
    # LibreOffice must see the system fonts (Times New Roman, STIX Two Math):
    export FONTCONFIG_FILE=/opt/homebrew/etc/fonts/fonts.conf FONTCONFIG_PATH=/opt/homebrew/etc/fonts
    soffice --headless --convert-to pdf la_overview.pptx
    pdfcrop --margins 1 la_overview.pdf ../la_overview.pdf
"""
from pathlib import Path
import re

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

OUT = Path(__file__).with_name('la_overview.pptx')
W, H = 12.2, 3.5                     # inches; same aspect class as Fig. 3

TEXT_FONT = 'Times New Roman'
MATH_FONT = 'Times New Roman'         # math letters, as in Fig. 3 (italic Times)
SYM_FONT = 'STIX Two Math'            # symbols missing in Times and accented letters
SYMBOLS = set('∈ℝ⊙ℎ')

BLACK = RGBColor(0, 0, 0)
WHITE = RGBColor(255, 255, 255)
GRAY_PANEL = RGBColor(0xF2, 0xF2, 0xF2)
GRAY_BOX = RGBColor(0xD0, 0xD0, 0xD0)
PEACH = RGBColor(0xFB, 0xE0, 0xD3)
YELLOW = RGBColor(0xFF, 0xF2, 0xCC)
BLUE_PANEL = RGBColor(0xDE, 0xEB, 0xF7)
BLUE_BOX = RGBColor(0xC5, 0xDC, 0xF2)
TOKEN_BLUE = RGBColor(0x1F, 0x9F, 0xD8)
TOKEN_GREEN = RGBColor(0x3E, 0x9B, 0x2A)
TOKEN_LIGHT = RGBColor(0xA9, 0xD1, 0x8E)
BRANCH = RGBColor(0x1F, 0x4E, 0x96)   # arrows of the new level branch

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
slide = prs.slides.add_slide(prs.slide_layouts[6])
shapes = slide.shapes


# --------------------------------------------------------------------------
# Text with a small math markup: $...$ is math; inside it ASCII letters and
# lowercase Greek are italic, _x/_{..} and ^x/^{..} are sub/superscripts.
# **...** outside math is bold.
# --------------------------------------------------------------------------
ITALIC = re.compile(r'[A-Za-zα-ωℓ]')


def _run(par, text, size, bold=False, italic=False, font=TEXT_FONT, baseline=0, color=BLACK):
    run = par.add_run()
    run.text = text
    f = run.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = color
    rpr = run._r.get_or_add_rPr()
    for tag in ('a:latin', 'a:ea', 'a:cs'):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set('typeface', font)
    if baseline:
        rpr.set('baseline', str(baseline))


def _math(par, src, size, color, baseline=0):
    i = 0
    while i < len(src):
        ch = src[i]
        if ch in '_^':
            shift = -25000 if ch == '_' else 30000
            i += 1
            if src[i] == '{':
                depth, j = 1, i + 1
                while depth:
                    depth += {'{': 1, '}': -1}.get(src[j], 0)
                    j += 1
                inner, i = src[i + 1:j - 1], j
            else:
                inner, i = src[i], i + 1
                # keep combining marks with their base letter
                while i < len(src) and '̀' <= src[i] <= 'ͯ':
                    inner += src[i]
                    i += 1
            _math(par, inner, size, color, shift)
            continue
        if ch == '𝟏':            # all-ones vector: upright bold 1, as \one in the paper
            _run(par, '1', size, bold=True, font=MATH_FONT, baseline=baseline, color=color)
            i += 1
            continue
        j = i + 1
        while j < len(src) and '̀' <= src[j] <= 'ͯ':
            j += 1
        tok = src[i:j]
        # Accented letters come from the math font (mark anchors place the
        # accent over the letter), except the overline U+0305, which Times
        # sets above the ascender of h.
        accented = j > i + 1 and '\u0305' not in tok
        if accented:
            tok = MATH_ITALIC.get(ch, ch) + tok[1:]
        font = SYM_FONT if (ch in SYMBOLS or ord(ch) > 0xFFFF or accented) else MATH_FONT
        italic = bool(ITALIC.match(ch)) and font == MATH_FONT   # STIX glyphs are already math italic
        _run(par, tok, size, italic=italic, font=font, baseline=baseline, color=color)
        i = j


MATH_ITALIC = {'h': 'ℎ', 'x': '𝑥', 'z': '𝑧', 'y': '𝑦'}


def write(tf, lines, size=13, color=BLACK, align=PP_ALIGN.CENTER, spacing=1.0):
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
                for part in re.split(r'(\*\*[^*]+\*\*)', seg):
                    if part.startswith('**'):
                        _run(par, part[2:-2], size, bold=True, color=color)
                    elif part:
                        _run(par, part, size, color=color)


def _plain(shape):
    """No theme style: explicit line/fill only, no shadow or glow."""
    style = shape._element.find(qn('p:style'))
    if style is not None:
        shape._element.remove(style)


def box(x, y, w, h, lines=(), fill=WHITE, line=BLACK, lw=1.5, dash=False, size=13,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12, anchor=MSO_ANCHOR.MIDDLE,
        align=PP_ALIGN.CENTER, spacing=1.0, margin=0.04):
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
            etree.SubElement(ln, qn('a:prstDash')).set('val', 'dash')
    _plain(s)
    tf = s.text_frame
    tf.word_wrap = False           # every line break is explicit
    tf.vertical_anchor = anchor
    for side in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, side, Inches(margin))
    if lines:
        write(tf, list(lines), size=size, align=align, spacing=spacing)
    return s


def label(x, y, w, h, lines, size=12, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, color=BLACK):
    tb = shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = anchor
    for side in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, side, 0)
    write(tf, list(lines), size=size, align=align, color=color)
    return tb


def path(points, color=BLACK, lw=1.5, arrow=True):
    """Polyline through points (inches) with an arrowhead at the end."""
    fb = shapes.build_freeform(Inches(points[0][0]), Inches(points[0][1]), scale=1.0)
    fb.add_line_segments([(Inches(px), Inches(py)) for px, py in points[1:]], close=False)
    s = fb.convert_to_shape()
    s.fill.background()
    s.line.color.rgb = color
    s.line.width = Pt(lw)
    _plain(s)
    if arrow:
        ln = s.line._get_or_add_ln()
        tail = etree.SubElement(ln, qn('a:tailEnd'))
        tail.set('type', 'triangle')
        tail.set('w', 'med')
        tail.set('len', 'med')
    return s


def tokens(x, y, n, color, w=0.2, h=0.13, gap=0.05):
    for k in range(n):
        box(x + k * (w + gap), y, w, h, fill=color, line=BLACK, lw=0.75,
            shape=MSO_SHAPE.RECTANGLE)


def curve(x, y, w, h, ys, color=BLACK, lw=1.0):
    """Small sparkline through normalized ys in [0, 1] inside the box (x, y, w, h)."""
    n = len(ys)
    pts = [(x + w * k / (n - 1), y + h * (1 - v)) for k, v in enumerate(ys)]
    path(pts, color=color, lw=lw, arrow=False)


# --------------------------------------------------------------------------
# Layout (inches).  Main path on top, level branch below.
# --------------------------------------------------------------------------
import math

YM = 0.98                      # center line of the main path
BH = 0.92                      # height of the main-path boxes

# 1. Input window with a small multi-cycle series.
IX, IW = 0.08, 1.28
box(IX, YM - BH / 2, IW, BH, fill=WHITE)
label(IX, YM - BH / 2 + 0.05, IW, 0.24, ['**Input Window**'], size=13)
wave = [0.5 + 0.32 * math.sin(2 * math.pi * k / 12) + 0.12 * math.sin(2 * math.pi * k / 4)
        + 0.004 * k for k in range(37)]
lo, hi = min(wave), max(wave)
curve(IX + 0.12, YM - 0.12, IW - 0.24, 0.3, [(v - lo) / (hi - lo) for v in wave])
label(IX, YM + 0.2, IW, 0.24, ['$x ∈ ℝ^L$'], size=13)

# 2. PhaseFormer backbone panel (unchanged): Fig. 3 colors.
PX, PW = 1.62, 3.32
box(PX, 0.12, PW, 1.72, fill=GRAY_PANEL, line=BLACK, lw=1.5, radius=0.06)
label(PX, 0.18, PW, 0.26, ['**PhaseFormer Backbone** (unchanged)'], size=13)
sub_y, sub_h = YM - 0.3, 0.62
x1, w1 = PX + 0.1, 0.98
x2, w2 = x1 + w1 + 0.17, 0.86
x3, w3 = x2 + w2 + 0.17, 0.92
box(x1, sub_y, w1, sub_h, ['Phase', 'Tokenization'], fill=WHITE, size=12, spacing=0.9)
box(x2, sub_y, w2, sub_h, ['Embedding'], fill=PEACH, size=12)
for off in (0.08, 0.04):       # stacked routing layers, as in Fig. 3
    box(x3 + off, sub_y - off, w3, sub_h, fill=WHITE, line=RGBColor(0xB0, 0xB0, 0xB0), lw=0.75,
        radius=0.03)
box(x3, sub_y, w3, sub_h, ['Cross-Phase', 'Routing'], fill=GRAY_BOX, size=12, spacing=0.9, radius=0.03)
label(x3 + w3 - 0.38, sub_y + sub_h + 0.03, 0.45, 0.2, ['$×N$'], size=12)
path([(x1 + w1, sub_y + sub_h / 2), (x2, sub_y + sub_h / 2)])
path([(x2 + w2, sub_y + sub_h / 2), (x3, sub_y + sub_h / 2)])
tokens(x1 + 0.08, sub_y + sub_h + 0.12, 4, TOKEN_GREEN, w=0.17, h=0.11, gap=0.04)
label(x1 + 0.97, sub_y + sub_h + 0.07, 1.0, 0.2, ['phase latents'], size=11, align=PP_ALIGN.LEFT)

# 3. Centered modulation (new).
MX, MW = 5.66, 2.06
box(MX, YM - BH / 2, MW, BH, ['**Centered Modulation**', '$h′_p = h_p + (h_p − h̅) ⊙ q$'],
    fill=BLUE_BOX, size=13, spacing=1.2)

# 4. Shared predictor (unchanged, Fig. 3 yellow).
SX, SW = 8.04, 1.06
box(SX, YM - BH / 2, SW, BH, ['**Shared**', '**Predictor** $Φ$'], fill=YELLOW, size=13, spacing=1.1)

# 5. Level/shape fusion (new).
FX, FW = 9.42, 2.0
box(FX, YM - BH / 2, FW, BH, ['**Level/Shape Fusion**',
                              '$M_c$:  $(1 − g_ℓ)F + g_ℓR$',
                              '$J_c$:  $(1 − g_s)F + g_sR$'],
    fill=BLUE_BOX, size=12.5, spacing=1.05)

# 6. Output.
OX, OW = 11.74, 0.38
box(OX, YM - 0.24, OW, 0.48, ['$ŷ$'], fill=WHITE, size=14)

# Main-path arrows.
path([(IX + IW, YM), (PX, YM)])
path([(PX + PW, YM), (MX, YM)])
label(PX + PW, YM - 0.3, MX - PX - PW, 0.24, ['$h_1, …, h_P$'], size=12)
path([(MX + MW, YM), (SX, YM)])
path([(SX + SW, YM), (FX, YM)])
label(SX + SW, YM - 0.3, FX - SX - SW, 0.24, ['$F$'], size=12)
path([(FX + FW, YM), (OX, YM)])

# --------------------------------------------------------------------------
# Level branch (new): dashed light-blue panel, as the routing panels of Fig. 3.
# --------------------------------------------------------------------------
LY, LH = 2.0, 1.42
LX0, LX1 = 0.42, 11.62
box(LX0, LY, LX1 - LX0, LH, fill=BLUE_PANEL, line=BLACK, lw=1.25, dash=True, radius=0.05)
label(0.86, LY + 0.06, 4.0, 0.26, ['**Level Branch** (rank 3, on cycle means)'], size=13,
      align=PP_ALIGN.LEFT)

YB = LY + 0.84                 # center line of the branch boxes
BBH = 0.94
# pooling box
CX, CW = 1.5, 2.62
box(CX, YB - BBH / 2, CW, BBH, ['**Cycle-Mean Pooling**',
                                '$z̄ ∈ ℝ^K$ from $x̃ − x̃_L 𝟏_L$',
                                '$t = E z̄ + e ∈ ℝ^3$'],
    fill=WHITE, size=12.5, spacing=1.15)
# state t as three blue tokens (rank-3 state)
TX = 4.86
tokens(TX, YB - 0.065, 3, TOKEN_BLUE)
label(TX - 0.15, YB + 0.12, 1.0, 0.22, ['state $t ∈ ℝ^3$'], size=12)
# fixed profiles: title, three profile sketches, readout
RX, RW = 8.0, 3.42
box(RX, YB - BBH / 2, RW, BBH, fill=WHITE)
label(RX, YB - BBH / 2 + 0.06, RW, 0.24, ['**Fixed Profiles** $Ψ_H$'], size=12.5)
for (name, prof, bx) in (('displacement', [0.5] * 9, RX + 0.16),
                         ('tilt', [k / 8 for k in range(9)], RX + 1.48),
                         ('curvature', [1 - ((k - 4) / 4) ** 2 for k in range(9)], RX + 2.2)):
    box(bx, YB - 0.14, 0.34, 0.2, fill=WHITE, line=RGBColor(0x90, 0x90, 0x90), lw=0.5,
        shape=MSO_SHAPE.RECTANGLE)
    curve(bx + 0.03, YB - 0.12, 0.28, 0.16, prof, color=BRANCH, lw=1.25)
    label(bx + 0.38, YB - 0.15, 0.95, 0.22, [name], size=10.5, align=PP_ALIGN.LEFT)
label(RX, YB + 0.17, RW, 0.24, ['$R = x̃_L 𝟏_H + Ψ_H(Nt + n)$'], size=12.5)

# Branch arrows (dark blue).
xin = IX + IW / 2
path([(xin, YM + BH / 2), (xin, YB), (CX, YB)], color=BRANCH)
path([(CX + CW, YB), (TX - 0.06, YB)], color=BRANCH)
xt = TX + 0.75 + 0.03
path([(xt, YB), (RX, YB)], color=BRANCH)
xq = MX + MW / 2
path([(xq, YB), (xq, YM + BH / 2)], color=BRANCH)
label(xq + 0.07, YM + BH / 2 + 0.1, 1.2, 0.24, ['$q = V_2V_1t$'], size=12.5, align=PP_ALIGN.LEFT,
      color=BRANCH)
xr = FX + FW / 2
path([(xr, YB - BBH / 2), (xr, YM + BH / 2)], color=BRANCH)
label(xr + 0.07, YM + BH / 2 + 0.1, 0.4, 0.24, ['$R$'], size=12.5, align=PP_ALIGN.LEFT, color=BRANCH)

prs.save(OUT)
print('wrote', OUT)
