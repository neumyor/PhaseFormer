"""Build the PhaseFormer overview (Fig. 3) as a PowerPoint slide.

A redraw of the conference figure (figures/overview.pdf of the ICLR version)
with the same layout, colors and content, at font sizes that are readable when
the figure is included at \\textwidth (see pptfig.py).  The series sketches are
schematic, as in the original.

    python make_overview.py               # writes phaseformer_overview.pptx next to this file
    ./export.sh phaseformer_overview 495
"""
import math
from pathlib import Path

from pptfig import *  # noqa: F401,F403

OUT = Path(__file__).with_name('phaseformer_overview.pptx')
new_slide(12.2, 3.58)

SERIES_GRAY = RGBColor(0x80, 0x80, 0x80)
DARK_GREEN = RGBColor(0x2E, 0x6B, 0x1F)


def noise(k, amp):
    return amp * (math.sin(2.3 * k) + 0.6 * math.sin(5.1 * k + 1.3))


def plateau(t, k=0):
    """Daily traffic-like profile on t in [0, 1): low night, ramp, noisy plateau, decay."""
    if t < 0.12:
        v = 0.12
    elif t < 0.28:
        v = 0.12 + 0.68 * (t - 0.12) / 0.16
    elif t < 0.78:
        v = 0.8 + 0.06 * math.sin(14 * t) + noise(k, 0.025)
    elif t < 0.95:
        v = 0.8 - 0.68 * (t - 0.78) / 0.17
    else:
        v = 0.12
    return v


def spike(t, k=0):
    """Profile with a morning spike, as in the de-tokenization sketch of Fig. 3."""
    if t < 0.06:
        return 0.15
    if t < 0.12:
        return 0.15 + 0.75 * (t - 0.06) / 0.06
    if t < 0.2:
        return 0.9 - 0.4 * (t - 0.12) / 0.08
    if t < 0.85:
        return 0.48 + 0.04 * math.sin(20 * t) + noise(k, 0.02)
    return 0.48 - 0.33 * min(1, (t - 0.85) / 0.1)


def series_box(x, y, w, h, cycles, prof, dividers=True):
    box(x, y, w, h, fill=WHITE, line=SERIES_GRAY, lw=2, shape=MSO_SHAPE.RECTANGLE)
    n = 40 * cycles
    ys = [prof((k % 40) / 40, k) for k in range(n)]
    curve(x + 0.04, y + 0.05, w - 0.08, h - 0.1, ys, lw=1.0)
    if dividers:
        for c in range(1, cycles):
            xd = x + w * c / cycles
            path([(xd, y + 0.03), (xd, y + h - 0.03)], color=SERIES_GRAY, lw=1.5, arrow=False,
                 dash='sysDash')


def phase_columns(x, y, w, h, ncol, prof, row_t, gap=0.03):
    """Phase-period matrix: one cycle per column drawn vertically; returns the
    x positions where the highlighted phase row crosses each column's curve."""
    hits = []
    for c in range(ncol):
        cx = x + c * (w + gap)
        box(cx, y, w, h, fill=WHITE, line=SERIES_GRAY, lw=2, shape=MSO_SHAPE.RECTANGLE)
        pts = []
        for k in range(41):
            t = k / 40
            v = prof(t, 7 * c + k)
            pts.append((cx + 0.02 + (w - 0.04) * v, y + 0.03 + (h - 0.06) * t))
        path(pts, color=BLACK, lw=1.0, arrow=False)
        v = prof(row_t, 7 * c + round(row_t * 40))
        hits.append(cx + 0.02 + (w - 0.04) * v)
    return hits


def phase_row(x0, x1, yc, hits, h=0.1):
    """Highlighted phase row with circles where it crosses the columns."""
    box(x0, yc - h / 2, x1 - x0, h, fill=TOKEN_LIGHT, line=DARK_GREEN, lw=1.5, shape=MSO_SHAPE.RECTANGLE)
    for hx in hits:
        r = 0.055
        box(hx - r, yc - r, 2 * r, 2 * r, fill=TOKEN_LIGHT, line=DARK_GREEN, lw=1.5, shape=MSO_SHAPE.OVAL)


def token_stack(x, y, w, h, n, dark, gap=0.04):
    for k in range(n):
        box(x, y + k * (h + gap), w, h, fill=TOKEN_GREEN if k == dark else TOKEN_LIGHT,
            line=DARK_GREEN if k == dark else RGBColor(0x8F, 0xB7, 0x7B), lw=1.25,
            shape=MSO_SHAPE.RECTANGLE)
    return y + dark * (h + gap) + h / 2


def chip(cx, cy, w=0.62, h=0.26):
    """Router icon: a flat chip in perspective (top face and two sides)."""
    t = 0.08
    top = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy), (cx, cy - h / 2)]
    left = [(cx - w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx - w / 2, cy + t), (cx - w / 2, cy)]
    right = [(cx + w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx + w / 2, cy + t), (cx + w / 2, cy)]
    path(left, color=RGBColor(0x3B, 0x4C, 0x9A), lw=0.75, arrow=False, fill=RGBColor(0x4E, 0x62, 0xB8), closed=True)
    path(right, color=RGBColor(0x3B, 0x4C, 0x9A), lw=0.75, arrow=False, fill=RGBColor(0x3B, 0x4C, 0x9A), closed=True)
    path(top, color=RGBColor(0x3B, 0x4C, 0x9A), lw=0.75, arrow=False, fill=RGBColor(0x8C, 0x9E, 0xE6), closed=True)
    inner = [(cx, cy - h / 4), (cx + w / 4, cy), (cx, cy + h / 4), (cx - w / 4, cy), (cx, cy - h / 4)]
    path(inner, color=RGBColor(0x6A, 0x7D, 0xD0), lw=0.75, arrow=False, fill=RGBColor(0x7A, 0x8D, 0xDE), closed=True)


def down(x, y0, y1, **kw):
    path([(x, y0), (x, y1)], **kw)


def token_bar(cx, y, n, color, w=0.13, h=0.15):
    """Contiguous small tokens (keys/values/queries) centered at cx."""
    x = cx - n * w / 2
    for k in range(n):
        box(x + k * w, y, w, h, fill=color, line=RGBColor(0x1F, 0x3A, 0x5F) if color == TOKEN_BLUE
            else DARK_GREEN, lw=1.25, shape=MSO_SHAPE.RECTANGLE)
    return x


def token_row(cx, y, n, color, w=0.36, h=0.13, gap=0.1):
    x = cx - (n * w + (n - 1) * gap) / 2
    tokens(x, y, n, color, w=w, h=h, gap=gap, lw=1.0)


# --------------------------------------------------------------------------
# Left column: tokenization, PhaseFormer, de-tokenization.
# --------------------------------------------------------------------------
BRACE = RGBColor(0x2F, 0x5D, 0x7C)
LX, LW = 0.04, 2.76
box(LX, 0.04, LW, 1.46, fill=WHITE, shape=MSO_SHAPE.RECTANGLE)
label(LX + 0.07, 0.06, 2.3, 0.3, ['**Phase Tokenization**'], size=TITLE, align=PP_ALIGN.LEFT)
label(LX + 0.03, 0.78, 0.24, 0.3, ['$\\b{X}$'], size=TITLE)
series_box(0.32, 0.74, 1.0, 0.38, 3, plateau)
brace(0.32 + 2 / 3, 1.32, 0.71, depth=0.05, up=True, color=BRACE)
label(0.86, 0.38, 0.6, 0.24, ['$L_{\\r{phase}}$'])
brace(0.32, 1.32, 1.15, depth=0.05, up=False, color=BRACE)
label(0.6, 1.22, 0.44, 0.24, ['$L_{\\r{in}}$'])
block_arrow(1.37, 0.86, 0.18, 0.14)
hits = phase_columns(1.6, 0.7, 0.18, 0.56, 3, plateau, 0.6)
row_y = 0.7 + 0.03 + 0.5 * 0.6
brace(1.6, 2.2, 0.67, depth=0.05, up=True, color=BRACE)
label(1.72, 0.36, 0.36, 0.24, ['$P_{\\r{in}}$'])
phase_row(1.57, 2.23, row_y, hits)
label(2.22, 0.36, 0.56, 0.28, ['$\\b{X}_{\\r{phase}}$'])
yc = token_stack(2.33, row_y - 0.32, 0.38, 0.1, 4, 2, gap=0.035)
path([(2.23, row_y), (2.33, row_y)], color=DARK_GREEN, lw=1.5)

path([(LX + LW / 2, 1.5), (LX + LW / 2, 1.62)])
box(LX, 1.62, LW, 0.38, ['**Phase Pathway**'], fill=GRAY_BOX, line=PURPLE, lw=1.75, size=22,
    shape=MSO_SHAPE.RECTANGLE)
path([(LX + LW / 2, 2.0), (LX + LW / 2, 2.12)])

box(LX, 2.12, LW, 1.42, fill=WHITE, shape=MSO_SHAPE.RECTANGLE)
label(LX + 0.07, 2.14, 2.6, 0.3, ['**Phase De-Tokenization**'], size=TITLE, align=PP_ALIGN.LEFT)
label(LX + 0.01, 2.88, 0.56, 0.3, ['$\\b{Y}_{\\r{phase}}$'])
hits = phase_columns(1.06, 2.76, 0.18, 0.56, 2, spike, 0.62)
row_y = 2.76 + 0.03 + 0.5 * 0.62
yc = token_stack(0.62, row_y - 0.32, 0.32, 0.1, 4, 2, gap=0.035)
phase_row(1.03, 1.48, row_y, hits)
path([(0.94, row_y), (1.03, row_y)], color=DARK_GREEN, lw=1.5)
brace(1.06, 1.45, 2.73, depth=0.05, up=True, color=BRACE)
label(1.03, 2.44, 0.46, 0.24, ['$P_{\\r{out}}$'])
block_arrow(1.52, 2.96, 0.18, 0.14)
series_box(1.74, 2.84, 0.78, 0.38, 2, spike)
brace(2.13, 2.52, 2.81, depth=0.05, up=True, color=BRACE)
label(2.03, 2.48, 0.6, 0.24, ['$L_{\\r{phase}}$'])
brace(1.74, 2.52, 3.25, depth=0.05, up=False, color=BRACE)
label(1.9, 3.24, 0.46, 0.24, ['$L_{\\r{out}}$'])
label(2.55, 2.88, 0.22, 0.3, ['$\\b{Y}$'], size=TITLE)

# --------------------------------------------------------------------------
# Middle column: the PhaseFormer stack.
# --------------------------------------------------------------------------
MX0, MX1 = 2.9, 4.64
box(MX0, 0.04, MX1 - MX0, 3.5, fill=WHITE, shape=MSO_SHAPE.RECTANGLE)
mc = 3.7
down(mc, 0.04, 0.22)
box(mc - 0.66, 0.22, 1.32, 0.38, ['Embedding'], fill=PEACH)
SX0, SX1, SY0, SY1 = 2.97, 4.43, 0.84, 2.52
for off in (0.12, 0.06):
    box(SX0 + off, SY0 - off, SX1 - SX0, SY1 - SY0, fill=WHITE, line=GRAY_LINE, lw=0.75,
        shape=MSO_SHAPE.RECTANGLE)
down(mc, 0.6, SY0)
box(SX0, SY0, SX1 - SX0, SY1 - SY0, fill=GRAY_PANEL, lw=1.0, shape=MSO_SHAPE.RECTANGLE)
label(SX0, SY0 + 0.05, SX1 - SX0, 0.5, ['**Cross-Phase**', '**Routing Layer**'], size=TITLE, spacing=0.92)
box(mc - 0.63, 1.46, 1.26, 0.42, ['Aggregation'], fill=BLUE_PANEL, lw=1.25, dash=True)
box(mc - 0.63, 2.0, 1.26, 0.42, ['Distribution'], fill=GREEN_PANEL, lw=1.25, dash=True)
label(4.17, 2.52, 0.45, 0.26, ['$×N_{r}$'], align=PP_ALIGN.LEFT)
down(mc, SY1, 2.82)
box(mc - 0.66, 2.82, 1.32, 0.38, ['Predictor'], fill=YELLOW)
down(mc, 3.2, 3.54)

# Zoom lines.
RX0 = 4.76
for (a, b) in (((LX + LW, 1.62), (MX0, 0.04)), ((LX + LW, 2.0), (MX0, 3.54)),
               ((SX1 + 0.12, SY0 - 0.12), (RX0, 0.04)), ((SX1 + 0.12, SY1 - 0.12), (RX0, 3.54))):
    path([a, b], lw=2.0, arrow=False, dash='sysDot')

# --------------------------------------------------------------------------
# Right panel: the two steps of a cross-phase routing layer.
# --------------------------------------------------------------------------
box(RX0, 0.04, 12.16 - RX0, 3.5, fill=GRAY_PANEL, shape=MSO_SHAPE.RECTANGLE)


def step_panel(x0, x1, title, fill):
    box(x0, 0.12, x1 - x0, 3.34, fill=fill, lw=1.25, dash=True, shape=MSO_SHAPE.RECTANGLE)
    label(x0 + 0.1, 0.15, x1 - x0 - 0.2, 0.3, [f'**{title}**'], size=TITLE, align=PP_ALIGN.LEFT)


def wbox(cx, sub, sup):
    box(cx - 0.4, 1.3, 0.8, 0.46, [f'$\\b{{W}}_{sub}^{{\\r{{{sup}}}}}$'], size=TITLE)


YT, YB_ = 1.96, 2.11           # token bars
YM0, YM1 = 2.29, 2.67          # multi-head attention
YO0, YO1 = 2.86, 3.36          # output box

# Phase-to-Router aggregation.
A0, A1 = 4.86, 8.36
step_panel(A0, A1, 'Phase-to-Router Aggregation', BLUE_PANEL)
box(5.0, 0.52, 1.96, 0.58, fill=WHITE)
token_row(5.98, 0.59, 4, TOKEN_GREEN)
label(5.0, 0.76, 1.96, 0.3, ['Phase Tokens $\\b{Z̃}$'])
kx, vx, qx = 5.52, 6.44, 7.7
chip(qx, 0.66)
label(qx - 0.6, 0.82, 1.2, 0.28, ['__Routers__ **R**'])
for cx, sub in ((kx, 'K'), (vx, 'V'), (qx, 'Q')):
    wbox(cx, sub, 'agg')
down(kx, 1.1, 1.3)
down(vx, 1.1, 1.3)
down(qx, 1.12, 1.3)
for cx, n, col, name in ((kx, 4, TOKEN_GREEN, '$K_z$'), (vx, 4, TOKEN_GREEN, '$V_z$'),
                         (qx, 3, TOKEN_BLUE, '$Q_r$')):
    down(cx, 1.76, YT)
    x = token_bar(cx, YT, n, col)
    label(x - 0.38, YT - 0.07, 0.34, 0.28, [name], align=PP_ALIGN.RIGHT)
    down(cx, YB_, YM0)
box(5.0, YM0, 3.22, YM1 - YM0, ['Multi-Head Attention'])
down(6.61, YM1, YO0)
box(5.0, YO0, 3.22, YO1 - YO0, fill=WHITE)
token_row(6.61, YO0 + 0.06, 3, TOKEN_BLUE)
label(5.0, YO0 + 0.21, 3.22, 0.3, ['Contextualized Routers $\\b{R}′$'])

# Router-to-Phase distribution.
D0, D1 = 8.48, 12.08
step_panel(D0, D1, 'Router-to-Phase Distribution', GREEN_PANEL)
box(8.62, 0.52, 1.82, 0.58, fill=WHITE)
token_row(9.53, 0.59, 3, TOKEN_BLUE)
label(8.62, 0.76, 1.82, 0.3, ['Contextualized $\\b{R}′$'])
box(10.56, 0.52, 1.38, 0.58, fill=WHITE)
token_row(11.25, 0.59, 4, TOKEN_GREEN, w=0.24, gap=0.08)
label(10.56, 0.76, 1.38, 0.3, ['Phase Tokens $\\b{Z̃}$'])
kx, vx, qx = 9.08, 9.98, 11.25
for cx, sub in ((kx, 'K'), (vx, 'V'), (qx, 'Q')):
    wbox(cx, sub, 'dist')
for cx in (kx, vx, qx):
    down(cx, 1.1, 1.3)
for cx, n, col, name in ((kx, 3, TOKEN_BLUE, '$K_r$'), (vx, 3, TOKEN_BLUE, '$V_r$'),
                         (qx, 4, TOKEN_GREEN, '$Q_z$')):
    down(cx, 1.76, YT)
    x = token_bar(cx, YT, n, col)
    label(x - 0.38, YT - 0.07, 0.34, 0.28, [name], align=PP_ALIGN.RIGHT)
    down(cx, YB_, YM0)
box(8.62, YM0, 3.32, YM1 - YM0, ['Multi-Head Attention'])
down(10.28, YM1, YO0)
box(8.62, YO0, 3.32, YO1 - YO0, fill=WHITE)
token_row(10.28, YO0 + 0.06, 4, TOKEN_GREEN)
label(8.62, YO0 + 0.21, 3.32, 0.3, ['Enhanced Phase Tokens $\\b{Z}_{\\r{attn}}$'])

# Curved hand-over of H from aggregation to distribution.
p0, p1, p2, p3 = (8.22, 3.1), (8.56, 3.04), (8.3, 0.8), (8.62, 0.7)
bez = []
for k in range(25):
    t = k / 24
    bez.append(tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d
                     for a, b, c, d in zip(p0, p1, p2, p3)))
path(bez, lw=1.5)

save(OUT)
