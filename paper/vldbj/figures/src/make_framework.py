"""PhaseFormer+ framework (Fig. 5): the three token pathways in one figure.

Merges the former Figs. 5 (la_overview) and 6 (phaseformer_overview) into one
slide in the same style: Times New Roman, rounded boxes with black outlines,
dashed panels with light fills, green/blue token bars and black arrows, the
conference colors for embedding (peach), routing (gray) and predictor (yellow).

Layout: three parallel bands, one per token, and a gated fusion on the right.
  phase band (green):     input window -> fold into phase tokens -> embedding
                          -> N_r cross-phase routing layers (phase tokens ->
                          a few routers -> phase tokens) -> (.) -> shared
                          predictor and de-tokenization -> phase forecast F
  amplitude band (orange): amplitude token q -> centered modulation, which
                          acts on the final phase latents at (.)
  level band (blue):      level token -> state t -> fixed profiles -> F_b;
                          the same state t gives the amplitude token q
  fusion: gate g_l on the cycle levels, gate g_s on the within-cycle shape.
The slide is 12.2 x 4.56 in (aspect ratio about 2.7); it is included at \\textwidth.

    python3 make_framework.py            # writes framework.pptx next to this file
    sh export.sh framework 495
"""
import math
from pathlib import Path

import pptfig
from pptfig import *  # noqa: F401,F403

pptfig.SUBSCALE = 1.35          # sub/superscripts print at about 6.7pt

OUT = Path(__file__).with_name('framework.pptx')
W, H = 12.2, 4.56
new_slide(W, H)

SERIES_GRAY = RGBColor(0x80, 0x80, 0x80)
DARK_GREEN = RGBColor(0x2E, 0x6B, 0x1F)
LIGHT_LINE = RGBColor(0x8F, 0xB7, 0x7B)
CHIP_EDGE = RGBColor(0x3B, 0x4C, 0x9A)
AMP_PANEL = RGBColor(0xFE, 0xF0, 0xDC)      # light orange: amplitude pathway band
AMP_BOX = RGBColor(0xFB, 0xD5, 0xA5)
AMP_LINE = RGBColor(0xC5, 0x5A, 0x11)       # arrows of the amplitude token q


def daily(t, k=0):
    """Daily traffic-like profile on t in [0, 1): night, morning and evening peaks."""
    base = 0.15 + 0.55 * math.exp(-((t - 0.33) / 0.07) ** 2) + 0.45 * math.exp(-((t - 0.72) / 0.09) ** 2)
    return base + 0.03 * math.sin(7.3 * k)


def chip(cx, cy, w=0.42, h=0.18):
    """Router icon (as in the conference figure): a flat chip in perspective."""
    t = 0.06
    top = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy), (cx, cy - h / 2)]
    left = [(cx - w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx - w / 2, cy + t), (cx - w / 2, cy)]
    right = [(cx + w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx + w / 2, cy + t), (cx + w / 2, cy)]
    path(left, color=CHIP_EDGE, lw=0.75, arrow=False, fill=RGBColor(0x4E, 0x62, 0xB8), closed=True)
    path(right, color=CHIP_EDGE, lw=0.75, arrow=False, fill=CHIP_EDGE, closed=True)
    path(top, color=CHIP_EDGE, lw=0.75, arrow=False, fill=RGBColor(0x8C, 0x9E, 0xE6), closed=True)


def column_tokens(x, y, n, w=0.13, h=0.5, gap=0.05, dark=None):
    """Phase tokens drawn as vertical green bars (columns of the fold)."""
    for k in range(n):
        box(x + k * (w + gap), y, w, h, fill=TOKEN_GREEN if k == dark else TOKEN_LIGHT,
            line=DARK_GREEN if k == dark else LIGHT_LINE, lw=1.0, shape=MSO_SHAPE.RECTANGLE)
    return x + n * (w + gap) - gap


def arrow(x0, y0, x1, y1, **kw):
    path([(x0, y0), (x1, y1)], **kw)



def header(y0, title, subtitle, color):
    label(1.62, y0 + 0.03, 7.0, 0.3, [f'**{title}**'], align=PP_ALIGN.LEFT)
    label(3.62, y0 + 0.05, 5.0, 0.28, [subtitle], size=SMALL,
          align=PP_ALIGN.LEFT, color=color)


# Band geometry (inches).
PB0, PB1, YP = 0.05, 1.92, 1.17          # phase band and its content line
AB0, AB1, YA = 2.04, 2.98, 2.62          # amplitude band
LB0, LB1, YL = 3.1, 4.5, 3.96            # level band
BX0, BX1 = 1.5, 8.86                     # common left and right edges of the bands

for y0, y1, fill in ((PB0, PB1, GREEN_PANEL), (AB0, AB1, AMP_PANEL), (LB0, LB1, BLUE_PANEL)):
    box(BX0, y0, BX1 - BX0, y1 - y0, fill=fill, lw=1.25, dash=True, radius=0.06)
header(PB0, 'Phase Pathway', 'phase tokens → shape', DARK_GREEN)
header(AB0, 'Amplitude Pathway', 'amplitude token → amplitude', AMP_LINE)
header(LB0, 'Level Pathway', 'level token → level', BRANCH)

# ---------------------------------------------------------------- input window
IX, IW, IH = 0.02, 1.44, 0.9
box(IX, YP - IH / 2, IW, IH)
label(IX, YP - IH / 2 + 0.03, IW, 0.3, ['**Input Window**'])
ys = [daily((k % 24) / 24, k) + 0.12 * (k // 24) / 3 for k in range(72)]
lo, hi = min(ys), max(ys)
curve(IX + 0.1, YP - 0.06, IW - 0.2, 0.3, [(v - lo) / (hi - lo) for v in ys])
for c in (1, 2):
    xd = IX + 0.1 + (IW - 0.2) * c / 3
    path([(xd, YP - 0.08), (xd, YP + 0.26)], color=SERIES_GRAY, lw=1.0, arrow=False, dash='sysDash')

# ---------------------------------------------------------------- phase band
FW, FH = 0.84, 0.84
FX, FY = 1.68, YP - FH / 2
box(FX, FY, FW, FH, fill=WHITE, line=SERIES_GRAY, lw=1.25, shape=MSO_SHAPE.RECTANGLE)
for r in range(3):
    ry = FY + 0.07 + r * (FH - 0.14) / 3
    pts = [(FX + 0.05 + (FW - 0.1) * k / 23, ry + 0.2 - 0.18 * (daily(k / 24, k + 24 * r) - 0.1))
           for k in range(24)]
    path(pts, color=BLACK, lw=0.9, arrow=False)
hx = FX + 0.05 + (FW - 0.1) * 8 / 23
box(hx - 0.045, FY + 0.02, 0.09, FH - 0.04, fill=None, line=DARK_GREEN, lw=1.75,
    shape=MSO_SHAPE.RECTANGLE)
label(FX - 0.12, FY + FH + 0.01, FW + 0.24, 0.26, ['fold by cycle'], size=SMALL)

EX, EW, EH = 2.72, 0.32, 1.24
box(EX, YP - EH / 2, EW, EH, fill=PEACH, radius=0.08)
lb = label(EX - 0.6, YP - 0.15, EW + 1.2, 0.3, ['Embedding'])
lb.rotation = -90

RX0, RX1 = 3.24, 6.0
RY0, RY1 = YP - EH / 2, YP + EH / 2
for off in (0.1, 0.05):
    box(RX0 + off, RY0 - off, RX1 - RX0, RY1 - RY0, fill=WHITE, line=GRAY_LINE, lw=0.75,
        shape=MSO_SHAPE.RECTANGLE)
box(RX0, RY0, RX1 - RX0, RY1 - RY0, fill=GRAY_PANEL, lw=1.0, shape=MSO_SHAPE.RECTANGLE)
label(RX0, RY0 + 0.03, RX1 - RX0, 0.3, ['**Cross-Phase Routing**'])
BW, BG, BH_ = 0.1, 0.05, 0.5
c0x = RX0 + 0.14
c0 = column_tokens(c0x, YP - BH_ / 2 + 0.04, 3, w=BW, h=BH_, gap=BG)
c1x = RX1 - 0.14 - (3 * BW + 2 * BG)
column_tokens(c1x, YP - BH_ / 2 + 0.04, 3, w=BW, h=BH_, gap=BG)
cx = (RX0 + RX1) / 2
for yy in (YP - 0.16, YP + 0.04, YP + 0.24):
    chip(cx, yy, w=0.36, h=0.15)
yc = YP + 0.04
arrow(c0 + 0.06, yc, cx - 0.24, yc)
arrow(cx + 0.24, yc, c1x - 0.06, yc)
label(c0, yc - 0.33, cx - 0.2 - c0, 0.26, ['gather'], size=SMALL)
label(cx + 0.2, yc - 0.33, c1x - cx - 0.2, 0.26, ['spread'], size=SMALL)
label(cx - 0.7, YP + 0.36, 1.4, 0.26, ['a few routers'], size=SMALL)
label(RX1 - 0.66, RY1 - 0.38, 0.56, 0.3, ['$×N_{r}$'], align=PP_ALIGN.RIGHT)

NX, ND = 6.7, 0.32                       # modulation node where q acts on h_p
box(NX - ND / 2, YP - ND / 2, ND, ND, ['$⊙$'], fill=AMP_BOX, lw=1.25, shape=MSO_SHAPE.OVAL,
    margin=0)
label(RX1 + 0.1, YP - 0.36, NX - ND / 2 - RX1 - 0.1, 0.3, ['$h_p$'])

PRX, PRW, PRH = 7.26, 1.44, 0.92
box(PRX, YP - PRH / 2, PRW, PRH, fill=YELLOW)
label(PRX, YP - PRH / 2 + 0.06, PRW, 0.56, ['**Shared**', '**Predictor** $Φ$'], spacing=0.95)
label(PRX, YP + 0.14, PRW, 0.28, ['+ de-tokenize'], size=SMALL)

arrow(IX + IW, YP, FX, YP)
arrow(FX + FW, YP, EX, YP)
arrow(EX + EW, YP, RX0, YP)
arrow(RX1 + 0.1, YP, NX - ND / 2, YP)
arrow(NX + ND / 2, YP, PRX, YP)

# ---------------------------------------------------------------- level band
LTX, LTW, LTH = 1.66, 1.86, 0.92
box(LTX, YL - LTH / 2, LTW, LTH, fill=WHITE)
label(LTX, YL - LTH / 2 + 0.03, LTW, 0.3, ['**Level Token**'])
for k, v in enumerate((0.35, 0.5, 0.42, 0.62, 0.72)):
    bx = LTX + 0.38 + k * 0.23
    box(bx, YL + 0.16 - 0.36 * v, 0.16, 0.36 * v, fill=BLUE_BOX, line=BRANCH, lw=1.0,
        shape=MSO_SHAPE.RECTANGLE)
label(LTX, YL + 0.17, LTW, 0.26, ['cycle means − last value'], size=SMALL, color=SERIES_GRAY)
xin = 0.71
path([(xin, YP + IH / 2), (xin, YL), (LTX, YL)], color=BRANCH)

TSX, TSW, TSH = 4.52, 1.22, 0.6
box(TSX, YL - TSH / 2, TSW, TSH, ['state $t ∈ ℝ^3$'], fill=WHITE)
arrow(LTX + LTW, YL, TSX, YL, color=BRANCH)

PFX, PFW, PFH = 6.12, 2.58, 0.92
box(PFX, YL - PFH / 2, PFW, PFH, fill=WHITE)
label(PFX, YL - PFH / 2 + 0.03, PFW, 0.3, ['**Fixed Profiles** $Ψ_{H}$'])
for k, (name, prof) in enumerate((('offset', [0.5] * 9), ('tilt', [i / 8 for i in range(9)]),
                                 ('curvature', [1 - ((i - 4) / 4) ** 2 for i in range(9)]))):
    icx = PFX + PFW * (k + 0.5) / 3
    box(icx - 0.2, YL - 0.08, 0.4, 0.24, fill=WHITE, line=RGBColor(0x90, 0x90, 0x90), lw=0.75,
        shape=MSO_SHAPE.RECTANGLE)
    curve(icx - 0.16, YL - 0.055, 0.32, 0.19, prof, color=BRANCH, lw=1.25)
    label(icx - 0.5, YL + 0.17, 1.0, 0.26, [name], size=SMALL)
arrow(TSX + TSW, YL, PFX, YL, color=BRANCH)

# ---------------------------------------------------------------- amplitude band
# The amplitude token q is computed from the level state t (riser), then scales the
# within-cycle part of the final phase latents at the node on the phase line.
QX = TSX + TSW / 2
qx0 = QX - 0.245
path([(QX, YL - TSH / 2), (QX, YA + 0.07)], color=AMP_LINE)
for k in range(3):
    box(qx0 + k * 0.18, YA - 0.07, 0.13, 0.14, fill=AMP_BOX, line=AMP_LINE, lw=1.0,
        shape=MSO_SHAPE.RECTANGLE)
label(qx0 - 0.4, YA - 0.15, 0.32, 0.3, ['$q$'], color=AMP_LINE, align=PP_ALIGN.RIGHT)
path([(qx0 + 0.49, YA), (NX, YA), (NX, YP + ND / 2)], color=AMP_LINE)
label(NX + 0.08, YA - 0.15, BX1 - NX - 0.1, 0.3, ['$h′_p = h_p + (h_p − h̅) ⊙ q$'],
      align=PP_ALIGN.LEFT, color=AMP_LINE)
label(NX + ND / 2, YP - 0.36, PRX - NX - ND / 2, 0.3, ['$h′_p$'])

# ---------------------------------------------------------------- fusion and output
GX, GW = 9.12, 2.52
GY, GH = AB0 - 0.14, AB1 - AB0 + 0.28
box(GX, GY, GW, GH, fill=WHITE)
YF = GY + GH / 2
label(GX, GY + 0.04, GW, 0.3, ['**Gated Fusion**'])
label(GX, YF - 0.12, GW, 0.3, ['level:  $(1 − g_ℓ)F + g_ℓF_{\\r{b}}$'])
label(GX, YF + 0.2, GW, 0.3, ['shape:  $(1 − g_s)F + g_sF_{\\r{b}}$'])
XR = GX + GW / 2
path([(PRX + PRW, YP), (XR, YP), (XR, GY)])
label(BX1 + 0.1, YP - 0.36, 0.5, 0.3, ['$F$'], align=PP_ALIGN.LEFT)
path([(PFX + PFW, YL), (XR, YL), (XR, GY + GH)], color=BRANCH)
label(BX1 + 0.1, YL - 0.36, 0.5, 0.3, ['$F_{\\r{b}}$'], align=PP_ALIGN.LEFT, color=BRANCH)
OX, OW = 11.82, 0.36
box(OX, YF - 0.23, OW, 0.46, ['$y\u0302$'], fill=WHITE)
arrow(GX + GW, YF, OX, YF)

save(OUT)
