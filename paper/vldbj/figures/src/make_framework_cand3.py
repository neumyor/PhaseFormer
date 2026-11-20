"""Framework figure of PhaseFormer+, candidate V3: tensor-centric data flow.

The figure follows the tensors rather than the modules: every intermediate
tensor is a small token stack or grid with its shape next to it, and every
module is a thin labeled box between two tensors.  One left-to-right flow
splits after the shared embedding into the phase pathway (top row) and the
trend pathway (bottom row) and joins again in the gated fusion.
Style of the conference figure (make_overview.py): Times, white modules with
black outlines, dashed light bands.  Colors follow Fig. 3: phase tokens blue,
level token green, routers indigo.  Symbols follow body/method.tex.

    python3 make_framework_cand3.py && ./export.sh framework_cand3 516
"""
import math
from pathlib import Path

from pptfig import *  # noqa: F401,F403
import pptfig

pptfig.SYMBOLS |= set('ΦΨΓℓ×′−')

OUT = Path(__file__).with_name('framework_cand3.pptx')
W, H = 12.2, 3.86
new_slide(W, H)

TITLE, MOD, SM = 16, 14, 12.5
INK = RGBColor(0x26, 0x26, 0x26)
EDGE = RGBColor(0x8C, 0x8C, 0x8C)          # band outlines
SERIES = RGBColor(0x40, 0x40, 0x40)
PH_ACC = RGBColor(0x15, 0x6F, 0xA8)        # phase pathway: arrows and labels
PH_LIGHT = RGBColor(0xB9, 0xDF, 0xF3)
TR_ACC = RGBColor(0x2E, 0x6B, 0x1F)        # trend pathway
RT = RGBColor(0x6A, 0x7D, 0xD0)            # router tokens
RT_LINE = RGBColor(0x3B, 0x4C, 0x9A)
FAN = RGBColor(0xA8, 0xB4, 0xC8)
FU_ACC = RGBColor(0xC0, 0x39, 0x2B)
FUSE_BAND = RGBColor(0xFD, 0xF0, 0xEA)
NEUTRAL_BAND = RGBColor(0xF7, 0xF7, 0xF7)


def band(x, y, w, h, fill, title=None, color=INK):
    box(x, y, w, h, fill=fill, line=EDGE, lw=1.0, dash=True, shape=MSO_SHAPE.RECTANGLE)
    if title:
        label(x + 0.08, y + 0.03, w - 0.16, 0.3, [f'**{title}**'], size=TITLE, align=PP_ALIGN.LEFT,
              color=color)


def module(x, y, w, h, lines, fill=WHITE, size=MOD):
    return box(x, y, w, h, list(lines), fill=fill, line=INK, lw=1.25, size=size, radius=0.08,
               spacing=0.92, color=INK)


def text(x, y, w, h, lines, size=SM, color=INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    return label(x, y, w, h, list(lines), size=size, color=color, align=align, anchor=anchor,
                 spacing=0.92)


def arrow(points, color=INK, lw=1.25, head=True, dash=None):
    path(points, color=color, lw=lw, arrow=head, dash=dash)


def stack(x, yc, n, fill, line, w=0.28, h=0.085, gap=0.033, hi=None, hi_fill=None):
    """n horizontal token bars stacked vertically, centered at yc; returns (top, bottom)."""
    tot = n * h + (n - 1) * gap
    y0 = yc - tot / 2
    for k in range(n):
        f = hi_fill if (hi is not None and k == hi) else fill
        box(x, y0 + k * (h + gap), w, h, fill=f, line=line, lw=0.9, shape=MSO_SHAPE.RECTANGLE)
    return y0, y0 + tot, [y0 + k * (h + gap) + h / 2 for k in range(n)]


def chip(cx, cy, w=0.5, h=0.21):
    """Router icon of the conference figure."""
    t = 0.07
    top = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy), (cx, cy - h / 2)]
    left = [(cx - w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx - w / 2, cy + t), (cx - w / 2, cy)]
    right = [(cx + w / 2, cy), (cx, cy + h / 2), (cx, cy + h / 2 + t), (cx + w / 2, cy + t), (cx + w / 2, cy)]
    path(left, color=RT_LINE, lw=0.75, arrow=False, fill=RGBColor(0x4E, 0x62, 0xB8), closed=True)
    path(right, color=RT_LINE, lw=0.75, arrow=False, fill=RT_LINE, closed=True)
    path(top, color=RT_LINE, lw=0.75, arrow=False, fill=RGBColor(0x8C, 0x9E, 0xE6), closed=True)
    inner = [(cx, cy - h / 4), (cx + w / 4, cy), (cx, cy + h / 4), (cx - w / 4, cy), (cx, cy - h / 4)]
    path(inner, color=RGBColor(0x6A, 0x7D, 0xD0), lw=0.75, arrow=False, fill=RGBColor(0x7A, 0x8D, 0xDE),
         closed=True)


def daily(u):
    return 0.55 * math.exp(-((u - 0.32) / 0.08) ** 2) + 0.75 * math.exp(-((u - 0.70) / 0.1) ** 2)


def series(x, y, w, h, cycles, color, drift=0.0, lw=1.0, samples=24):
    ys = [daily((k / samples) % 1.0) + drift * k / (cycles * samples) for k in range(cycles * samples + 1)]
    lo, hi = min(ys), max(ys)
    curve(x, y, w, h, [(v - lo) / (hi - lo) for v in ys], color=color, lw=lw)


def steps(x, y, w, h, levels, color, lw=1.5):
    n = len(levels)
    pts = []
    for k, v in enumerate(levels):
        pts += [(x + w * k / n, y + h * (1 - v)), (x + w * (k + 1) / n, y + h * (1 - v))]
    path(pts, color=color, lw=lw, arrow=False)


def profile(x, y, w, h, kind, color):
    pts = []
    for k in range(21):
        u = k / 20
        v = {'offset': 0.5, 'tilt': 0.1 + 0.8 * u, 'curve': 0.95 - 3.4 * (u - 0.5) ** 2}[kind]
        pts.append((x + w * u, y + h * (1 - v)))
    path(pts, color=color, lw=1.4, arrow=False)


def operator(x, y, r=0.11, color=INK):
    box(x - r, y - r, 2 * r, 2 * r, fill=WHITE, line=color, lw=1.25, shape=MSO_SHAPE.OVAL)
    a = r * 0.55
    path([(x - a, y), (x + a, y)], color=color, lw=1.25, arrow=False)
    path([(x, y - a), (x, y + a)], color=color, lw=1.25, arrow=False)


def fan(src, dst, x0, x1, color=FAN):
    for ys in src:
        for yd in dst:
            path([(x0, ys), (x1, yd)], color=color, lw=0.6, arrow=False)


# ---------------------------------------------------------------- geometry
P, K, M, KP = 5, 4, 3, 2
YP, YT = 1.30, 3.03                    # rows of the phase and the trend pathway
TOP, MID, BOT = 0.04, 2.13, H - 0.04   # band edges
IN_X1 = 1.72
EMB_X0, EMB_X1 = 1.8, 2.74
BAND_X0 = 2.82
FU_X0 = 9.92
BAND_X1 = FU_X0 - 0.08
TW = 0.28                              # width of a token bar

# ---------------------------------------------------------------- folding (input)
band(0.04, TOP, IN_X1 - 0.04, BOT - TOP, NEUTRAL_BAND, 'Folding')
text(0.8, 0.08, IN_X1 - 0.86, 0.28, ['$x∈ℝ^L$'], size=SM, align=PP_ALIGN.RIGHT)
series(0.2, 0.42, 1.4, 0.32, K, SERIES, drift=0.6)
for k in range(1, K):
    xd = 0.2 + 1.4 * k / K
    path([(xd, 0.4), (xd, 0.77)], color=EDGE, lw=0.75, arrow=False, dash='sysDash')

GX, CW, CH = 0.6, 0.12, 0.155
GY = 1.84
GW = P * CW
fold_x = GX + GW / 2
arrow([(fold_x, 0.82), (fold_x, GY - 0.03)])
text(0.06, 0.98, fold_x - 0.1, 0.46, ['normalize,', 'fold'], align=PP_ALIGN.RIGHT)
lo, hi_ = RGBColor(0xF2, 0xF4, 0xF6), RGBColor(0xA9, 0xB4, 0xC0)
HL = 3                                  # highlighted phase token (column)
means = []
for k in range(K):
    row = []
    for p in range(P):
        v = min(1.0, 0.1 + 0.75 * daily((p + 0.5) / P) + 0.07 * k)
        row.append(v)
        fill = RGBColor(*(round(a + (b - a) * v) for a, b in zip(lo, hi_)))
        if p == HL:
            fill = RGBColor(*(round(a + (b - a) * v) for a, b in zip(PH_LIGHT, TOKEN_BLUE)))
        box(GX + p * CW, GY + k * CH, CW, CH, fill=fill, line=EDGE, lw=0.6, shape=MSO_SHAPE.RECTANGLE)
    means.append(sum(row) / P)
box(GX + HL * CW, GY, CW, K * CH, fill=None, line=PH_ACC, lw=1.75, shape=MSO_SHAPE.RECTANGLE)
text(GX - 0.3, GY, 0.26, K * CH, ['$K$'], size=SM)
text(GX - 0.2, GY + K * CH + 0.02, GW + 0.4, 0.22, ['$P$ phases'], size=SM)
# level token: the row means, one per cycle
LX = GX + GW + 0.2
arrow([(GX + GW + 0.02, GY + K * CH / 2), (LX - 0.03, GY + K * CH / 2)], TR_ACC, lw=1.0)
for k, v in enumerate(means):
    t = (v - min(means)) / (max(means) - min(means))
    fill = RGBColor(*(round(a + (b - a) * (0.2 + 0.8 * t)) for a, b in zip(TOKEN_LIGHT, TOKEN_GREEN)))
    box(LX, GY + k * CH, CW, CH, fill=fill, line=TR_ACC, lw=0.9, shape=MSO_SHAPE.RECTANGLE)
text(0.1, GY + K * CH + 0.22, 1.5, 0.24, ['$\\b{X̃}∈ℝ^{K×P}$'], size=SM)
# the two token types leave on the rows of the two pathways
hx = GX + (HL + 0.5) * CW
arrow([(hx, GY - 0.02), (hx, YP), (EMB_X0, YP)], PH_ACC)
text(hx + 0.04, YP - 0.44, IN_X1 - hx, 0.42, ['phase', 'token $x̃_p$'], color=PH_ACC, align=PP_ALIGN.LEFT)
lx = LX + CW / 2
arrow([(lx, GY + K * CH), (lx, YT), (EMB_X0, YT)], TR_ACC)
text(0.1, YT + 0.04, EMB_X0 - 0.28, 0.42, ['level token', '$r̅∈ℝ^K$'], color=TR_ACC, align=PP_ALIGN.RIGHT)
text(0.08, 3.52, IN_X1 - 0.12, 0.24, ['row means = cycle levels'], size=11, color=TR_ACC)

# ---------------------------------------------------------------- shared embedding
module(EMB_X0, 1.0, EMB_X1 - EMB_X0, 2.34, ['**Shared**', '**Embedding**', '$W_e, b_e$'], fill=YELLOW, size=13)

# ---------------------------------------------------------------- phase pathway (top row)
band(BAND_X0, TOP, BAND_X1 - BAND_X0, MID - TOP - 0.04, BLUE_PANEL, 'Phase Pathway', PH_ACC)
x = 2.94
z0, z1, zs = stack(x, YP, P, PH_LIGHT, PH_ACC)
arrow([(EMB_X1, YP), (x - 0.02, YP)], PH_ACC)
text(x - 0.32, z0 - 0.27, TW + 0.64, 0.24, ['$\\b{Z}∈ℝ^{P×d}$'])
x2 = 3.54
arrow([(x + TW, YP), (x2 - 0.02, YP)], PH_ACC)
text(x + 0.1, z1 + 0.02, x2 - x + 0.08, 0.42, ['+ pos.', 'emb.'])
_, _, zts = stack(x2, YP, P, PH_LIGHT, PH_ACC)
text(x2 - 0.1, z0 - 0.27, TW + 0.2, 0.24, ['$\\b{Z̃}$'])

# routing layer, drawn as a bow tie P -> M -> P
RL0, RL1 = 3.9, 7.52
RY0, RY1 = 0.38, 2.0
box(RL0, RY0, RL1 - RL0, RY1 - RY0, fill=WHITE, line=INK, lw=1.0, dash=True, shape=MSO_SHAPE.RECTANGLE)
text(RL0, 0.1, RL1 - RL0, 0.28, ['**Cross-Phase Routing Layer**'], size=MOD)
xr = 4.98
r0, r1, rs = stack(xr, YP, M, RT, RT_LINE)
fan(zts, rs, x2 + TW, xr)
xa = 6.12
_, _, zas = stack(xa, YP, P, PH_LIGHT, PH_ACC)
fan(rs, zas, xr + TW, xa)
cx = xr + TW / 2
chip(cx, 0.52, w=0.36, h=0.15)
arrow([(cx, 0.66), (cx, r0 - 0.02)], RT_LINE, lw=1.0)
text(cx + 0.05, 0.76, 0.3, 0.22, ['$\\b{R}$'], color=RT_LINE, align=PP_ALIGN.LEFT)
text(RL0 + 0.02, 0.45, cx - 0.22 - RL0 - 0.02, 0.4, ['Aggregation', '$\\r{MHA}(\\b{R},\\b{Z̃},\\b{Z̃})$'], size=11,
     align=PP_ALIGN.RIGHT)
text(cx + 0.22, 0.45, 1.2, 0.4, ['Distribution', '$\\r{MHA}(\\b{Z̃},\\b{R}′,\\b{R}′)$'], size=11,
     align=PP_ALIGN.LEFT)
text(xr - 0.4, r1 + 0.12, TW + 0.8, 0.24, ['$\\b{R}′∈ℝ^{M×d}$'], color=RT_LINE)
text(xa - 0.08, z0 - 0.27, TW + 0.24, 0.24, ['$\\b{Z}_{\\r{attn}}$'])
BL0, BL1 = 6.56, 7.44
arrow([(xa + TW, YP), (BL0 - 0.02, YP)])
module(BL0, 0.98, BL1 - BL0, 0.64, ['Add & LN', 'MLP', 'Add & LN'], size=SM)
text(RL1 - 0.62, RY1 - 0.28, 0.58, 0.26, ['$×N_r$'], size=MOD, align=PP_ALIGN.RIGHT)

xh = 7.64
arrow([(BL1, YP), (xh - 0.02, YP)])
stack(xh, YP, P, PH_LIGHT, PH_ACC)
text(xh - 0.1, z0 - 0.27, TW + 0.2, 0.24, ['$\\b{H}$'])
text(xh - 0.3, z1 + 0.02, TW + 0.6, 0.24, ['$h_p∈ℝ^d$'])
PH0, PH1 = 8.01, 8.75
arrow([(xh + TW, YP), (PH0 - 0.02, YP)])
module(PH0, 0.98, PH1 - PH0, 0.64, ['Shared', 'Predictor', '$Φ$'], size=SM)
GF = 8.86
arrow([(PH1, YP), (GF - 0.02, YP)])
fh = 0.118
for p in range(P):
    for j in range(KP):
        box(GF + j * 0.13, YP - P * fh / 2 + p * fh, 0.13, fh, fill=PH_LIGHT, line=PH_ACC, lw=0.75,
            shape=MSO_SHAPE.RECTANGLE)
text(GF - 0.5, z0 - 0.27, 0.86, 0.24, ['$F_p∈ℝ^{K′}$'], align=PP_ALIGN.RIGHT)
text(GF - 0.25, z1 + 0.02, 0.76, 0.24, ['$P×K′$'])
WX = 9.36
arrow([(GF + 0.26, YP), (WX - 0.03, YP)], PH_ACC)
series(WX, YP - 0.13, 0.4, 0.26, KP, PH_ACC, lw=1.1)
text(WX - 0.12, YP - 0.4, 0.64, 0.24, ['$F∈ℝ^H$'], color=PH_ACC)
text(WX - 0.12, YP + 0.16, 0.64, 0.22, ['unfold'], size=11.5)

# ---------------------------------------------------------------- trend pathway (bottom row)
band(BAND_X0, MID + 0.04, BAND_X1 - BAND_X0, BOT - MID - 0.04, GREEN_PANEL, 'Trend Pathway', TR_ACC)
xz = 2.94
box(xz, YT - 0.0425, TW, 0.085, fill=TOKEN_LIGHT, line=TR_ACC, lw=0.9, shape=MSO_SHAPE.RECTANGLE)
arrow([(EMB_X1, YT), (xz - 0.02, YT)], TR_ACC)
text(xz - 0.2, YT - 0.33, TW + 0.4, 0.24, ['$z̅∈ℝ^d$'])
text(xz - 0.08, YT + 0.08, 1.0, 0.24, ['= mean of $z_p$'], align=PP_ALIGN.LEFT)
E0, E1 = 3.84, 4.76
arrow([(xz + TW, YT), (E0 - 0.02, YT)], TR_ACC)
module(E0, YT - 0.27, E1 - E0, 0.54, ['Linear', 'Encoder $E$'], size=SM)
xt = 5.0
for k in range(3):
    box(xt, YT - 0.165 + k * 0.11, 0.11, 0.11, fill=TOKEN_GREEN if k == 1 else TOKEN_LIGHT, line=TR_ACC,
        lw=0.9, shape=MSO_SHAPE.RECTANGLE)
arrow([(E1, YT), (xt - 0.02, YT)], TR_ACC)
text(xt - 0.3, YT - 0.43, 0.72, 0.24, ['$t∈ℝ^r$'])
B0, B1 = 5.78, 7.6
arrow([(xt + 0.11, YT), (B0 - 0.02, YT)], TR_ACC)
text(xt + 0.1, YT - 0.29, B0 - xt - 0.1, 0.24, ['$Γt+c_Γ$'])
box(B0, YT - 0.5, B1 - B0, 1.0, fill=WHITE, line=INK, lw=1.25, radius=0.08)
text(B0, YT - 0.48, B1 - B0, 0.26, ['Horizon Basis $Ψ_H$'], size=SM)
for i, kind in enumerate(('offset', 'tilt', 'curve')):
    cx = B0 + 0.33 + i * 0.6
    profile(cx - 0.18, YT - 0.18, 0.36, 0.2, kind, TR_ACC)
    text(cx - 0.3, YT + 0.04, 0.6, 0.2, [kind], size=11.5)
text(B0, YT + 0.25, B1 - B0, 0.2, ['fixed or learned'], size=11.5)
OX = 7.94
arrow([(B1, YT), (OX - 0.12, YT)], TR_ACC)
operator(OX, YT)
arrow([(OX, YT + 0.5), (OX, YT + 0.12)], TR_ACC, lw=1.0)
text(OX - 0.45, YT + 0.5, 0.9, 0.24, ['anchor $r̅_K$'])
SX = 8.28
arrow([(OX + 0.11, YT), (SX - 0.02, YT)], TR_ACC)
steps(SX, YT - 0.16, 0.6, 0.32, [0.25, 0.45, 0.7, 0.9], TR_ACC)
text(SX - 0.05, YT - 0.46, 0.75, 0.24, ['$F_b∈ℝ^H$'], color=TR_ACC)

# ---------------------------------------------------------------- gated fusion
band(FU_X0, TOP, W - 0.04 - FU_X0, BOT - TOP, FUSE_BAND, 'Gated Fusion', FU_ACC)
OL, OS, OY = 10.42, 11.02, 2.12
# F from the top, F_b from the bottom; each enters both operators
arrow([(WX + 0.42, YP), (OS, YP), (OS, OY - 0.12)], PH_ACC)
arrow([(OL, YP), (OL, OY - 0.12)], PH_ACC)
arrow([(SX + 0.6, YT), (OS, YT), (OS, OY + 0.12)], TR_ACC)
arrow([(OL, YT), (OL, OY + 0.12)], TR_ACC)
operator(OL, OY)
operator(OS, OY)
arrow([(OL + 0.11, OY), (OS - 0.12, OY)])
text(OL - 0.3, 0.84, 0.6, 0.42, ['levels', '$M_c$'], color=FU_ACC)
text(OS - 0.3, 0.84, 0.6, 0.42, ['shape', '$J_c$'], color=FU_ACC)
text(OL - 0.5, 1.56, 0.46, 0.24, ['$1−g_ℓ$'], align=PP_ALIGN.RIGHT)
text(OS - 0.5, 1.56, 0.46, 0.24, ['$1−g_s$'], align=PP_ALIGN.RIGHT)
text(OL - 0.42, 2.48, 0.38, 0.24, ['$g_ℓ$'], align=PP_ALIGN.RIGHT)
text(OS - 0.42, 2.48, 0.38, 0.24, ['$g_s$'], align=PP_ALIGN.RIGHT)
# fused forecast, de-normalized
OUTX = 11.86
arrow([(OS + 0.11, OY), (OUTX, OY), (OUTX, 2.42)])
text(OS + 0.1, OY - 0.28, W - 0.08 - OS - 0.1, 0.24, ['de-normalize'], size=11.5)
series(11.46, 2.46, 0.64, 0.36, 3, FU_ACC, drift=0.5, lw=1.2)
text(11.36, 2.86, 0.84, 0.26, ['$ŷ∈ℝ^H$'], color=FU_ACC)

save(OUT)
