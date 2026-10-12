"""Build the PhaseFormer-LA overview (Fig. 4) as a PowerPoint slide.

The style follows the PhaseFormer overview of Fig. 3 (make_overview.py):
Times New Roman text with bold titles, rounded boxes with black outlines,
dashed panels with light fills, green/blue token bars and black arrows.
Components of the conference PhaseFormer keep their Fig. 3 colors (embedding
peach, routing gray, predictor yellow); the new components of PhaseFormer-LA
are blue.  Font sizes and export: see pptfig.py.

    python make_la_overview.py            # writes la_overview.pptx next to this file
"""
import math
from pathlib import Path

from pptfig import *  # noqa: F401,F403

OUT = Path(__file__).with_name('la_overview.pptx')
new_slide(12.2, 4.25)

YM = 1.18                      # center line of the main path
BH = 1.08                      # height of the main-path boxes

# 1. Input window with a small multi-cycle series.
IX, IW = 0.05, 1.45
box(IX, YM - BH / 2, IW, BH)
label(IX, YM - BH / 2 + 0.06, IW, 0.28, ['**Input Window**'])
wave = [0.5 + 0.32 * math.sin(2 * math.pi * k / 12) + 0.12 * math.sin(2 * math.pi * k / 4)
        + 0.004 * k for k in range(37)]
lo, hi = min(wave), max(wave)
curve(IX + 0.14, YM - 0.12, IW - 0.28, 0.32, [(v - lo) / (hi - lo) for v in wave])
label(IX, YM + 0.24, IW, 0.26, ['$x ∈ ℝ^L$'])

# 2. PhaseFormer backbone panel (unchanged), vertical flow as in Fig. 3.
PX, PW = 1.64, 2.5
PY0, PY1 = 0.06, 2.38
box(PX, PY0, PW, PY1 - PY0, fill=GRAY_PANEL, radius=0.06)
label(PX, PY0 + 0.08, PW, 0.5, ['**PhaseFormer Backbone**', '(unchanged)'], size=TEXT, spacing=0.95)
sx, sw, sh = PX + 0.14, 1.94, 0.4
ys = (0.72, 1.28, 1.84)
box(sx, ys[0], sw, sh, ['Phase Tokenization'], fill=WHITE, size=TEXT)
box(sx, ys[1], sw, sh, ['Embedding'], fill=PEACH, size=TEXT)
for off in (0.08, 0.04):       # stacked routing layers, as in Fig. 3
    box(sx + off, ys[2] - off, sw, sh, fill=WHITE, line=GRAY_LINE, lw=0.75, radius=0.05)
box(sx, ys[2], sw, sh, ['Cross-Phase Routing'], fill=GRAY_BOX, size=TEXT, radius=0.05)
label(sx + sw + 0.1, ys[2] + 0.06, 0.3, 0.3, ['$×N_{r}$'], size=SMALL, align=PP_ALIGN.LEFT)
for y0, y1 in ((ys[0] + sh, ys[1]), (ys[1] + sh, ys[2] - 0.08)):
    path([(sx + sw / 2, y0), (sx + sw / 2, y1)])

# 3. Centered modulation (new).
MX, MW = 5.0, 2.46
box(MX, YM - BH / 2, MW, BH, ['**Centered Modulation**', '$h′_p = h_p + (h_p − h̅) ⊙ q$'],
    fill=BLUE_BOX, size=TEXT, spacing=1.25)

# 4. Shared predictor (unchanged, Fig. 3 yellow).
SX, SW = 7.74, 1.24
box(SX, YM - BH / 2, SW, BH, ['**Shared**', '**Predictor** $Φ$'], fill=YELLOW, size=TEXT, spacing=1.15)

# 5. Level/shape fusion (new).
FX, FW = 9.28, 2.38
box(FX, YM - BH / 2, FW, BH, ['**Level/Shape Fusion**',
                              '$M_c$:  $(1 − g_ℓ)F + g_ℓF_{\\r{b}}$',
                              '$J_c$:  $(1 − g_s)F + g_sF_{\\r{b}}$'],
    fill=BLUE_BOX, size=TEXT, spacing=1.08)

# 6. Output.
OX, OW = 11.8, 0.38
box(OX, YM - 0.27, OW, 0.54, ['$ŷ$'], fill=WHITE, size=TITLE)

# Main-path arrows.
path([(IX + IW, YM), (PX, YM)])
path([(PX + PW, YM), (MX, YM)])
label(PX + PW, YM - 0.36, MX - PX - PW, 0.28, ['$h_1, …, h_P$'])
path([(MX + MW, YM), (SX, YM)])
path([(SX + SW, YM), (FX, YM)])
label(SX + SW, YM - 0.36, FX - SX - SW, 0.28, ['$F$'])
path([(FX + FW, YM), (OX, YM)])

# --------------------------------------------------------------------------
# Level branch (new): dashed light-blue panel, as the routing panels of Fig. 3.
# --------------------------------------------------------------------------
LY, LH = 2.6, 1.6
LX0, LX1 = 0.36, 11.7
box(LX0, LY, LX1 - LX0, LH, fill=BLUE_PANEL, lw=1.25, dash=True, radius=0.05)
label(0.9, LY + 0.07, 5.0, 0.3, ['**Level Branch** (rank 3, on cycle means)'], align=PP_ALIGN.LEFT)

YB = LY + 0.92                 # center line of the branch boxes
BBH = 1.08
# pooling box
CX, CW = 1.2, 2.9
box(CX, YB - BBH / 2, CW, BBH, ['**Cycle-Mean Pooling**',
                                '$r̄ − x̃_L 𝟏_K ∈ ℝ^K$ from $x̃ − x̃_L 𝟏_L$',
                                '$t = E(r̄ − x̃_L 𝟏_K) + c_E ∈ ℝ^3$'],
    size=TEXT, spacing=1.12)
# state t as three blue tokens (rank-3 state)
TX = 4.66
tend = tokens(TX, YB - 0.08, 3, TOKEN_BLUE, w=0.24, h=0.16, gap=0.06)
label(TX - 0.25, YB + 0.14, 1.3, 0.26, ['state $t ∈ ℝ^3$'], size=SMALL)
# fixed profiles: title, three profile sketches, readout
RX, RW = 7.15, 4.4
box(RX, YB - BBH / 2, RW, BBH)
label(RX, YB - BBH / 2 + 0.07, RW, 0.28, ['**Fixed Profiles** $Ψ_H$'])
for (name, prof, bx) in (('displacement', [0.5] * 9, RX + 0.22),
                         ('tilt', [k / 8 for k in range(9)], RX + 1.86),
                         ('curvature', [1 - ((k - 4) / 4) ** 2 for k in range(9)], RX + 2.82)):
    box(bx, YB - 0.14, 0.42, 0.24, fill=WHITE, line=RGBColor(0x90, 0x90, 0x90), lw=0.5,
        shape=MSO_SHAPE.RECTANGLE)
    curve(bx + 0.04, YB - 0.115, 0.34, 0.19, prof, color=BRANCH, lw=1.25)
    label(bx + 0.48, YB - 0.155, 1.2, 0.26, [name], size=SMALL, align=PP_ALIGN.LEFT)
label(RX, YB + 0.2, RW, 0.28, ['$F_{\\r{b}} = x̃_L 𝟏_H + Ψ_H(Γt + c_Γ)$'])

# Branch arrows (dark blue).
xin = IX + IW / 2
path([(xin, YM + BH / 2), (xin, YB), (CX, YB)], color=BRANCH)
path([(CX + CW, YB), (TX - 0.07, YB)], color=BRANCH)
path([(tend + 0.04, YB), (RX, YB)], color=BRANCH)
xq = MX + MW / 2
path([(xq, YB), (xq, YM + BH / 2)], color=BRANCH)
label(xq + 0.08, YM + BH / 2 + 0.12, 1.5, 0.28, ['$q = V_2V_1t$'], align=PP_ALIGN.LEFT, color=BRANCH)
xr = FX + FW / 2
path([(xr, YB - BBH / 2), (xr, YM + BH / 2)], color=BRANCH)
label(xr + 0.08, YM + BH / 2 + 0.12, 0.4, 0.28, ['$F_{\\r{b}}$'], align=PP_ALIGN.LEFT, color=BRANCH)

save(OUT)
