"""Fig. 6 of Section 4.1: the three tokens on one real window.

Four days of Traffic sensor 175 (Tuesday 2018-04-03 to Friday 2018-04-06, test
period), normalized as the model sees it.  Same style as Fig. 1(a): black
series, dashed gray cycle boundaries, colored markers on the series.
  phase token (blue, as in Fig. 1): the values at one hour of every day;
  level token (navy, as in Figs. 2 and 4): the day means, measured from the
      last value of the window (dashed line);
  amplitude (orange, as in Figs. 2 and 4): the swing of every day, which the
      amplitude token scales; it follows the level (busy weekdays swing wider).
The slide is 6.0 in wide and included at \\linewidth (239pt).

    python3 make_three_tokens.py <all_datasets>
    sh export.sh three_tokens 239
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pptfig
from pptfig import *  # noqa: F401,F403

START, SENSOR, P, K = 15384, 175, 24, 4
HOUR = 10                                    # hour of the phase token shown

frame = pd.read_csv(os.path.join(sys.argv[1], 'traffic', 'traffic.csv'))
dates = pd.to_datetime(frame.iloc[:, 0])
# assert dates.iloc[START].hour == 0 and dates.iloc[START].dayofweek == 1
x = frame.iloc[START:START + K * P, 1 + SENSOR].to_numpy('float64')
z = (x - x.mean()) / x.std()
X = z.reshape(K, P)
means = X.mean(1)

PHASE = RGBColor(0x1F, 0x9F, 0xD8)
LEVEL = RGBColor(0x1B, 0x3A, 0x6B)
AMP = RGBColor(0xE0, 0x7B, 0x39)
GRID = RGBColor(0xB8, 0xB8, 0xB8)

pptfig.SUBSCALE = 1.35
FS, FSS = 13.5, 12.5                         # 7.5pt and 6.9pt printed
OUT = Path(__file__).with_name('three_tokens.pptx')
W, H = 6.0, 3.25
new_slide(W, H)

X0, X1, Y0, Y1 = 0.42, 5.92, 0.90, 2.78      # raised Y0 to shorten the plot area
LO, HI = -1.75, 2.45                         # value range


def px(t):
    return X0 + 0.04 + (X1 - X0 - 0.08) * t / (K * P - 1)


def py(v):
    return Y1 - (Y1 - Y0) * (v - LO) / (HI - LO)


# frame, ticks and cycle boundaries
path([(X0, Y0), (X0, Y1), (X1, Y1)], lw=1.0, arrow=False)
for v in (-1, 0, 1, 2):
    path([(X0 - 0.05, py(v)), (X0, py(v))], lw=1.0, arrow=False)
    label(X0 - 0.42, py(v) - 0.1, 0.34, 0.2, [str(v) if v >= 0 else f'−{-v}'], size=FSS,
          align=PP_ALIGN.RIGHT)
for k in range(1, K):
    xb = (px(k * P - 1) + px(k * P)) / 2
    path([(xb, Y0), (xb, Y1)], color=GRID, lw=1.25, arrow=False, dash='dash')
for k, day in enumerate(dates.iloc[START:START + K * P:P]):
    label(px(k * P), Y1 + 0.03, px(k * P + P - 1) - px(k * P), 0.22,
          [day.strftime('%a %m-%d')], size=FSS)

# level token: day means only (no last-value dashed line)
for k in range(K):
    path([(px(k * P), py(means[k])), (px(k * P + P - 1), py(means[k]))], color=LEVEL, lw=3.0,
         arrow=False)

# amplitude: double-headed arrow spanning exact min-to-max, placed right of peak
for k in range(K):
    hi_idx = int(np.argmax(X[k]))
    lo_idx = int(np.argmin(X[k]))
    hi, lo = X[k, hi_idx], X[k, lo_idx]
    # place arrow 2 columns to the right of the peak, clamped inside the cycle
    t_arr = min(k * P + hi_idx + 2, k * P + P - 2)
    path([(px(t_arr), py(lo)), (px(t_arr), py(hi))], color=AMP, lw=1.5)
    path([(px(t_arr), py(hi)), (px(t_arr), py(lo))], color=AMP, lw=1.5)

# series (drawn on top of level lines, below phase dots)
path([(px(t), py(v)) for t, v in enumerate(z)], lw=1.25, arrow=False)

# phase token: one hour of every day, connected by a thick line
pts = [(px(k * P + HOUR), py(X[k, HOUR])) for k in range(K)]
path(pts, color=PHASE, lw=2.5, arrow=False)
R = 0.055
for cx, cy in pts:
    box(cx - R, cy - R, 2 * R, 2 * R, fill=PHASE, line=BLACK, lw=1.0, shape=MSO_SHAPE.OVAL)

# key above the plot — three items only, evenly spaced
y = 0.2
# phase: dot then label — start further right so the legend dot doesn't sit on the plot line
box(0.45 - R, y - R, 2 * R, 2 * R, fill=PHASE, line=BLACK, lw=1.0, shape=MSO_SHAPE.OVAL)
label(0.58, y - 0.12, 1.55, 0.24, ['**phase token** $x̃_p$'], size=FS, align=PP_ALIGN.LEFT,
      color=PHASE)
# level: short thick line then label
path([(2.22, y), (2.52, y)], color=LEVEL, lw=3.0, arrow=False)
label(2.58, y - 0.12, 1.55, 0.24, ['**level token** $r̄_k$'], size=FS, align=PP_ALIGN.LEFT,
      color=LEVEL)
# amplitude: small double arrow then label
path([(4.32, y + 0.11), (4.32, y - 0.11)], color=AMP, lw=1.5)
path([(4.32, y - 0.11), (4.32, y + 0.11)], color=AMP, lw=1.5)
label(4.42, y - 0.12, 1.55, 0.24, ['**amplitude** $a_k$'], size=FS, align=PP_ALIGN.LEFT, color=AMP)

save(OUT)
