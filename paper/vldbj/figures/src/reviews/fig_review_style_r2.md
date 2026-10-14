# Round 2: style and font re-review of efficiency_plot.pdf and efficiency_bars.pdf

## Verdict: PASS after one must-fix (the "10⁹" tick label is cut off)

Checked: both PDFs at 150 and 500 dpi, pdffonts, make_efficiency.py, and main.pdf pages 3 and 20.

## Status of the round-1 issues

| # | Issue | Status |
|---|---|---|
| P1 | Font family | Resolved. pdffonts shows only ArialMT and Arial-BoldMT. Fig. 1(b) now matches the Arial ticks of Fig. 1(a). Sizes: SMALL prints at 7.0pt and the tick labels at about 7.6pt; the bar panel titles are regular weight at about 8.6pt. |
| P2 | Crossformer label crossing the frame | Resolved. The label is now above and to the right of its bubble, inside the frame with room to spare. |
| P3 | PhaseFormer+ highlighted differently in the two figures | Resolved. Both figures now use a blue fill with a red outline and a bold red label. |
| P4 | Y-axis titles of the bars too close to the previous panel | Resolved. Each title now sits next to its own tick labels. The gap to the "10⁻¹" label in panel (c) is tight, about 2pt printed, but acceptable. |
| P5 | Gray too dark to tell apart in grayscale | Resolved. #A6ADB5 has a luma of about 172, against about 115 to 127 for the green and blue, so it separates in grayscale. It is also close to the grays of the coupling and token_energy figures. |
| P6 | Tick marks, minus sign, bar outline | Resolved. Outward ticks are drawn (0.75pt on the slide, about 0.43pt printed), the exponents use a true minus sign (U+2212), the PhaseFormer+ bar is inset from the baseline, and the Electricity FLOPs axis now goes up to 10³. |
| P7 | Legends | Mostly resolved. The swatches are filled like the markers, TimeBase's "Phase and patch" class is in both legends, the bars legend is centered with even gaps, and its last item is capitalized. Not done: the Fig. 1(b) legend is still below the plot (see the optional note). |
| P8 | Title weight and text color | Resolved. The titles are regular weight, and all text is INK #222222. |

## Must-fix

1. **The "10⁹" tick label of Fig. 1(b) is cut off at the slide edge.**
   - In `scatter()`, the label is centered on X1 = 5.88in, but the slide is only 6.0in wide. The
     superscript "9" runs past the slide edge and is clipped. This is visible in the 500 dpi
     render, and is just at the margin on page 3.
   - Fix: widen the slide to `new_slide(6.08, 3.2)`, or lower X1 to about 5.80.
   - After the change, re-export the figure and check that the page-3 width still lines up with
     Fig. 1(a).

## Optional, not blocking

- The plot box of Fig. 1(b) still sits about 0.15in higher than the box of Fig. 1(a) on page 3,
  and is shorter. Moving the legend into the empty lower-right part of the plot and lowering
  the slide height to about 2.95in would line the two boxes up. This is cosmetic.
- In the bars, the superscripts of the tick labels nearly touch the tick marks, for example
  "10⁸" in panel (b). Moving the tick-label box left by about 0.03in, from `x0 - 0.56` to
  `x0 - 0.59`, would give them room.
