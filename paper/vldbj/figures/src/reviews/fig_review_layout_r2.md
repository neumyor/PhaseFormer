# Layout re-review, round 2: efficiency figures

Checked against: `efficiency_plot.pdf` (22:59:05), `efficiency_bars.pdf` (23:01:52) and the rebuilt
`main.pdf` (23:01:54). I checked p.3 and p.20 of main.pdf at 300 dpi and measured text and shape
boxes with pymupdf. "pt-n" means native PDF points; "pt-p" means printed points.

## Verdict: NOT YET PASS — two must-fix items (both small)

### Must-fix 1: Fig. 1(b), the "10^9" exponent is clipped at the right edge
In both the standalone PDF and main.pdf p.3, only the left half of the "9" prints. The text box
ends at x = 433.3 pt-n, and the page is 433.0 pt-n wide.
- **Cause:** with `X1 = 5.88`, the 10^9 tick label (`px(1e9) - 0.3` … `+0.6`) is centred at
  5.88 in. Its glyphs run to about 6.02 in, past the 6.0 in slide, and LibreOffice clips at the
  slide edge.
- **Fix:** use `new_slide(6.1, 3.2)` and keep `X1`, or reduce `X1` to 5.80. Either way the
  10^9 exponent must sit at least 1 pt-n inside the crop.

### Must-fix 2: Fig. 7, the PhaseFormer+ x-labels sit on the legend in all four panels
The rotation is now -60°, so the labels reach lower than before. In every panel the bold red "P"
has its lowest corner at y = 234.0 pt-n. The legend text's ascenders and caps start at about
233.5 pt-n, with the baseline at 242.5. The red labels land directly above the legend items:
| Panel | PhaseFormer+ "P" at x (pt-n) | Legend item below it |
|---|---|---|
| (a) | 113.8 | "Phase-based" |
| (b) | 329.8 | "Phase and patch" |
| (c) | 545.8 | "level" |
| (d) | 761.8 | "(≤2.1K / ≤1.0K)" |

The inked gap is 0–2 pt-n (≤ 1 pt-p). On p.20 the labels visibly touch the legend line, and the
"P" of panel (c) reads as part of the legend.
- **Fix:** set `LY = 3.26` → `3.40`, and `new_slide(12.2, 3.55)` → `new_slide(12.2, 3.69)`. This
  gives a gap of at least 0.1 in. You could shorten the rotated labels instead, but that is
  unnecessary.

## Round-1 items: status
| # | Item | Status |
|---|------|--------|
| 1 | Fig. 7 y titles beside the wrong panel | **Resolved.** For (b), "Parameters" now sits about 19 pt-n from frame (a) and about 4 pt-n from its own ticks; (c) and (d) are the same. |
| 2 | Crossformer label crossing the frame | **Resolved.** The label is in the top-right corner, ending at x 416.1 inside the frame edge at 423.3, and clears all markers. |
| 3 | "11.7 GFLOPs" over "SparseTSF" | **Resolved.** SparseTSF now starts at y 70 pt-n, against the iTransformer label bottom at 64. |
| 4 | TimeBase label touching the SparseTSF marker | **Resolved.** The label is below the marker, with a gap of at least 5 pt-n. |
| 5 | Panel (c) bar at the frame top; ranges | **Resolved.** Both FLOPs panels span 10^-4 to 10^3, and the Crossformer bar has a decade of headroom. |
| 6 | Exponent size; minus sign | **Resolved.** Exponents print at 5.05 (scatter) and 5.16 (bars) pt-p; a true minus sign is used, and the exponents read at the 100 dpi page render. |
| 7 | TimeBase double ring unexplained | **Resolved.** There is a "Phase and patch" legend entry in both figures. |
| 8 | Scatter legend position | **Accepted as is.** It is a single row below the axis title, which is fine. |
| 9 | Fig. 7 legend spacing and wording | **Resolved.** The legend is centred within about 12 pt-n, the gaps are 7–18 pt-n, and the wording is shorter. |
| 10 | Filled swatches | **Resolved.** |

Not blocking: the TimeMixer and iTransformer markers still overlap. That reflects the data and is
acceptable. Fig. 1(a) and 1(b) now both use sans-serif fonts, so the earlier font mismatch is gone.

Once the two fixes are re-exported (`sh export.sh efficiency_plot 243; sh export.sh efficiency_bars 495`),
I expect to pass both figures.
