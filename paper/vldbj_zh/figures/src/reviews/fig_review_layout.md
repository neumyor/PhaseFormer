# Layout and readability review: efficiency figures (print size)

Reviewer focus: layout and readability at print size.
Files: `figures/efficiency_plot.pdf` (Fig. 1(b), 428x224 pt native, printed at 243 pt, scale 0.568) and
`figures/efficiency_bars.pdf` (Fig. 7, 877x233 pt native, printed at 495 pt, scale 0.564).
Source: `figures/src/make_efficiency.py` (with `pptfig.py`). The measurements below come from text
and shape boxes extracted with pymupdf, in native PDF points ("pt-n") and printed points ("pt-p"),
and from renders of main.pdf p.3 and p.20 at 100 and 300 dpi.

## Verdict

**Both figures need minor revision before submission.** Each reads at a glance: PhaseFormer+ is
the red marker at the bottom left in Fig. 1(b), and the red-outlined, red-labelled bar in all four
panels of Fig. 7. Fonts are uniform: body text 7.1 pt-p, panel titles 8.5 pt-p, Times throughout,
all vector, with no raster images. Nothing is clipped by the page. Several specific problems are
visible in print, though:

- In Fig. 1(b), one label crosses the plot frame and two labels nearly collide.
- In Fig. 7, the y-axis titles sit next to the wrong panel.
- In Fig. 7, the panel (c) bar reaches the top of its frame.
- In both figures the superscripts are 4.7 pt-p.

None of these needs a redesign. Each is a change to a few coordinates.

---

## Fig. 1(b): `scatter()`

### P1 (must fix): the Crossformer label runs across the right frame
The frame's right edge is at x = 418.3 pt-n. "259 GFLOPs" ends at x = 421.8 and "Crossformer" ends
at 417.9, so the text sits on the border line. You can see this in the 300 dpi page render.
The label cannot fit to the right of the marker: the marker's right edge is at 350.3 pt-n and
the text is 66 pt-n wide.
- **Change:** in `place`, set `'Crossformer': (0.12, -0.54, PP_ALIGN.LEFT)`. This puts the
  label above and to the right of the marker, in the empty top-right corner: about x 338–404,
  y 17–45 pt-n. It clears the Crossformer marker by about 6 pt-n and the iTransformer marker
  by more than 20 pt-n, and stays inside the frame.

### P1: the iTransformer and SparseTSF labels nearly collide
- **Measurement:** "11.7 GFLOPs" (x 208–277, baseline about 62 pt-n) sits directly above
  "SparseTSF" (x 185–240, cap top about 64 pt-n).
- **Effect:** the inked gap is only about 2 pt-n, or 1.2 pt-p, over a 32 pt-n horizontal
  overlap. In print the two labels merge into one four-line block. A reader can pair
  "11.7 GFLOPs" with SparseTSF.
- **Change:** move the SparseTSF label down and slightly right:
  `'SparseTSF': (0.30, 0.14, PP_ALIGN.LEFT)`. Its first line then starts at about 73 pt-n, clear
  of the iTransformer label. Nothing lies below it in that region.
- **Alternative:** put the label under the marker, centred: `(0.0, 0.36, PP_ALIGN.CENTER)`.

### P1: the TimeBase label nearly touches the SparseTSF marker
"8.44 MFLOPs" ends at x = 146.2 pt-n, and the SparseTSF circle begins at 147.6 pt-n. The gap is
1.8 pt-p, so the label looks attached to the SparseTSF marker.
- **Change:** move the label below the marker, left-aligned, into the empty area at MAE
  0.26–0.275: `'TimeBase': (-0.17, 0.39, PP_ALIGN.LEFT)`. This gives about x 45–117,
  y 96–123 pt-n, which clears the y tick labels (x < 35) and the outer dashed ring (bottom at
  92.6 pt-n).

### P2: the 10^k superscripts are 4.7 pt-p
The base digits are 7.1 pt-p, but the exponents are 4.71 pt-p. That is because `SUBSCALE = 1.15`
and LibreOffice shrinks baseline-shifted text to 58%. At 100 dpi the exponents on p.3 cannot be
read. At 300 dpi they can, but only just. The exponent is the only information in a log tick
label, so it should be at least 5 pt-p.
- **Change (local, keeps other figures unchanged):** set the tick labels at a larger size, e.g.
  `label(..., [f'$10^{e}$'], size=13.5)` in the x-tick loop. Base size 13.5 gives about 7.7 pt-p
  for the digits and about 5.1 pt-p for the exponent.
- **Better:** add an optional sub/superscript scale argument to `pptfig.label`/`_math`, e.g.
  1.3. This gives about 5.3 pt-p exponents with the digits at 7.1 pt-p.

### P2: TimeBase's two rings are not explained
TimeBase is drawn with a blue solid ring plus an outer green dashed ring (`cat == 'both'`). The
legend has no entry for this and the caption does not mention it. A reader sees one anomalous
marker and must guess why.
- **Change:** add one clause to the caption, e.g. "TimeBase, drawn with both rings, is both
  phase- and patch-based".
- **Alternative:** add a fourth legend item, "Both", drawn with the same two rings.

### P2: the legend wastes a full row; move it into the empty lower-right corner
The region from 10^5 to 10^9 and MAE 0.244–0.259 is empty: x 220–418, y 140–177 pt-n,
below the PatchTST label (which ends at y 129). The legend row (y 2.98 in) takes about 8% of the
figure height. If you stack the legend in three rows at the lower right, the frame can grow.
- **Legend:** in the legend code, set `x, y = 4.72, 1.86` and step `y += 0.21` instead of
  `x += 1.25`.
- **Frame:** set `Y1 = 2.80`, and move the x tick labels and the axis title down by the same
  amount.
- **Slide size:** keep `new_slide(6.0, 3.22)` so that the height still matches Fig. 1(a).
- **Effect:** the taller frame spreads the crowded MAE 0.28–0.30 cluster by about 10% vertically.

### P3: smaller issues
- **TimeMixer/iTransformer overlap:** the two markers overlap by about 9 pt-n, and TimeMixer is
  drawn on top. This is acceptable because the data are close. If you want to reduce it, change
  the slope in `diameter()` from `0.072` to `0.065`. This shrinks the largest markers by about
  0.05 in and also gives Crossformer and iTransformer daylight; they touch now, with a gap of
  -1 pt-n.
- **Legend swatches:** the legend uses hollow rings (`oval(..., None, col, ...)`), but the plot
  markers are filled with light colours. Fill the swatches with `LIGHT[col]` so they match.
- **Top margin:** the top of "TimeMixer" is about 2.4 pt-p below the top frame edge.
  Setting `my1 = 0.308` adds breathing room.
- **FEDformer:** FEDformer is silently omitted because its MAE of 0.400 is off-scale, while the
  introduction says "eight baselines". Consider a caption note, e.g. "FEDformer (MAE 0.400) lies
  above the plotted range".
- **Fig. 1(a) mismatch (sibling panel, outside this script):** Fig. 1(a) uses sans-serif
  (Helvetica) tick labels while Fig. 1(b) uses Times. A faint vertical rule is also visible at
  the left edge of (a). Placed side by side, the mismatch is visible.

### What works
- **PhaseFormer+ highlight:** the saturated red fill with a blue phase ring and the bold red
  label is the only saturated marker. It is found immediately and still shows its category.
- **Labels:** every model is labelled directly with its FLOPs. Showing MFLOPs next to GFLOPs
  carries the message without needing a size key.
- **Marker sizes:** printed diameters run from 8 to 24 pt-p. They are distinguishable and do not
  crowd the plot.
- **Legend placement:** the legend is centred within about 8 pt-n of the frame centre.

---

## Fig. 7: `bars()`

### P1 (must fix): y-axis titles sit next to the wrong panel
Panels (b), (c) and (d) each have a rotated y title ("Parameters", "GFLOPs", "Parameters") about
3 pt-n (1.9 pt-p) to the right of the *previous* panel's frame. They are about 28 pt-n from their
own tick labels. For example, "Parameters" spans x 215–229, while frame (a) ends at 211.9 and the
(b) ticks start at about 257.
- **Effect:** in print they read as right-hand axis titles of panels (a), (b) and (c).
- **Cause:** the code is
  `tb = label(x0 - 0.72 - 0.4, ...)`, which centres the title at `x0 - 0.72`.
- **Change:** use `label(x0 - 0.43 - 0.4, ...)`, which centres it at `x0 - 0.43`. The widest
  tick, "10^-3", has its ink left edge at about `x0 - 0.31`. The title then sits about 0.06 in
  from its own ticks and about 0.36 in from the previous frame.
- **Alternative:** drop the y titles. The panel titles already name the quantity, and the unit
  can go into the title, e.g. "(a) FLOPs on Traffic (G)". This also frees width for `W`.

### P1: the panel (c) Crossformer bar reaches the frame top
On Electricity, Crossformer uses 96.6 GFLOPs = 10^1.985, and panel (c) has `hi = 2.0`. The bar
therefore fills its slot up to the frame line. It looks clipped, and readers cannot tell whether
it exceeds the axis.
- **Change:** give both FLOPs panels the same range, `('(a) ...', 2, 'GFLOPs', -4.0, 3.0)` and
  `('(c) ...', 4, 'GFLOPs', -4.0, 3.0)`.
- **Tick step:** set it to 1 in the tick loop: `range(int(lo), int(hi) + 1)`, instead of
  `1 if hi - lo <= 6 else 2`. That gives 8 labels about 19 pt-n apart; each label is about
  12 pt-n tall, so they fit.
- **Bonus:** a shared range lets readers compare Traffic and Electricity directly. The two
  Parameters panels already share 10^2–10^8.

### P2: superscripts are 4.68 pt-p, and the minus sign is a hyphen
This is the same exponent-size issue as in Fig. 1(b). On these tick labels the exponent carries
all the information. The negative exponents also use an ASCII hyphen ("-3"), which at 4.7 pt-p
is a barely visible dash.
- **Change:** in the y-tick loop, enlarge the tick labels or their exponents to at least
  5 pt-p (see Fig. 1(b), P2), and emit a true minus sign:
  `f'$10^{{{e}}}$'.replace('-', '−')`. Times has U+2212. Check that `_math` routes it to a
  font that has the glyph.

### P3: legend spacing and wording
- **Spacing:** the legend uses a fixed `x += 1.35`. The gap from "Others" (ending at 385.6 pt-n)
  to the dashed swatch (about 431 pt-n) is about 45 pt-n. The gaps between the other entries are
  about 13.5 pt-n, more than three times smaller.
- **Off-centre:** with `LX = 1.9`, the legend's centre is about 33 pt-n right of the figure's
  centre.
- **Change:** advance `x` by the measured text width plus a constant gap. If you want to keep the
  PhaseFormer+ item visually separate, use a deliberate gap of about 0.35 in. Then recompute `LX`
  so that the whole legend is centred.
- **Wording:** "level and amplitude pathways of PhaseFormer+ (at most about 2.1K)" starts in
  lowercase and is long, and "at most about" reads awkwardly. Suggest
  "PhaseFormer+ level and amplitude pathways (≤ 2.1K)".

### P3: smaller issues
- **Panel (d) title width:** the "(d) Parameters on Electricity" title (x 689.5–875.4) is about
  30 pt-n wider than its panel and ends 1.6 pt-n from the page edge. It is not clipped, but it
  is tight. Adding about 0.1 in to the slide width (`new_slide(12.3, 3.35)`) would give it a
  margin.
- **Dashed segment:** the dashed PhaseFormer+ extension in (b) and (d) is only about 6 pt-p tall
  on the log axis, but its outline and fill are distinct and it reads. No change is needed. If
  you want more contrast, set the dashed segment's `lw` from 1.5 to 1.75 to match the solid bar.

### What works
- **Sort order:** sorted descending, each panel forms a staircase. PhaseFormer+ happens to fall in
  the same (7th) slot in all four panels, so the eye finds it at the same place each time.
- **Rotated labels:** the 45° labels are 7.1 pt-p, anchored at their bars, and do not collide
  with tick labels or with neighbouring panels; the closest pair, (b) TimeBase and (c)
  Crossformer, has a clear gap. The red bold PhaseFormer+ label is the strongest element in
  each panel.
- **Grid:** the light grid, thin frames and 0.62-slot bar width give a clean, uncluttered field.

---

## Priority summary

| # | Fig. | Change | Where in `make_efficiency.py` |
|---|------|--------|-------------------------------|
| 1 | 7 | Move the y titles beside their own panel | `bars()`: `label(x0 - 0.72 - 0.4, ...)` → `label(x0 - 0.43 - 0.4, ...)` |
| 2 | 1(b) | Move the Crossformer label off the right frame | `place['Crossformer'] = (0.12, -0.54, LEFT)` |
| 3 | 1(b) | Separate the SparseTSF label from "11.7 GFLOPs" | `place['SparseTSF'] = (0.30, 0.14, LEFT)` |
| 4 | 1(b) | Move the TimeBase label off the SparseTSF marker | `place['TimeBase'] = (-0.17, 0.39, LEFT)` |
| 5 | 7 | Give the panel (c) bar headroom; share the FLOPs ranges | `panels`: both FLOPs `lo, hi = -4.0, 3.0`; tick step 1 |
| 6 | both | Exponents ≥ 5 pt-p; true minus sign | tick `label(..., size=13.5)` or a sub/superscript scale argument; `.replace('-', '−')` |
| 7 | 1(b) | Explain the TimeBase double ring | caption or a legend item |
| 8 | 1(b) | Move the legend into the lower-right corner; enlarge the frame | legend at `(4.72, 1.86)`, stacked; `Y1 = 2.80` |
| 9 | 7 | Even, centred legend spacing; shorter fourth item | `bars()` legend loop, `LX` |
| 10 | 1(b) | Filled legend swatches, `my1 = 0.308`, optional `diameter` slope 0.065 | `scatter()` |

After the edits, re-export with `sh export.sh efficiency_plot 243; sh export.sh efficiency_bars 495`.
Then re-check p.3 and p.20 of main.pdf at 100 dpi, in particular whether the exponents can be
read.
