# Style and font consistency review: efficiency_plot.pdf (Fig. 1(b)) and efficiency_bars.pdf

Reviewer view: do the two new figures match the paper's existing figures in style and fonts?
Renders checked: both figures on their own (150 and 500 dpi), and main.pdf page 3 (Fig. 1)
and page 20 (efficiency figure), compared with la_overview, phaseformer_overview, token_energy,
coupling, spectrum_traffic, phase_token.png and router_number_effect_*_lines.png.

## Verdict: PASS WITH CHANGES

The colors, the print sizes (SMALL 12.5pt x 0.563 = 7.0pt, the same as the 7pt of token_energy
and coupling) and the frame weight (1.0pt x 0.563 = 0.56pt, against axes.linewidth 0.6) are
right. Both figures follow the same rules, so they look like a pair. Four problems remain:

- They use the font of the schematic figures (Times), but every other data plot in the paper
  uses a sans-serif font.
- Fig. 1(b) has a label that runs into the plot frame.
- PhaseFormer+ is highlighted one way in Fig. 1(b) and the opposite way in the bars figure.
- In the bars figure, each panel's y-axis title sits closer to the panel on its left than to its
  own axis.

## What already matches (keep)
- PHASE = TOKEN_BLUE #1F9FD8 is the same blue as the phase token in Fig. 1(a) and as BLUE in
  make_coupling.py. PATCH = TOKEN_GREEN #3E9B2A matches the patch-token green of Fig. 1(a).
  RED #D62728 is the same as RED in make_coupling.py and make_spectrum_traffic.py.
- GRID #E2E2E2 at 0.75pt (about 0.42pt printed) is close to the light solid grid of Fig. 1(a)
  and the gridlines of spectrum_traffic.
- Legends have no frame and sit outside the plot, as in token_energy and coupling.
- Fig. 1(b) already tells the classes apart without color, by outline style (solid, dash,
  sysDot). This is good.

## Change requests (in order of priority)

### P1. Switch both figures to Arial, the font of the paper's data plots
Every data plot in the paper uses a sans-serif font: token_energy and coupling (matplotlib, Arial
7pt), spectrum_traffic (a PowerPoint chart, FONT = "Arial", 7pt), and the tick labels of
phase_token.png and router_*_lines.png. Only the schematic figures (Figs. 3 and 4, la_overview
and phaseformer_overview) use Times. In Fig. 1, (a) has Arial-like ticks and (b) has Times ticks,
side by side in one figure. This is the most visible mismatch in the paper.
- In make_efficiency.py, before scatter() and bars() run, set `pptfig.MATH_FONT = 'Arial'`.
  `_math` looks MATH_FONT up when it is called, so the digits of `$10^{e}$` will change.
- Changing `pptfig.TEXT_FONT` will **not** work: `_run(..., font=TEXT_FONT)` binds the default
  value when the function is defined. Either give `write()`/`label()` a font parameter, or
  change `_run`'s default to `font=None` and resolve it as `font or TEXT_FONT` at call time.
  Then set the font to Arial for this script only, so Figs. 3 and 4 keep Times.
- Keep the sizes as they are (SMALL = 7pt printed). Arial is wider than Times, so check the label
  offsets in `place` and the legend steps again (see P2 and P7).
- Export needs Arial through the brew fontconfig environment variables, as
  make_spectrum_traffic.py already does.
- If the authors want to keep Times on purpose, as the PowerPoint family style, then at least
  Fig. 1(a) must change too. Since phase_token.png is a raster carried over from ICLR, switching
  the two new figures to Arial is much cheaper.

### P2. Fig. 1(b): the Crossformer label crosses the right edge of the frame
At print size, "Crossformer / 259 GFLOPs" overlaps the right frame line by about 0.06in on the
slide; the frame cuts through "r" and "s". It also sticks out past the plot in main.pdf, page 3.
- Fix it in `scatter()`. The Crossformer bubble's radius is about 0.29in, so `dx` cannot drop
  below 0.36. Change the axis range instead: set `lx1 = 9.4` and keep the ticks at
  `range(2, 10)`, which moves Crossformer about 0.2in to the left.
- Alternatively, put the label below the bubble with `(0.0, 0.36, PP_ALIGN.CENTER)`, but that
  space is tight against PatchTST.
- Arial (P1) makes this label wider still, so recheck it after the font change.

### P3. Highlight PhaseFormer+ the same way in both figures
- Fig. 1(b) draws it as a red fill with a blue outline: `oval(cx, cy, d, RED, PHASE, 1.75)`.
- efficiency_bars draws it as a blue fill with a red outline: `fill=PHASE, line=RED, lw=1.75`.

This is the same model with opposite encodings. The red fill also cancels its "phase-based = blue"
class in Fig. 1(b).

Recommended fix: use "class color fill + red outline" in both. In `scatter()`, use
`oval(cx, cy, d, PHASE, RED, 1.5)`, with a solid PHASE fill so it stands out from the
LIGHT[PHASE] bubbles. Keep the bold red label, which already matches the bold red tick label in
the bars figure.

If the red dot is preferred, which is a stronger signal, make the PhaseFormer+ bars red-filled
too. Either choice works, as long as both figures use the same one.

### P4. efficiency_bars: each y-axis title is too close to the panel on its left
In `bars()`, the unit label is centered at `x0 - 0.72`. Its glyphs reach to about
`x0 - 0.80`, while the panel on its left ends at `x0 - GAP = x0 - 0.85`. So "Parameters" in
panel (b) is about 0.05in from the frame of (a), but about 0.2in from its own tick labels. The
title looks as if it belongs to panel (a). Any one of these fixes it:
- (a) Move the label in to center at about `x0 - 0.60`. The widest tick label, `10^-3`, takes up
  about 0.40in to the right of `x0 - 0.44`.
- (b) Raise `GAP` from 0.85 to about 1.0 and lower `W` from 2.15 to 2.05, which keeps the total
  width at 12.2in.
- (c) Drop the rotated unit labels and put the unit in the panel title, for example
  "(a) Traffic FLOPs (G)" and "(b) Traffic parameters". This also frees width.

### P5. Lighter "Others" gray, so the bar classes stay apart in grayscale
In the bars, OTHER #8C8C8C, PATCH #3E9B2A and PHASE #1F9FD8 all have luma of about 115 to 140.
Printed in black and white, the three classes cannot be told apart, and the bars have no dash
style to fall back on.
- Make the gray lighter, to about #B4BEC9 (OTHER in make_coupling.py) or #C9CDD2 (REST in
  make_token_energy.py). This also brings the paper's neutral gray into line.
- Change `OTHER` and `LIGHT[OTHER]` together. The light version could be about #E4E7EA.
- Optional: give patch-based bars a pattern fill, which keeps blue and green apart without
  color.

### P6. Axis ticks: add short tick marks, use a real minus sign, and keep the bar off the frame
- Both figures have tick labels but no tick marks. Every other data plot in the paper has short
  outward ticks (0.6pt wide, 2.5pt long). At a scale of 0.563, that is a 1.0pt line about 0.06in
  long. Add them with the existing `vline`/`hline` helpers:
  - in `scatter()`, at each x tick 10^e and each y tick m;
  - in `bars()`, at each y tick.

  Use `color=INK, lw=1.0` and draw them outside the frame.
- The exponents use an ASCII hyphen: `f'$10^{{{e}}}$'` with e < 0 gives "10-3". Use
  U+2212 instead, for example `str(e).replace('-', '−')`. Both Times and Arial have this
  glyph.
- The PhaseFormer+ bar outline (lw 1.75, centered on the edge of the shape) reaches below the
  baseline Y1 and crosses the bottom of the frame. Inset the box by half the line width, or lower
  lw to 1.25.
- In panel (c), Crossformer is 96.6 GFLOPs, about 10^1.98, so the bar touches the top of the
  frame. Either accept it or raise `hi` to 3.0 for that panel.
- Keep the full closed frame. It matches Fig. 1(a), the figure beside 1(b), and both new figures
  use it. Open L-shaped axes as in token_energy and coupling would be fine too, but use the same
  axes in both new figures.

### P7. Legends
- In Fig. 1(b), the legend swatches are outline-only (`oval(..., None, col, ...)`), but the
  markers are filled with LIGHT[...]. Pass `LIGHT[col]` as the fill so each swatch looks like
  its marker.
- In Fig. 1(b), TimeBase gets a dashed green ring ('both') that the legend and caption never
  explain. Either explain it, or drop it so TimeBase is a plain phase marker, as in the bars,
  where 'both' maps to PHASE.
- **Move the Fig. 1(b) legend inside the plot.** The empty area at the lower right
  (MAE 0.244 to 0.26, parameters 10^5 to 10^9) can hold a vertical legend. Then cut the slide
  height in `new_slide(6.0, 3.22)` to about 2.95in. That brings the aspect ratio to about 2.04,
  which matches phase_token.png (1947x953 pixels, 2.04). Now, at equal widths, the plot box of
  (b) on page 3 sits about 12pt higher than the plot of (a) and is shorter. With the legend
  inside, the two plot areas line up.
- In the bars, the gap between "Others" and the dashed swatch is wider than the other gaps,
  because `x += 1.35` is a fixed step. Step by the text width instead.
- Shorten the last legend entry and capitalize it like the others, for example
  "PhaseFormer+ level and amplitude pathways (≤ 2.1K)". The caption already explains it.

### P8. Type weight and color
- The bar panel titles use TEXT 15pt bold, about 8.5pt bold printed: the heaviest text in any
  data figure. token_energy and coupling use only a bold "(a)" of 7.5 to 8pt. Lower them to about
  13.5pt (7.6pt printed), with "(a)" bold and the rest regular. If P4 (c) is taken, use that
  wording.
- Text colors are mixed. Model labels use INK #222222, but tick labels and axis titles use the
  default BLACK of `label()`. Pass `color=INK` to every `label()` call, to match INK in the
  matplotlib figures.

## Priority summary
- **Must fix:** P1 (font family), P2 (label crossing the frame), P3 (consistent highlight),
  P4 (y-axis title placement).
- **Should fix:** P5 (grayscale gray), P6 (tick marks and minus sign), P7 (legend swatches,
  legend inside Fig. 1(b) to line it up with 1(a)).
- **Nice to have:** P8.
