# Aesthetic reviews of the redrawn efficiency figures (2026-10-14)

Figures: `../../efficiency_plot.pdf` (Fig. 1(b)) and `../../efficiency_bars.pdf` (Section 6.3), drawn by `../make_efficiency.py` into `../efficiency_plot.pptx` and `../efficiency_bars.pptx`.

Three independent reviewers:
- **style** (`fig_review_style*.md`): consistency of style and font with the existing figures
- **layout** (`fig_review_layout*.md`): layout and readability at print size
- **accuracy** (`fig_review_accuracy*.md`): accuracy of the data and captions

Rounds and outcome:
- Round 1: all three passed with changes.
- Round 2: accuracy passed; style passed with one must-fix (the clipped 10^9 tick); layout raised two must-fixes (the 10^9 tick, labels touching the legend).
- Round 3: layout raised one new collision (y-title against the tick labels); the stated fix was applied.

Changes made after round 1:
- Arial, as in the other data plots
- per-dataset level/amplitude bound (2.1K on Traffic, 1.0K on Electricity)
- one PhaseFormer+ highlight style
- label collisions resolved
- tick marks and true minus signs
- lighter gray
- legends reworked, with TimeBase shown as "phase and patch"
- y-titles moved
- FLOPs panels on a common 1e-4 to 1e3 range
- captions state the encodings
