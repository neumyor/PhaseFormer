# Review: redrawn efficiency figures (accuracy of data, encoding and captions)

Figures: `figures/efficiency_plot.pdf` (Fig. 1(b)) and `figures/efficiency_bars.pdf` (Fig. `fig:flops_params_visualization`, Section 6.3), both from `figures/src/make_efficiency.py`. I rendered both at 150 dpi, and I also rendered the old Fig. 1(b) from commit 49f1049.

## Verdict

**Accept with minor revisions.** Every plotted baseline value matches its source exactly. Moving from the old figure's numbers to the supplement-table numbers is right, and it is what makes the Introduction's "about 36,000" and "about 99.99%" claims hold. One real data problem remains: on Electricity the dashed level/amplitude segment is drawn with the Traffic bound of 2.1K. The true Electricity bound is about 1.0K, so the drawn increment is about twice too large. The other issues are gaps in the captions. They are not wrong numbers, but each one could mislead a careful reader: TimeBase's "both" status, the omission of FEDformer, the log-scaled marker size, the horizon basis of parameters and FLOPs, and the FLOPs of PhaseFormer+ in Fig. 1(b).

## 1. Data against the sources

**Parameters and FLOPs (`tab:params_flops_details`, body/appendix.tex lines 133-141).** All 18 baseline values per dataset (Traffic and Electricity, parameters and FLOPs) in `MODELS` match the table digit for digit. This covers PatchTST, iTransformer, Crossformer, FEDformer, TimeBase, SparseTSF, FITS and TimeMixer. PhaseFormer+ uses 1.156K / 13.9 (Traffic) and 1.156K / 5.18 (Electricity), which is the conference row.

**Traffic average MAE (Table `main_results`).** All nine values match: 0.252, 0.266, 0.290, 0.282, 0.400, 0.278, 0.280, 0.290, 0.298. The rendered marker positions agree with these values.

**Units.** The script reads the table's unitless "FLOPS" column as MFLOPs. This is consistent with three things:
- the conference bar PNG (`flops_traffic.png` puts PatchTST at about 500 GFLOPs, which is 498,577 MFLOPs);
- the CycleNet and PatchMLP rows, which carry an explicit "M";
- the magnitudes (13.9 MFLOPs for a model of about 1.2K parameters over 862 channels is about 16K FLOPs per channel, which is plausible per window).

The table caption still gives no unit, and the figures now print "MFLOPs/GFLOPs" on the strength of an inference. See CR-7.

**Consistency with the text.**
- 498,577.49 / 13.9 = 35,869. With the at most 0.23% of extra FLOPs from the level and amplitude pathways it is about 35,790. Both round to "about 36,000" (Introduction).
- 1 - 13.9/498,577 = 99.997% and 1 - 13.9/259,210 = 99.995%. Both give "about 99.99%" (Introduction, Section 6.3, Section 6.7 summary, Conclusion).
- "PatchTST holds 7.6M parameters" matches 7.589M.

**The old figures contradicted these claims.** The old Fig. 1(b) used 113 MFLOPs for PhaseFormer and 168 GFLOPs for PatchTST, a ratio of about 1,500, not 36,000. It also placed PatchTST at about 3-4M parameters (the text says 7.6M), PhaseFormer below 10^3 parameters, iTransformer at 2 GFLOPs and TimeMixer at 6 GFLOPs. None of these match the table. Switching to the table numbers removes a real inconsistency, so it is a correction, not a cosmetic change. The old bar PNGs were already on the table basis (for example PhaseFormer at about 0.014-0.025 GFLOPs), so the bars change only in labels, categories and the Autoformer removal.

## 2. The PhaseFormer+ encoding

**Traffic.** The bound is faithful. The recorded breakdown (docs/agent-log.md, Traffic-720) is level pathway 105 (fixed profiles, K=30, independent of H) + amplitude 262 (w=2 on all Traffic settings, so the same at every horizon) + gates 2C = 1,724. That totals 2,091, about 2.1K, and matches "at most about 2.1K ... the largest increment being on Traffic". Placing PhaseFormer+ at 1,156 + 2,100 = 3,256 in Fig. 1(b), and stacking the dashed segment up to 3,256 in panel (b), are both correct. The bound is a conservative upper bound: Traffic-336 switches the amplitude pathway off at inference but still carries its parameters.

**Electricity.** The encoding is not faithful. The script adds `EXTRA = 2100` in panel (d) as well. On Electricity C = 321, so the gates add 642. With every Electricity setting on fixed profiles and w=2, the increment is at most 105 + 262 + 642 = 1,009, about 1.0K (assuming the same latent width d as Traffic, which the 14-262 range implies is the maximum). The bar therefore tops out at about 3.26K instead of about 2.17K, about 0.18 decades too high on the log axis, which is visible. It also contradicts the text, which says the increment is largest on Traffic. The legend wording "(at most about 2.1K)" is technically still an upper bound, but drawing the Traffic value on Electricity misstates the figure's own data. See CR-1.

**FLOPs drawn as the phase pathway.** This is consistent with Section 6.3 (an extra 0.09%-0.23% on Traffic and Electricity, which is invisible on a log axis and changes no label at three significant figures). The bars caption explains it. The Fig. 1 caption does not: it says only that the parameters include the extra pathways, so a reader cannot tell that the 13.9 MFLOPs label covers the phase pathway only. See CR-4.

**Parameters on the conference basis (1.156K).** This holds, and it matches "about 1.2K parameters" everywhere in the text.

## 3. Captions

**Fig. 1 (body/introduction.tex, line 24).**
- Axes: "average MAE over the four horizons against the number of parameters". Correct, but the x-axis is logarithmic (10^2 to 10^9) and the caption does not say so. The tick labels show it, so this is minor.
- Marker size: "growing with the FLOPs per forecast" is true but under-specified. The diameter is linear in log10(FLOPs) (`0.11 + 0.072*(log10(MFLOPs)+1)`). PatchTST has about 283,000 times the FLOPs of FITS but only about 9 times the marker area. A reader who assumes area-proportional sizing will badly underestimate the gap that the figure is meant to show. The printed FLOPs labels rescue the figure, but the caption should say "logarithmically" and point to the labels. See CR-3.
- Horizon and averaging: "(input length 720, horizon 96)" follows the FLOPs clause, so it reads as applying to the FLOPs only. In fact the parameters are also at H=96, while the MAE averages four horizons. Say so explicitly. See CR-5.
- No explanation of the dashed green ring around TimeBase. The old figure had a "Both" legend entry, and the new legend lists only three categories. See CR-2.
- No mention that FEDformer, one of the eight baselines, is left out. See CR-2.

**fig:flops_params_visualization (body/experiments.tex, line 102).**
- "logarithmic scales", "input length of 720 steps and a horizon of 96", "FLOPs per forecast": correct and complete.
- "Phase-based models are blue": TimeBase is drawn plain blue, although Section 6.1 says it "integrates patch and phase paradigms" and Fig. 1(b) marks it as both. The two figures disagree and neither caption says how "both" is coloured. See CR-2.
- "the dashed part adds the at most about 2.1K parameters": wrong for Electricity, as shown in Section 2 above. See CR-1.
- "all models": the figure shows PhaseFormer+ and the eight baselines, but the supplement table it is drawn from also lists CycleNet and PatchMLP, and Section 6.3 discusses CycleNet's parameters right next to the figure. "the eight baselines" would be exact. See CR-6.

## 4. Categories

- Patch-based PatchTST, Crossformer and TimeMixer; phase-based SparseTSF; others FITS and FEDformer (frequency-domain) and iTransformer (whole sequence). All of these match the Section 6.1 taxonomy.
- TimeBase "both": correctly encoded in Fig. 1(b) (blue fill with a dashed green ring) but not in the legend or caption. In the bars it is plain blue with no cue. See CR-2.
- Autoformer is not among the eight baselines of Section 6.1, so dropping it from the bars (the old PNGs had it) is correct.
- FEDformer in Fig. 1(b): MAE 0.400 is far above the 0.244-0.306 range, and extending the axis would squash the remaining points into a band, so omitting it is a reasonable choice. Because the caption implies a full comparison, though, the omission has to be stated. The bars do include FEDformer, which is correct.

## Prioritized change requests

**CR-1 (high, data). Use a dataset-specific bound for the dashed segment.** In panel (d) (Electricity), draw the level/amplitude increment with its Electricity bound (105 + 262 + 642 = 1,009, about 1.0K; confirm against the recorded `parameter_breakdown` of the Electricity settings) instead of `EXTRA = 2100`. For example, make `EXTRA` a per-dataset value {Traffic: 2100, Electricity: ~1010}. Then update the legend text, e.g. "level and amplitude pathways of PhaseFormer+ (at most about 2.1K on Traffic, 1.0K on Electricity)", and the bars caption, e.g. "the dashed part adds the parameters of its level and amplitude pathways (at most about 2.1K on Traffic and 1.0K on Electricity)". No experimental number changes; this only stops plotting the Traffic bound on Electricity. If the authors prefer not to introduce a new number, the minimum fix is to state in the caption that the Traffic bound is drawn on both datasets, but the dataset-specific bar is more accurate.

**CR-2 (high, caption and legend). Disclose TimeBase "both" and the FEDformer omission.**
- Fig. 1: add a fourth legend item (a blue circle with a dashed green ring, labelled "Both"), or add a caption clause such as "TimeBase, which combines phase and patch tokens, is ringed in green". Also add "FEDformer (MAE 0.400) lies above the plotted range."
- Bars: give TimeBase a cue (for example a green dashed outline on its blue bar, plus a legend entry), or add a caption clause such as "TimeBase, which combines both, is drawn blue".

**CR-3 (medium, caption). State that marker size is logarithmic in FLOPs.** For example: "marker diameter grows logarithmically with the FLOPs per forecast, which are printed under each name". Without this the visual ratio understates the cost gap by several orders of magnitude.

**CR-4 (medium, caption). In Fig. 1, state what the FLOPs label of PhaseFormer+ covers.** For example: "...plus the at most about 2.1K parameters of its level and amplitude pathways; its FLOPs are those of the phase pathway, to which these pathways add less than 0.25%." This matches the bars caption and Section 6.3.

**CR-5 (medium, caption). Make the horizon basis explicit in Fig. 1.** For example: "average MAE over the four horizons against the number of parameters (logarithmic), with marker size growing with the FLOPs per forecast; parameters and FLOPs are for input length 720 and horizon 96."

**CR-6 (low, caption). Replace "all models" with "PhaseFormer+ and the eight baselines"** in the caption of fig:flops_params_visualization, and optionally in the sentence "the FLOPs and the number of parameters of all tested models" in Section 6.3. This keeps a reader from expecting CycleNet and PatchMLP, which the supplement table lists.

**CR-7 (low, source table; affects the figures' printed units). Give `tab:params_flops_details` an explicit unit and basis.** Its caption ("Parameters and FLOPS across models for different datasets") gives no unit. The rows mix unitless MFLOPs with "720.27M"-style entries for CycleNet and PatchMLP. Adding "FLOPs in MFLOPs per input window, L=720, H=96" (and writing the CycleNet/PatchMLP entries without the "M", values unchanged) would make "per forecast" and the figures' MFLOPs/GFLOPs labels traceable. This changes only the presentation of the numbers, not the numbers.

**CR-8 (cosmetic).** In Fig. 1(b) the "Crossformer / 259 GFLOPs" label runs past the right frame of the plot. Nudge it left or below the marker. In the bars, the Crossformer bar in panel (c) nearly reaches the top of the frame (96.6 GFLOPs against 10^2). That is acceptable, but raising `hi` to 3 would give it headroom.

No change to any experimental number is proposed, and parameter counts stay on the conference basis (phase pathway 1.156K).
