# Round 2: efficiency figures (accuracy of data, encoding and captions)

## Verdict: PASS. No must-fix items remain.

I re-rendered both figures (dated 2026-10-14 22:59) at 150 dpi and re-read the script and both captions. The `MODELS` data is unchanged and still matches `tab:params_flops_details` and Table main_results.

| Round-1 item | Status |
|---|---|
| CR-1: Electricity bound | **Resolved.** `EXTRA = {'Traffic': 2100, 'Electricity': 1000}`. In panel (d) the dashed segment now ends at about 2.2K, against about 3.3K on Traffic. The legend reads "(≤2.1K / ≤1.0K)", and the caption gives "at most about 2.1K on Traffic and 1.0K on Electricity". The Electricity figure (105 + at most 262 + 642, about 1.0K) is consistent with Table la_config. |
| CR-2: TimeBase "both" | **Resolved.** There is a "Phase and patch" legend entry in both figures. The bars give TimeBase a dashed green outline on a blue fill, and the Fig. 1(b) ring is explained in both captions. |
| CR-2: FEDformer omitted from Fig. 1(b) | **Resolved.** The caption says "FEDformer (MAE 0.400) lies outside the plotted range". |
| CR-3: marker size is logarithmic | **Resolved.** The caption says it "grows logarithmically with the FLOPs per forecast, printed below each name". |
| CR-4: PhaseFormer+ FLOPs in Fig. 1 | **Resolved.** The caption says its FLOPs "are those of the phase pathway, to which they add less than 0.25%". |
| CR-5: horizon basis | **Resolved.** The caption has "average MAE over the four horizons against the number of parameters at horizon 96 (input length 720)". |
| CR-6: "all models" | **Resolved in the caption.** It now says "PhaseFormer+ and the eight baselines". |
| CR-8: Crossformer label overflow | **Resolved.** The label is inside the frame. Panel (c) now has headroom. |

The Introduction and Section 6.3 claims ("about 36,000", "about 99.99%", "7.6M") still match the plotted data.

## Optional, not blocking

1. Section 6.3, line 110, still says "all tested models" next to the figure. "PhaseFormer+ and the eight baselines" would match the caption.
2. The caption of `tab:params_flops_details` still gives no FLOPs unit (round-1 CR-7). The figures' MFLOPs/GFLOPs labels are correct on the conference basis, so this is only a matter of traceability in the supplement.
3. The legend's "(≤2.1K / ≤1.0K)" does not say which dataset is which, but the caption does, so this is acceptable.
