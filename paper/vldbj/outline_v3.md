# Proposed section outline v3 (Sections 3–8)

Status: proposal; no .tex file has been changed. Revised from outline_v2.md after a second review round
(three reviewers; all three: "adopt with changes").
Principle: the conference block comes first and stays intact; the extension follows as one block that mirrors it
(problem → theory → model → evaluation), in the order of the introduction.

Legend: [C] conference material (editorial changes only, marked blue), [N] new. Existing labels are kept unless noted.
Items marked (DECIDE) are the two choices on which the reviewers disagree.

---

## Conference block

### 3 Motivations: Phase Tokenization and Its Stability [C] (`sec:motivations`)
- Insight 1, Insight 2, Theorem 1 and its takeaways — unchanged
- REMOVED: the paragraph "What the theorem does not cover: the cycle axis" → rewritten into 6.1
- Title: "Motivations:" kept so the conference section stays recognizable; the subtitle exposes Theorem 1 (blue)

### 4 PhaseFormer [C] (`sec:method`; renamed from "Methodology", blue)
- 4.1 Data Pre-Processing; 4.2 Phase-based Routing Transformer (4.2.1–4.2.3); 4.3 Complexity of PhaseFormer — unchanged
- 4.4 Discussion of the Periodicity Assumption (renamed from "Discussion", blue, so it does not repeat the title of
  Section 9; content unchanged)

### 5 Experiments [C] (`sec:exp`; current Section 7)
- 5.1 Long-term Time Series Forecasting (`sec:exp_main`)
  - "Setup for PhaseFormer-LA" → 8.1
  - "Results of PhaseFormer-LA" → end of 8.4
  - the LA sentence at the end of "Efficiency Comparison" (experiments.tex l.145) → 7.3 (architectural count) and
    8.1 / App. D (per-setting range 119–679)
  - (DECIDE) the PhaseFormer-LA row of `tab:main_results` and its caption sentence — see choice A below
- 5.2 Ablation Studies and Analysis; 5.3 Sensitivity Analysis of Phase Length Choices; 5.4 Case Study — unchanged

## Extension block

### 6 Changes Across Cycles: How Far PhaseFormer Can Follow Them [N] (`sec:theory`)
- Opening (new, short): restates the premise of Theorem 1 in one sentence (a common shape change); points to the
  case study of 5.4, where phases 1 and 9 share a 7-day cycle but drift in opposite directions — a change across
  cycles that Theorem 1 does not cover; gives the roadmap, optionally with `tab:two_envelopes` as a preview.
- 6.1 Level Drift and Level-Dependent Amplitude (new label `sec:theory_changes`)
  - the moved paragraph, rewritten in the K×P notation of 6.2 (one mapping to Theorem 1: D = K cycles, H = P phases);
    its pointers to `sec:theory`/`sec:extension` are removed because it now sits inside Section 6
  - the two changes `eq:cycle_axis_changes`, why each breaks Theorem 1, why level-dependent amplitude leaves the
    patch subspace unchanged
  - the multiplicative seasonal model `eq:mult_model`, introduced here once
    ("the multiplicative seasonal model of Section 3" in theory.tex l.28 is retargeted)
  - the questions the section answers: for level drift, can PhaseFormer represent it, how far can it follow it, what
    carries the rest; for level-dependent amplitude, how far, and what carries the rest (the "represent" question is
    promised only where a theorem answers it)
- 6.2 Notation and Error Decomposition (`sec:theory_setup`; renamed from "Setup and the three-way error split")
  - a self-contained recap of the notation (it no longer says "we use the notation of Section 4"), stating the
    transpose relative to Section 4 and the meanings of ℓ and H in this section
  - LN₂ and the shared predictor, the three-way error split, the level demand Λ
  - the Bayes-forecast and common-shift sentences move to the head of 6.3 (they use the cycle means defined here)
- 6.3 Level Drift (run-in paragraphs named after the results)
  - Capability: Theorem 2
  - Level envelope: Theorem 3, Corollary (steps and trends), the capability–envelope trade-off
  - Low-rank level complement: Theorem 4 (`sec:theory_complement` dropped; it is never referenced)
- 6.4 Level-Dependent Amplitude (`sec:theory_amplitude`)
  - Amplitude envelope and multiplicative demand: Theorem 5(a)(b)
  - Level-invariant modulation: Theorem 5(c)(d)
  - PhaseFormer-L stays defined here (it is the object of Theorem 5); see choice B
  - optional: split Theorem 5 into `thm:amplitude` (a, b) and `thm:modulation` (c, d)
- 6.5 Summary and Scope: `tab:two_envelopes` and the Scope paragraph
- heading vocabulary: "complement" throughout the headings (the theorem names and the table use it)

### 7 PhaseFormer-LA [N] (`sec:extension`)
- opening: architecture unchanged; Fig. `la_overview`; results-to-components map (last rows of `tab:two_envelopes`);
  the sentence citing "Sections ext_branch and ext_fusion" is rewritten
- 7.1 Level Branch and Gated Fusion (`sec:ext_branch`; current 6.1 + 6.2; `sec:ext_fusion` dropped): the branch, the
  level and shape gates, the whole-cycle requirement; "the branch alone gives PhaseFormer-L (Section 6.4)"
- 7.2 Centered Amplitude Modulation (`sec:ext_modulation`; plus the "Preserved guarantees" sentence)
- 7.3 Nesting and Cost (`sec:ext_properties`): exact nesting, architectural parameter count (105 + k(3 + d)) and
  complexity, mirroring 4.3

### 8 Evaluating the Theory and PhaseFormer-LA [N] (`sec:ext_exp`; renamed from "Extension Experiments")
- 8.1 Setup (`sec:ext_setup`): absorbs "Setup for PhaseFormer-LA" (Huber loss, 30 epochs, patience 8, seeds
  2021–2023) and the per-setting parameter range; names the two LA results distinctly — the per-setting configuration
  of App. D (used against the baselines) and the fixed-protocol runs of the ladder
- 8.2 Level–Amplitude Coupling in the Benchmarks (`sec:exp_coupling`)
- 8.3 Envelopes and Level Hand-over on Trained Checkpoints (`sec:exp_theory`; moved before the ladder, which refers to it
  twice); absorbs "Where the gain occurs" (per-window gain by |Λ| quintile) from the mechanism subsection, since it is
  the result promised in intro P8
- 8.4 From PhaseFormer to PhaseFormer-LA (`sec:exp_ladder`), ending with the comparison against the baselines
  (the moved "Results of PhaseFormer-LA" paragraph, merged with the ladder's own "17 of 28" sentence; this keeps the
  ladder's existing promise that the conference numbers are discussed at the end)
- 8.5 Mechanism of the Modulation (`sec:exp_mechanism`; synthetic series, cycle-mean invariance, what drives it)
- 8.6 Capacity of the Level Branch (`sec:exp_capacity`)
- extension_experiments.tex l.5 (question list) and l.51 (pointer to the main table) follow the new order

### 9 Discussion and Conclusion — unchanged in this proposal

---

## Consequences outside Sections 3–8
- Intro, "Relation to the conference version": "Sections 3–5 follow the conference version …; Sections 6–8 and
  Appendices B–D are new" (also fixes the existing range `app:proofs`–`app:ext_results`, which omits `app:la_config`);
  the "except for … end of Section 3" clause is deleted; if choice A keeps the LA row, add "except for the
  PhaseFormer-LA row of Table X".
- Intro, contributions: references become monotone (3; 4–5; 6; 6–7; 8).
- main.tex input order: motivations, method, experiments, theory, extension, extension_experiments.
- appendix_theory.tex l.186 ("main-table PhaseFormer-LA of Section exp_main") follows choice A.
- Every other \ref resolves unchanged (checked by the reviewers: 4.3 and 5.1 cite `sec:motivations`, statements.tex cites
  `sec:exp_main`, the intro cites `sec:exp_coupling` and `sec:exp_theory`, theorem numbers stay 1–5).

## (DECIDE) Choices on which the reviewers disagree
A. Where PhaseFormer-LA meets the baselines
   - A1 keep the blue LA row in the conference main table, with a caption pointer to Sections 7–8; no new table
     (reviewer A: no duplication, the headline table shows the final model)
   - A2 remove the row; add to 8.4 a full table in the format of `tab:main_results` with all baselines, the conference
     PhaseFormer, the retrained PhaseFormer of the ladder protocol, and PhaseFormer-LA, marks recomputed
     (reviewer C; reviewer B wanted a compact version) — the conference block stays pure and the protocols are explicit,
     at the cost of repeating the baseline numbers
B. Generic theory statements (PhaseFormer-L and the gates only in Section 7, Theorem 5 stated for an abstract fused model)
   - B1 defer; keep PhaseFormer-L defined in 6.4 (reviewer A, given the deadline; the appendix proofs cite the model
     equations of Section 7)
   - B2 do it in the same pass as the move (reviewers B and C)
