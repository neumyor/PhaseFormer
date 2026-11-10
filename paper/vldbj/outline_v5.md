# Section outline v5 (integrated structure, revised after the review of v4)

Status: proposal; no .tex file has been changed. Revises outline_v4.md; all three reviewers: "adopt with changes".
Principle: one paper, every kind of material in one place — observations (3), model (4), theory of the model (5),
experiments (6). Old and new are distinguished by blue text, blue headings of all-new subsections, and an itemized
"Relation to the conference version" paragraph.

Legend: [C] conference, [N] new, [C+N] conference with new parts. Labels move with their text unless noted.

---

### 3 Motivations [C+N] (`sec:motivations`) — properties of the tokens and the data, independent of any model
- 3.1 Phase Tokens Are Stable and Low-Dimensional [C]: Insight 1, Insight 2, Theorem 1 and its takeaways
  (Theorem 1 stays here; its proof stays in A.7)
- 3.2 Changes Across Cycles [N, blue heading] (new label `sec:mot_changes`)
  - the current paragraph "What the theorem does not cover", with `eq:cycle_axis_changes` and `eq:mult_model`
    (both equations live here; Section 5 refers back)
  - Insight 3 (moved from current 8.2 with `tab:coupling`, label `sec:exp_coupling` kept): cycle means and
    within-cycle amplitudes are strongly correlated in every benchmark, far above an additive control; reframed as a
    model-free observation (no "Theorem 5(b) predicts", no forward claim about where PhaseFormer performs best);
    one sentence reconciling it with Insight 1 (stability over weeks vs coupling within a 30-cycle window)
  - closes on the question: how far can a model with fixed weights follow these changes, and what must be added?

### 4 Methodology [C+N] (`sec:method`)
- 4.1 Data Pre-Processing [C]: normalization, phase tokenization
- 4.2 Phase-based Routing Transformer [C+N]: 4.2.1 Embedding, 4.2.2 Cross-Phase Routing, 4.2.3 Predictor;
  the routing-unit equations (attention, LN₁, MLP, LN₂, residuals; now only in theory.tex and the preamble of App. B)
  and the predictor in the form Φh_p + c_Φ are added in blue, so the architecture the theorems analyze is specified here
- 4.3 PhaseFormer-LA: Supplying the Missing Level and Amplitude [N, blue heading] (`sec:extension`; Fig. `la_overview`)
  - opening: recalls in the intro's words what a trained PhaseFormer cannot reach (two envelopes; the missing level is a
    combination of a few fixed profiles predictable from the cycle means; the missing amplitude is the cycle shape scaled
    with the level), each with a pointer to Section 5; then the one-channel notation the components need
    (K, P, K′, L, H, the normalized window, the folded matrix and its relation to the phase matrix of 4.1, cycle means,
    M_c, J_c) — fixed here and reused by Section 5
  - 4.3.1 Level Branch and Gated Fusion (`sec:ext_branch`; current 6.1 + 6.2; `sec:ext_fusion` dropped): what the
    branch is and why it reads only cycle means and uses a few smooth profiles (from 3.2), the level and shape gates,
    the whole-cycle requirement; PhaseFormer-L is named here
  - 4.3.2 Centered Amplitude Modulation (`sec:ext_modulation`; current 6.3): the modulation fully defined here
    (Theorem 5(c)(d) will refer to it); the one-line property that centering leaves every future cycle mean unchanged
  - Exact nesting (current 6.4, label `sec:ext_properties` moves here): LA ⊃ L ⊃ PhaseFormer, identical forecasts at
    initialization
  - configuration selection (grid search, the rule that selects decoders/modulation) moves to 6.1, so 4.3 stays
    architectural
- 4.4 Complexity [C+N]: current 4.3 + one blue paragraph with the architectural cost of the two components
  (105 + k(3 + d) parameters plus gates); the per-setting range moves to 6.3 / App. D
- 4.5 Discussion of the Periodicity Assumption [C] (current 4.4, retitled so it does not repeat Section 7's title)

### 5 How Far PhaseFormer Can Follow Changes Across Cycles [N, blue heading] (`sec:theory`)
- opening: the three questions (represent? how far? what carries the rest?), answered for level drift and for
  level-dependent amplitude; optionally `tab:two_envelopes` as a roadmap
- 5.1 Error Decomposition (`sec:theory_setup`): refers to the notation of 4.3 and the changes of 3.2 instead of
  redefining them; the three-way error split; the demands Λ and Ω; the stylized model is cited from 3.2
- 5.2 Level Drift (Theorem 2 capability, Theorem 3 level envelope, Corollary, trade-off, Theorem 4 low-rank level
  complement; run-in paragraphs, or level-3 headings)
- 5.3 Level-Dependent Amplitude (`sec:theory_amplitude`; Theorem 5: amplitude envelope, multiplicative demand,
  level-invariant modulation; refers to the modulation of 4.3.2; optional split into two theorems)
- 5.4 From the Envelopes to PhaseFormer-LA: `tab:two_envelopes`; the design-to-theorem map (each design choice of
  4.3 and the theorem part that requires it — the "follows Theorem 4 term by term" sentences move here); the current
  "Preserved guarantees" paragraph; the Scope paragraph, ending on "whether an envelope binds is a property of a
  checkpoint and a dataset" (→ 6.4)
- the theory text deletes its own re-definitions (the "We call PhaseFormer-L the model …" paragraph, `eq:modulation`
  as a duplicate of `eq:la_modulation`)

### 6 Experiments [C+N] (`sec:exp`; current Sections 7 and 8 merged)
- 6.1 Setup (`sec:exp_main` stays here; statements.tex cites it): datasets, baselines, implementation; the training
  protocol and configuration selection of PhaseFormer-LA; names for the four runs that appear
  (PhaseFormer conference, PhaseFormer retrained, PhaseFormer-LA fixed protocol, PhaseFormer-LA per-setting of App. D)
  and which table uses which
- 6.2 Main Results (new label): one table `tab:main_results` with the baselines, PhaseFormer and PhaseFormer-LA; the
  caption states which LA run it reports and how the best/second-best marks treat it
- 6.3 Efficiency: current "Efficiency Comparison" + the LA parameter counts in the text
- 6.4 Validating the Theory [N, blue heading] (`sec:exp_theory`; `sec:ext_exp` retargeted here)
  - Envelopes and level hand-over on trained checkpoints, with "where the gain occurs" (gain by |Λ| quintile)
  - Capacity of the level branch (`sec:exp_capacity`; a test of Theorem 4)
  - Mechanism of the modulation (`sec:exp_mechanism`; synthetic series, cycle-mean invariance, what drives it)
- 6.5 Ablation Studies [C+N]
  - Routers [C]: number of routers, effectiveness of cross-phase routing
  - From PhaseFormer to PhaseFormer-LA [N] (`sec:exp_ladder`): the component ladder; it now refers back to 6.4; the
    moved "Results of PhaseFormer-LA" paragraph is merged with it so the "17 of 28" comparison is stated once
- 6.6 Sensitivity to the Phase Length [C] (`sec:phase_length`)
- 6.7 Case Study [C] (`sec:exp_case`)
- the five-question roadmap of current Section 8 becomes one sentence in the Section 6 opening or is dropped

### 7 Discussion and Conclusion — unchanged in this proposal

### Appendices — order unchanged (A conference supplement incl. the proof of Theorem 1; B proofs for Section 5;
C extended LA results; D per-setting LA configuration)

---

## Consequences
- Intro, contributions: C1 Section 3; C2 Sections 4.1–4.2 and 6; C3 Section 5; C4 Sections 4.3 and 5; C5 Sections 3.2
  and 6 (wording "derive" in C4 stays justified by 4.3's opening and 5.4).
- Intro, "Relation to the conference version": an itemized list of the new units (3.2 with Table coupling; the routing
  equations in 4.2; 4.3; the LA paragraph of 4.4; Section 5; in Section 6 the LA protocol, the LA row, 6.4 and the ladder
  in 6.5; Appendices B–D, fixing the current range that omits D; the extended related work; the limitations); one
  sentence on what is unchanged or only renumbered. Moved conference text stays black; the cover letter maps
  conference sections to journal sections.
- Cross-references to retarget (from the reviewers' check): `sec:ext_exp` (intro C5, theory Scope, appendix_theory);
  `theory.tex` l.28 "model of Section motivations"; `appendix_theory.tex` l.186 → 6.2; `extension_experiments.tex` l.5,
  l.51, l.78, l.80; `experiments.tex` l.133 and l.145; `sec:ext_setup` citations inside the LA method text.
- main.tex: theory.tex, extension.tex and extension_experiments.tex are absorbed (extension → method, extension
  experiments → experiments); the input order becomes motivations, method, theory, experiments.
- Known issues to fix in the body revision, not structural: Theorem 1's statement uses M and N′ without definition and
  its proof setup names the matrix axes inconsistently; MMD's P, Q clash with the period P; H means phases (Theorem 1),
  heads (4.4) and horizon (Section 5) — use d_h for heads.
