# Proposed section outline v4 (integrated, "scheme 2")

Status: proposal; no .tex file has been changed. Supersedes outline_v2.md and outline_v3.md (two-block order,
rejected by the authors as scattered and redundant).
Principle: one paper, one storyline, every kind of material in exactly one place —
one motivation section (observations), one method section (PhaseFormer and its level–amplitude completion),
one theory section (all theorems), one experiments section. Old and new are distinguished by blue text and by the
"Relation to the conference version" paragraph, not by the section structure.

Legend: [C] conference material, [N] new, [C+N] conference section with new subsections. Existing labels are kept
(they move with their text) unless noted.

---

### 3 Motivations [C+N] (`sec:motivations`) — what the data show
- 3.1 Phase Tokens Are Stable and Low-Dimensional [C]
  - Insight 1 (globally vs locally stationary), Insight 2 (lower-dimensional subspace), the PCA figure — unchanged
  - Theorem 1 moves out to 5.1; a one-sentence forward pointer remains ("Theorem 1 explains why", Section 5.1)
- 3.2 Changes Across Cycles [N] (new label `sec:mot_changes`)
  - the current paragraph "What the theorem does not cover", rewritten without equations: level drift and
    level-dependent amplitude, why they are not covered by the stability of 3.1
  - Insight 3 (moved from current 8.2, `sec:exp_coupling`, Table `tab:coupling`): level-dependent amplitude is
    pronounced in the benchmarks (correlation of cycle means and within-cycle amplitudes against an additive control);
    reframed as an observation, not as a test of a theorem
  - closes on what a forecaster must therefore handle: the level and an amplitude that scales with it

### 4 Methodology [C+N] (`sec:method`) — the model
- 4.1 Data Pre-Processing [C]: normalization, phase tokenization (the folding notation is fixed here once and used by
  every later section)
- 4.2 Phase-based Routing Transformer [C]: 4.2.1 Embedding, 4.2.2 Cross-Phase Routing, 4.2.3 Predictor
- 4.3 Level–Amplitude Completion: PhaseFormer-LA [N] (`sec:extension` moves here; Fig. `la_overview`)
  - opening: the backbone of 4.2 is unchanged; two components address the two changes of 3.2; their forms are
    justified in Section 5 (forward pointer, one sentence)
  - 4.3.1 Level Branch and Gated Fusion (`sec:ext_branch`; current 6.1 + 6.2; PhaseFormer-L = backbone + branch)
  - 4.3.2 Centered Amplitude Modulation (`sec:ext_modulation`; current 6.3)
- 4.4 Complexity [C+N] (current 4.3 + the cost part of current 6.4: exact nesting, parameter count, complexity of the
  two components)
- 4.5 Discussion [C] (current 4.4: periodic alignment, failure modes, padding)

### 5 Theoretical Analysis [N, with Theorem 1 from C] (`sec:theory`) — why the model is built this way
- 5.1 Stability of Phase Tokens [C] (Theorem 1 and its takeaways, moved from Section 3; label `thm:stability` kept)
- 5.2 Setup and Error Decomposition [N] (`sec:theory_setup`): the cycle-axis changes `eq:cycle_axis_changes` and the
  multiplicative seasonal model `eq:mult_model` (formal versions of 3.2), LN₂ and the shared predictor, the three-way
  error split, the demands Λ and Ω
- 5.3 Level Drift: Capability and the Level Envelope [N] (Theorem 2, Theorem 3, Corollary, the trade-off)
- 5.4 What Lies Outside the Envelopes [N] (Theorem 4: low-rank level complement; Theorem 5: amplitude envelope,
  multiplicative demand, level-invariant modulation) — or split level / amplitude as 5.4 and 5.5 (open choice 2)
- 5.5 Implications for PhaseFormer-LA [N] (Table `tab:two_envelopes` + Scope): each component of 4.3 supplies exactly
  the part outside one envelope; whether an envelope binds is a property of a checkpoint and a dataset (→ 6.5)
- Because the model is defined in Section 4, the theorems can name PhaseFormer, PhaseFormer-L and PhaseFormer-LA
  directly; no forward dependency.

### 6 Experiments [C+N] (`sec:exp`; current Sections 7 and 8 merged; `sec:ext_exp` becomes an alias of a subsection
or is retargeted)
- 6.1 Setup [C+N] (`sec:exp_main` setup part): datasets, baselines, implementation; the training protocol of
  PhaseFormer-LA (current "Setup for PhaseFormer-LA" + current 8.1)
- 6.2 Main Results [C+N]: one table `tab:main_results` with the baselines, PhaseFormer and PhaseFormer-LA; per-horizon
  results in the appendix
- 6.3 Efficiency [C+N]: current "Efficiency Comparison" + the PhaseFormer-LA parameter counts
- 6.4 Ablation Studies [C+N]
  - Routers [C]: number of routers, effectiveness of cross-phase routing
  - From PhaseFormer to PhaseFormer-LA [N] (`sec:exp_ladder`): the component ladder
  - Capacity of the level branch [N] (`sec:exp_capacity`)
- 6.5 Validating the Theory [N] (`sec:exp_theory`)
  - Envelopes and level hand-over on trained checkpoints
  - Mechanism of the modulation (synthetic multiplicative series, cycle-mean invariance, what drives it, where the gain
    occurs) (`sec:exp_mechanism`)
- 6.6 Sensitivity to the Phase Length [C] (`sec:phase_length`)
- 6.7 Case Study [C] (`sec:exp_case`)

### 7 Discussion and Conclusion — unchanged in this proposal

### Appendices
- A Supplementary material of the conference version (A.7, the proof of Theorem 1, stays or moves to B — open choice 4)
- B Proofs for Section 5
- C Extended results of PhaseFormer-LA; D Per-setting configuration of PhaseFormer-LA

---

## Consequences
- Intro: contributions' section references change (C1: Sections 3 and 5.1; C2: Sections 4 and 6; C3: Section 5;
  C4: Sections 4.3 and 5; C5: Section 6); the "Relation to the conference version" paragraph is rewritten as a list of
  the new subsections (3.2, 4.3, part of 4.4, 5.2–5.5, the new parts of 6, Appendices B–D). Intro P4's
  "Section 3 confirms on real data" still holds; intro P6's pointer to `sec:exp_coupling` now lands in 3.2.
- Two PhaseFormer-LA results exist today (the per-setting configuration of App. D in the main table, the fixed-protocol
  runs in the ladder); with one experiments section they must be named distinctly in 6.1.
- main.tex input order unchanged except that theory.tex absorbs Theorem 1 from motivations.tex, extension.tex is merged
  into method.tex, and extension_experiments.tex is merged into experiments.tex.

## Open choices for the reviewers
1. Theorem 1 in 5.1 (this proposal: all theory in one place, Section 3 purely observational) or kept in Section 3?
2. Section 5 middle: one subsection "What lies outside the envelopes" (Theorems 4–5) or one subsection per change
   (level: Theorems 2–4; amplitude: Theorem 5)?
3. Insight 3 (coupling) in 3.2, or kept in the experiments (6.5)?
4. Proof of Theorem 1: stay in A.7 or move to B with the other proofs?
5. Does 4.3 need more than a one-sentence forward pointer to justify the forms of the components before Section 5?
