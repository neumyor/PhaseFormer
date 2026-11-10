# Proposed section outline v2 (Sections 3–8)

Status: proposal for review; no .tex file has been changed.
Scope: level-1 and level-2 headings, their order and titles, and which existing material goes under which heading.
Principle: the conference block comes first and stays intact; the extension follows as one block that mirrors it
(problem → theory → model → evaluation), in the order of the introduction (P4–P5 PhaseFormer and its results,
then P6–P8 changes across cycles, envelopes, PhaseFormer-LA).

Legend: [C] conference material (editorial changes only), [N] new in the journal version.
Labels in parentheses are kept, so every \ref in the introduction still resolves.

---

## Conference block

### 3 Motivations [C] (`sec:motivations`)
- Insight 1: phase tokens are globally stationary, patch tokens locally stationary (unchanged)
- Insight 2: phase tokens lie in a lower-dimensional subspace (unchanged)
- Theorem 1 (Phase Tokenization Stability) and its takeaways (unchanged; proof stays in Appendix A.7)
- REMOVED: the blue paragraph "What the theorem does not cover: the cycle axis" → moves to 6.1

### 4 PhaseFormer [C] (`sec:method`; title changed from "Methodology", editorial)
- 4.1 Data Pre-Processing (unchanged)
- 4.2 Phase-based Routing Transformer (4.2.1 Embedding, 4.2.2 Cross-Phase Routing, 4.2.3 Predictor; unchanged)
- 4.3 Complexity of PhaseFormer (unchanged)
- 4.4 Discussion (periodic alignment, failure modes, padding; unchanged — the failure-mode paragraph is the natural
  hand-off to Section 6)

### 5 Experiments [C] (`sec:exp`; current Section 7)
- 5.1 Long-term Time Series Forecasting (`sec:exp_main`): datasets, baselines, main results, efficiency.
  REMOVED: the blue paragraphs "Setup for PhaseFormer-LA" (→ 8.1) and "Results of PhaseFormer-LA" (→ 8.3), and the
  PhaseFormer-LA rows of Table `tab:main_results` (→ 8.3). The main table returns to the conference version.
- 5.2 Ablation Studies and Analysis (unchanged)
- 5.3 Sensitivity Analysis of Phase Length Choices (unchanged)
- 5.4 Case Study (unchanged)

## Extension block

### 6 Changes Across Cycles: How Far PhaseFormer Can Follow Them [N] (`sec:theory`; current Section 5)
- 6.1 Level Drift and Level-Dependent Amplitude (new label, e.g. `sec:theory_changes`)
  - moved from the end of Section 3: the two changes \equref{eq:cycle_axis_changes}, why each breaks Theorem 1,
    why level-dependent amplitude leaves the patch subspace unchanged
  - the multiplicative seasonal model \equref{eq:mult_model}, introduced here once (now split between Sec. 3 and 5.1)
  - the three questions per change (represent? how far? what carries the rest?) and the roadmap of the section
- 6.2 Setup and Error Decomposition (`sec:theory_setup`): notation for one window, LN₂ and the shared predictor,
  the three-way error split, the level demand Λ (current 5.1 minus the model, which moved to 6.1)
- 6.3 Level Drift: Capability, Envelope and a Low-Rank Missing Part
  (current 5.2–5.4, as run-in paragraphs, not level-3 headings)
  - Capability: Theorem 2 (`thm:capability`)
  - Level envelope: Theorem 3 (`thm:level_envelope`), Corollary (`cor:steps`), the capability–envelope trade-off
  - What lies outside: Theorem 4 (`thm:complement`, `sec:theory_complement` kept on this subsection)
- 6.4 Level-Dependent Amplitude: Envelope and a Level-Invariant Modulation (`sec:theory_amplitude`; current 5.5)
  - Amplitude envelope and multiplicative demand: Theorem 5(a)(b)
  - Level-invariant modulation: Theorem 5(c)(d)
- 6.5 Summary and Scope: Table `tab:two_envelopes` and the Scope paragraph (now buried in 5.5)

Content change attached to this section (requires rewriting theorem statements, so it can be deferred to the body
revision): the theory refers to a generic level branch whose output lies in a span of fixed profiles, and to a generic
centered modulation; the concrete names and sizes (PhaseFormer-L, rank 3, the modulation equation of the model) are
defined only in Section 7. Optional: split Theorem 5 into an envelope theorem and a modulation theorem.

### 7 PhaseFormer-LA [N] (`sec:extension`; current Section 6)
- opening paragraph: the architecture of PhaseFormer is unchanged; Fig. `la_overview`; the map from results to
  components (Table `tab:two_envelopes`, last rows)
- 7.1 Gated Level Branch (current 6.1 + 6.2; `sec:ext_branch`, with `sec:ext_fusion` kept on the fusion paragraph):
  the cycle-mean branch, the gated fusion, the whole-cycle requirement; PhaseFormer-L is named here
- 7.2 Centered Amplitude Modulation (`sec:ext_modulation`; current 6.3, plus the "Preserved guarantees" sentence)
- 7.3 Nesting and Cost (`sec:ext_properties`; current 6.4 minus "Preserved guarantees"): exact nesting, parameters and
  complexity, mirroring 4.3

### 8 Evaluating the Theory and PhaseFormer-LA [N] (`sec:ext_exp`; current Section 8, title changed from "Extension Experiments")
- 8.1 Setup (`sec:ext_setup`; absorbs "Setup for PhaseFormer-LA" from current 7.1)
- 8.2 Level–Amplitude Coupling in the Benchmarks (`sec:exp_coupling`)
- 8.3 Accuracy Against the Baselines (new label): the moved "Results of PhaseFormer-LA" paragraph and a compact table of
  dataset averages (PhaseFormer, PhaseFormer-LA, the strongest baselines); per-horizon results stay in Table `tab:la_full`
  in the appendix
- 8.4 From PhaseFormer to PhaseFormer-LA (`sec:exp_ladder`)
- 8.5 Envelopes and Level Hand-over on Trained Checkpoints (`sec:exp_theory`)
- 8.6 Mechanism of the Modulation (`sec:exp_mechanism`)
- 8.7 Capacity of the Level Branch (`sec:exp_capacity`)

### 9 Discussion and Conclusion (unchanged in this proposal)

---

## Consequences outside Sections 3–8
- Introduction, "Relation to the conference version": becomes "Sections 3–5 follow the conference version … Sections 6–8
  … are new"; the "except for the new discussion of changes across cycles at the end of Section 3" clause is deleted.
- Introduction, contributions: section references become monotone (3; 4–5; 6; 6–7; 8); contribution 2 cites
  `sec:method` and `sec:exp`, contribution 5 cites `sec:ext_exp`.
- main.tex: input order becomes motivations, method, experiments, theory, extension, extension_experiments.
- Appendices: unchanged order (A conference supplement, B proofs for Section 6, C extended LA results, D LA configuration).
- Floats: the main-results table loses the LA rows; Section 8.3 gains one compact table.

## Open choices for the reviewers
1. Two-block order (this proposal) versus keeping theory and model before all experiments (current order) — is the
   two-block order the better fit for a VLDBJ extension? What does it cost?
2. Main table: remove the LA rows and add a compact table in 8.3 (this proposal), or keep the LA rows in the
   conference main table with a forward pointer to Section 7?
3. Section 3 title: keep "Motivations" (this proposal, to keep the conference block recognizable) or rename to expose
   Theorem 1 (e.g. "Phase Tokenization and Its Stability")?
4. Order inside Section 8: coupling → accuracy → ladder → envelopes → mechanism → capacity (this proposal), or follow
   contribution 5 (coupling → envelopes → ablation → accuracy)?
5. Generic theory statements (PhaseFormer-L named only in Section 7) now, or deferred to the body revision?
