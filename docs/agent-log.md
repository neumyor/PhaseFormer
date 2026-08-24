# Agent Maintenance Log

## 2026-08-24 — Pure Phase Modeling (phase-only forecasting, no residual)

Implemented 4 warm-start pure-phase modules (commits 1653cd1, 00f09dc) and ran
the next-stage plan (`docs/PhaseFormer_pure_phase_plan.md`)
at full budget: MultiScalePhase (period-axis long view, zeta gate), PhaseDeformation
(rate+stretch -> cumsum displacement warp), PhaseGraph (circular message passing),
TrajectoryDecoder (per-slot polynomial over the future axis). 7 modes registered
(multiscale_phase / phase_deformation / phase_geo / phase_graph / predictor_mlp /
trajectory_decoder / pure_full). Report:
`docs/PhaseFormer_pure_phase_results.md`.

- **Result (61/70 runs; 9 missing — Traffic h720 trajectory_decoder+pure_full,
  ETTh1 h720 all 7; user stopped the run mid-batch-2)**:
  - representation/evolution/interaction modules are parity with original:
    avg ΔMSE multiscale +0.53%, deformation −0.09%, phase_geo −0.16%,
    phase_graph −0.10%, predictor_mlp +0.03% (no consistent wins).
  - **TrajectoryDecoder is catastrophic** on 3/5 datasets (ETTm1 +90.5%/+71.8%,
    Electricity +26%, Traffic h336 +59.4%); mild improvement only on ETTh1/ETTh2.
    Analysis: it makes output smoother (−5.4% |dy|) but destroys phase peak
    alignment (peak_shift 3.67 vs 3.24). pure_full inherits the failure
    (avg +33.5%; best single result ETTh2 h720 −4.2%).
  - Deformation field learned compression (s≈0.67) but cumulative displacement
    <0.1 slot — numerically near-inactive. Multiscale zeta gate IS open
    (mean|ζ|≈0.17, 99% dims) but no MSE benefit.
  - **Conclusion: the "adaptive phase geometry" narrative is not supported** —
    pure-phase gains ≤±0.5% and the trajectory decoder dominates negatively.
- Artifacts: `research_runs/pure_phase_summary.csv`, `research_runs/pure_phase_analysis/`
  (4 CSVs + figures/), per-run `research_runs/dyn_phase_full/dynphase_*_<mode>_*/`.

## 2026-08-12 — Reliability-aware Adaptive Phase Evolution (RAPE)

New mechanism (67bb537): compose the adaptive phase warp + amplitude
calibration with a per-sample, per-channel ReliabilityGate. The gate
g=sigmoid(MLP(history volatility, linear slope, same-slot phase instability,
adaptation magnitude)) fuses `h~ = g*h_adapted + (1-g)*h_identity`, letting the
model fall back to the original fixed-grid phase prior on stable strong-period
windows. Zero-init gate -> g=0.5 at construction; warp+amp are identity then,
so the fused output equals the identity phase for any g (warm start). Mutually
exclusive with phase_align/phase_warp/phase_amp_calib, constructed last.
37/37 tests pass. Audit set in `research_runs/phase_rape_full/` (six files +
figures). Reuses `scripts/analyze_experiment.py`, extended with reliability-gate
activity + configurable report labels.

- Stage A (30%/8ep, val-only, 10 settings x original/warp/amp_calib/rape):
  rape improves Weather h192 (−6.45%) and slightly mitigates the ETTm1
  amp_calib regression; near-neutral elsewhere.
- Stage B (full budget, seed 2021, test eval, `research_runs/phase_rape_runs/`,
  10 settings, paired original + phase_rape):
  - dMAE improves on 6/10 (ETTh1 96/192, ETTh2 96, ETTm1 96/192, Weather 192);
    dMSE improves on 5/10 (ETTh1 96/192, ETTm1 192, Weather 96/192).
  - **Weather h192 beats the gold standard on both metrics** (dMSE +0.41%,
    dMAE +0.00% at 4-decimal precision; marginal, single-seed). ETTh1 h192
    beats gold on MSE (+1.66%) but not MAE (−0.84%); dMSE −3.36% is the largest
    improvement seen across all mechanisms so far.
  - Regressions: ETTm2 96 (+1.13/+1.97), ETTh2 192 (+1.07/+1.60), ETTm2 192
    (+0.91/+0.20), Weather 96 (+0.26/−0.65).
  - vs amp_calib (no gate, prior round): the gate helps ETTh1 96/192 (dMSE
    −0.01 vs +0.82; −3.36 vs −1.69) and Weather h192 (−0.90 vs −0.72), but is
    neutral-to-worse on ETTh2 192, ETTm1 192, ETTm2 192.
  - Reliability gate activity (mean g over test): high on 8/10 settings
    (0.70-0.92), lowest on ETTm2 192 (0.42) and Weather 96 (0.61). The gate
    mostly commits to the adapted representation rather than selectively
    falling back to the original phase prior; the "reliability-aware
    selection" is only weakly realized.
  - Training cost: candidate ~1.5-2.9x slower than original (Weather h192
    2120s vs 745s; ETTh1 96 146s vs 83s).
- Conclusion: no stable cross-task gain; two genuinely positive settings
  (ETTh1 h192 MSE, Weather h192 dual-metric gold beat) both improve over the
  no-gate amp_calib, but the benefit is dataset-dependent and within
  single-seed spread. Mechanism stays flag-gated and out of `_LATEST_POLICY`;
  the gate is not a reliable cross-task fix.

## 2026-08-12 — Phase-conditioned Amplitude Calibration

New mechanism (4afc634): phase-conditioned amplitude calibration builds on the
adaptive phase warp representation. `src/models/phase_amp_calib.py`
(`PhaseAmpCalibration`, flag `use_phase_amp_calib`) predicts per phase slot a
scale `alpha_l` and shift `beta_l` from the phase-slot position and per-slot
statistics of the phase history (mean/std/abs-mean/last period/linear trend),
then applies `h'[l,k] = alpha_l*h[l,k] + beta_l` broadcast over the period axis.
Zero-init final layer warm-starts at identity (alpha=1, beta=0). Module
constructed last so flag-off keeps baseline initialization; `phase_amp_calib`
ablation mode = `phase_warp` + `use_phase_amp_calib`. 31/31 tests pass. Audit
set in `research_runs/phase_amp_full/` (six files + figures). Reusable analysis
tool added as `scripts/analyze_experiment.py` (validated against phase_warp_full).

- Stage A (30%/8ep, val-only, `research_runs/phase_amp_screen/`, 10 settings x
  original/warp/amp_calib): dataset-dependent. amp_calib improves Weather
  (h192 dMAE −4.76%, h96 −1.83%) and mildly ETTh1/ETTh2 96; regresses ETTm1
  (h96 +2.78% MAE/+5.17% MSE) and mildly ETTm2.
- Stage B (full budget, seed 2021, test eval, `research_runs/phase_amp_runs/`,
  10 settings, paired original + phase_amp_calib):
  - dMAE improves on 6/10 (ETTh1 192 −0.13, ETTh2 96 −1.38, ETTm1 96 −0.54,
    ETTm1 192 −0.50, Weather 96 −0.03, Weather 192 −0.46); dMSE improves on 6/10.
  - Regressions: ETTm2 96 (+1.83/+2.01), ETTh2 192 (+0.62/+0.81), ETTh1 96
    (+0.09/+0.82), ETTm2 192 (+0.59/−0.34).
  - **No setting beats the gold standard on both MSE and MAE.** Weather 192
    beats gold on MSE (+0.25%) but not MAE (−0.07%); Weather 96 beats gold on
    MAE (+0.20%) but not MSE (−0.51%).
  - The screen's strong Weather signal (−4.76% at h192) collapsed to −0.46% at
    full budget; the ETTm1 screen regression inverted to slight improvement.
  - Calibration activity (mean |alpha−1| over test): most active ETTh1 (~0.79)
    and Weather 192 (~0.77), near-inactive ETTm2 192 (0.08); high activity with
    no net gain. beta small (<0.35). max_scale=2.0 permits alpha<0 (sign-flip),
    and the old log-alpha diagnostic nans showed it does occur.
  - Training cost: candidate ~1.7–2x slower than original (ETTm1 96 576s vs
    292s; Weather h192 1509s vs 751s; ETTh1 96 138s vs 82s).
  - Sample-level (per-cell delta_mae): ETTm2 96 42.6% cells improve (57.4%
    regress, net +0.00475), ETTh2 96 59.5% improve (net −0.00476); no dominant
    structural signature across groups beyond the aggregate sign.
- Conclusion: no stable cross-task gain, consistent with the phase_align and
  phase_warp explorations — the fixed phase grid is not the bottleneck on this
  grid, and adding a per-slot amplitude branch costs ~2x training for no net
  benefit. Mechanism stays flag-gated and out of `_LATEST_POLICY`. Diagnostic
  hook fixed to |alpha−1| (a820c2a) because log alpha nans when alpha≤0.

## 2026-08-12 — Simplified report archive validation

- Reduced ZIP validation to three practical checks: successful extraction,
  presence of the Markdown and referenced figures, and valid relative image
  links after extraction.
- Replaced the three detailed archive validation flags with one
  `archive_checked` status.

## 2026-08-12 — Portable Markdown report bundle

- Replaced the experiment PDF artifact with `objective_error_analysis.zip`.
- Required the archive to contain only the byte-identical Markdown report and
  the exact `figures/` images it references, using portable relative paths.
- Added ZIP integrity, path-safety, member-whitelist, byte-equivalence, and
  extracted-link validation; prohibited PDF generation.
- Updated the research guide and active experiment plan to use the same
  six-file Markdown-plus-ZIP contract.

## 2026-08-12 — Strict multi-setting experiment artifact layout

- Tightened `experiment-and-error-analysis` so every experiment directory has
  exactly six audit files plus one `figures/` directory.
- Prohibited retained checkpoints, command files, environment snapshots, logs,
  full predictions, temporary files, and per-setting output files inside an
  experiment directory.
- Required all settings from one run to share `run.yaml`, `results.csv`,
  `sample_errors.csv`, `selected_cases.npz`, and one Markdown/PDF report pair,
  with an explicit `setting` identifier in every applicable artifact.
- Updated the repository research guide and active search plan to use the same
  strict whitelist.
- Validation: checked Skill metadata, setting coverage requirements, directory
  whitelist language, repository references, whitespace, and the staged diff.

## 2026-08-11 — Adaptive Phase Warping exploration

Follow-up to Phase Alignment (2ab472b, 3b805d4, 08c74e4): replace the bounded
per-token phase correction with a monotonic, data-driven phase warp. A speed
field from `[value, time marks]` defines a normalized cumulative-sum map from
time-in-cycle to continuous phase (phi[0]=0, phi[L-1]=L-1), expressing
per-stage compression/stretch while preserving order; uniform speed reduces to
the identity grid (warm start). `use_phase_warp` flag, mutually exclusive with
`use_phase_align`, module constructed last. 26/26 tests pass. Audit set per
`experiment-and-error-analysis` skill in `research_runs/phase_warp_full/`.

- Stage A (30%/8ep, val-only): same sign pattern as Phase Alignment — 192
  horizons slightly positive (ETTm1 192 +0.54, Weather 192 +0.50), ETTm1 96 and
  Weather 96 eliminated.
- Report regenerated 2026-08-12 per the updated `experiment-and-error-analysis`
  skill contract: audit set in `research_runs/phase_warp_full/` is now exactly
  the six files (run.yaml, results.csv, sample_errors.csv, selected_cases.npz,
  objective_error_analysis.md, objective_error_analysis.zip) plus `figures/`
  over all 10 settings (single sample_errors.csv / selected_cases.npz with
  `setting` identifiers; ZIP = Markdown + referenced figures, byte-identical;
  PDF removed). Raw training runs preserved under `research_runs/phase_warp_runs/`.
- Stage B (full budget, seed 2021, test): no stable cross-task gain. vs matched
  original — clearly negative ETTm2 96 (dMSE -2.38%), mild positive on 192-horizon
  tasks (ETTm1 192, ETTm2 192, Weather 192). Weather 192 is the only task beating
  the gold standard on both metrics (dMSE +0.17%, dMAE +0.21%), within single-seed
  noise. Result mirrors Phase Alignment, consistent with screening.
- Sample-level (Weather 192, ETTm2 96): Weather 192 54.1% of cells improve (net
  -0.0018 delta_mae), improvement concentrated in later horizon segments and NOT
  from peak/std alignment (peak closer 1/10, std closer 0/10); ETTm2 96 53.1%
  regress (net +0.0032), regression cases show peak farther from truth in 8/10.
- Conclusion: no significant stable gain; mechanism flag-gated and out of
  `_LATEST_POLICY`. Same verdict as Phase Alignment — the fixed phase grid is not
  the bottleneck on this diagnostic grid.

## 2026-08-11 — Adaptive Phase Alignment exploration

New mechanism (b2d06ba, d1d2be1, 626b0f2): replace the fixed `time % period_len`
phase assignment with a learned continuous phase per time point. A small MLP
(`src/models/phase_align.py`, `PhaseAlignment`) maps `[RevIN value, time-mark]`
to a residual delta from the position-in-cycle; input evidence is soft-scattered
onto the two neighbouring phase slots via linear interpolation (k=2). Output
grid stays fixed, so reconstruction is unchanged. Flag-gated
(`use_phase_align`), module constructed last in `__init__` so toggling the flag
does not shift shared-module initialization; flag-off path byte-identical.
`x_mark_enc` (previously unused) now feeds the estimator; must `.float()` because
training passes it as float64.

- Tests: `tests/test_phase_align.py` (forward shape, zero-delta identity,
  flag-on@init ≈ flag-off, plumbing, mark-dim fallback). 20/20 pass.
- Stage A (30% data / 8 ep, paired same-budget original, val-only): 6/10 tasks
  slightly positive (+0.02..+0.43), 4/10 negative; 3 eliminated (ETTm1 96
  −0.81, ETTm2 96 −0.41, Weather 96 −2.43).
- Stage B (full budget, seed 2021, test eval, `research_runs/phase_align_full/`):
  no task beats the gold standard on both MSE and MAE (matched original reruns
  themselves sit 0.5-5% above gold). vs matched original: ETTm1 192 is the only
  clear dual-metric gain (MSE −1.26%, MAE −0.77%); ETTh2 96 (−1.13/−0.84) and
  ETTm2 96 (−1.34/−0.72) clearly regress; the rest are neutral or mixed. No
  cross-task stable direction; horizon split leans positive at 192, negative at 96.
- Estimator activity diagnostic (mean |delta| on test, of 24 slots): ETTm1 192
  0.108, ETTm2 96 0.140, Weather 96 0.038 — active but tiny (<1% of the cycle);
  the model finds little benefit in deviating from the fixed phase grid.
- Bad cases: worst-sample MSE roughly unchanged; ETTm1 192 and Weather 96 top
  cases improve slightly.
- Conclusion: no significant stable gain (advantage < single-seed spread, per
  `EXPERIMENT_SEARCH_PLAN.md`). Mechanism stays flag-gated and out of
  `_LATEST_POLICY`; treated as an exploration without a clear positive signal.

## 2026-08-11 — Cross-agent experiment analysis skill

- Added the project-level `experiment-and-error-analysis` Skill under
  `.claude/skills/`, with a Codex-compatible entry under `.agents/skills/`.
- Added native repository entry rules for both Codex (`AGENTS.md`) and Claude
  Code (`CLAUDE.md`) with identical trigger boundaries.
- Renamed the shared maintenance policy to `MANAGE_RULES.md` and updated all
  repository references.
- Integrated the Skill into `HOW_TO_DO_RESEARCH.md` and explicitly allowed
  test-set-driven model/configuration selection when the complete search trail
  is retained and the resulting reports disclose test-set selection.
- Validation: checked Skill metadata and structure, link resolution, all
  repository references, whitespace, and the staged diff. No model code or
  experiment results changed.

## 2026-08-11 — Original PhaseFormer gold standard

- Transcribed the user-provided paper Table 5 screenshot into
  `docs/PhaseFormer_gold_standard.md`.
- Recorded 28 original PhaseFormer results covering ETTh1, ETTh2, ETTm1,
  ETTm2, Weather, Electricity, and Traffic at horizons 96, 192, 336, and 720,
  with input length 720 and explicit MSE/MAE column ordering.
- Defined the fixed comparison formula, dual-metric claim rule, matched-rerun
  distinction, and update authority. Exchange remains intentionally unset
  because it is absent from the supplied source image.
- Updated `MANAGE_RULES.md`, `HOW_TO_DO_RESEARCH.md`, and
  `EXPERIMENT_SEARCH_PLAN.md` so future improvement claims use this fixed gold
  standard instead of silently replacing it with a retrained baseline.
- Validation: manually cross-checked all 28 rows against the source image and
  verified the Markdown table contains 7 datasets × 4 horizons with both
  metrics. No training or model behavior changed.

## 2026-07-26 — Training protocol and maintainability repair

- Fixed the `ett_all` train/validation/test dataset selection condition.
- Made the effective loss name authoritative and retained legacy Huber flags
  only as compatibility metadata.
- Changed official and research runners to evaluate the lowest-validation-loss
  checkpoint and use that same model for bad-case export.
- Replaced the standalone Traffic training loop with the shared preset runner,
  removing per-epoch access to the test set.
- Moved PhaseFormer weak-period and phase-adaptation helpers into
  `src/models/phase_adapters.py` while preserving public imports and state-dict
  keys.
- Added a uv project definition with separate core, development, and GIFT-Eval
  dependency groups.
- Validation commands:
  - `uv run pytest -q` — 7 passed.
  - `uv run python -m compileall -q config src scripts run_*.py`.
  - All seven official dataset entry points completed `--help` smoke checks.
  - `smoke_best_checkpoint_protocol_20260726b` completed a two-epoch GPU
    training/test cycle and restored `checkpoints/best.ckpt` before evaluation.
- Environment: NVIDIA GeForce RTX 4090; PyTorch 2.7.1+cu126; CUDA 12.6.
- Protocol compatibility: historical benchmark files used last-epoch weights.
  New best-checkpoint results require matched original/latest reruns and must
  not be compared directly with those historical metrics.
- Completed matched best-checkpoint regressions:
  - ETTm2 96: MAE -3.85%, MSE -5.82%.
  - ETTh2 720: MAE -4.75%, MSE -4.33%.
  - Exchange 96: MAE -13.27%, MSE -16.93%.
  - Weather 96: MAE -4.45%, MSE -1.85%.
  - Electricity 336: MAE -2.28%, MSE -2.31%.
- Traffic 96 batch64 was blocked by another process occupying 18.9 GiB GPU
  memory. The official batch8 setting entered training successfully but was
  stopped because completing both 30-epoch runs under contention was
  impractically slow. No Traffic metric is claimed from these incomplete runs.

## 2026-08-10 — Weak-residual branch refactor and cleanup

- Branch renamed `phaseformer-weather-electricity-presets` → `weak-residual-phaseformer`
  (confirmed independent from `main`, which removed the weak/adaptive residual line).
- Extracted the shared training protocol into `src/training/runner.py`
  (`build_logger`, `build_trainer`, `restore_best_checkpoint`); refactored the
  four previously duplicated Trainer assemblies (`run_ett_latest.py`,
  `scripts/benchmark_phaseformer_suite.py`, `scripts/research_weather_weak.py`,
  `scripts/search_phaseformer.py`) to use it. Best-checkpoint restore now has a
  single implementation.
- Converted the 37-branch `get_latest_overrides` if-ladder into a declarative
  `_LATEST_POLICY` table keyed by `(dataset, horizon)` with a per-dataset
  full-horizon fallback and the original guardrail default. Verified
  behaviorally identical for all 32 dataset×horizon tasks; added
  `LatestPolicyTableTests` in `tests/test_presets_and_loss.py`.
- Unified dataset entry: `run_ett_latest.py --datasets` runs multiple datasets;
  thin `run_*.py` wrappers unchanged. `run_all_experiments.py` marked
  deprecated (superseded by `scripts/run/*.sh` + benchmark suite).
- Archived 18 unused `src/models/layers/*` legacy modules to
  `archive/layers_legacy/` (the active model only imports
  `SelfAttention_Family.py`), with an explaining README.
- Archived `iteration_brief.md` / `iteration_log.md` to `docs/archive/` and
  repointed references in `MANAGE_RULES.md` / `HOW_TO_DO_RESEARCH.md` to the archived
  paths, clarifying the current active plan/log are `EXPERIMENT_SEARCH_PLAN.md`
  and `docs/agent-log.md`.
- Removed tracked `.DS_Store` files and added `.DS_Store` to `.gitignore`.
- Environment note: sandbox lacks the repo's locked deps (torch/lightning), so
  verification was static (AST parse + behavioral-equivalence simulation for the
  presets table). Full `uv run pytest` and a GPU smoke run should be executed in
  the real `raft`/`py310` environment to confirm runtime equivalence.

## 2026-08-11 — Weather 192 weak-period mechanism exploration

Follow-up to the original-vs-latest benchmark: the current `_LATEST_POLICY`
table has no entry for (Weather, 192), so `latest` falls back to the original
guardrail. Question: which weak-period mechanisms are actually useful for
Weather 720→192? Ran a validation-isolated search following
`EXPERIMENT_SEARCH_PLAN.md` (val-only until a frozen winner).

### Protocol

- Entry point: `scripts/search_phaseformer.py` (fixed a startup import-order bug
  — `from src...` ran before the `sys.path.insert`, so the script failed with
  `ModuleNotFoundError: No module named 'src'` when invoked directly).
- Stages: period screen → mechanism screen (30% / 8ep) → full-budget confirm
  (100% / 30ep, seeds 2021+2022) → 3-seed test was truncated by user decision
  after the 2-seed confirm proved stable.
- All runs: val-only (no test read during search), loss=huber, lr 0.001,
  batch 64, period search {12, 24, 48}.

### Results (val, period 48)

| run | seeds | avg val_MAE | avg val_MSE | dMAE% | dMSE% |
|---|---|---|---|---|---|
| **channel_residual** (gate 0.5) | 2 | 0.29925 | 0.43405 | **−4.93** | **−4.39** |
| channel_adaptive (channel head + adaptive gate) | 1 | 0.30001 | 0.43241 | −4.69 | −4.75 |
| phase_stack (uncert+level+hifreq+sparse) | 2 | 0.31090 | 0.45132 | −1.23 | −0.59 |
| adaptive_g02 (shared head + adaptive gate) | 1 | 0.31405 | 0.44393 | −0.23 | −2.22 |
| original | 2 | 0.31477 | 0.45400 | — | — |

### Findings

- **Period 48 wins** for Weather 192 (val MAE 0.341 vs 0.346 / 0.363 for 12 / 24).
  Note Weather 96's enhanced preset uses period 12 — the optimal cycle length
  differs across horizons.
- **Channel-wise weak-period residual head is the only robust winner**, stable
  across seeds (0.2993 / 0.2992). It extrapolates a per-channel centered
  trajectory + persistence anchor, gated at 0.5.
- **Adaptive gate adds nothing** on top of the fixed channel head (channel vs
  channel+adaptive ≈ equal); on the shared head it is a mild regression.
- **Phase adapters (uncertainty/level/hifreq/sparse) give only ~−1%** here, far
  below the Weather-96 preset's benefit — their effect does not transfer to the
  192-horizon setting.
- time_mark and phase_local_trend are clearly negative / no-op for this task.

### Artifacts

- Search runner output: `research_runs/weather192_explore/runs/` (per-experiment
  `metrics.csv`, `config.json`, best checkpoint).
- Logs: `~/.claude/jobs/eee0ff88/tmp/full/` and `tmp/final/`.
- The 3-seed `--evaluate-test` confirm round was launched then stopped by user
  request ("无需三个seed了，可以结束了"); no test-set numbers were produced.
  Test-set validation of the channel-residual winner is still outstanding.

### Open question

Whether to promote a channel_residual entry for (Weather, 192) into
`_LATEST_POLICY` — and whether the same mechanism helps Weather 336/720, which
currently also fall back to the original guardrail.

## 2026-08-12 — Compress experiment analysis Skill

- Condensed `.claude/skills/experiment-and-error-analysis/SKILL.md` from 300
  to 168 lines while retaining its experiment protocol, six-file artifact
  whitelist, unified multi-setting schema, test-set-selection disclosure,
  programmatic case selection, objective reporting, and Markdown/figure ZIP.
- Simplified repeated validation language into four required checks, consistent
  with the existing lightweight-validation requirement.
- Validation: Skill schema passed `quick_validate.py`; measured at 2,159
  `o200k_base` tokens and 2,577 `cl100k_base` tokens.

## 2026-08-24 — Residual topology plan and implementation

- Added `docs/PhaseFormer_residual_topology_plan.md` as the experiment anchor.
  It preregisters R0 original, R1 full-forecast convex output residual, R2
  zero-initialized additive output correction, R3 one-shot latent long skip,
  R4 layer-wise latent injection, and R5 R2+R4 hybrid across four representative
  settings.
- Implemented the residual primitives and PhaseFormer wiring, registered all five
  candidate modes in presets/search, and added the resumable
  `scripts/run_residual_topology.py` scheduler with validation-screen/full-confirm
  stages and matched-delta summaries.
- Preserved comparison fairness by constructing the R1 control head after all
  shared modules so feature flags do not shift shared RNG initialization. R2--R5
  are exact zero-initialized warm starts; the residual master switch disables all
  new paths.
- Verification: Python compilation passed; the complete suite passed `90/90`;
  Stage A dry-run produced 24 commands and the frozen-candidate Stage B example
  produced four commands. Tests cover forward shapes, finite values, exact shared
  initialization, zero-init equivalence, gradients, optimizer movement, one-layer
  R3/R4 equivalence, multi-layer depth, 321-channel input, and summary arithmetic.
- Per the revised user scope, no training was launched, no test split was read,
  and no experimental result or error-analysis package was generated.
