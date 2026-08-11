# Agent Maintenance Log

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
  `docs/PHASEFORMER_GOLD_STANDARD.md`.
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
- Archived `ITERATION_BRIEF.md` / `ITERATION_LOG.md` to `docs/archive/` and
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
