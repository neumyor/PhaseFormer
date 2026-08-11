# Agent Maintenance Log

## 2026-08-11 — Replace LPTD with Phase-Anchored Forecasting

- Rejected and removed active LPTD code after ten matched tasks showed an
  average -0.13% MSE / -0.09% MAE change versus original and consistent
  regression versus latest and gold. Archived the full result and minimal
  diagnostic evidence in `docs/archive/LPTD_FAILURE_ANALYSIS.md`.
- Added the parameter-free `PhaseAnchorTransform`. It gathers the most recent
  real observation for every phase slot, centers phase trajectories around
  those anchors, and restores the anchors after the unchanged PhaseFormer
  predictor generates future displacements.
- For incomplete final periods, missing slots are filled from their matching
  real phase anchors instead of circular padding; the mechanism requires at
  least one complete input period.
- Replaced the LPTD preset, search mechanism, and tests with `phase_anchor` and
  documented its single-path formulation and phase-translation equivariance.
- Runtime validation was not executed because this workstation has no usable
  project environment. Static review and test definitions cover complete and
  incomplete periods, coordinate reconstruction, and equivariance. Accuracy
  remains unverified until a matched experiment is run.

## 2026-08-11 — Replace failed CPTD with LPTD

- Rejected and removed the active output-space CPTD implementation after the
  user-provided evaluation regressed on all eight ETT tasks. Average regression
  versus the fixed gold standard was about 15.7% MAE / 37.0% MSE; ETTm1-192
  reached 46.1% MAE / 124.8% MSE regression.
- Archived the result table and root-cause analysis in
  `docs/archive/CPTD_FAILURE_ANALYSIS.md` instead of erasing negative evidence.
- Added the default-off `LatentPhaseTransportDecoder`. LPTD keeps phase-local
  routed tokens, predicts horizon- and phase-specific circular transport kernels,
  and decodes with the original horizon-specific linear form. It does not copy
  observed period profiles, pool away phase details, accumulate level state, or
  create a time-domain residual forecast.
- Replaced CPTD configuration, search, and test definitions with `lptd`; added
  identity-containment, shape, equivariance, and diagnostic test definitions.
- Updated the README, active search plan, and detailed design documentation.
- Runtime validation remains pending because this machine has no usable project
  environment. No LPTD accuracy improvement is claimed.

## 2026-08-11 — Circular Phase Transport Decoder implementation

- Added the default-off `CircularPhaseTransportDecoder` in
  `src/models/phase_transport.py` as a phase-native replacement for the direct
  value predictor. It uses convex recent-period memory, bounded circular phase
  shifts, bounded amplitude evolution, and cumulative period-level evolution;
  it does not create or blend a time-domain forecast.
- Integrated CPTD into `PhaseFormer` with an explicit incompatibility guard for
  weak-period residual branches. For non-divisible lookbacks, transport memory
  uses complete periods ending at the last real observation rather than the
  model's circular right-padding.
- Added the `phase_transport` preset and validation-isolated search mechanism,
  configuration propagation, interpretable diagnostics, and focused unit-test
  definitions for shape, initialization, equivariance, and normalized weights.
- Documented motivation, formulation, defaults, invariants, novelty boundary,
  and the required first-round evaluation in
  `docs/PHASE_TRANSPORT_DECODER.md`; updated the README and active experiment
  search plan.
- Per user instruction, no unit, integration, or training tests were run because
  this machine has no usable project environment. Verification was limited to
  code review, AST parsing, Markdown/config consistency, and whitespace checks.
- Accuracy status: unvalidated. No improvement over the fixed PhaseFormer gold
  standard is claimed by this implementation commit.

## 2026-08-11 — Original PhaseFormer gold standard

- Transcribed the user-provided paper Table 5 screenshot into
  `docs/PHASEFORMER_GOLD_STANDARD.md`.
- Recorded 28 original PhaseFormer results covering ETTh1, ETTh2, ETTm1,
  ETTm2, Weather, Electricity, and Traffic at horizons 96, 192, 336, and 720,
  with input length 720 and explicit MSE/MAE column ordering.
- Defined the fixed comparison formula, dual-metric claim rule, matched-rerun
  distinction, and update authority. Exchange remains intentionally unset
  because it is absent from the supplied source image.
- Updated `AGENT.md`, `HOW_TO_DO_RESEARCH.md`, and
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
  repointed references in `AGENT.md` / `HOW_TO_DO_RESEARCH.md` to the archived
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
