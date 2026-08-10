# Agent Maintenance Log

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
