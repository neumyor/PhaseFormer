# PhaseFormer Weak-Period Iteration Log

## Iteration 0 - Initialization

- Goal: understand the repository, select a weak-period benchmark, and prepare auditable experiment tooling before changing behavior.
- Files read: `AGENT.md`, `HOW_TO_DO_RESEARCH.md`, `README.md`, `src/models/PhaseFormer.py`, `src/models/pl_bases/default_module.py`, `src/dataset/data_info.py`, `src/dataset/data_factory.py`, `src/dataset/data_loader.py`, `src/utils/metrics.py`, `run_weather.py`, `run_electricity.py`.
- Dataset choice: Weather 720 -> 96 for the first comparable experiment. It is local, multivariate, and less strictly periodic than traffic/electricity.
- Baseline plan: run `scripts/research_weather_weak.py --variant baseline --horizon 96 --epochs 30`.
- Initial hypothesis list:
  - H1 Weak-period residual path: add a direct temporal extrapolation branch to compensate drift and phase jitter that fixed phase routing can miss.
  - H2 Adaptive period mixture: mix daily and shorter pseudo-period views to reduce sensitivity to a single `period_len`.
  - H3 Robust objective tuning: adjust Huber delta or loss composition to reduce spike-dominated gradients.
- Selected first hypothesis: H1, because it is local, default-off, easy to ablate, and directly targets weak periodicity without changing splits or metrics.
- Smoke test:
  - Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant baseline --horizon 96 --epochs 1 --percent 5 --batch-size 32 --num-workers 0 --run-id smoke_weather96_baseline`
  - Result path: `research_runs/smoke_weather96_baseline/`
  - Result: completed on CUDA; test MAE 0.292750, test MSE 0.239384.
  - Comparability: not an effect conclusion because it used only 5% train data and 1 epoch.
- Tooling decision: final metrics use Lightning `trainer.test`, matching the official run scripts more closely; bad cases are sampled from the first 8 test batches and capped at 10 cases per iteration to keep analysis efficient.
- Evidence status: pending full baseline and H1 experiment.

## Iteration 1 - Full Weather Baseline

- Goal: establish the current-version baseline for Weather 720 -> 96 using the official Weather hyperparameters.
- Hypothesis under test: none; this is the comparison baseline.
- Experiment ID: `weather96_baseline_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant baseline --horizon 96 --epochs 30 --percent 100 --batch-size 16 --num-workers 0 --run-id weather96_baseline_e30_seed2021`
- Key parameters: Weather, lookback 720, horizon 96, seed 2021, batch size 16, max epochs 30, Huber-enabled MSE setting inherited from `run_weather.py`, no weak-period residual branch.
- Comparability: this will be the baseline for the first H1 experiment.
- Result path: `research_runs/weather96_baseline_e30_seed2021/`
- Result: early stopped after 17 completed epochs; test MAE 0.196280, test MSE 0.148928, elapsed 496.8 s.
- Improvement target for success: MAE < 0.176652 and MSE < 0.134035.
- Bad case summary: top sampled bad cases are concentrated in adjacent samples from test batch 4 and one from batch 5; sampled worst case MSE ranges from 0.372711 to 0.393980, with MAE around 0.406 to 0.419. This pattern suggests a localized regime/window where the phase-only path underfits level or drift, consistent with H1.
- Iteration decision: keep baseline as current reference and run H1 weak-period residual path next.

## Iteration 2 - H1 Weak-Period Residual Path

- Goal: test whether adding a direct temporal residual branch improves Weather 720 -> 96 weak-period forecasting.
- Candidate hypothesis: H1 Weak-period residual path.
- Mechanism: an NLinear-style head extrapolates the centered recent trajectory on the normalized input and blends it with the PhaseFormer phase prediction via a learned per-channel gate initialized to 0.2.
- Theory intuition: when periodicity is weak or phase-shifted, fixed phase tokens with `period_len=24` can over-emphasize same-phase history and miss short-term drift. A residual persistence/extrapolation path supplies a low-frequency anchor while the phase path keeps periodic structure.
- Risk: the residual path may dominate and reduce phase-specific modeling, especially if the gate grows too high.
- Experiment ID: `weather96_trend_residual_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant trend_residual --horizon 96 --epochs 30 --percent 100 --batch-size 16 --num-workers 0 --run-id weather96_trend_residual_e30_seed2021`
- Comparability: identical data split, horizon, seed, batch size, max epochs, loss setting, and metric path as the baseline, with only H1 enabled.
- Result: pending.
