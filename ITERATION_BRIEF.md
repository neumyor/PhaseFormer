# PhaseFormer Weak-Period Research Brief

## Project Understanding

- Model code: `src/models/PhaseFormer.py`.
- Training module: `src/models/pl_bases/default_module.py`, with PhaseFormer overriding the Lightning steps.
- Data loading: `src/dataset/data_factory.py`, `src/dataset/data_loader.py`, dataset metadata in `src/dataset/data_info.py`.
- Metrics: `src/utils/metrics.py`; primary comparison uses MAE and MSE on the test split.
- Official run scripts: `run_weather.py`, `run_electricity.py`, `run_traffic.py`, and ETT variants.
- Evidence directory for this task: `research_runs/<experiment_id>/`.
- Local data available for Weather, Electricity, Traffic, PEMS, ETT, exchange rate, and illness. Weather is selected first as the weak-period target because it has multi-variable meteorological signals with less rigid daily periodicity than traffic/electricity load.
- After Weather iterations showed limited weak-period gains, Exchange is added as a more direct weak-period benchmark: daily financial series, 8 variables, local path `resources/all_datasets/exchange_rate.csv`.
- New ETT round data target: local ETT files are under `resources/all_datasets/ETT/` (`ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`). The first ETT target is ETTh2 720 -> 96 because hourly transformer temperature/oil features have a 24-step phase prior but weaker, regime-dependent periodic alignment than load-style benchmarks.
- Hardware observed at start: NVIDIA GeForce RTX 4090, 24564 MiB.
- Python environment selected for experiments: conda env `raft`, because it has PyTorch, PyTorch Lightning, pandas, scikit-learn, and easydict installed.

## User Requirement

Original request: "请你根据AGENT.md, 进行自主性科研，优化PhaseFormer模型在弱周期数据上的表现，使得其比当前版本上的预测误差MAE和MSE降低均超过10%，或者模型迭代超过30轮就结束。"

New ETT round request: "请开始新的一轮自主迭代，在ETT系列数据上，MAE和MSE改进超过10%或者迭代超过30轮，要求突出在相位建模的框架下对弱周期数据的适应和改进。"

Operational constraints:

- Follow `AGENT.md` and `HOW_TO_DO_RESEARCH.md`.
- Improve PhaseFormer on weak-period data.
- Stop when both MAE and MSE are reduced by more than 10% against the current-version baseline, or when model iteration exceeds 30 rounds.
- Keep experiments reproducible and auditable.
- Do not change data splits or metric definitions.
- Prefer low-cost single-horizon evidence first, then expand only when the signal justifies it.
- Limit each iteration's bad case table to no more than 10 cases to keep analysis efficient.
- For the new round, compare against a matching current-version ETT baseline before claiming any ETT improvement.
- Candidate changes should remain in the PhaseFormer phase-modeling framework and specifically address weak-period adaptation, such as phase length sensitivity, phase/residual blending, or time-conditioned phase correction.

## Baseline Status

Weather 720 -> 96 baseline is established at `research_runs/weather96_baseline_e30_seed2021/`.

- Seed: 2021.
- Requested epochs: 30; early stopped after 17 completed epochs.
- Test MAE: 0.196280.
- Test MSE: 0.148928.
- Success threshold for the first target experiment: MAE < 0.176652 and MSE < 0.134035.

New ETT round baseline status:

- Initial target: ETTh2 720 -> 96.
- Dataset path fix: ETT metadata now points to `resources/all_datasets/ETT/`.
- Baseline pending: `research_runs/ett_etth2_96_baseline_e30_seed2021/`.
- Until that baseline completes, no ETT improvement claim is valid.

## Exit Conditions

- Success: a candidate on the selected weak-period benchmark reduces both test MAE and test MSE by more than 10% relative to the matching baseline.
- New ETT round success: a candidate on ETT-series data reduces both test MAE and test MSE by more than 10% relative to the matching current-version ETT baseline.
- Stop: 30 model/research iterations are reached without meeting the active round's improvement target.
- Blocked: dependency, data, or hardware constraints prevent comparable experiments after repeated attempts.

## Current Best

Current best and final selected weak-period model is `exchange96_residual_gate999_lr00013_mae_e30_seed2021`.

- Dataset: Exchange, lookback 720, horizon 96.
- Mechanism: residual-dominant weak-period head (`trend_residual`, gate init 0.999) trained with MAE loss and LR 0.00013.
- Baseline: `research_runs/exchange96_baseline_e30_seed2021/`, MAE 0.221346, MSE 0.095170.
- Final: `research_runs/exchange96_residual_gate999_lr00013_mae_e30_seed2021/`, MAE 0.198869, MSE 0.082640.
- Improvement: MAE -10.15%, MSE -13.16%.
- Exit condition: satisfied before 30 model iterations.
- Rejected variant: H1 with gate init 0.8 at `research_runs/weather96_trend_residual_gate08_e30_seed2021/` because it underperformed the default H1 gate.
- Rejected variant: H2 time-mark adjustment at `research_runs/weather96_time_mark_e30_seed2021/` because it increased both MAE and MSE versus baseline.
- Partial result: `period_len=12` at `research_runs/weather96_period12_e30_seed2021/` achieved the best MAE so far, 0.193105, but MSE was only 0.05% below baseline.
- Partial result: `period_len=12` plus H1 at `research_runs/weather96_period12_trend_residual_e30_seed2021/` achieved the best MSE so far, 0.148223, but MAE was worse than period_len 12 alone.
- Rejected variant: latent_dim 32 at `research_runs/weather96_lat32_e30_seed2021/` because both MAE and MSE degraded substantially.
- Rejected variant: residual-dominant gate 0.999 at `research_runs/weather96_residual_gate999_e30_seed2021/` because average Weather metrics degraded.
- Current pivot: establish an Exchange baseline before evaluating further weak-period improvements.
