# PhaseFormer Weak-Period Research Brief

## Project Understanding

- Model code: `src/models/PhaseFormer.py`.
- Training module: `src/models/pl_bases/default_module.py`, with PhaseFormer overriding the Lightning steps.
- Data loading: `src/dataset/data_factory.py`, `src/dataset/data_loader.py`, dataset metadata in `src/dataset/data_info.py`.
- Metrics: `src/utils/metrics.py`; primary comparison uses MAE and MSE on the test split.
- Official run scripts: `run_weather.py`, `run_electricity.py`, `run_traffic.py`, and ETT variants.
- Evidence directory for this task: `research_runs/<experiment_id>/`.
- Local data available for Weather, Electricity, Traffic, PEMS, ETT, exchange rate, and illness. Weather is selected first as the weak-period target because it has multi-variable meteorological signals with less rigid daily periodicity than traffic/electricity load.
- Hardware observed at start: NVIDIA GeForce RTX 4090, 24564 MiB.
- Python environment selected for experiments: conda env `raft`, because it has PyTorch, PyTorch Lightning, pandas, scikit-learn, and easydict installed.

## User Requirement

Original request: "请你根据AGENT.md, 进行自主性科研，优化PhaseFormer模型在弱周期数据上的表现，使得其比当前版本上的预测误差MAE和MSE降低均超过10%，或者模型迭代超过30轮就结束。"

Operational constraints:

- Follow `AGENT.md` and `HOW_TO_DO_RESEARCH.md`.
- Improve PhaseFormer on weak-period data.
- Stop when both MAE and MSE are reduced by more than 10% against the current-version baseline, or when model iteration exceeds 30 rounds.
- Keep experiments reproducible and auditable.
- Do not change data splits or metric definitions.
- Prefer low-cost single-horizon evidence first, then expand only when the signal justifies it.
- Limit each iteration's bad case table to no more than 10 cases to keep analysis efficient.

## Baseline Status

Pending. Smoke test passed at `research_runs/smoke_weather96_baseline/`, but it used 5% train data and 1 epoch, so it is not an effect baseline. The first comparable baseline will use Weather 720 -> 96, seed 2021, 30 requested epochs, 100% train data, and the existing Weather configuration from `run_weather.py`.

## Exit Conditions

- Success: a candidate on the selected weak-period benchmark reduces both test MAE and test MSE by more than 10% relative to the matching baseline.
- Stop: 30 model/research iterations are reached without meeting the improvement target.
- Blocked: dependency, data, or hardware constraints prevent comparable experiments after repeated attempts.

## Current Best

Pending baseline.
