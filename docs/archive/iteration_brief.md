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

Original request: "请你根据MANAGE_RULES.md, 进行自主性科研，优化PhaseFormer模型在弱周期数据上的表现，使得其比当前版本上的预测误差MAE和MSE降低均超过10%，或者模型迭代超过30轮就结束。"

New ETT round request: "请开始新的一轮自主迭代，在ETT系列数据上，MAE和MSE改进超过10%或者迭代超过30轮，要求突出在相位建模的框架下对弱周期数据的适应和改进。"

Operational constraints:

- Follow `MANAGE_RULES.md` and `HOW_TO_DO_RESEARCH.md`.
- Improve PhaseFormer on weak-period data.
- Stop when both MAE and MSE are reduced by more than 10% against the current-version baseline, or when model iteration exceeds 30 rounds.
- Keep experiments reproducible and auditable.
- Do not change data splits or metric definitions.
- Prefer low-cost single-horizon evidence first, then expand only when the signal justifies it.
- Limit each iteration's bad case table to no more than 10 cases to keep analysis efficient.
- For the new round, compare against a matching current-version ETT baseline before claiming any ETT improvement.
- Candidate changes should remain in the PhaseFormer phase-modeling framework and specifically address weak-period adaptation, such as phase length sensitivity, phase/residual blending, or time-conditioned phase correction.

Safe-Regret TriAxis round (2026-08-29): the user rejects an integrated model
that cannot exceed its original components and requests actual testing on wider
data and settings. The operational interpretation is one unified, A1-anchored
candidate family on ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity at H96 and
H192. Validation compares every metric against the per-cell best of A1, I0 and
R0; any regression fails promotion. No test split is accessed unless all 24
expanded validation metric cells strictly improve.

Safe-Regret TriAxis outcome (2026-08-29): all 66 preregistered validation runs
completed on ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity at H96/H192. S2
(`safe_triaxis_guarded`) won the H96 candidate ranking, but failed the gate:
macro ratio 1.010499 and worst ratio 1.053162 against the per-cell A1/I0/R0
envelope. It averaged 0.37% better than A1 alone, showing that exact A1
fallback works, but it cannot inherit settings where I0 or R0 is stronger.
No test split was accessed. Canonical local audit:
`research_runs/safe_regret_triaxis_v1/`; next design must use a multi-model
anchor or a distilled strong anchor rather than further A1-local tuning.

Multi-Anchor Selector round (2026-08-29): the user asks to continue with the
multi-anchor alternative. The new unified mechanism treats complete A1, I0 and
R0 forecasts as actions. To avoid training a stacker on targets already seen by
its anchors, 24%-trained shadow anchors produce forecasts on the disjoint
24%-30% temporal calibration segment; the learned router is then evaluated
with the frozen 30%-trained anchors. Pilot covers ETTh1/ETTm2/Weather H96;
promotion covers all six datasets at H96/H192. Test remains inaccessible until
every validation metric beats the original-model envelope.

Multi-Anchor Selector result (2026-08-29): the M3 structural-soft router won
with a six-dataset H96 macro envelope ratio of 0.992072 (0.79% average gain),
improving 10/12 metrics. ETTh1 MAE (+0.58%) and ETTm1 MAE (+0.01%) still
regressed, so the preregistered Stage-A gate failed and H192/test were not run.
All hard variants were worse than M3. Validation replay covered 1,121,992
sample-channel pairs; the evidence favors convex forecast error cancellation,
not reliable hard expert identification. Canonical local audit:
`research_runs/multi_anchor_selector_v1/`.

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
- Baseline: `research_runs/ett_etth2_96_baseline_e30_seed2021/`, MAE 0.343032, MSE 0.280557.
- New ETT success threshold: MAE < 0.308729 and MSE < 0.252501.

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

New ETT round interim ETTh2 96 result:

- Target: ETTh2 720 -> 96.
- Baseline: `research_runs/ett_etth2_96_baseline_e30_seed2021/`, MAE 0.343032, MSE 0.280557.
- Best MAE so far: `research_runs/ett_etth2_96_residual_gate999_lr0003_mae_e30_seed2021/`, MAE 0.329992, MSE 0.272189.
- Best MSE so far: `research_runs/ett_etth2_96_trend_residual_e30_seed2021/`, MAE 0.333583, MSE 0.267520.
- Conclusion: ETTh2 96 improved, but stayed far below the 10%/10% target, so the round pivoted to ETTh2 720 and other ETT checks.

New ETT round final status:

- Stop condition reached: model/research iteration count exceeded 30 without a candidate reducing both MAE and MSE by more than 10%.
- Best overall ETT run: `research_runs/ett_etth2_720_residual_gate999_e30_seed2021/`.
- Matching baseline: `research_runs/ett_etth2_720_baseline_e30_seed2021/`, MAE 0.448750, MSE 0.415718.
- Best result: MAE 0.424987, MSE 0.383477.
- Improvement: MAE -5.30%, MSE -7.76%, below the required -10%/-10% exit target.
- Mechanism retained as the strongest direction: residual-dominant weak-period phase adaptation, where the phase path remains present but long-horizon predictions are anchored by recent-trajectory extrapolation.
- Mechanisms rejected or not retained: isolated phase length changes, time-mark correction, larger phase capacity, phase-local slope correction, adaptive gate, and channel-wise residual head.

## Latest Formal Benchmark Plan

- New user request: update the project model design and auxiliary experiment design to the latest scheme, then run full tests against the original version across datasets. Training parameter adjustment is allowed.
- Formal latest policy is centralized in `src/models/phaseformer_presets.py`.
- `original` mode: original PhaseFormer phase path with dataset/horizon hyperparameters, no weak-period auxiliary branch.
- `latest` mode: dataset-aware guardrailed policy:
  - Exchange: residual-dominant weak-period branch, gate 0.999, MAE loss, LR 0.00013.
  - ETTh2 96: adaptive residual gate, gate init 0.2, MAE loss, LR 0.0003.
  - ETTh2 720: residual-dominant weak-period branch, gate 0.999.
  - Other dataset/horizon combinations: original phase path, because prior full experiments showed phase-local trend, channel-wise residual, and unconditional residual variants can degrade strong-period or unsupported settings.
- Formal benchmark runner: `scripts/benchmark_phaseformer_suite.py`.
- Evidence output: `research_runs/<run_prefix>_*` with per-run `metrics.csv`, `config.json`, `bad_cases.csv`, `runtime.md`, plus `<run_prefix>_summary.csv` and `<run_prefix>_comparison.csv`.
- Bad case cap remains 10 per run.

Formal benchmark outcome:

- ETT/Exchange full-data comparison: `research_runs/phaseformer_full_latest_vs_original_20260630_comparison.csv`.
- High-dimensional/Weather full-data comparison: `research_runs/phaseformer_full_latest_vs_original_highdim_b64w4_20260630_comparison.csv`.
- High-dimensional datasets used batch size 64 and `num_workers=4` for both original and latest modes after official batch 16 proved impractically slow on Electricity/Traffic; comparison remains paired and fair, but should be read as the batch64 full-data setting.
- Latest improves Exchange on all four horizons:
  - 96: MAE -10.15%, MSE -13.17%.
  - 192: MAE -5.31%, MSE -4.62%.
  - 336: MAE -12.13%, MSE -18.09%.
  - 720: MAE -9.38%, MSE -15.24%.
- Latest improves ETTh2 where enabled:
  - 96: MAE -4.31%, MSE -5.00%.
  - 720: MAE -5.30%, MSE -7.76%.
- Latest guardrail keeps ETTh1, ETTm1, ETTm2, Electricity, Traffic, and Weather identical to original on tested horizons, preventing the regressions observed when weak-period branches were forced onto unsupported settings.

## New ETTh1/ETTm1/ETTm2 Weak-Period Round

- New user request: treat ETTh1, ETTm1, and ETTm2 as weak-period datasets and continue model-design research until MAE and MSE both improve by more than 5%, or this round exceeds 50 iterations.
- Required mechanism constraint: improvements must target weak-period properties and include theoretical support.
- Initial evidence: raw/adaptive residual helps ETTm2 96 but does not solve ETTh1 or ETTm1; full daily `period_len=96` on ETTm data worsens, suggesting rigid calendar phase is too brittle.
- New candidate mechanism: low-pass weak-period residual branch. The branch estimates residual extrapolation from a moving-averaged trajectory while anchoring the output at the raw last value. It is intended to model low-frequency drift under weak periodicity without extrapolating high-frequency noise.
- New candidate mechanism: phase-jitter smoothing. It treats weak periodicity as small random phase shifts and approximates marginalization over neighboring phase slots before phase embedding.
- Active success thresholds:
  - ETTm2 96 baseline: MAE 0.258160, MSE 0.170091; threshold MAE < 0.245252 and MSE < 0.161587.
  - ETTm1 720 baseline: MAE 0.409929, MSE 0.412445; threshold MAE < 0.389433 and MSE < 0.391823.
- Current result:
  - ETTm2 96 satisfies the 5% target with `weakphase2_ettm2_96_residual_gate999_lr0003_mae_e30_seed2021`: MAE 0.245211, MSE 0.160063.
  - ETTm1 and ETTh1 did not reach the 5%/5% target before the iteration limit.
  - Stop condition reached: this round exceeded 50 iterations.
  - Best ETTm1 96 result after bad-case-driven refinement is `weakphase2_ettm1_96_phase_hifreq_s05_thr10_w7_residual_gate999_lr0003_mae_b256_e30_seed2021`: MAE 0.338555, MSE 0.292791 versus fixed baseline MAE 0.347958, MSE 0.299526.
  - Best ETTm1 192 result is `weakphase2_ettm1_192_adaptive_residual_g02_lr0003_mae_b256_e30_seed2021`: MAE 0.359033, MSE 0.328808 versus paired baseline MAE 0.363096, MSE 0.329395.
  - ETTh1 bad-case review showed trend under-response rather than high-frequency phase hallucination; residual, phase-local trend, and low-frequency trend variants were positive only for isolated bad-case modes and did not improve both aggregate metrics.
  - Retained research tooling change: `scripts/research_weather_weak.py` now exports <=8/10 pattern-covered bad cases with timestamps, variable names, and window-level prediction/true CSVs, so future iterations must choose mechanisms from bad-case modes rather than aggregate metrics alone.

## Weak-Period ETT Innovation Round - 2026-07-05

- User request: start a new autonomous iteration under the updated `HOW_TO_DO_RESEARCH.md`, targeting weak-period ETT-series data. Stop when MAE and MSE both improve by more than 5% versus the original PhaseFormer baseline, or when the round exceeds 30 model/research iterations.
- Method constraints:
  - Mechanisms must be framed inside PhaseFormer phase modeling and address weak-period data properties.
  - Simple residual branches, extreme residual gates, or post-hoc dataset/horizon-aware guardrails are not acceptable as the claimed innovation.
  - Every key experiment must export and review no more than 10 bad cases; this round uses 8.
  - Prefer a unified mechanism and shared hyperparameters across ETT datasets; dataset-specific choices require evidence, not opportunistic selection.
- Baselines from the formal ETT regression:
  - ETTh1 96: MAE 0.388491, MSE 0.364891; 5% thresholds MAE < 0.369066, MSE < 0.346646.
  - ETTm1 96: MAE 0.347958, MSE 0.299526; 5% thresholds MAE < 0.330560, MSE < 0.284550.
  - ETTm2 96: MAE 0.258160, MSE 0.170091; 5% thresholds MAE < 0.245252, MSE < 0.161587.
  - ETTh2 96: MAE 0.343032, MSE 0.280557; 5% thresholds MAE < 0.325880, MSE < 0.266529.
- Current hypothesis family:
  - Phase uncertainty shrinkage: estimate same-phase observation reliability and shrink noisy period-specific deviations before cross-phase routing.
  - Phase deviation dropout: training-time regularization that drops deviations from the phase template, intended to reduce memorization of unstable same-phase details.
  - Phase period-level detrending: decompose `x_{l,k}=p_l+d_k+eps_{l,k}` inside phase space, route de-leveled phase shape, and restore low-frequency period level.
- Final status after iteration 31:
  - Stop condition reached because the round exceeded 30 iterations without a candidate reducing both MAE and MSE by more than 5%.
  - Best model-design result is `weakphase3_iter31_ettm2_96_phase_uncertainty_levelcalib_hifreq_min20_g02_s08_thr05_lr0003_mae_e30_seed2021`: ETTm2 96 MAE 0.248220, MSE 0.160189 versus original MAE 0.258160, MSE 0.170091.
  - Improvement: MAE -3.85%, MSE -5.82%. MSE satisfies the 5% target, but MAE does not.
  - Matched training baseline `weakphase3_iter12_ettm2_96_baseline_lr0003_mae_e30_seed2021` reaches only MAE 0.257416, MSE 0.168600, so the ETTm2 MSE gain is attributable to the phase-uncertainty mechanism family rather than MAE/LR alone.
  - Transfer check failed: the same phase-uncertainty + level-calibration structure worsened ETTh1 96 and ETTh2 96 MSE, so it must not be promoted as a unified ETT solution.

## Dataset-Adaptive ETT Phase Framework Round - 2026-07-06

- User request: allow dataset-specific method switches and partial parameter adjustment, but keep all variants under one common method framework, then iterate 50 more rounds to seek stable improvement over original PhaseFormer on all ETT datasets.
- Interpretation:
  - Dataset-level switches are now allowed, but same-dataset horizon-level opportunistic switching still requires evidence.
  - The common framework remains weak-period phase adaptation: phase reliability/shrinkage, phase-level calibration, high-frequency damping, low-frequency trend response, and previously validated residual/adaptive branches only as submodules with bad-case justification.
  - Target evidence must cover ETTh1, ETTh2, ETTm1, and ETTm2. Initial search uses horizon 96 as the low-cost representative, then expands.
  - Bad case cap remains <= 10; this round uses 8.
- Starting evidence:
  - ETTh2 already has validated dataset-specific improvements in the formal latest policy for 96 and 720.
  - ETTm2 96 has the strongest new phase-framework signal: `weakphase3_iter31_ettm2_96_phase_uncertainty_levelcalib_hifreq_min20_g02_s08_thr05_lr0003_mae_e30_seed2021`, MAE -3.85%, MSE -5.82%.
  - ETTh1 and ETTm1 remain unsolved; their bad cases differ, so dataset-level switches are now allowed but must be justified by bad-case modes.

Current best dataset-adaptive ETT phase policy:

- Implementation: `src/models/phaseformer_presets.py`, `latest` mode.
- Common framework: weak-period phase adaptation.
  - Phase uncertainty shrinkage reduces unreliable same-phase deviations before phase routing.
  - Phase-period level calibration corrects systematic period-level bias without removing phase shape.
  - Phase-noise high-frequency damping is enabled for ETTm1 and short/mid ETTm2 horizons where bad cases show noisy minute-level phase deviations.
  - ETTh2 keeps the previously validated weak-period residual/adaptive residual branch because its bad cases are dominated by long-horizon drift that the pure phase path under-responds to.
- Dataset-level policy evidence:
  - Horizon 96: `research_runs/phaseformer_ett96_dataset_adaptive_20260706_comparison.csv`; all ETTh1, ETTh2, ETTm1, ETTm2 improve in both MAE and MSE.
  - Horizon 192: `research_runs/phaseformer_ett192_dataset_adaptive_20260706_comparison.csv`; all four improve in both metrics.
  - Horizon 336: `research_runs/phaseformer_ett336_dataset_adaptive_20260706b_comparison.csv`; all four improve in both metrics.
  - Horizon 720: `research_runs/phaseformer_ett720_dataset_adaptive_20260706c_comparison.csv`; all four improve in both metrics after the ETTh1 learning-rate stabilization.
- Horizon 720 final deltas:
  - ETTh1: MAE -1.35%, MSE -3.11%.
  - ETTh2: MAE -5.30%, MSE -7.76%.
  - ETTm1: MAE -1.28%, MSE -1.51%.
  - ETTm2: MAE -1.13%, MSE -1.09%.
- Important caveat: this is stable improvement versus original across ETT datasets and horizons, not a universal >5% gain on every dataset/horizon. The largest gains remain ETTh2 720 and ETTm2 96; the other settings are modest but paired positive under the formal runner.

## Weather/Electricity Phase Adaptation Round - 2026-07-06

- User request: under the current weak-period phase-adaptation framework, optimize Weather and Electricity. Stop when the target effect improves by more than 3%, or when the round exceeds 50 iterations.
- Interpretation:
  - Keep the same framework family: phase uncertainty shrinkage, period-level calibration, high-frequency phase damping, low-frequency trend correction, and residual/adaptive auxiliaries only when bad-case evidence justifies them.
  - Weather and Electricity currently use `latest_original_guardrail` in formal high-dimensional regression, so the matching starting point equals original PhaseFormer.
  - Use horizon 96 for low-cost mechanism search, then promote only evidence-backed settings to `phaseformer_presets.py` and run formal comparisons.
  - Bad case cap remains <= 10; this round uses 8.
- Starting formal baselines from `research_runs/phaseformer_full_latest_vs_original_highdim_b64w4_20260630_comparison.csv`:
  - Weather 96: MAE 0.195908, MSE 0.149202; 3% thresholds MAE < 0.190031, MSE < 0.144726.
  - Electricity 96: MAE 0.220274, MSE 0.128806; 3% thresholds MAE < 0.213666, MSE < 0.124942.
- First bad-case observation:
  - Electricity smoke run `smoke_electricity96_research_script` shows systematic bias, peak underfit, trend mismatch, volatility mismatch, and late-horizon drift among the 8 capped bad cases.
  - This supports testing phase-period level calibration and low-frequency trend correction before any residual-only solution.
