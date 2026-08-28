# Agent Maintenance Log

## 2026-08-27 — Periodic-residual next-stage 288-run formal matrix completed

- 完成预注册 288-run 矩阵（12 setting × 8 mode × 3 seed；ETTh1/ETTh2/ETTm1/
  ETTm2/Weather/Electricity × horizon 96/192，lookback 720、period 24；
  full-train、best-val checkpoint、单次 test 读取）。4 张 GPU 并行（
  `scripts/_gpu_periodic_residual_runner.py` 按命令轮转分片），全部 run 正常完成，
  无缺失/重复 key。
- 汇总器生成 `research_runs/periodic_residual_next_stage_v1/formal_summary.csv` 与
  `decision_summary.json`；结果回填至
  `docs/PhaseFormer_periodic_residual_next_stage.md` §3.2/§3.3，结论写入 §4。
- 机制诊断（`scripts/collect_mechanism_diagnostics.py`，seed 2021、best.ckpt 前向）
  输出 `mechanism_diagnostics.csv`：D1 内容检索熵随样本变化未塌缩但 gate 只在
  Electricity 打开；D2 内层周期 gate 持续偏低；D3 路由按数据集选周期（ETTh→P24、
  ETTm1→P96、Weather→P12）但 correction gate 几乎恒为 0。
- 结论：**没有候选满足替换 A2 的统一门槛**。I0（`rcrf_icpt_none`）达到 8/12
  双指标改善（宏平均 0.9969，Weather/Electricity 稳定超 Golden），但 ETTh2-96
  MSE 回退 +6.5% 被挡在门槛外；D1/D2/D3 均在 ±0.6% 内、机制 gate 收敛到零。
  先前“ICPT 系统性弱于 NLinear”的结论只在 ETTh2 上成立。原始 checkpoint 与
  metrics 保留在被 `.gitignore` 忽略的 `research_runs/periodic_residual_next_stage_v1/`。

## 2026-08-27 — ICPT ETTh2/ETTm2 formal test rerun

- 按 full-train、best-validation checkpoint、single test read 协议完成
  ETTh2-720 与 ETTm2-96 的 `RCRF+NLinear`、旧 ICPT decoder、full-horizon ICPT，
  共 18 个 seed/model runs；GPU 为 RTX 4090。
- 汇总结果写入 `docs/PhaseFormer_icpt_test_results.md`；原始 checkpoint 与运行产物
  保留在被 `.gitignore` 忽略的 `research_runs/icpt_etth2_ettm2_full_20260827/`。
- 结论：full-horizon ICPT 两个 setting 均优于旧 decoder，但未稳定超过
  RCRF+NLinear 或固定 Golden。

## 2026-08-26 — Merge experiment plans and results into closed-loop experiment files

- 根据用户要求，将动态相位、Pure Phase、残差拓扑、Golden 组合和周期位置编码路线整理为“一条实验路线一个文件”，每个文件统一包含：设想、整体计划、实现与结果、最终结论。
- 新增：`docs/PhaseFormer_dynamic_phase_experiment.md`、`docs/PhaseFormer_pure_phase_experiment.md`、`docs/PhaseFormer_residual_topology_experiment.md`、`docs/PhaseFormer_gold_combo_experiment.md`、`docs/PhaseFormer_periodic_residual_pe_experiment.md`。
- `intercycle patch residual` 按用户要求未纳入；原始 plan/results 文件保留为审计来源。
- 验证：静态检查新增文档结构与 git diff；未重新运行训练实验。

## 2026-08-25 — Golden combo stability experiment (gold_combo_stability_v1)

Running `docs/PhaseFormer_gold_combo_plan.md` end-to-end (user authorized full
run on all 4 GPUs). Implementation committed `a5f0b1f` (RCRF module +
`gold_combo_*` preset modes), tooling `7694579` (analyze/fill scripts).

- **RCRF** (`ReliabilityCoupledResidualFusion`): reliability
  `r = Var_l(mean_k x) / (Var_l(mean_k x) + mean_l Var_k x + eps)` computed from
  the **pre-shrinkage** phase series; sensitivity `s = s_max·tanh(s_raw)` with
  `s_raw` initialized at `atanh(s0/s_max)` (s0=0 ⇒ α=0.5 constant = fixed-gate
  warm start); `alpha = sigmoid(logit(α₀) + s·(1−r))`, sample×channel.
- **Stage A** (validation-only, 30% data, 8 epochs, seed 2021, 18/18 runs;
  `test_mse/test_mae` empty = no test loader; unique config hashes). 6-ratio
  score: s2 0.80473 < adaptive 0.80720 < s0 0.80739 < fixed 0.80827.
  **Frozen candidate: `gold_combo_reliability_s2`** (selection source
  `validation_only`, test not read before freeze). record:
  `research_runs/gold_combo_screen_runs/freeze_record.json`.
- **Stage B** complete (27/27, all 4 GPUs): original/latest/frozen × 3 settings
  × seeds 2021/2022/2023. Frozen candidate `gold_combo_reliability_s2`:
  - ETTh2-720: 3-seed mean MSE 0.394228±0.005051, MAE 0.429443±0.002123 —
    **stable**, above Golden (0.402/0.436) +1.93%/+1.50%.
  - ETTm2-96: MSE 0.159755±0.000180, MAE 0.245331±0.000280 — **stable**, above
    Golden (0.163/0.256) +1.99%/+4.17%; also beats `latest` both metrics.
  - Electricity-336: all 3 seeds below Golden (MSE 0.162954/0.164409/0.164977,
    MAE 0.253420/0.254921/0.255533) but MSE mean+std 0.16516 crosses Golden 0.165
    by a rounding-level margin → NOT a stable gain per plan; slight regression vs
    `latest` (+0.47%/+0.51%). 3-seed mean vs Golden is −0.54%/−0.92% (improvement).
  - **Cross-dataset success criterion MET** (2/3 stable + remaining ≤1% regression
    vs Golden). Honest caveat recorded: no rounding-level margin claimed as stable.
- RCRF activity (r-α corr ≈ −1.0 across all 9 setting×seed): ETTh2 r=0.193→α≈0.77-0.81
  (sens 1.65-1.86); ETTm2 r=0.019→α≈0.87 (sens 1.97-2.01, low-reliability leans
  residual); Electricity r=0.772→α≈0.31 (sens 0.78-0.94, high-reliability leans
  phase — the mechanism behind the small regression vs `latest`).
- Smoke (3 settings) validated finite val loss, best.ckpt, validation-only
  isolation. Unit tests green (incl. 15 RCRF + gold_combo preset tests).
- Audit package `research_runs/gold_combo_stability_v1/` complete + validated:
  six-file protocol + figures/ (18 referenced PNGs, ZIP byte-identical), npz 2.2MB
  (269 aligned selected cells), sample_errors.csv per-cell 704MB (gitignored).
  Tables filled `docs/PhaseFormer_gold_combo_experiment_tables.md`; results doc
  `docs/PhaseFormer_gold_combo_results.md`. Committed via SSH over 443.

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

## 2026-08-24 — Residual topology experiments executed (Stage A + Stage B)

- Executed the plan end-to-end on 4× A100-40GB (multi-GPU via
  `CUDA_VISIBLE_DEVICES`). **Stage A**: 24 validation-screen runs
  (`search_phaseformer.py --stage mechanism_screen_1`, 30% data, ≤8 epochs,
  no `--evaluate-test`). **Stage B**: 12 full-budget confirm runs
  (`benchmark_phaseformer_suite.py`, 100% data, ≤30 epochs, val early stop +
  best ckpt, test metrics). Tests passed 90/90 before launch.
- **R3≡R4 equivalence verified numerically** on ETTh2-h720 (1 layer): identical
  val_mae=0.66184554, val_mse=0.82789717, params=734 → implementation correct.
- **Stage A freeze** (score = 0.5·ΔMAE% + 0.5·ΔMSE%): R1 convex (15.55) and R2
  additive (13.59) → R0+R1+R2 advanced; all candidates 4/4 settings both-metric
  improvement, no regression.
- **Stage B result (test, positive = improvement)**: residual output fusion is
  cross-setting inconsistent — ETTh2-h720 **strong** (R1 ΔMAE +5.75/ΔMSE +7.66,
  R2 +5.69/+7.56), Electricity **mild** (+0.81/+1.57, +0.41/+1.32), ETTh1/ETTm1
  neutral-to-slightly-negative (R1 −0.75/−0.06, −0.19/−0.83). Reproduces the
  prior dynamic-phase finding exactly. R1 ≥ R2 on 3/4 settings → H2 ("additive
  correction beats convex fusion") **not supported**; R3/R4/R5 provide no
  additional benefit.
- Judgment call disclosed: plan gated Electricity behind "前三项通过且仍有正向
  信号"; borderline, but ran it (extra ~1 GPU·h) to complete all 4 planned
  settings, consistent with prior full-budget residual evidence.
- Single-seed only; **`_LATEST_POLICY` not updated**. No champion topology.
- Artifacts: `research_runs/residual_topology_screen_runs/` (24 metrics.csv +
  `screen_summary.csv` + `stage_a_selection_notes.md`), `research_runs/
  residual_topology_full_runs/` (12 metrics.csv + per-setting `*_summary.csv` +
  `full_summary.csv`). Report: `docs/PhaseFormer_residual_topology_results.md`.
- Plan §4 (sample-level error analysis package at `research_runs/
  residual_topology_v1/`) was **not produced** — see report; flag if needed.

## 2026-08-25 — Output-residual layerwise variants (A1/A2) screened and confirmed

- Completed the output×depth design-space cell the first round left open: R1/R2
  had only single-point output fusion; added **A1** `residual_output_layerwise_convex`
  (R1 convex fusion applied at each routing depth) and **A2**
  `residual_output_layerwise_additive` (R2 additive correction at each depth).
  Implemented via `PhaseSlotResidualHead` (zero-init Linear(seq_len→P) in the
  phase-slot domain (B,C,24,30); `anchor=True` = convex/persistence, `anchor=False`
  = additive/warm-start), intermediate gates shape (1,enc_in,1,1), constructed only
  for `phase_layers−1` intermediate depths. 1-layer ⇒ A1≡R1, A2≡R2 exactly.
- Tests extended (90→99/99): module broadcast/anchor tests; one-layer reduction to
  parent; multilayer warm-start (A2 == original); closed-gate A1 == R1; gate/head
  receive gradients; master-switch disable. Feature-flag init isolation preserved.
- **Stage A** (validation, 8 added runs): A1 ≥ R1 on all settings (avg 15.72 vs
  15.55), A2 < R2 (13.42 vs 13.59). Strict freeze top-2 = A1+R1; per user request
  to compare both layerwise forms, sent **A1+A2** to Stage B (deviation disclosed
  in `stage_a_selection_notes.md`).
- **Stage B** (test, 8 runs, 20 total with reused originals): **layerwise does NOT
  transfer** — all multilayer settings A1 ≤ R1 and A2 ≤ R2 except A2@Electricity
  (+0.59/+1.83 vs R2 +0.41/+1.32). Test-set avg score R1 1.75 > R2 1.53 > A1 1.38 >
  A2 1.31. **Stage A validation signal reversed on test** (A1≥R1 on val vs A1<R1 on
  test everywhere) — a clean screen-vs-confirm divergence, consistent with the
  single-seed / validation-not-guarantee protocol caveat.
- 1-layer degeneracy verified numerically (ETTh2 A1≡R1, A2≡R2 byte-identical
  metrics). All deltas recomputed from on-disk `*_summary.csv` and match
  `full_summary.csv`. Report updated with §3.2 four-form comparison and H6.
  Conclusion unchanged: single-point output convex fusion (R1) remains the
  correct insertion point; layerwise cascade not adopted. `_LATEST_POLICY` not
  updated (single seed).

## 2026-08-25 — ETTm2 RCRF sample-level analysis

- Ran a matched ETTm2-h96 comparison of ordinary PhaseFormer versus
  `gold_combo_reliability_s2` with lookback 720, batch 256, MAE loss, lr 3e-4,
  best-validation checkpoints, and seeds 2021/2022/2023. The raw runs are under
  `research_runs/ettm2_rcrf_sample_raw/`.
- RCRF improved every seed. Mean test MSE changed 0.167989 → 0.159761 (4.90%);
  mean test MAE changed 0.256186 → 0.245333 (4.24%). These are matched-rerun
  deltas, not replacements for `docs/PhaseFormer_gold_standard.md`.
- Added `scripts/analyze_ettm2_rcrf_samples.py` to reconstruct all six
  checkpoints and export sample×channel errors, phase/residual branch outputs,
  reliability `r`, gate `alpha`, dataset statistics, deterministic categories,
  non-overlapping Top-K cases, and Chinese matplotlib figures.
- Operational “significant stable improvement” means all three seeds improve
  and mean relative window MAE improves by at least 10%: 2,035/11,425 windows
  (17.81%). It is explicitly not a statistical-significance claim. Net
  regression occurs on 2,697 windows (23.61%).
- Version-controlled user-facing report:
  `docs/ETTm2_RCRF_sample_analysis/ETTm2_RCRF_sample_analysis.md`. The 11
  generated figures and portable ZIP remain local-only under ignored paths.
  Canonical six-file audit package:
  `research_runs/ettm2_rcrf_sample_analysis_v1/`. The report opens with a
  plain-language evidence summary: strong improvements overrepresent drift
  windows (38.38% vs 28.14% among net regressions), while net regressions
  overrepresent high-volatility windows (21.65% vs 11.99%); nearly identical
  alpha values in both groups identify gate saturation/discrimination as the
  next mechanism to test.
- Validation passed: 54 relevant unit tests; six checkpoint metrics reproduced
  within 1e-5; exported branches and gates reconstruct the final RCRF output
  within 2e-5; 239,925 sample-error rows were re-aggregated; Top-K, setting
  coverage, Chinese glyph rendering, Markdown references, directory whitelist,
  and byte-identical ZIP members were checked.
- Corrected the ETT dataset roots in `src/dataset/data_info.py` from the absent
  `resources/all_datasets/ETT-small` directory to the repository's actual
  `resources/all_datasets/ETT` directory. No model architecture or default
  hyperparameter was changed.

## 2026-08-26 — Periodic position encoding for the RCRF residual branch

- Implemented a flag-isolated `PeriodPositionEncodedResidualHead`: a shared
  NLinear delta is blended with a position-similarity periodic retrieval delta
  before the unchanged outer RCRF. Added seven controlled PE presets: ST-Informer,
  single-cycle, fixed harmonics, Traffic hybrid, Time2Vec, learnable Fourier
  features (LFF), and calendar cycles. RoPE was excluded because NLinear has no
  query/key and adding attention would confound the architecture comparison.
- Stage A completed 24 validation-only screens (30% data, at most 8 epochs,
  seed 2021, no test read). LFF froze first with six-ratio mean `0.9995488` and
  worst `1.0003643`; Time2Vec was second. Stage B completed all 18 current-RCRF
  versus LFF runs across ETTh2-720, ETTm2-96, Electricity-336 and three seeds.
- Mean MSE/MAE current RCRF→LFF: ETTh2 `0.394228/0.429443 →
  0.393591/0.428967` (+0.162%/+0.111%); ETTm2 `0.159762/0.245333 →
  0.159678/0.245196` (+0.052%/+0.056%); Electricity `0.164114/0.254625 →
  0.164260/0.254876` (−0.089%/−0.099%). The pre-registered cross-dataset
  effectiveness rule passes, but LFF is not a universal RCRF improvement.
- Relative to fixed Golden, LFF is stably better on ETTh2 and ETTm2. Across all
  18 dataset×seed×metric cells, 17 are below Golden; Electricity seed-2022 MSE
  `0.165042` is the sole exception versus `0.165`.
- Canonical audit `research_runs/periodic_residual_pe_v1/` contains 5,028,081
  sample×channel rows, 270 programmatically selected cases, 44 Chinese
  matplotlib figures and the exact ZIP whitelist. All 18 checkpoints reproduced
  logged metrics within 1e-5; setting/case/CSV/NPZ/report/ZIP validation passed.
- Environment fallback: base conda, Python 3.13.5, torch 2.7.1+cu126, RTX 4090;
  the documented py310 path was absent. Results doc:
  `docs/PhaseFormer_periodic_residual_pe_results.md`.

## 2026-08-26 — Generated asset history cleanup

- Rewrote the branch commits after `be8a22e` to remove the ETTm2 report's 11
  generated PNGs and ZIP from version control while preserving them locally.
- Added ignore rules for `docs/**/figures/` and `docs/*.zip`; reports, code,
  numeric results, and experiment conclusions are unchanged.

## 2026-08-26 — ICPT periodic residual follow-up plan (design only)

- Closed the NLinear+periodic-PE round and designed its successor without
  implementing or running experiments. The proposed Inter-Cycle Patch
  Transformer (ICPT) treats each complete `P=24` cycle as a token, models
  cycle-to-cycle motif evolution, and replaces only the NLinear residual head;
  the current PhaseFormer phase path and outer RCRF equation stay fixed.
- The complementarity claim is structural and pre-registered: PhaseFormer
  summarizes the same-phase axis of the cycle matrix, whereas ICPT embeds each
  complete-cycle row and models the inter-cycle axis. Controls include last-cycle
  repetition, CycleNet-style recurrent template, ICPT without PE, ICPT-only,
  fixed fusion, non-period-aligned patches, no anchor, and no attention.
- Planned a validation-only screen of nine PE variants plus no-PE: fixed/learned
  absolute, Time2Vec, RoPE, relative bias, ALiBi, LFF, absolute+relative, and
  calendar. Calendar is ranked separately because it consumes real timestamp
  information. A frozen index-PE must beat ICPT-none, not only NLinear.
- Formal confirmation covers six datasets/settings, three seeds, matched current
  RCRF and fixed Golden comparisons, resource accounting, internal attention/
  gate diagnostics and programmatic sample errors. Pre-registered adoption
  requires at least 4/6 settings to improve both mean metrics, all remaining
  regressions ≤0.5%, and at least 4/6 settings to stably beat Golden before any
  optional 28-task expansion.
- The design, validation gates, executed results, and stop decision were later
  consolidated into `docs/PhaseFormer_intercycle_patch_residual_experiment.md`.
  No code, checkpoint, validation metric or test metric was produced in this
  design-only step.

## 2026-08-26 — ICPT periodic residual experiment: Stage 0 pass, Stage A gate failure

Executed the pre-registered ICPT plan
(`docs/PhaseFormer_intercycle_patch_residual_experiment.md`) under full-GPU
authorization. Implementation committed `372a5af` (ICPT module, PE variants,
PhaseFormer wiring), presets/runner `086f241`, GPU parallel runner + analyzer
`bca8909`.

- **Stage 0**: `pytest tests/ -q` all green (124 existing + 15 new ICPT tests);
  P0–P9 PE forward/backward finite with gradients; flag-off paths untouched.
- **Stage A** (architecture screen, validation-only, 30% data, ≤8 epochs, seed
  2021): 16 runs over 4 settings × {A2 gold_combo, A3 repeat-last-cycle,
  A4 CycleNet-style, A5 ICPT-none} on GPUs 0/1. Metrics in
  `research_runs/phaseformer_icpt_pe_screen/screen_summary.csv`.
- **A5 vs A2 gate** (8 ratios = 4 settings × MSE/MAE): mean **1.137**, worst
  **1.278**; only ETTh2-720 improves both metrics (0.960/0.973). Gate failed —
  neither mean<1 nor ≥3/4 settings both-metric improve holds.
- **Architecture diagnosis**: A3 RepeatLastCycle (≈0.7–4.7K params) is near
  parity only on ETTh2-720, regresses 15–60% elsewhere; A4 CycleNet
  (≈ A2 param count) is numerically within 1.3% of A2 on all 4 settings, with
  no statistical claim from the single seed; A5 ICPT (24.7K–28.2K params, far smaller than NLinear) beats A2 only
  on ETTh2-720, regresses 7–28% on the other three.
- **Decision per plan §13**: Stage A architecture gate failed → **ICPT main line
  stopped**; no PE freeze, no Stage B/C/D. `freeze_record.json` written with
  `stage_a_passed: false`; test set was never read.
- Plan doc updated: tables 9.1/9.2 filled with actuals, 9.3–9.8 marked 不适用,
  §7 B/C/D sections marked 未运行, status header reflects the stop.

## 2026-08-27 — GitHub SSH-over-443 route documented

- Verified pull/push route: GitHub's `ssh.github.com:443` through the local
  SOCKS5 proxy at `127.0.0.1:7897`.
- Added reusable temporary `core.sshCommand` examples to `AGENTS.md`; the
  commands leave the configured remote URL unchanged.

## 2026-08-27 — ICPT report consolidation and result review

- Consolidated the ICPT plan and filled Stage A results into
  `docs/PhaseFormer_intercycle_patch_residual_experiment.md`, following the
  repository's four-section closed-loop report format.
- Recomputed the reported A5-vs-A2 percentage changes: ETTh2-720 improves
  3.98%/2.67% MSE/MAE, while ETTm2, Electricity, and Weather regress by
  7.35%–27.77%. The pre-registered Stage A failure decision is unchanged.
- Clarified that Stage B/C/D and formal Golden comparison were not run, so the
  experiment neither ranks position encodings nor supports a Golden-beating
  claim. The locally generated screen CSV is absent from the current checkout,
  which limits independent run-level re-aggregation.

## 2026-08-27 — ICPT full-horizon head experiment preregistration

- Started a new, separately identified ICPT experiment at
  `docs/PhaseFormer_icpt_horizon_head_experiment.md`; the stopped decoder-based
  ICPT result remains unchanged.
- Replaced future-query decoding in the candidate with an ordered flattened
  full-horizon head and restored last-value centering/anchoring. With
  `d_model=24`, the `30×24→H` prediction matrix matches NLinear's `720→H`
  matrix size; the cycle encoder is the only additional capacity.
- Pre-registered a validation-only four-setting screen of none plus eight index
  position encodings and a separately ranked calendar encoding. All encodings
  will run; no-position is an ablation rather than a gate that blocks PE tests.
- Formal three-seed test and Golden comparison are allowed only after a frozen
  candidate beats the matched NLinear validation gate.

## 2026-08-27 — ICPT full-horizon head experiment: validation gate failure

- Implemented the ordered `30×24→H` full-horizon ICPT head, last-value
  centering/anchoring, cycle-anchor control, and nine index/calendar position
  variants. The legacy decoder remains the default flag-off path.
- Stage 0 passed: 146 repository tests, finite forward/backward for every
  candidate, exact zero-init last-value persistence, history-only calendar
  invariance, and two ETTm2 5%/1-epoch GPU smoke runs. The full-horizon matrix
  matches NLinear's `720→H`; total residual-head overhead ranges from 8.07% at
  H=96 to 1.08% at H=720.
- Stage A completed all 48 validation-only runs on ETTh2-720, ETTm2-96,
  Electricity-336, and Weather-336 (seed 2021, 30% train, at most 8 epochs),
  with no test loader and no OOM. All candidates improved both metrics only on
  ETTh2.
- `sincos_relative` had the best eight-ratio mean versus matched RCRF-NLinear
  at 0.999544, but its worst ratio was 1.041909 and it improved both metrics in
  only 1/4 settings. Calendar also failed (mean 1.002364, worst 1.042364).
  Consequently no candidate was frozen and formal three-seed testing was not
  run.
- Relative to the stopped decoder ICPT, the new no-PE head recovered roughly
  18.4%/12.6% MSE/MAE on ETTm2, 13.6%/5.2% on Electricity, and 20.2%/16.7%
  on Weather. This validates the head/anchor diagnosis but not stable
  superiority over NLinear. Full results and the stop decision are in
  `docs/PhaseFormer_icpt_horizon_head_experiment.md`.

## 2026-08-27 — Periodic-complementary residual next-stage preregistration

- Pre-registered three NLinear-preserving residual directions: content-aware
  phase-template-error memory, dual-reliability LFF routing, and an adaptive
  12/24/48/96 multi-period bank.
- The new plan re-evaluates both decoder and full-horizon no-PE ICPT without an
  early architecture gate. All eight matched modes must cover ETTh1/ETTh2/
  ETTm1/ETTm2/Weather/Electricity at lookback 720 and horizons 96/192, with
  three seeds: 288 formal runs in total.
- The plan discloses prior ETTh2/ETTm2 test exposure, freezes all candidates
  before further tests, and uses RCRF+NLinear+LFF as the primary incumbent.
  Protocol, success rules and empty result tables are in
  `docs/PhaseFormer_periodic_residual_next_stage.md`.

## 2026-08-27 — Periodic-complementary residual candidates implemented

- Implemented `PhaseErrorPeriodicMemoryHead`,
  `DualReliabilityPeriodicFusion`, and `AdaptiveMultiPeriodResidualHead` in
  `src/models/periodic_residual_experts.py`. D1/D3 start exactly as NLinear;
  D2 preserves the old LFF component outputs but replaces its global blend with
  sample/channel residual-cycle reliability.
- Added isolated presets `rcrf_phase_error_memory`,
  `rcrf_dual_reliability_lff`, and `rcrf_multiperiod`; existing NLinear, LFF
  and both ICPT paths remain unchanged by default.
- Added a formal runner/summarizer that expands the frozen six-dataset,
  96/192, three-seed matrix to 36 commands and 288 model runs. Summarization
  refuses incomplete/duplicate matrices and computes sample std, A2 ratios,
  stable-Golden counts and the pre-registered replacement gate.
- Verification: 160 repository unit tests passed; full PhaseFormer forwards at
  both horizons, actual `720→192` finite backward, exact NLinear warm starts,
  normalized/sample-varying diagnostics, all dataset presets, dry-run count and
  synthetic summarization were checked. No training/test experiment was run.
  Code commit: `d1ab49e`.

## 2026-08-28 — TriAxis 自验证三专家实验在 validation 门槛停止

- 实现 PhaseFormer/NLinear/旧 decoder ICPT 三个原子专家与单一历史路由器；T0 固定均匀，T1
  使用结构统计，T2 使用历史内伪预测风险。推理路由不读取 future value、future mark 或专家预测。
- T2 训练目标加入专家辅助损失 0.2 和 oracle 路由 KL 0.1；旧 preset flag-off state dict 不变。
- 验证：168 项单元测试通过；ETTh2、ETTm2、Weather、Electricity 的 L720→H96、seed 2021、
  30% train、8 epoch validation-only 共 20 个 run 完成。
- T2 的 8 指标宏平均比值 1.0005、最差 1.0426，只在 2/4 setting 双指标改善；T0/T1 也失败。
  按预注册规则停止，不读取 test，不更新 A1/RCRF+NLinear incumbent。
- 三专家逐点 oracle 宏平均改善 47.80%，但实际路由命中率只有 34.54%–39.27%，说明瓶颈是
  历史代理风险与未来专家 regret 的错配，而不是专家完全缺乏互补性。
- 审计产物：`research_runs/triaxis_self_validating_v1/`；实现 commit `e313ee4`。原始 checkpoint
  和训练日志只保留在被忽略的 scratch 目录，不加入版本控制。

## 2026-08-28 — TriAxis v2 多截点滚动校准仍在 validation 门槛停止

- 修正 v1 的单截点代理错配：对最近四个历史目标周期按未来 1–4 个周期的相同 lead 做
  rolling-origin 回测，输出三专家风险及跨 origin 方差。R0 只把证据作为特征，R1 强制低风险
  单调先验，R2 再加周期级 soft-oracle KL。实现 commit `d7ecc7f`。
- Stage 0：174 项仓库测试通过；新增 H96/H192 shape、严格历史因果、线性/周期回测、风险单调、
  不确定性收缩、梯度和完整 PhaseFormer forward 测试；ETTm2 5%/1 epoch GPU smoke 通过。
- Stage A：ETTh2/ETTm2/Weather/Electricity，L720→H96、P24、seed 2021、30% train、最多
  8 epoch、validation-only，完成 R0/R1/R2 共 12 个新 run，并复用 A1/I0/T2-v1 配对结果。
- R0/R1/R2 的 8 指标宏平均比值分别为 0.992243/0.999310/1.007830，最差比值分别为
  1.026184/1.015926/1.042114；双指标改善为 2/4、2/4、1/4，全部未通过预注册 gate。
  R0 改善 Weather 和 Electricity，也改善 ETTh2 MAE，但 ETTm2 MSE/MAE 回退 2.62%/1.42%。
- 结论：多截点等 horizon 特征相对 T2-v1 有效，但伪风险排序不够可靠；强制风险单调和周期级
  路由监督都使宏平均更差。按规则停止，未访问 test，A1/RCRF+NLinear incumbent 不变。
- 三专家 validation 优势：ETTm2 的轨迹专家四个 24 步段都第一，领先第二名 10.7%–29.8%；
  ETTh2 的周期间专家在 1–24 领先 23.9%，且高 lag-24/低形状创新区间胜率显著提高；Weather
  和 Electricity 的较远区间更多由相位专家占优。共得到 48 个满足 n、lift 和 bootstrap CI
  约束的优势区间，但 R0 的滚动风险首选命中率仅约 30.7%–41.8%。
- 审计：`scripts/analyze_triaxis_rolling_calibration.py` 在 validation 上复算 A1/T2-v1/R0 指标，
  误差均 `<1e-5`；本地 `research_runs/triaxis_rolling_calibration_v2/` 含 1,022,522 条
  sample×channel 记录、9 个程序化去重案例、7 张中文图和已校验 ZIP。该目录被忽略，不提交
  426 MiB 的样本 CSV 或图片；代码与数值结论写入仓库文档。
- 关键命令：`python scripts/search_phaseformer.py ... --mechanism
  <triaxis_rolling_features|triaxis_rolling_prior|triaxis_rolling_calibrated> --lookback 720 --horizon 96
  --percent 30 --max-epochs 8 --seed 2021 --loss huber`；审计命令：
  `python scripts/analyze_triaxis_rolling_calibration.py`（RTX 4090，torch 2.4.1+cu121）。
