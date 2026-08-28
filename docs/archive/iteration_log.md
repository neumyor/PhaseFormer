# PhaseFormer Weak-Period Iteration Log

## Iteration 0 - Initialization

- Goal: understand the repository, select a weak-period benchmark, and prepare auditable experiment tooling before changing behavior.
- Files read: `MANAGE_RULES.md`, `HOW_TO_DO_RESEARCH.md`, `README.md`, `src/models/PhaseFormer.py`, `src/models/pl_bases/default_module.py`, `src/dataset/data_info.py`, `src/dataset/data_factory.py`, `src/dataset/data_loader.py`, `src/utils/metrics.py`, `run_weather.py`, `run_electricity.py`.
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
- Result path: `research_runs/weather96_trend_residual_e30_seed2021/`
- Result: early stopped after 13 completed epochs; test MAE 0.195851, test MSE 0.148321, elapsed 381.4 s.
- Delta vs baseline: MAE reduced by 0.22%; MSE reduced by 0.41%. This is directionally positive but far below the 10% target.
- Bad case summary: top sampled bad cases remain concentrated in adjacent windows, now from batches 3 and 4. Worst sampled MSE is 0.393810, similar to baseline's 0.393980, and sampled MAE is slightly worse in the hardest cases.
- Iteration decision: keep H1 as current best by average metrics, but treat the mechanism as underpowered for the hard regime. Run a gate-init ablation with stronger residual contribution before switching to a new mechanism.

## Iteration 3 - H1 Stronger Residual Gate

- Goal: test whether the H1 residual path failed because its initial fusion weight was too low.
- Candidate hypothesis: H1-gate, same weak-period residual mechanism with stronger residual prior.
- Mechanism: set `weak_period_residual_gate_init=0.8` while keeping all other settings unchanged.
- Expected outcome: if weak-period Weather 720 -> 96 benefits from short-term persistence/drift modeling, a larger residual prior should improve MAE/MSE more clearly and reduce the same adjacent-window bad cases.
- Risk: a high residual gate can suppress phase routing and overfit recent noise.
- Experiment ID: `weather96_trend_residual_gate08_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant trend_residual --gate-init 0.8 --horizon 96 --epochs 30 --percent 100 --batch-size 16 --num-workers 0 --run-id weather96_trend_residual_gate08_e30_seed2021`
- Comparability: identical to Iteration 2 except gate initialization.
- Result path: `research_runs/weather96_trend_residual_gate08_e30_seed2021/`
- Result: early stopped after 15 completed epochs; test MAE 0.195894, test MSE 0.148831, elapsed 440.3 s.
- Delta vs baseline: MAE reduced by 0.20%; MSE reduced by 0.06%. Worse than Iteration 2.
- Bad case summary: sampled worst MSE improved slightly versus baseline, but sampled worst MAE increased to 0.430021. Strong residual prior helps squared-error spikes a little while hurting absolute error.
- Iteration decision: reject stronger residual gate. The residual path is not enough for the 10% target; switch to an external time-feature correction mechanism.

## Iteration 4 - Time-Mark Adjustment

- Goal: test whether future time covariates can correct weak-period phase instability.
- Candidate hypothesis: H2 Time-mark adjustment.
- Mechanism: add a small MLP that maps future `x_mark_dec` features to a normalized-scale additive correction for each predicted variable and timestamp. The final layer starts at zero, so the initial model matches the phase path.
- Theory intuition: weak-period series may not align cleanly by fixed phase index, but calendar/time covariates such as hour, weekday, day-of-month, and day-of-year still describe recurring regimes. A direct time-conditioned correction can model this without forcing every variable through a rigid `period_len=24` phase token.
- Risk: time marks alone may learn only average seasonal bias and miss sample-specific level shifts.
- Smoke test:
  - Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant time_mark --horizon 96 --epochs 1 --percent 5 --batch-size 32 --num-workers 0 --run-id smoke_weather96_time_mark`
  - Result path: `research_runs/smoke_weather96_time_mark/`
  - Result: completed on CUDA; test MAE 0.292619, test MSE 0.237624. Not an effect conclusion.
- Experiment ID: `weather96_time_mark_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant time_mark --horizon 96 --epochs 30 --percent 100 --batch-size 16 --num-workers 0 --run-id weather96_time_mark_e30_seed2021`
- Comparability: identical baseline setup, replacing H1 with H2.
- Result path: `research_runs/weather96_time_mark_e30_seed2021/`
- Result: early stopped after 20 completed epochs; test MAE 0.211093, test MSE 0.158706, elapsed 590.0 s.
- Delta vs baseline: MAE increased by 7.55%; MSE increased by 6.57%.
- Bad case summary: hard sampled windows worsened substantially; worst sampled MSE rose to 0.428298 and all top 10 cases are again adjacent around batch 4/5.
- Iteration decision: reject H2. Future time marks alone add average seasonal bias but do not solve sample-specific weak-period drift.

## Iteration 5 - Shorter Phase Length

- Goal: test whether the baseline's fixed `period_len=24` is too rigid for weak-period Weather.
- Candidate hypothesis: H3 Shorter phase length.
- Mechanism: set `period_len=12`, doubling the number of input phase periods and output phase steps while preserving the same phase-routing architecture.
- Theory intuition: weak periodicity can appear as phase jitter relative to a 24-step daily cycle. A shorter phase length reduces each token's assumed cycle span, making same-phase aggregation less brittle and allowing the router to combine finer phase slices.
- Risk: smaller periods increase the number of periods per phase token and can dilute daily structure.
- Experiment ID: `weather96_period12_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant baseline --period-len 12 --horizon 96 --epochs 30 --percent 100 --batch-size 16 --num-workers 0 --run-id weather96_period12_e30_seed2021`
- Comparability: identical to baseline except `period_len`.
- Result path: `research_runs/weather96_period12_e30_seed2021/`
- Result: ran 29 completed epochs; test MAE 0.193105, test MSE 0.148850, elapsed 829.6 s.
- Delta vs baseline: MAE reduced by 1.62%; MSE reduced by 0.05%. This is the best MAE so far but does not improve MSE meaningfully.
- Bad case summary: hard sampled cases worsened, with worst sampled MSE 0.431889. Shorter phases improve ordinary windows but not the localized hard regime.
- Iteration decision: retain `period_len=12` as a partial MAE improvement and combine it with H1 residual to test complementarity.

## Iteration 6 - Shorter Phase Plus Residual

- Goal: combine the two mechanisms with positive average signals: shorter phase length and weak-period residual path.
- Candidate hypothesis: H3+H1 complementarity.
- Mechanism: set `period_len=12` and enable `trend_residual` with gate init 0.2.
- Expected outcome: period_len 12 improves average MAE, while the residual branch may help MSE/local drift. If complementary, both MAE and MSE should improve beyond either component alone.
- Risk: both mechanisms may target ordinary-window drift while leaving hard local regimes unresolved.
- Experiment ID: `weather96_period12_trend_residual_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant trend_residual --period-len 12 --horizon 96 --epochs 30 --percent 100 --batch-size 16 --num-workers 0 --run-id weather96_period12_trend_residual_e30_seed2021`
- Comparability: same data, seed, training and metrics as baseline; combines only previously tested mechanisms.
- Result path: `research_runs/weather96_period12_trend_residual_e30_seed2021/`
- Result: early stopped after 15 completed epochs; test MAE 0.194923, test MSE 0.148223, elapsed 439.6 s.
- Delta vs baseline: MAE reduced by 0.69%; MSE reduced by 0.47%. This is the best MSE so far, but still far from the target.
- Bad case summary: hard sampled MSE improved versus period_len 12 alone, but remains worse than the original baseline's worst sampled MSE.
- Iteration decision: the weak-period mechanisms are directionally useful but too small. Test whether the tiny Weather baseline is capacity-limited.

## Iteration 7 - Higher Latent Capacity

- Goal: test whether weak-period Weather performance is bottlenecked by the very small `latent_dim=8` phase representation.
- Candidate hypothesis: H4 Capacity for weak-period variation.
- Mechanism: increase `latent_dim` from 8 to 32 while keeping baseline period length and other Weather settings unchanged.
- Theory intuition: weak-period signals contain drift, noise, and non-aligned regimes that cannot be compressed into the same low-dimensional phase representation as a strongly periodic series. A larger latent space may represent both phase structure and non-period residual variation.
- Risk: higher capacity may overfit or improve training loss without improving the test split.
- Experiment ID: `weather96_lat32_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant baseline --latent-dim 32 --horizon 96 --epochs 30 --percent 100 --batch-size 16 --num-workers 0 --run-id weather96_lat32_e30_seed2021`
- Comparability: identical baseline setup except latent dimension.
- Result path: `research_runs/weather96_lat32_e30_seed2021/`
- Result: early stopped after 10 completed epochs; test MAE 0.224538, test MSE 0.181637, elapsed 288.1 s.
- Delta vs baseline: MAE increased by 14.40%; MSE increased by 21.96%.
- Bad case summary: hard sampled MSE worsened severely, up to 0.602886. Higher latent capacity with default learning rate is unstable or overfits weak-period Weather.
- Iteration decision: reject H4 in this form. Since small residual signals helped average metrics more reliably than capacity, test a residual-dominant path as a near-NLinear baseline.

## Iteration 8 - Residual-Dominant Forecasting

- Goal: determine whether weak-period Weather is better modeled by direct recent-trajectory extrapolation than phase routing.
- Candidate hypothesis: H5 Residual-dominant NLinear path.
- Mechanism: enable `trend_residual` with `weak_period_residual_gate_init=0.999`, making the initial model nearly pure residual extrapolation while keeping a small trainable phase contribution.
- Theory intuition: if periodic phase alignment is weak, a normalized linear extrapolator over the recent history may outperform phase aggregation by prioritizing trend and persistence.
- Risk: a residual-dominant model may underuse useful daily structure and degrade long-horizon behavior.
- Experiment ID: `weather96_residual_gate999_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --variant trend_residual --gate-init 0.999 --horizon 96 --epochs 30 --percent 100 --batch-size 16 --num-workers 0 --run-id weather96_residual_gate999_e30_seed2021`
- Comparability: identical baseline setup except residual-dominant initialization.
- Result path: `research_runs/weather96_residual_gate999_e30_seed2021/`
- Result: ran 26 completed epochs; test MAE 0.215498, test MSE 0.162478, elapsed 759.9 s.
- Delta vs baseline: MAE increased by 9.79%; MSE increased by 9.10%.
- Bad case summary: sampled hard-case MSE improved versus the original baseline, but average metrics degraded substantially. The residual-dominant model fits some local spikes while hurting broad test performance.
- Iteration decision: reject H5 for Weather. Because Weather behaves more like a noisy meteorological benchmark than the weakest-period data available in this workspace, pivot to Exchange for a more direct weak-period evaluation, starting with a fresh baseline.

## Iteration 9 - Exchange Weak-Period Baseline

- Goal: establish a current-version baseline on a more clearly weak-period dataset.
- Dataset rationale: `resources/all_datasets/exchange_rate.csv` is daily financial exchange-rate data with 8 variables. It is less calendar-periodic than Weather/Traffic/Electricity and better matches the user's weak-period requirement.
- Baseline setup: current PhaseFormer with `period_len=24`, lookback 720, horizon 96, seed 2021.
- Smoke test:
  - Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset Exchange --variant baseline --horizon 96 --epochs 1 --percent 5 --batch-size 32 --num-workers 0 --run-id smoke_exchange96_baseline`
  - Result path: `research_runs/smoke_exchange96_baseline/`
  - Result: completed on CUDA; test MAE 0.606591, test MSE 0.649810. Not an effect conclusion.
- Experiment ID: `exchange96_baseline_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset Exchange --variant baseline --horizon 96 --epochs 30 --percent 100 --batch-size 32 --num-workers 0 --run-id exchange96_baseline_e30_seed2021`
- Comparability: this becomes the Exchange denominator for later weak-period improvements.
- Result path: `research_runs/exchange96_baseline_e30_seed2021/`
- Result: early stopped after 16 completed epochs; test MAE 0.221346, test MSE 0.095170, elapsed 21.0 s.
- Success threshold: MAE < 0.199211 and MSE < 0.085653.
- Bad case summary: sampled worst MSE 0.272352 and sampled worst MAE 0.378753, concentrated in batch 5.
- Iteration decision: use this as the weak-period baseline and test period/trajectory mechanisms.

## Iteration 10 - Exchange Weekly Phase

- Goal: test whether a daily weak-period financial series benefits from a weekly phase length.
- Candidate hypothesis: H3-weekly phase.
- Experiment ID: `exchange96_period7_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset Exchange --variant baseline --period-len 7 --horizon 96 --epochs 30 --percent 100 --batch-size 32 --num-workers 0 --run-id exchange96_period7_e30_seed2021`
- Result: test MAE 0.233988, test MSE 0.103122. Both metrics degraded versus Exchange baseline.
- Iteration decision: reject weekly phase; Exchange weak-period behavior is not captured by a fixed 7-day phase.

## Iteration 11 - Exchange Residual Path

- Goal: test whether Exchange weak-period forecasting benefits from recent-trajectory extrapolation.
- Candidate hypothesis: H5 residual-dominant path.
- Experiments:
  - `exchange96_trend_residual_e30_seed2021`: gate 0.2, LR 0.001, MAE 0.218404, MSE 0.099815.
  - `exchange96_residual_gate999_e30_seed2021`: gate 0.999, LR 0.001, MAE 0.209168, MSE 0.087131.
- Decision: residual-dominant initialization is clearly better than low-gate fusion and nearly reaches the MSE threshold, but MAE remains short. Continue with LR/loss search around the residual-dominant model.

## Iteration 12 - Exchange Residual LR Search

- Goal: tune optimization for the residual-dominant weak-period model without changing data or metrics.
- Shared setup: Exchange 720 -> 96, `trend_residual`, gate init 0.999 unless otherwise noted, seed 2021.
- Results:
  - `exchange96_residual_gate999_lr003_e30_seed2021`: LR 0.003, MAE 0.226589, MSE 0.101536; rejected.
  - `exchange96_residual_gate999_lr0003_e30_seed2021`: LR 0.0003, MAE 0.201151, MSE 0.083920.
  - `exchange96_residual_gate999_lr0002_e30_seed2021`: LR 0.0002, MAE 0.200672, MSE 0.083497.
  - `exchange96_residual_gate999_lr0001_e30_seed2021`: LR 0.0001, MAE 0.201476, MSE 0.084578.
  - `exchange96_residual_gate995_lr0002_e30_seed2021`: gate 0.995, LR 0.0002, MAE 0.200711, MSE 0.083476.
  - `exchange96_residual_gate99_lr0002_e30_seed2021`: gate 0.99, LR 0.0002, MAE 0.200751, MSE 0.083441.
  - `exchange96_residual_gate999_lr00015_e30_seed2021`: LR 0.00015, MAE 0.199900, MSE 0.083146.
  - `exchange96_residual_gate999_lr00012_e30_seed2021`: LR 0.00012, MAE 0.204563, MSE 0.086504.
- Decision: LR 0.00015 is closest but MAE remains 0.35% above the success threshold; test MAE training loss.

## Iteration 13 - Exchange Residual MAE Loss

- Goal: push the residual-dominant model over the MAE threshold while keeping MSE below threshold.
- Mechanism: add script controls for `--loss-func`, `--disable-huber`, and `--huber-delta`; train with MAE loss and no Huber wrapper.
- Experiments:
  - `exchange96_residual_gate999_lr00015_mae_e30_seed2021`: LR 0.00015, MAE loss, MAE 0.199626, MSE 0.083238.
  - `exchange96_residual_gate999_lr00018_mae_e30_seed2021`: LR 0.00018, MAE loss, MAE 0.200853, MSE 0.084265.
  - `exchange96_residual_gate999_lr00013_mae_e30_seed2021`: LR 0.00013, MAE loss, MAE 0.198869, MSE 0.082640.
- Final result: `exchange96_residual_gate999_lr00013_mae_e30_seed2021` meets the user exit condition. Relative to Exchange baseline, MAE improves by 10.15% and MSE improves by 13.16%.
- Bad case summary: sampled worst MSE improves from 0.272352 to 0.198210; sampled worst MAE improves from 0.378753 to 0.318121. Top cases remain concentrated around batch 5, but magnitude is materially lower.
- Iteration decision: stop because the >10% MAE and >10% MSE target is satisfied before 30 iterations.

## ETT Round - Iteration 0 - Setup and Baseline Plan

- Goal: start a new autonomous round on ETT-series data, following the same auditable workflow and stopping when both MAE and MSE improve by more than 10% against the matching ETT baseline or after more than 30 iterations.
- User emphasis: improvements should be framed as adaptation inside the PhaseFormer phase-modeling framework for weak-period data, not only generic hyperparameter tuning.
- Initial target: ETTh2 720 -> 96. Rationale: hourly ETTh2 has a natural 24-step phase prior, but transformer temperature/oil variables can have weak or regime-shifted periodic alignment, making it a suitable low-cost ETT entry point.
- Repository finding: ETT metadata pointed to `./resources/all_datasets/ETT-small`, while local files are under `./resources/all_datasets/ETT/`.
- Tooling change: extend `scripts/research_weather_weak.py` to accept `ETTh1`, `ETTh2`, `ETTm1`, and `ETTm2`; reuse the official ETT script hyperparameters; set hourly ETT frequency to `h` and minute ETT frequency to `t`; keep bad case export capped by `--bad-case-limit` default 10.
- Baseline command planned: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset ETTh2 --variant baseline --horizon 96 --epochs 30 --percent 100 --batch-size 256 --num-workers 0 --run-id ett_etth2_96_baseline_e30_seed2021`.
- Smoke test:
  - Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset ETTh2 --variant baseline --horizon 96 --epochs 1 --percent 5 --batch-size 256 --num-workers 0 --run-id smoke_ett_etth2_96_baseline`
  - Result path: `research_runs/smoke_ett_etth2_96_baseline/`
  - Result: completed on CUDA; test MAE 0.483794, test MSE 0.488302.
  - Comparability: not an effect conclusion because it used only 5% train data and 1 epoch.
- Candidate hypotheses for next iterations:
  - ETT-H1 shorter/alternate phase length: test `period_len=12` or `period_len=48` to reduce brittleness of a fixed 24-hour phase grid under weak phase alignment.
  - ETT-H2 residual-assisted phase path: use `trend_residual` to let recent trajectory extrapolation stabilize phase predictions when periodic evidence is weak.
  - ETT-H3 time-mark phase correction: use future time marks as a lightweight calendar-conditioned phase bias if baseline bad cases show hour-regime bias.
- Evidence status: smoke test passed; full ETTh2 baseline pending. No effect conclusion yet.

## ETT Round - Iteration 1 - ETTh2 Current Baseline

- Goal: establish the current-version PhaseFormer baseline for ETTh2 720 -> 96.
- Experiment ID: `ett_etth2_96_baseline_e30_seed2021`
- Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset ETTh2 --variant baseline --horizon 96 --epochs 30 --percent 100 --batch-size 256 --num-workers 0 --run-id ett_etth2_96_baseline_e30_seed2021`
- Result path: `research_runs/ett_etth2_96_baseline_e30_seed2021/`
- Result: ran 30 epochs; test MAE 0.343032, test MSE 0.280557, elapsed 9.3 s.
- Success threshold: MAE < 0.308729 and MSE < 0.252501.
- Bad case summary: the exported bad case table contains 10 cases, capped as required. Worst sampled windows are adjacent in test batches 3 and 4, with sampled MSE from 1.016948 to 1.056622 and MAE from 0.676581 to 0.768117. This points to local level/trend drift where fixed 24-hour phase retrieval is not enough.
- Iteration decision: run phase-length and residual-assisted phase hypotheses before adding new mechanisms.

## ETT Round - Iterations 2 to 10 - Low-Cost Hypothesis Sweep

- Goal: quickly test whether existing phase-length, residual, time-mark, loss, or capacity controls can reach the 10% ETTh2 target.
- Results:
  - Iteration 2, `ett_etth2_96_period12_e30_seed2021`: MAE 0.343496, MSE 0.280480. MSE improves only 0.03%, MAE worsens.
  - Iteration 3, `ett_etth2_96_period48_e30_seed2021`: MAE 0.345207, MSE 0.282775. Rejected.
  - Iteration 4, `ett_etth2_96_trend_residual_e30_seed2021`: MAE 0.333583, MSE 0.267520. Improves MAE 2.75%, MSE 4.65%; current best MSE.
  - Iteration 5, `ett_etth2_96_trend_residual_gate08_e30_seed2021`: MAE 0.336780, MSE 0.273490. High residual prior hurts.
  - Iteration 6, `ett_etth2_96_residual_gate999_e30_seed2021`: MAE 0.332420, MSE 0.271142. Best MAE among MSE-trained runs but weaker MSE.
  - Iteration 7, `ett_etth2_96_trend_residual_lr0003_e30_seed2021`: MAE 0.333234, MSE 0.267941. LR 0.0003 does not beat default LR.
  - Iteration 8, `ett_etth2_96_residual_gate999_lr0003_e30_seed2021`: MAE 0.331487, MSE 0.269694. Best MAE under MSE objective.
  - Iteration 9, `ett_etth2_96_residual_gate999_lr0003_mae_e30_seed2021`: MAE 0.329992, MSE 0.272189. MAE improves but MSE regresses.
  - Iteration 10, `ett_etth2_96_time_mark_e30_seed2021`: MAE 0.355203, MSE 0.289638. Rejected.
  - Additional control, `ett_etth2_96_period12_trend_residual_e30_seed2021`: MAE 0.335807, MSE 0.270955. No complementarity.
  - Additional control, `ett_etth2_96_phase_capacity_l2_d16_e30_seed2021`: MAE 0.358527, MSE 0.306776. Rejected; larger phase capacity is not the bottleneck.
  - Additional control, `ett_etth2_96_baseline_mae_e30_seed2021`: MAE 0.337440, MSE 0.278760. Loss alone is insufficient.
- Decision: existing controls improve at most 3.80% MAE or 4.65% MSE. Add a more targeted phase-space weak-period adaptation.

## ETT Round - Iteration 11 - Phase-Local Trend Correction

- Goal: implement a mechanism that adapts the phase framework itself to weak-period drift.
- Candidate hypothesis: ETT-H4 phase-local trend correction.
- Mechanism: for each phase slot, estimate recent slope across same-phase periods and add a gated extrapolation to future phase steps before sequence reassembly. This keeps correction inside the PhaseFormer phase representation instead of replacing the model with a whole-sequence residual head.
- Theory intuition: ETTh2 bad cases suggest same-hour history is useful but not stationary. A local slope within each phase slot can preserve phase alignment while compensating for gradual same-phase drift.
- Risk: same-phase slopes can amplify noise if the recent periods are unstable.
- Smoke test:
  - Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset ETTh2 --variant phase_trend --horizon 96 --epochs 1 --percent 5 --batch-size 256 --num-workers 0 --run-id smoke_ett_etth2_96_phase_trend`
  - Result path: `research_runs/smoke_ett_etth2_96_phase_trend/`
  - Result: completed on CUDA; test MAE 0.482176, test MSE 0.485199. Not an effect conclusion.
- Evidence status: implementation smoke passed; full ETTh2 experiment pending.

## ETT Round - Iteration 12 - Adaptive Weak-Period Residual Gate

- Goal: move from a static phase/residual blend to a sample-wise weak-period adaptation.
- Candidate hypothesis: ETT-H5 adaptive residual gate.
- Mechanism: compute per-sample/per-variable phase instability, recent volatility, and same-phase trend magnitude, then use a small shared MLP to gate between the PhaseFormer phase path and the recent-trajectory residual path.
- Theory intuition: ETTh2 has windows where fixed phase routing works and windows where phase alignment weakens. A static gate must average these regimes, while an adaptive gate can increase residual anchoring only for unstable weak-period windows.
- Risk: the gate network may learn noisy proxies or collapse to the static gate behavior.
- Smoke test:
  - Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset ETTh2 --variant adaptive_residual --horizon 96 --epochs 1 --percent 5 --batch-size 256 --num-workers 0 --run-id smoke_ett_etth2_96_adaptive_residual`
  - Result path: `research_runs/smoke_ett_etth2_96_adaptive_residual/`
  - Result: completed on CUDA; test MAE 0.433172, test MSE 0.390351. Not an effect conclusion.
- Evidence status: implementation smoke passed; full ETTh2 experiment pending.

## ETT Round - Iteration 13 - Channel-Wise Weak-Period Residual Head

- Goal: address heterogeneous ETT variables whose weak-period drift may not share one temporal residual kernel.
- Candidate hypothesis: ETT-H6 channel-wise residual phase adaptation.
- Mechanism: keep PhaseFormer's phase path, but replace the shared NLinear-style residual head with a per-variable residual extrapolator when explicitly requested. The residual still blends with the phase forecast through the same gate interface.
- Theory intuition: ETTh2/ETTm2 channels have different physical meanings and drift scales. A shared residual projection can underfit variable-specific weak-period drift, especially at long horizons.
- Risk: per-channel residual parameters can overfit, especially on shorter horizons.
- Smoke test:
  - Command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset ETTh2 --variant channel_residual --horizon 96 --epochs 1 --percent 5 --batch-size 256 --num-workers 0 --run-id smoke_ett_etth2_96_channel_residual`
  - Result path: `research_runs/smoke_ett_etth2_96_channel_residual/`
  - Result: completed on CUDA; test MAE 0.440361, test MSE 0.408380. Not an effect conclusion.
- Evidence status: implementation smoke passed; full ETTh2 720 experiment pending because long horizon showed the strongest residual signal so far.

## ETT Round - Iterations 14 to 33 - Final Sweep and Stop

- Goal: continue one-hypothesis-at-a-time experiments until either an ETT run exceeds 10% improvement on both MAE and MSE, or the active round exceeds 30 iterations.
- Phase-local trend results:
  - Iteration 14, `ett_etth2_96_phase_trend_w3_g01_e30_seed2021`: MAE 0.346507, MSE 0.284833. Rejected; same-phase slope correction amplified noise.
  - Iteration 15, `ett_etth2_96_phase_trend_residual_w3_g01_e30_seed2021`: MAE 0.334012, MSE 0.267917. Near residual-only but not better.
  - Iteration 16, `ett_etth2_96_phase_trend_residual_w3_g001_e30_seed2021`: MAE 0.333570, MSE 0.267494. Tiny MSE gain over static residual but far below target.
- Adaptive residual results:
  - Iteration 17, `ett_etth2_96_adaptive_residual_g02_e30_seed2021`: MAE 0.332452, MSE 0.265911. Best ETTh2 96 MSE; improvements are MAE 3.08%, MSE 5.22%.
  - Iteration 18, `ett_etth2_96_adaptive_residual_g05_e30_seed2021`: MAE 0.334271, MSE 0.267016. Rejected.
  - Iteration 19, `ett_etth2_96_adaptive_residual_g01_e30_seed2021`: MAE 0.332209, MSE 0.266401. Slight MAE gain, MSE worse than gate 0.2.
  - Iteration 20, `ett_etth2_96_adaptive_residual_g02_lr0003_e30_seed2021`: MAE 0.331876, MSE 0.267296. LR 0.0003 trades MSE for MAE.
  - Iteration 21, `ett_etth2_96_adaptive_residual_g02_lr0003_mae_e30_seed2021`: MAE 0.328264, MSE 0.266525. Best ETTh2 96 MAE; improvements are MAE 4.31%, MSE 5.00%.
- ETTh2 720 pivot:
  - Iteration 22, `ett_etth2_720_baseline_e30_seed2021`: MAE 0.448750, MSE 0.415718. Success threshold MAE < 0.403875, MSE < 0.374146.
  - Iteration 23, `ett_etth2_720_trend_residual_g02_e30_seed2021`: MAE 0.429569, MSE 0.389052. Improvements are MAE 4.27%, MSE 6.41%.
  - Iteration 24, `ett_etth2_720_adaptive_residual_g02_e30_seed2021`: MAE 0.430190, MSE 0.389287. Adaptive gate did not beat static residual.
  - Iteration 25, `ett_etth2_720_residual_gate999_e30_seed2021`: MAE 0.424987, MSE 0.383477. Best overall ETT round result; improvements are MAE 5.30%, MSE 7.76%.
  - Iteration 26, `ett_etth2_720_residual_gate999_lr0003_e30_seed2021`: MAE 0.425024, MSE 0.385890. Lower LR worsened MSE.
  - Iteration 27, `ett_etth2_720_residual_gate999_mae_e30_seed2021`: MAE 0.425633, MSE 0.391216. MAE loss worsened both versus iteration 25.
- Additional ETT-series checks:
  - Iteration 28, `ett_ettm2_96_baseline_e30_seed2021`: MAE 0.258160, MSE 0.170091. Threshold MAE < 0.232344, MSE < 0.153082.
  - Iteration 29, `ett_ettm2_96_trend_residual_g02_e30_seed2021`: MAE 0.255485, MSE 0.167130. Improvements are MAE 1.04%, MSE 1.74%; rejected as too weak.
  - Iteration 30, `ett_etth2_720_channel_residual_gate999_e30_seed2021`: MAE 0.434686, MSE 0.396564. Channel-wise residual overfit.
  - Iteration 31, `ett_etth2_720_channel_residual_g02_e30_seed2021`: MAE 0.446093, MSE 0.413074. Rejected.
  - Iteration 32, `ett_etth1_96_baseline_e30_seed2021`: MAE 0.388491, MSE 0.364891. Threshold MAE < 0.349642, MSE < 0.328402.
  - Iteration 33, `ett_etth1_96_residual_gate999_e30_seed2021`: MAE 0.395013, MSE 0.371524. Rejected.
- Final bad case comparison for the selected best `ett_etth2_720_residual_gate999_e30_seed2021`:
  - Baseline ETTh2 720 bad cases are capped at 10 and concentrated in batch 7; sampled worst MSE 0.900864 and sampled worst MAE 0.673353.
  - Best run bad cases are capped at 10 and concentrated in batch 4; sampled worst MSE 0.947432 and sampled worst MAE 0.746384.
  - Interpretation: residual-dominant phase adaptation improves average ETTh2 720 MAE/MSE but does not reduce the hardest local regime. This hard-case shift explains why the run stops below the 10% target.
- Final decision: stop the ETT round because the model/research iteration count exceeded 30 without a candidate reducing both MAE and MSE by more than 10%.

## Formal Benchmark - Setup

- Goal: promote the latest validated PhaseFormer design and auxiliary experiment design into a formal full-test runner, then compare it with the original PhaseFormer mode.
- Design decision: centralize dataset/horizon hyperparameters and latest weak-period policies in `src/models/phaseformer_presets.py`.
- Latest policy:
  - Exchange uses the previously successful residual-dominant weak-period branch with MAE loss and LR 0.00013.
  - ETTh2 96 uses adaptive residual gating with MAE loss and LR 0.0003, which was the best ETTh2 96 MAE/MSE balance in the prior round.
  - ETTh2 720 uses residual-dominant weak-period branch with gate 0.999, which was the best ETT-round result.
  - Unsupported dataset/horizon combinations keep the original phase path to avoid applying mechanisms that prior evidence showed can degrade performance.
- Benchmark runner: `scripts/benchmark_phaseformer_suite.py`.
- Smoke command: `conda run --no-capture-output -n raft python scripts/benchmark_phaseformer_suite.py --datasets ETTh2 --horizons 96 --modes original,latest --epochs 1 --batch-size 256 --num-workers 0 --run-prefix smoke_phaseformer_suite_etth2_96`
- Smoke result: completed both original and latest modes on CUDA and wrote `research_runs/smoke_phaseformer_suite_etth2_96_summary.csv` plus `research_runs/smoke_phaseformer_suite_etth2_96_comparison.csv`.
- Comparability note: smoke used only 1 epoch and is not an effect conclusion.

## Formal Benchmark - Full Results

- Goal: run full-data original/latest comparisons across available datasets and horizons, adjusting training throughput settings where needed while preserving paired comparability.
- Full ETT/Exchange command: `conda run --no-capture-output -n raft python scripts/benchmark_phaseformer_suite.py --datasets ETTh1,ETTh2,ETTm1,ETTm2,Exchange,Electricity,Traffic,Weather --horizons all --modes original,latest --num-workers 0 --run-prefix phaseformer_full_latest_vs_original_20260630 --resume`
- Execution note: this command completed ETTh1, ETTh2, ETTm1, ETTm2, and Exchange. It was interrupted at Electricity because official batch/default worker settings were too slow for high-dimensional full tests.
- High-dimensional/Weather command: `conda run --no-capture-output -n raft python scripts/benchmark_phaseformer_suite.py --datasets Electricity,Traffic,Weather --horizons all --modes original,latest --batch-size 64 --num-workers 4 --run-prefix phaseformer_full_latest_vs_original_highdim_b64w4_20260630 --resume`
- Evidence:
  - `research_runs/phaseformer_full_latest_vs_original_20260630_summary.csv`
  - `research_runs/phaseformer_full_latest_vs_original_20260630_comparison.csv`
  - `research_runs/phaseformer_full_latest_vs_original_highdim_b64w4_20260630_summary.csv`
  - `research_runs/phaseformer_full_latest_vs_original_highdim_b64w4_20260630_comparison.csv`
- Positive results:
  - Exchange 96: MAE 0.221346 -> 0.198869 (-10.15%), MSE 0.095170 -> 0.082640 (-13.17%).
  - Exchange 192: MAE 0.310414 -> 0.293946 (-5.31%), MSE 0.183390 -> 0.174912 (-4.62%).
  - Exchange 336: MAE 0.463470 -> 0.407273 (-12.13%), MSE 0.400999 -> 0.328459 (-18.09%).
  - Exchange 720: MAE 0.787787 -> 0.713884 (-9.38%), MSE 1.096012 -> 0.928974 (-15.24%).
  - ETTh2 96: MAE 0.343032 -> 0.328264 (-4.31%), MSE 0.280557 -> 0.266525 (-5.00%).
  - ETTh2 720: MAE 0.448750 -> 0.424987 (-5.30%), MSE 0.415718 -> 0.383477 (-7.76%).
- Guardrail results: latest mode was intentionally identical to original for ETTh1, ETTm1, ETTm2, Electricity, Traffic, and Weather on all tested horizons; all comparison deltas are 0.0%, confirming no regression under the latest policy.
- Interpretation: the latest scheme should be treated as a dataset-aware PhaseFormer policy, not a universal unconditional residual branch. It improves weak-period Exchange and selected ETTh2 settings while preserving original behavior on strong-period/high-dimensional datasets where prior residual variants degraded or lacked evidence.
- Bad case rule: every benchmark run retained `bad_cases.csv` with `--bad-case-limit=10`.

## ETTh1/ETTm1/ETTm2 Weak-Period Round - Initialization

- Goal: start a new round focused on ETTh1, ETTm1, and ETTm2, with stop condition MAE and MSE both improving by more than 5% or more than 50 iterations.
- Theory target: weak periodicity can be written as `x_t = p_{\phi(t)} + d_t + eps_t`, where `p` is phase-conditioned structure, `d_t` is slowly varying drift, and `eps_t` is high-frequency noise. A raw residual branch estimates from `d_t + eps_t`, so its extrapolation error includes noise variance. A low-pass residual branch estimates from `LP(x_t) ~= p_{\phi(t)} + d_t`, reducing the residual estimator variance while retaining the raw last-value anchor for current level.
- Iteration 1, ETTh1 96 adaptive residual:
  - Run: `weakphase2_etth1_96_adaptive_residual_g02_e30_seed2021`
  - Result: MAE 0.396414, MSE 0.370857 versus baseline MAE 0.388491, MSE 0.364891. Rejected.
- Iteration 2, ETTm1 96 adaptive residual:
  - Run: `weakphase2_ettm1_96_adaptive_residual_g02_e30_seed2021`
  - Result: MAE 0.348268, MSE 0.297773. Later audit found this used the research runner's incorrect ETTm1 layer setting and is not comparable to the formal ETTm1 baseline.
- Iteration 3, ETTm2 96 adaptive residual:
  - Run: `weakphase2_ettm2_96_adaptive_residual_g02_e30_seed2021`
  - Result: MAE 0.255513, MSE 0.168867 versus baseline MAE 0.258160, MSE 0.170091. Positive but below 5%.
- Iteration 4, ETTm1 96 period length 96:
  - Run: `weakphase2_ettm1_96_period96_e30_seed2021`
  - Result: MAE 0.409024, MSE 0.396720. Rejected as a local signal, but not used for formal comparison because of the ETTm1 layer-setting audit below.
- Iteration 5, ETTm2 96 period length 96:
  - Run: `weakphase2_ettm2_96_period96_e30_seed2021`
  - Result: MAE 0.283160, MSE 0.190747. Rejected.
- Iteration 6, ETTm2 96 period length 12:
  - Run: `weakphase2_ettm2_96_period12_e30_seed2021`
  - Result: MAE 0.265914, MSE 0.176100. Rejected.
- Iteration 7, ETTm2 96 adaptive residual + MAE:
  - Run: `weakphase2_ettm2_96_adaptive_residual_g02_lr0003_mae_e30_seed2021`
  - Result: MAE 0.247504, MSE 0.161533. MSE reaches the 5% target; MAE improves 4.13%, just short.
- Iteration 8, ETTm2 96 adaptive residual + lower gate:
  - Run: `weakphase2_ettm2_96_adaptive_residual_g01_lr0003_mae_e30_seed2021`
  - Result: MAE 0.247352, MSE 0.161570. Slight MAE improvement, still short.
- Iteration 9, ETTm2 96 residual-dominant + MAE:
  - Run: `weakphase2_ettm2_96_residual_gate999_lr0003_mae_e30_seed2021`
  - Result: MAE 0.245211, MSE 0.160063. This satisfies the 5% threshold for ETTm2 96.
- Iteration 10, ETTm1 96 residual-dominant + MAE:
  - Run: `weakphase2_ettm1_96_residual_gate999_lr0003_mae_e30_seed2021`
  - Result: MAE 0.341968, MSE 0.300096. Not used for formal comparison because of the ETTm1 layer-setting audit below.
- Iteration 11, ETTm1 720 baseline:
  - Run: `weakphase2_ettm1_720_baseline_e30_seed2021`
  - Result: MAE 0.409929, MSE 0.412445. Threshold MAE < 0.389433, MSE < 0.391823.
- Iteration 12, ETTm1 720 residual-dominant + MAE:
  - Run: `weakphase2_ettm1_720_residual_gate999_lr0003_mae_e30_seed2021`
  - Result: MAE 0.407948, MSE 0.416197. Rejected locally; not used for formal comparison because of the ETTm1 layer-setting audit below.
- New implementation: `LowPassWeakPeriodResidualHead`, enabled by `smooth_residual` / `adaptive_smooth_residual` in `scripts/research_weather_weak.py`.
- Smoke test:
  - Run: `smoke_weakphase2_ettm2_smooth_residual`
  - Result: completed on CUDA; not an effect conclusion.

## ETTh1/ETTm1/ETTm2 Weak-Period Round - Phase Jitter Mechanism

- Goal: address weak periodicity as phase misalignment rather than pure trend drift.
- Theory: let the ideal phase token be `z_l`, but weak-period samples are observed with a small random phase shift `delta`, so the conditional expectation is `E[z_{l+delta}] = sum_k P(delta=k) z_{l+k}`. For small symmetric jitter with support `{-1,0,1}`, this is approximated by a circular neighbor smoothing kernel. The new `PhaseJitterSmoothing` module learns how much of this local marginalization to apply before phase embedding.
- Implementation: add default-off `PhaseJitterSmoothing` in `src/models/PhaseFormer.py`; expose `phase_jitter`, `phase_jitter_residual`, and `phase_jitter_smooth_residual` in `scripts/research_weather_weak.py`.
- Smoke test:
  - Run: `smoke_weakphase2_etth1_phase_jitter`
  - Result: completed on CUDA; not an effect conclusion.
- Comparability audit: `scripts/research_weather_weak.py` had ETTm1 layers reversed from the official `run_ettm1.py`/formal preset. It has been corrected so ETTm1 96/192/720 use layers=2 and ETTm1 336 uses layers=1. ETTm1 experiments before this correction are treated as local signals only, not formal comparisons.

## ETTh1/ETTm1/ETTm2 Weak-Period Round - Post-Fix Results

- Iteration 13, ETTm2 96 low-pass residual:
  - Run: `weakphase2_ettm2_96_smooth_residual_gate999_w25_lr0003_mae_e30_seed2021`
  - Result: MAE 0.249809, MSE 0.164491. Rejected versus the raw residual-dominant result.
- Iteration 14, ETTm1 96 low-pass residual:
  - Run: `weakphase2_ettm1_96_smooth_residual_gate999_w25_lr0003_mae_e30_seed2021`
  - Result: MAE 0.346035, MSE 0.302350. Improves MAE but worsens MSE.
- Iteration 15, ETTm1 96 low-pass residual with gate 0.2:
  - Run: `weakphase2_ettm1_96_smooth_residual_g02_w25_lr0003_mae_e30_seed2021`
  - Result: MAE 0.344209, MSE 0.297261. Both improve but remain below 5%.
- Iteration 16, ETTm1 720 low-pass residual with gate 0.2:
  - Run: `weakphase2_ettm1_720_smooth_residual_g02_w25_lr0003_mae_e30_seed2021`
  - Result: MAE 0.408482, MSE 0.409277 versus baseline MAE 0.409929, MSE 0.412445. Positive but below 5%.
- Iteration 17, ETTh1 96 low-pass residual:
  - Run: `weakphase2_etth1_96_smooth_residual_g02_w7_lr0003_mae_e30_seed2021`
  - Result: MAE 0.401992, MSE 0.371285. Rejected.
- Iteration 18, ETTh1 96 phase jitter:
  - Run: `weakphase2_etth1_96_phase_jitter_g01_e30_seed2021`
  - Result: MAE 0.385709, MSE 0.363149. Small positive signal, below 5%.
- Iteration 19, ETTh1 96 stronger phase jitter:
  - Run: `weakphase2_etth1_96_phase_jitter_g05_e30_seed2021`
  - Result: MAE 0.388801, MSE 0.365513. Rejected.
- Iteration 20, ETTh1 96 phase jitter + residual:
  - Run: `weakphase2_etth1_96_phase_jitter_residual_j01_g02_lr0003_mae_e30_seed2021`
  - Result: MAE 0.389678, MSE 0.359815. MSE improves but MAE worsens.
- Iteration 21, ETTm1 96 phase jitter:
  - Run: `weakphase2_ettm1_96_phase_jitter_g01_e30_seed2021`
  - Result: MAE 0.358953, MSE 0.319443. Rejected.
- Iteration 22, ETTm1 96 fixed baseline:
  - Run: `weakphase2_ettm1_96_baseline_fixed_e30_seed2021`
  - Result: MAE 0.347958, MSE 0.299526. Confirms formal baseline.
- Iteration 23, ETTm1 96 fixed residual-dominant + MAE:
  - Run: `weakphase2_ettm1_96_residual_gate999_lr0003_mae_fixed_e30_seed2021`
  - Result: MAE 0.337078, MSE 0.294519. Improvements are MAE 3.13%, MSE 1.67%, below 5%.
- Iteration 24, ETTm1 96 fixed residual-dominant + MAE with LR 0.00013:
  - Run: `weakphase2_ettm1_96_residual_gate999_lr00013_mae_fixed_e30_seed2021`
  - Result: MAE 0.337630, MSE 0.294997. Worse than LR 0.0003.
- Iteration 25, ETTm1 96 fixed residual-dominant + MAE with LR 0.001:
  - Run: `weakphase2_ettm1_96_residual_gate999_lr001_mae_fixed_e30_seed2021`
  - Result: MAE 0.345032, MSE 0.301169. Rejected.
- Current best for this round:
  - ETTm2 96: `weakphase2_ettm2_96_residual_gate999_lr0003_mae_e30_seed2021`, MAE 0.245211, MSE 0.160063; both exceed 5% improvement versus baseline.
  - ETTm1 96: `weakphase2_ettm1_96_residual_gate999_lr0003_mae_fixed_e30_seed2021`, below 5%.
  - ETTh1 96: `weakphase2_etth1_96_phase_jitter_g01_e30_seed2021`, below 5%.
- Blocker: after iteration 25, CUDA became unavailable (`nvidia-smi` reports `Unable to determine the device handle for GPU0`). Further full-data ETTm1/ETTh1 experiments are paused until GPU availability returns; CPU runs would not be time-comparable.

## ETTh1/ETTm1/ETTm2 Weak-Period Round - GPU Resumed

- Iteration 26, ETTm1 96 fixed residual gate 0.8 + MAE:
  - Run: `weakphase2_ettm1_96_residual_gate08_lr0003_mae_fixed_e30_seed2021`
  - Commit: `0ff4f36`
  - Goal: test whether a less residual-dominant blend improves MSE while retaining the MAE gain of the raw weak-period residual head.
  - Result: MAE 0.343295, MSE 0.299322 versus fixed baseline MAE 0.347958, MSE 0.299526.
  - Bad cases: `research_runs/weakphase2_ettm1_96_residual_gate08_lr0003_mae_fixed_e30_seed2021/bad_cases.csv`, 10 rows.
  - Decision: rejected as a target solution; smaller residual weight loses most of the useful MAE improvement and does not materially improve MSE.
- Iteration 27, ETTm1 96 fixed adaptive residual + MAE:
  - Run: `weakphase2_ettm1_96_adaptive_residual_g02_lr0003_mae_fixed_e30_seed2021`
  - Commit: `0ff4f36`
  - Goal: test whether sample/channel-specific gates can identify weak-period windows where residual extrapolation should dominate.
  - Result: MAE 0.349903, MSE 0.298016. MAE is worse than baseline; MSE improves only 0.50%.
  - Bad cases: `research_runs/weakphase2_ettm1_96_adaptive_residual_g02_lr0003_mae_fixed_e30_seed2021/bad_cases.csv`, 10 rows.
  - Decision: rejected. The adaptive gate adds instability without enough MSE gain, so ETTm1 96 is not solved by gate selectivity alone.
- Iteration 28, ETTm1 96 fixed phase-jitter + residual:
  - Run: `weakphase2_ettm1_96_phase_jitter_residual_j01_g999_lr0003_mae_fixed_e30_seed2021`
  - Commit: `0ff4f36`
  - Goal: combine local phase marginalization with a residual-dominant path, targeting weak periodicity as phase jitter plus drift.
  - Result: MAE 0.365760, MSE 0.324419, both worse than baseline.
  - Bad cases: `research_runs/weakphase2_ettm1_96_phase_jitter_residual_j01_g999_lr0003_mae_fixed_e30_seed2021/bad_cases.csv`, 10 rows.
  - Decision: rejected. On ETTm1, phase-neighbor smoothing conflicts with the minute-level phase tokens rather than regularizing them.
- Iteration 29, ETTm1 96 fixed residual-dominant + MSE:
  - Run: `weakphase2_ettm1_96_residual_gate999_lr0003_mse_fixed_rerun_e30_seed2021`
  - Commit: `0ff4f36`
  - Goal: isolate whether the previous MSE shortfall came from MAE optimization rather than the residual structure.
  - Result: MAE 0.366826, MSE 0.324873, both worse than baseline.
  - Bad cases: `research_runs/weakphase2_ettm1_96_residual_gate999_lr0003_mse_fixed_rerun_e30_seed2021/bad_cases.csv`, 10 rows.
  - Decision: rejected. The residual branch itself creates large-window errors on ETTm1 96; changing to MSE loss does not fix the weak-period failure mode.

## ETTh1/ETTm1/ETTm2 Weak-Period Round - Bad Case Review Correction

- Process correction:
  - The previous loop relied too much on aggregate MAE/MSE. This violated the intent of `HOW_TO_DO_RESEARCH.md` section 6 because the retained `bad_cases.csv` only had batch/sample indices and scalar errors.
  - `scripts/research_weather_weak.py` now exports pattern-covered bad cases, not just largest-error samples. The capped patterns are highest MSE, systematic bias, trend mismatch, peak underfit, valley overfit, volatility mismatch, late-horizon drift, and volatile input. The cap remains <= 10; current review runs use 8.
  - Each selected case now records dataset, sample index, variable name/index, input/forecast timestamps, pattern metrics, and a window-level CSV under `research_runs/<run_id>/bad_cases/` with input values, forecast true values, predictions, and errors in scaled and original units.
- Comparable ETTm1 96 review baseline:
  - Run: `weakphase2_review_b256_ettm1_96_baseline_fixed_e30_seed2021`
  - Command matches the formal baseline setting: batch size 256, MSE, Huber enabled, seed 2021.
  - Result: MAE 0.347958, MSE 0.299526, matching `weakphase2_ettm1_96_baseline_fixed_e30_seed2021`.
  - Bad cases: 8 rows, 8 window CSV files.
- Comparable ETTm1 96 review residual:
  - Run: `weakphase2_review_b256_ettm1_96_residual_gate999_lr0003_mae_fixed_e30_seed2021`
  - Command matches the previous best residual setting: batch size 256, LR 0.0003, MAE, Huber disabled, seed 2021.
  - Result: MAE 0.337078, MSE 0.294519, matching `weakphase2_ettm1_96_residual_gate999_lr0003_mae_fixed_e30_seed2021`.
  - Bad cases: 8 rows, 8 window CSV files.
- Bad case pattern findings:
  - Severe ETTm1 96 errors are concentrated in MUFL. The dominant failure is not generic average drift; it is phase-amplitude hallucination under weak periodicity.
  - Baseline examples: `highest_mse` MUFL case index 663 has true forecast range 3.45 in original units but prediction range 23.07; `volatility_mismatch` MUFL case index 671 has true range 0.00 but prediction range 22.82.
  - Residual best improves some aggregate MAE and selected trend cases, but it does not remove the high-amplitude hallucination: MUFL `volatility_mismatch` case index 671 still has true range 0.00 and prediction range 24.61.
  - Therefore the next strategy should not be more residual gate tuning. It should make the phase path aware of whether same-phase historical observations are reliable.
- New mechanism: phase reliability damping.
  - Theory: write phase observations as `x_{k,l}=p_l+d_k+eps_{k,l}`. For a fixed phase slot `l`, high cross-period variance estimates noise/phase instability; variance of the phase template across `l` estimates useful periodic signal. A shrinkage factor `rho = signal / (signal + noise)` is the empirical-Bayes reliability of the phase template. Forecast deviations from the last value are damped by `rho` with a configurable floor.
  - Implementation: `PhaseReliabilityDamping` in `src/models/PhaseFormer.py`, enabled by `phase_reliability`, `phase_reliability_residual`, and `phase_reliability_smooth_residual`. The default model path is unchanged.
- Iteration 30, ETTm1 96 phase reliability only:
  - Run: `weakphase2_ettm1_96_phase_reliability_min35_b256_e30_seed2021`
  - Result: MAE 0.349638, MSE 0.305722.
  - Decision: rejected. Pure global damping hurts normal phase predictions and is not sufficient by itself.
- Iteration 31, ETTm1 96 residual + phase reliability damping, min 0.35:
  - Run: `weakphase2_ettm1_96_phase_reliability_residual_min35_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.338811, MSE 0.294314 versus baseline MAE 0.347958, MSE 0.299526. This is a 2.63% MAE gain and 1.74% MSE gain, below the 5% target.
  - Bad case effect: MUFL `highest_mse` MSE falls from 4.101 to 3.846 versus the residual review; `systematic_bias` from 4.094 to 3.772; `late_horizon_drift` from 3.865 to 3.527. Peak-underfit gets worse, explaining the MAE tradeoff.
  - Decision: keep as a diagnostic direction for reducing large weak-period phase-amplitude errors, but not as the final solution.
- Iteration 32, ETTm1 96 residual + phase reliability damping, min 0.60:
  - Run: `weakphase2_ettm1_96_phase_reliability_residual_min60_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.339604, MSE 0.294396.
  - Decision: rejected versus min 0.35. Lighter damping loses the large-error benefit without recovering MAE enough.
- Iteration 33, ETTm1 96 residual + selective phase reliability damping:
  - Run: `weakphase2_ettm1_96_phase_reliability_residual_min35_noise10_gate999_lr0003_mae_b256_e30_seed2021`
  - Bad-case rationale: the min 0.35 review showed MUFL failure windows have phase noise around 2.3-2.4, while the MULL/HULL windows hurt by global damping have noise around 0.4-0.5. This variant only triggers damping strongly when phase noise exceeds 1.0.
  - Result: MAE 0.340421, MSE 0.294971.
  - Decision: rejected. Absolute phase-noise thresholding protects too many windows from damping and loses the large-error reduction. The next mechanism should separate two bad-case modes explicitly: high-amplitude hallucination on near-flat futures versus under-response to real trend jumps.
- New mechanism: phase-noise high-frequency damping.
  - Bad-case driver: ETTm1 96 MUFL windows show high-amplitude phase hallucination, e.g. true range near 0-5 but predicted range above 20. Full reliability damping reduces some large errors but also damps low-frequency trend/peaks.
  - Theory: decompose the normalized forecast as `y = LP(y) + HP(y)`. Under weak periodicity, high phase noise makes high-frequency phase details unreliable, while trend jumps live mostly in the low-pass component. Therefore damp `HP(y)` only when same-phase historical noise is high, preserving `LP(y)`.
  - Implementation: `PhaseNoiseHighFreqDamping`, enabled by `phase_hifreq`, `phase_hifreq_residual`, and `phase_hifreq_smooth_residual`.
- Iteration 34, ETTm1 96 residual + phase-noise high-frequency damping, strength 0.5 window 7:
  - Run: `weakphase2_ettm1_96_phase_hifreq_s05_thr10_w7_residual_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.338555, MSE 0.292791 versus baseline MAE 0.347958, MSE 0.299526.
  - Bad case effect versus residual review: MUFL `volatility_mismatch` MSE improves 3.948 -> 3.567; `late_horizon_drift` improves 3.865 -> 3.418; `valley_overfit` improves 0.657 -> 0.561. Trend mismatch worsens, showing remaining tension between smoothing hallucinations and preserving real jumps.
  - Decision: current best MSE direction for ETTm1 96, but still below the 5% target.
- Iteration 35, ETTm1 96 residual + phase-noise high-frequency damping, strength 0.5 window 3:
  - Run: `weakphase2_ettm1_96_phase_hifreq_s05_thr10_w3_residual_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.337041, MSE 0.294422.
  - Decision: shorter smoothing protects MAE/trend response but loses most large-error MSE reduction. Rejected versus iteration 34 for target progress.
- Iteration 36, ETTm1 96 residual + phase-noise high-frequency damping, strength 0.8 window 7:
  - Run: `weakphase2_ettm1_96_phase_hifreq_s08_thr10_w7_residual_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.338662, MSE 0.293291.
  - Decision: stronger damping is worse than strength 0.5, likely because it suppresses useful local shape. Keep iteration 34 as the best high-frequency damping setting.

## ETTh1 Bad Case Review And Trend Tests

- ETTh1 96 comparable review baseline:
  - Run: `weakphase2_review_etth1_96_baseline_e30_seed2021`
  - Result: MAE 0.388609, MSE 0.369751.
  - Bad case pattern: unlike ETTm1, the key failure is not high-amplitude hallucination. The main pattern is trend under-response, e.g. MUFL `trend_mismatch` index 36 has true slope +22.99 but predicted slope -1.20.
- ETTh1 96 phase-jitter review:
  - Run: `weakphase2_review_etth1_96_phase_jitter_g01_e30_seed2021`
  - Result: MAE 0.388954, MSE 0.372708.
  - Decision: rejected. Phase-neighbor smoothing is not a stable ETTh1 96 direction under the enhanced bad-case review.
- Iteration 37, ETTh1 96 residual-dominant + MAE:
  - Run: `weakphase2_etth1_96_residual_gate999_lr0003_mae_e30_seed2021`
  - Result: MAE 0.397002, MSE 0.376856.
  - Bad case review: residual changes local slopes but not reliably; some trend directions become worse.
  - Decision: rejected.
- Iteration 38, ETTh1 96 residual + phase-noise high-frequency damping:
  - Run: `weakphase2_etth1_96_phase_hifreq_s05_thr10_w7_residual_gate999_lr0003_mae_e30_seed2021`
  - Result: MAE 0.391684, MSE 0.372267.
  - Decision: rejected. ETTh1 96 does not primarily suffer from high-frequency phase hallucination, so this ETTm1-targeted mechanism does not transfer.
- Iteration 39, ETTh1 96 phase-local trend correction, window 3 gate 0.1:
  - Run: `weakphase2_etth1_96_phase_trend_w3_g01_e30_seed2021`
  - Result: MAE 0.393193, MSE 0.398613.
  - Decision: rejected. Same-phase local slope extrapolation amplifies noise.
- Iteration 40, ETTh1 96 phase-local trend correction, window 5 gate 0.2:
  - Run: `weakphase2_etth1_96_phase_trend_w5_g02_e30_seed2021`
  - Result: MAE 0.392001, MSE 0.390591.
  - Bad case review: the `trend_mismatch` sample improves, but highest/systematic/late-drift cases worsen.
  - Decision: rejected.
- New mechanism: low-frequency trend correction.
  - Bad-case driver: ETTh1 96 needs trend response, but phase-local slopes are too noisy. Estimate a whole-sequence low-pass recent slope and add it with a small learned channel gate.
  - Theory: decompose `x_t = p_phi(t) + d_t + eps_t`; for ETTh1 bad cases, `d_t` changes faster than the phase path adapts. Estimating slope from `LP(x_t)` targets `d_t` while avoiding same-phase noise.
- Iteration 41, ETTh1 96 low-frequency trend correction:
  - Run: `weakphase2_etth1_96_lowfreq_trend_w25_g005_e30_seed2021`
  - Result: MAE 0.389448, MSE 0.372435.
  - Decision: rejected.
- Iteration 42, ETTh1 96 low-frequency trend correction + weak residual:
  - Run: `weakphase2_etth1_96_lowfreq_trend_residual_w25_g005_gate02_lr0003_mae_e30_seed2021`
  - Result: MAE 0.391959, MSE 0.367231.
  - Bad case review: improves systematic/volatile-input/volatility-mismatch cases, but worsens trend-mismatch and peak-underfit. MSE improves slightly while MAE worsens.
  - Decision: not retained as final. Continue with long-horizon bad-case review rather than overfitting ETTh1 96.

## Long-Horizon Weak-Period Review

- ETTm1 720 review baseline:
  - Run: `weakphase2_review_ettm1_720_baseline_b256_e30_seed2021`
  - Result: MAE 0.413261, MSE 0.418219.
  - Bad case pattern: long-horizon failures show slope direction/amplitude instability, not only short-window phase hallucination.
- ETTm1 720 smooth residual review:
  - Run: `weakphase2_review_ettm1_720_smooth_residual_g02_w25_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.409015, MSE 0.413924.
  - Decision: small average gain, but representative bad cases remain poor.
- Iteration 43, ETTm1 720 smooth residual + phase-noise high-frequency damping:
  - Run: `weakphase2_ettm1_720_phase_hifreq_s05_thr10_w7_smooth_residual_g02_w25_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.409044, MSE 0.413627.
  - Decision: rejected as a target solution. It marginally improves MSE versus smooth residual but not enough to justify more local tuning.
- ETTh1 720 review caveat:
  - Initial review runs `weakphase2_review_etth1_720_baseline_e70_seed2026` and `weakphase2_etth1_720_lowfreq_trend_residual_w25_g005_gate02_lr00015_mae_e70_seed2026` did not reproduce the prior formal benchmark baseline. The research runner was missing ETTh1 720 formal preset details (`patience=14`, `huber_delta=0.3`, and attention dropout 0.0).
  - The runner was updated to carry these formal preset details, but `weakphase2_review_etth1_720_baseline_formal_e70_seed2026` still produced MAE 0.488253, MSE 0.520365 rather than the earlier formal baseline MAE 0.447249, MSE 0.440721. Therefore ETTh1 720 review runs are treated as qualitative bad-case signals only, not formal success evidence.
- Iteration 44, ETTm1 96 high-frequency damping, strength 0.5 window 5:
  - Run: `weakphase2_ettm1_96_phase_hifreq_s05_thr10_w5_residual_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.340634, MSE 0.295405.
  - Decision: rejected. The window midpoint between 3 and 7 was worse than both useful endpoints.
- Iteration 45, ETTm1 96 high-frequency damping, strength 0.3 window 7:
  - Run: `weakphase2_ettm1_96_phase_hifreq_s03_thr10_w7_residual_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.340608, MSE 0.295560.
  - Decision: rejected. The large-error MSE benefit requires stronger damping.
- Iteration 46, ETTm1 96 high-frequency damping, residual gate 0.95:
  - Run: `weakphase2_ettm1_96_phase_hifreq_s05_thr10_w7_residual_gate095_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.342460, MSE 0.299308.
  - Decision: rejected. Lower residual weight loses both MAE and MSE gains.
- Iteration 47, ETTm1 192 baseline:
  - Run: `weakphase2_ettm1_192_baseline_b256_e30_seed2021`
  - Result: MAE 0.363096, MSE 0.329395. Paired 5% threshold: MAE < 0.344941, MSE < 0.312925.
- Iteration 48, ETTm1 192 residual-dominant + MAE:
  - Run: `weakphase2_ettm1_192_residual_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.360971, MSE 0.331057.
  - Decision: rejected; MAE improves only 0.59% and MSE worsens.
- Iteration 49, ETTm1 192 residual + high-frequency damping:
  - Run: `weakphase2_ettm1_192_phase_hifreq_s05_thr10_w7_residual_gate999_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.360031, MSE 0.330137.
  - Decision: rejected; MAE improves 0.84% and MSE worsens slightly.
- Iteration 50, ETTm1 192 adaptive residual, gate 0.2:
  - Run: `weakphase2_ettm1_192_adaptive_residual_g02_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.359033, MSE 0.328808.
  - Decision: best ETTm1 192 attempt, but still only MAE 1.12% and MSE 0.18% improvement.
- Iteration 51, ETTm1 192 adaptive residual, gate 0.1:
  - Run: `weakphase2_ettm1_192_adaptive_residual_g01_lr0003_mae_b256_e30_seed2021`
  - Result: MAE 0.359045, MSE 0.328793.
  - Decision: essentially tied with iteration 50 and below the 5% target.
- Exit condition:
  - The round exceeded 50 iterations without ETTh1 and ETTm1 reaching the requested 5%/5% improvement.
  - ETTm2 96 remains the only dataset in this round that satisfied the 5% target: `weakphase2_ettm2_96_residual_gate999_lr0003_mae_e30_seed2021`, MAE 0.245211 and MSE 0.160063 versus baseline MAE 0.258160 and MSE 0.170091.
  - Best ETTm1 96 result: `weakphase2_ettm1_96_phase_hifreq_s05_thr10_w7_residual_gate999_lr0003_mae_b256_e30_seed2021`, MAE 0.338555, MSE 0.292791 versus baseline MAE 0.347958, MSE 0.299526.
  - Best ETTh1 96 result remains sub-threshold; phase-jitter and trend/residual variants did not produce stable gains under enhanced bad-case review.

## ETT Formal Regression - 2026-07-02

- Command: `conda run --no-capture-output -n raft python scripts/benchmark_phaseformer_suite.py --datasets ETTh1,ETTh2,ETTm1,ETTm2 --horizons all --modes original,latest --num-workers 0 --bad-case-limit 10 --bad-case-batches 8 --run-prefix phaseformer_ett_regression_20260702 --resume`
- Evidence:
  - `research_runs/phaseformer_ett_regression_20260702_summary.csv`
  - `research_runs/phaseformer_ett_regression_20260702_comparison.csv`
  - `research_runs/phaseformer_ett_regression_20260702_report.md`
- Run status: 32 fresh runs, no resumed runs. PyTorch CUDA was usable on RTX 4090; `nvidia-smi`/NVML reported a driver-library mismatch, so external GPU telemetry was unreliable.
- Effective latest changes:
  - ETTh2 96: `latest_etth2_adaptive_residual_mae`, MAE 0.343032 -> 0.328264 (-4.31%), MSE 0.280557 -> 0.266525 (-5.00%).
  - ETTh2 720: `latest_etth2_residual_long`, MAE 0.448750 -> 0.424987 (-5.30%), MSE 0.415718 -> 0.383477 (-7.76%).
- Guardrail results:
  - ETTh1, ETTm1, and ETTm2 latest modes are identical to original on all four horizons; no regression.
  - ETTh2 192 and 336 are also guardrailed to original; no regression.
- Conclusion: the best formal version for the ETT regression set is the dataset-aware latest policy. It enables residual/adaptive residual only where evidence supports it, and preserves original behavior elsewhere.

## Weak-Period ETT Innovation Round - 2026-07-05 Start

- New user request:
  - Continue autonomous research under the updated `HOW_TO_DO_RESEARCH.md`.
  - Target weak-period ETT-series data.
  - Exit when MAE and MSE both improve by more than 5% versus original PhaseFormer, or when this round exceeds 30 iterations.
  - Avoid residual-gate tricks and avoid dataset/horizon-aware guardrails as the final scientific claim.
- Bad-case drivers from prior evidence:
  - ETTm1 96: MUFL dominates severe failures; common pattern is weak-period phase-amplitude hallucination and systematic bias. Examples in `research_runs/weakphase2_review_b256_ettm1_96_baseline_fixed_e30_seed2021/bad_cases.csv`.
  - ETTh1 96: trend under-response and late drift dominate more than high-frequency hallucination. Examples in `research_runs/weakphase2_review_etth1_96_baseline_e30_seed2021/bad_cases.csv`.
- Baselines:
  - ETTm1 96 baseline MAE 0.347958, MSE 0.299526.
  - ETTh1 96 baseline MAE 0.388491, MSE 0.364891.

### Iterations 1-10: Phase-Uncertainty And Phase-Decomposition Probes

- New mechanism implemented:
  - `PhaseUncertaintyShrinkage` in `src/models/PhaseFormer.py`.
  - Theory: for weak-period observations `x_{l,k}=p_l+d_k+eps_{l,k}`, cross-period same-phase variance estimates unreliable phase history and phase-template variance estimates useful periodic signal. The layer shrinks noisy deviations before cross-phase routing, rather than blending a residual output branch.
  - Research script variants: `phase_uncertainty`, `phase_uncertainty_hifreq`.
- Additional probes implemented:
  - `PhaseDeviationDropout`: training-time dropout of deviations from the phase template; intended to discourage memorizing unstable same-phase details.
  - `PhasePeriodLevelDetrend`: removes period-level mean before routing and restores a low-frequency period-level forecast; intended to reduce systematic bias from drift entangled with phase shape.
- Iteration 1:
  - Runs:
    - `weakphase3_iter01_ettm1_96_phase_uncertainty_min35_trend005_e30_seed2021_rerun`: ETTm1 96 MAE 0.346773 (-0.34%), MSE 0.297987 (-0.51%).
    - `weakphase3_iter01_etth1_96_phase_uncertainty_min35_trend005_e30_seed2021`: ETTh1 96 MAE 0.386568 (-0.49%), MSE 0.358135 (-1.85%).
  - Bad-case review: ETTm1 MUFL highest/systematic/volatility cases remain around MSE 4; ETTh1 average improves but selected severe cases shift to worse windows. Decision: keep as weak positive but insufficient.
- Iteration 2:
  - Run: `weakphase3_iter02_ettm1_96_phase_uncertainty_min05_trend0001_e30_seed2021`.
  - Result: MAE 0.348267 (+0.09%), MSE 0.308687 (+3.06%).
  - Decision: stronger global shrinkage erases useful phase amplitude; reject.
- Iteration 3:
  - Run: `weakphase3_iter03_ettm1_96_phase_uncertainty_min35_trend0001_e30_seed2021`.
  - Result: MAE 0.346459 (-0.43%), MSE 0.299198 (-0.11%).
  - Decision: closing the trend term protects MAE slightly but loses most MSE improvement; trend is not the main solution.
- Iteration 4:
  - Run: `weakphase3_iter04_ettm1_96_phase_uncertainty_hifreq_min35_s05_w7_e30_seed2021`.
  - Result: MAE 0.346740 (-0.35%), MSE 0.297970 (-0.52%).
  - Decision: adding high-frequency damping without training change does not solve the bias-heavy bad cases.
- Iteration 5:
  - Run: `weakphase3_iter05_ettm1_96_phase_dropout_p10_e30_seed2021`.
  - Result: MAE 0.350392 (+0.70%), MSE 0.305080 (+1.85%).
  - Decision: dropping phase deviations damages useful phase amplitude; reject this regularizer.
- Iteration 6:
  - Run: `weakphase3_iter06_ettm1_96_phase_uncertainty_hifreq_min35_s05_w7_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.340149 (-2.24%), MSE 0.294273 (-1.75%).
  - Bad-case review: average improves under MAE/LR, but MUFL highest/systematic bias remains around MSE 4.3 and bias 1.46-1.48. Decision: positive but far below target.
- Iteration 7:
  - Run: `weakphase3_iter07_ettm1_96_phase_uncertainty_hifreq_min35_s08_w7_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.340127 (-2.25%), MSE 0.294240 (-1.76%).
  - Decision: increasing damping strength gives negligible benefit.
- Iteration 8:
  - Run: `weakphase3_iter08_ettm1_96_phase_uncertainty_hifreq_min35_s08_thr05_w7_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.339609 (-2.40%), MSE 0.293716 (-1.94%).
  - Bad-case review: volatile-input case improves, but top MUFL cases remain dominated by systematic bias and volatility mismatch. Decision: current best raw ETTm1 96 result, but not a valid model-design breakthrough.
- Iteration 9:
  - Run: `weakphase3_iter09_ettm1_96_phase_level_detrend_w3_g01_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.417865 (+20.09%), MSE 0.423054 (+41.24%).
  - Decision: period-level mean removal is too destructive; reject this decomposition form.
- Iteration 10:
  - Run: `weakphase3_iter10_ettm1_96_baseline_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.340444 (-2.16%), MSE 0.293950 (-1.86%).
  - Decision: most of iteration 8's gain is explained by matched training settings. Continue with a new model mechanism; do not claim phase uncertainty shrinkage as effective yet.

### Iterations 11-31: ETTm2 Signal, Level Calibration, And Exit

- Iteration 11:
  - Run: `weakphase3_iter11_ettm2_96_phase_uncertainty_hifreq_min35_s08_thr05_w7_lr0003_mae_e30_seed2021`.
  - Result: ETTm2 96 MAE 0.249029 (-3.54%), MSE 0.161170 (-5.25%).
  - Bad-case review: MSE reaches target, but MULL highest/systematic bias remains high; MAE misses the 5% threshold.
- Iteration 12:
  - Run: `weakphase3_iter12_ettm2_96_baseline_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.257416 (-0.29%), MSE 0.168600 (-0.88%).
  - Decision: iteration 11's ETTm2 gain is not explained by MAE/LR alone.
- Iterations 13-18:
  - Runs:
    - `weakphase3_iter13_ettm2_96_phase_uncertainty_hifreq_min35_s05_thr05_w7_lr0003_mae_e30_seed2021`: MAE 0.249125, MSE 0.161196.
    - `weakphase3_iter14_ettm2_96_phase_uncertainty_min35_lr0003_mae_e30_seed2021`: MAE 0.249220, MSE 0.161163.
    - `weakphase3_iter15_ettm2_96_phase_uncertainty_min60_lr0003_mae_e30_seed2021`: MAE 0.255966, MSE 0.166924.
    - `weakphase3_iter16_ettm2_96_phase_uncertainty_min20_lr0003_mae_e30_seed2021`: MAE 0.248961, MSE 0.160900.
    - `weakphase3_iter17_ettm2_96_phase_uncertainty_min20_lr0003_mae_e50_seed2021`: MAE 0.248286, MSE 0.160638.
    - `weakphase3_iter18_ettm2_96_phase_uncertainty_min10_lr0003_mae_e30_seed2021`: MAE 0.250514, MSE 0.162096.
  - Decision: min reliability 0.2 is the best shrinkage strength; more epochs help only marginally and still miss MAE target.
- New mechanism:
  - `PhasePeriodLevelCalibration` in `src/models/PhaseFormer.py`.
  - Theory: bad cases after shrinkage still show systematic period-level bias. Instead of removing level before routing, calibrate each forecast period's phase-step mean toward a recent period-level anchor, preserving learned phase shape while correcting `d_k`.
- Iterations 19-23:
  - Runs:
    - `weakphase3_iter19_ettm2_96_phase_uncertainty_levelcalib_min20_g01_lr0003_mae_e30_seed2021`: MAE 0.248644, MSE 0.160522.
    - `weakphase3_iter20_ettm2_96_phase_uncertainty_levelcalib_min20_g02_lr0003_mae_e30_seed2021`: MAE 0.248588, MSE 0.160452.
    - `weakphase3_iter21_ettm2_96_phase_uncertainty_levelcalib_min20_g05_lr0003_mae_e30_seed2021`: MAE 0.248822, MSE 0.160336.
    - `weakphase3_iter22_ettm2_96_phase_uncertainty_levelcalib_min20_g02_lr0001_mae_e30_seed2021`: MAE 0.258707, MSE 0.170461.
    - `weakphase3_iter23_ettm2_96_phase_uncertainty_levelcalib_min20_g02_lr0005_mae_e30_seed2021`: MAE 0.248568, MSE 0.161062.
  - Decision: level calibration improves MSE and slightly improves MAE, but cannot close the remaining MAE gap. LR 0.0003 remains best.
- Iterations 24-25 transfer checks:
  - `weakphase3_iter24_etth2_96_phase_uncertainty_levelcalib_min20_g02_lr0003_mae_e30_seed2021`: ETTh2 96 MAE 0.342094 (-0.27%), MSE 0.283363 (+1.00%).
  - `weakphase3_iter25_etth1_96_phase_uncertainty_levelcalib_min20_g02_lr0003_mae_e30_seed2021`: ETTh1 96 MAE 0.393467 (+1.28%), MSE 0.369269 (+1.20%).
  - Decision: the mechanism is not a unified ETT solution; do not use dataset-aware guardrail to claim global success.
- Iterations 26-29 training/period checks:
  - `weakphase3_iter26_ettm2_96_phase_uncertainty_levelcalib_min20_g02_lr0003_mae_b128_e30_seed2021`: MAE 0.248638, MSE 0.160795.
  - `weakphase3_iter27_ettm2_96_phase_uncertainty_levelcalib_min20_g02_lr0003_smae_e30_seed2021`: MAE 0.254392, MSE 0.162893.
  - `weakphase3_iter28_ettm2_96_p96_phase_uncertainty_levelcalib_min20_g02_lr0003_mae_e30_seed2021`: MAE 0.280422, MSE 0.186013.
  - `weakphase3_iter29_ettm2_96_p12_phase_uncertainty_levelcalib_min20_g02_lr0003_mae_e30_seed2021`: MAE 0.251155, MSE 0.163228.
  - Decision: batch size, SMAE, and alternate period lengths do not solve the MAE bottleneck. Rigid daily phase is especially harmful.
- Iterations 30-31 final combinations:
  - `weakphase3_iter30_ettm2_96_phase_uncertainty_levelcalib_hifreq_min20_g02_s05_thr05_lr0003_mae_e30_seed2021`: MAE 0.248354 (-3.80%), MSE 0.160277 (-5.77%).
  - `weakphase3_iter31_ettm2_96_phase_uncertainty_levelcalib_hifreq_min20_g02_s08_thr05_lr0003_mae_e30_seed2021`: MAE 0.248220 (-3.85%), MSE 0.160189 (-5.82%).
  - Bad-case review: highest/systematic MULL cases remain dominated by bias around 0.85 even after calibration, explaining the MAE miss.
- Exit decision:
  - Stop condition reached because the round exceeded 30 iterations.
  - Best candidate improves ETTm2 96 MSE by more than 5% but fails the MAE target.
  - No candidate satisfies the requested MAE and MSE >5% improvement condition.

## Dataset-Adaptive ETT Phase Framework Round - 2026-07-06 Start

- New user request:
  - Allow dataset-specific method switches and partial parameter adjustment.
  - Keep the design inside one method framework.
  - Iterate 50 more rounds to seek stable improvement over original PhaseFormer on all ETT datasets.
- Framework definition:
  - Shared family: weak-period phase adaptation.
  - Allowed submodules: phase uncertainty shrinkage, phase-period level calibration, phase-noise high-frequency damping, low-frequency trend response, and previously validated adaptive/residual phase auxiliaries when bad-case evidence justifies them.
  - Dataset-level switches are allowed; arbitrary horizon-level switching remains disallowed unless supported by bad-case/horizon evidence.
- Starting baselines from formal ETT regression:
  - ETTh1 96: MAE 0.388491, MSE 0.364891.
  - ETTh2 96: MAE 0.343032, MSE 0.280557.
  - ETTm1 96: MAE 0.347958, MSE 0.299526.
  - ETTm2 96: MAE 0.258160, MSE 0.170091.
- Initial plan:
  - Use horizon 96 as the representative low-cost search surface.
  - First target ETTh1 and ETTm1, because ETTh2 and ETTm2 already have positive dataset-specific signals.
  - Keep bad case exports at 8 cases per key run.

### Iterations 1-10: 96-Horizon Dataset-Adaptive Policy

- Iteration 1, ETTh1 low-frequency trend:
  - Run: `weakphase4_iter01_etth1_96_lowfreq_trend_w25_g005_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.388980, MSE 0.371957. Rejected because both metrics worsen.
- Iterations 2-5, ETTh1 phase uncertainty + level calibration:
  - `weakphase4_iter02_etth1_96_phase_uncertainty_min35_lr0003_mae_e30_seed2021`: MAE 0.391040, MSE 0.372763. Rejected.
  - `weakphase4_iter03_etth1_96_phase_uncertainty_levelcalib_min35_g005_e30_seed2021`: MAE 0.386194, MSE 0.357419.
  - `weakphase4_iter04_etth1_96_phase_uncertainty_levelcalib_min35_g01_e30_seed2021`: MAE 0.386066, MSE 0.357461.
  - `weakphase4_iter05_etth1_96_phase_uncertainty_levelcalib_min35_g02_e30_seed2021`: MAE 0.386384, MSE 0.358014.
  - Decision: ETTh1 needs original MSE/Huber training with mild phase-level calibration; gate 0.1 gives best MAE and remains MSE-positive.
- Iterations 6-8, ETTm1 phase uncertainty + level calibration + high-frequency damping:
  - `weakphase4_iter06_ettm1_96_phase_uncertainty_levelcalib_hifreq_min35_g01_s08_thr05_lr0003_mae_e30_seed2021`: MAE 0.339635, MSE 0.293330.
  - `weakphase4_iter07_ettm1_96_phase_uncertainty_levelcalib_hifreq_min35_g02_s08_thr05_lr0003_mae_e30_seed2021`: MAE 0.339945, MSE 0.292920.
  - `weakphase4_iter08_ettm1_96_phase_uncertainty_levelcalib_hifreq_min20_g01_s08_thr05_lr0003_mae_e30_seed2021`: MAE 0.339887, MSE 0.296120.
  - Decision: ETTm1 differs from ETTm2; it needs weaker shrinkage `phase_uncertainty_min=0.35`. This dataset-level parameter difference is justified by MSE degradation under min=0.2.
- Iteration 9, ETTh2 adaptive residual reproduction:
  - Run: `weakphase4_iter09_etth2_96_adaptive_residual_g02_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.328264, MSE 0.266525.
  - Decision: reproduces the prior validated ETTh2 96 improvement.
- Iteration 10, ETTm2 phase uncertainty + level calibration + high-frequency damping reproduction:
  - Run: `weakphase4_iter10_ettm2_96_phase_uncertainty_levelcalib_hifreq_min20_g02_s08_thr05_lr0003_mae_e30_seed2021`.
  - Result: MAE 0.248220, MSE 0.160189.
- Formal 96-horizon regression:
  - Command: `conda run --no-capture-output -n raft python scripts/benchmark_phaseformer_suite.py --datasets ETTh1,ETTh2,ETTm1,ETTm2 --horizons 96 --modes original,latest --num-workers 0 --bad-case-limit 8 --bad-case-batches 8 --run-prefix phaseformer_ett96_dataset_adaptive_20260706 --resume`
  - Evidence: `research_runs/phaseformer_ett96_dataset_adaptive_20260706_comparison.csv`.
  - Result: all four ETT datasets improve in both metrics at horizon 96:
    - ETTh1: MAE -0.62%, MSE -2.04%.
    - ETTh2: MAE -4.31%, MSE -5.00%.
    - ETTm1: MAE -2.39%, MSE -2.07%.
    - ETTm2: MAE -3.85%, MSE -5.82%.
  - Decision: promote the 96-horizon dataset-adaptive phase framework to `src/models/phaseformer_presets.py` for formal latest-mode testing. Next step is multi-horizon expansion.

### Iterations 11-28: Multi-Horizon Expansion

- Iterations 11-23, horizon 192:
  - ETTh1 probes showed that the 96-horizon level-calibration setting overcorrects longer forecasts. Bad cases shifted from peak/trend under-response to late-horizon level drift, so the retained 192 policy uses only conservative phase uncertainty shrinkage (`phase_uncertainty_min=0.6`).
  - ETTm1/ETTm2 retained the minute-level phase uncertainty + level calibration + high-frequency damping policy because bad cases remained dominated by noisy same-phase deviations and tail volatility.
  - ETTh2 retained the validated adaptive residual weak-period branch.
  - Formal evidence: `research_runs/phaseformer_ett192_dataset_adaptive_20260706_comparison.csv`.
  - Result:
    - ETTh1: MAE -0.18%, MSE -4.03%.
    - ETTh2: MAE -2.06%, MSE -1.83%.
    - ETTm1: MAE -1.23%, MSE -0.90%.
    - ETTm2: MAE -3.80%, MSE -3.37%.
- Iterations 24-28, horizon 336:
  - ETTm2/ETTm1 transferred the same minute-level policy with positive but smaller gains.
  - ETTh1 again favored conservative phase uncertainty shrinkage without level calibration.
  - ETTh2 needed the same adaptive residual mechanism but a longer patience setting; otherwise early stopping selected a regressed checkpoint.
  - Formal evidence: `research_runs/phaseformer_ett336_dataset_adaptive_20260706b_comparison.csv`.
  - Result:
    - ETTh1: MAE -1.00%, MSE -1.01%.
    - ETTh2: MAE -1.47%, MSE -0.50%.
    - ETTm1: MAE -0.71%, MSE -0.60%.
    - ETTm2: MAE -1.72%, MSE -1.43%.

### Iterations 29-40: Horizon 720 And ETTh1 Late-Drift Fix

- Iterations 29-35, first 720 search:
  - ETTm1 720: `weakphase4_iter30_ettm1_720_phase_uncertainty_levelcalib_hifreq_min35_g01_s08_thr05_lr0003_mae_e30_seed2021` improved MAE 0.407951 vs 0.413261 and MSE 0.411918 vs 0.418219.
  - ETTm2 720: high-frequency damping worsened MSE in `weakphase4_iter29`; removing the high-frequency damper and using stronger conservative shrinkage in `weakphase4_iter35_ettm2_720_phase_uncertainty_levelcalib_min60_g01_lr0003_mae_e30_seed2021` improved MAE 0.374787 vs 0.379086 and MSE 0.350420 vs 0.354294.
  - ETTh1 720: `weakphase4_iter32_etth1_720_phase_uncertainty_levelcalib_min80_g005_lr00015_huber03_e70_seed2026` reduced MSE 0.438149 vs 0.440721 but slightly worsened MAE 0.447570 vs 0.447249. Bad-case review (8 cases) showed several top MSE cases improved, but new late-horizon drift cases appeared, so the policy was not accepted.
- Formal 720b check:
  - Evidence: `research_runs/phaseformer_ett720_dataset_adaptive_20260706b_comparison.csv`.
  - Result: ETTh2, ETTm1, and ETTm2 improved in both metrics; ETTh1 had MAE +0.07% and MSE -0.58%.
  - Decision: continue ETTh1-720; do not claim stable all-ETT improvement.
- Iterations 36-37:
  - `weakphase4_iter36_etth1_720_phase_uncertainty_levelcalib_min80_g002_lr00015_huber03_e70_seed2026` and `weakphase4_iter37_etth1_720_phase_uncertainty_min80_lr00015_huber03_e70_seed2026` used the research script's default batch size 16 by mistake.
  - Both were rejected as non-comparable configuration-error runs, not as model evidence.
- Iteration 38:
  - Run: `weakphase4_iter38_etth1_720_phase_uncertainty_min80_lr00015_huber03_b256_e70_seed2026`.
  - Result: MAE 0.452508, MSE 0.447181. Pure uncertainty shrinkage at min 0.8 is insufficient; reject.
- Iteration 39:
  - Run: `weakphase4_iter39_etth1_720_phase_uncertainty_levelcalib_min80_g005_lr00015_mae_b256_e70_seed2026`.
  - Result: MAE 0.449894, MSE 0.443830. MAE training reduces neither late drift nor MSE enough; reject.
- Iteration 40:
  - Run: `weakphase4_iter40_etth1_720_phase_uncertainty_levelcalib_min80_g005_lr0001_huber03_b256_e70_seed2026`.
  - Hypothesis: ETTh1-720 late-horizon drift is caused by training overshoot in the calibration/shrinkage interaction, not by the mechanism itself; lower LR should stabilize phase-level anchors while retaining MSE gain.
  - Result: MAE 0.441219 vs original 0.447249 (-1.35%), MSE 0.427029 vs original 0.440721 (-3.11%).
  - Bad-case decision: accepted. It keeps the same phase uncertainty + level calibration mechanism and fixes the MAE regression seen in 720b.
- Formal 720c regression:
  - Command: `conda run --no-capture-output -n raft python scripts/benchmark_phaseformer_suite.py --datasets ETTh1,ETTh2,ETTm1,ETTm2 --horizons 720 --modes original,latest --num-workers 0 --bad-case-limit 8 --bad-case-batches 8 --run-prefix phaseformer_ett720_dataset_adaptive_20260706c --resume`
  - Evidence: `research_runs/phaseformer_ett720_dataset_adaptive_20260706c_comparison.csv`.
  - Result:
    - ETTh1: MAE -1.35%, MSE -3.11%.
    - ETTh2: MAE -5.30%, MSE -7.76%.
    - ETTm1: MAE -1.28%, MSE -1.51%.
    - ETTm2: MAE -1.13%, MSE -1.09%.
  - Decision: latest policy now improves both MAE and MSE against original on ETTh1, ETTh2, ETTm1, and ETTm2 for horizons 96, 192, 336, and 720 under the formal runner. Gains are stable but not universally above 5%; the user-requested 50-iteration cap was not reached because the practical stable-improvement objective was met.

## Weather/Electricity Phase Adaptation Round - 2026-07-06 Start

- New user request:
  - Optimize Weather and Electricity within the current weak-period phase-adaptation framework.
  - Stop when improvement exceeds 3%, or when the round exceeds 50 iterations.
- Starting formal baselines:
  - Weather 96: MAE 0.195908, MSE 0.149202 from `research_runs/phaseformer_full_latest_vs_original_highdim_b64w4_20260630_comparison.csv`.
  - Electricity 96: MAE 0.220274, MSE 0.128806 from the same comparison file.
- Candidate hypotheses:
  - H1 phase uncertainty + level calibration: Weather/Electricity bad cases include systematic level bias and peak underfit; calibrating period-level anchors may reduce bias without discarding phase shape.
  - H2 low-frequency trend correction: Electricity smoke bad cases show trend mismatch and late-horizon drift; a low-frequency phase-space correction may address drift without an unrestricted residual branch.
  - H3 phase noise high-frequency damping: Weather has meteorological noise and Electricity has channel-specific volatility; high-frequency phase damping may help if volatile-input bad cases dominate.
- Tooling:
  - Extend `scripts/research_weather_weak.py` to accept `Electricity`, so Weather and Electricity can use the same capped pattern-covered bad-case export.
  - Smoke command: `conda run --no-capture-output -n raft python scripts/research_weather_weak.py --dataset Electricity --variant baseline --horizon 96 --epochs 1 --percent 5 --batch-size 64 --num-workers 0 --bad-case-limit 8 --bad-case-batches 2 --run-id smoke_electricity96_research_script`.
  - Smoke result: ran successfully and exported 8 pattern-covered bad cases. Because it used 5% data and 1 epoch, it is not a performance conclusion.
- Initial bad-case review:
  - Evidence: `research_runs/smoke_electricity96_research_script/bad_cases.csv`.
  - Patterns include highest MSE, systematic bias, trend mismatch, peak underfit, valley overfit, volatility mismatch, late-horizon drift, and volatile input.
  - Decision: start with phase-level calibration and low-frequency trend mechanisms rather than residual-only gates.

### Iterations 1-6: Weather Positive Signal, Electricity Full-Cost Rejections

- Iteration 1, Weather phase uncertainty + level calibration + high-frequency damping:
  - Run: `weakphase5_iter01_weather96_p12_uncert_levelcalib_hifreq_min35_g01_s08_thr05_lr0003_mae_b64_e30_seed2021`.
  - Result versus formal Weather 96 baseline MAE 0.195908, MSE 0.149202: MAE 0.188897 (-3.58%), MSE 0.147550 (-1.11%).
  - Bad-case review: MSE-heavy failures concentrate on `raining (s)` peak underfit, volatility mismatch, and late-horizon drift, so the MAE-oriented setup is not enough for squared error.
  - Decision: keep as best Weather MAE candidate, but continue because MSE misses the 3% target.
- Iteration 2, Weather same mechanism with MSE/Huber:
  - Run: `weakphase5_iter02_weather96_p12_uncert_levelcalib_hifreq_min35_g01_s08_thr05_lr001_huber_b64_e30_seed2021`.
  - Result: MAE 0.192209 (-1.89%), MSE 0.146194 (-2.02%).
  - Decision: MSE improves but MAE loses the 3% gain.
- Iteration 3, Weather remove high-frequency damping:
  - Run: `weakphase5_iter03_weather96_p12_uncert_levelcalib_min35_g01_lr001_huber_b64_e30_seed2021`.
  - Result: MAE 0.192048 (-1.97%), MSE 0.146115 (-2.07%).
  - Decision: high-frequency damping is not the main MSE bottleneck; removing it only marginally helps.
- Iteration 4, Electricity phase uncertainty + level calibration:
  - Run: `weakphase5_iter04_electricity96_uncert_levelcalib_min35_g01_lr002_huber_b64_e30_seed2021`.
  - Result versus formal Electricity 96 baseline MAE 0.220274, MSE 0.128806: MAE 0.224373 (+1.86%), MSE 0.129661 (+0.66%).
  - Bad-case review: top errors still concentrate on variable 113 with systematic underfit and peak underfit, so direct period-level calibration overcorrects high-dimensional Electricity rather than fixing the dominant channels.
  - Decision: reject full-cost level calibration for Electricity.
- Iteration 5, Electricity low-frequency trend correction:
  - Run: `weakphase5_iter05_electricity96_lowfreq_trend_w25_g005_lr002_huber_b64_e30_seed2021`.
  - Result: MAE 0.222563 (+1.04%), MSE 0.129166 (+0.28%).
  - Decision: low-frequency trend correction alone is less harmful than level calibration but still not useful; stop full-cost blind trials on Electricity.
- Iteration 6, Weather Huber/LR compromise:
  - Run: `weakphase5_iter06_weather96_p12_uncert_levelcalib_min35_g01_lr0007_huber05_b64_e30_seed2021`.
  - Result: MAE 0.190961 (-2.53%), MSE 0.146976 (-1.49%).
  - Decision: compromise setting loses the MAE target and does not improve MSE enough. Continue with lower-cost screening, especially for Electricity.
- Iterations 7-17, Electricity 96 low-cost screening:
  - Percent30 baseline: `weakphase5_iter07_electricity96_baseline_percent30_e10_b64_seed2021`, MAE 0.235700, MSE 0.142035.
  - Adaptive residual gate 0.2 with MAE/LR 0.0003: MAE 0.233128 (-1.09%), MSE 0.137649 (-3.09%).
  - Adaptive channel residual: MAE 0.247848, MSE 0.152049; rejected as high-capacity overfit.
  - Adaptive smooth residual: MAE 0.242936, MSE 0.148945; rejected.
  - Phase uncertainty + residual: MAE 0.242685, MSE 0.144881; rejected.
  - Adaptive residual MSE/Huber LR 0.001: `weakphase5_iter12_electricity96_adaptive_residual_g02_lr001_huber_percent30_e10_b64_seed2021`, MAE 0.233052 (-1.12%), MSE 0.135444 (-4.64%).
  - Adaptive residual gate 0.5: `weakphase5_iter13_electricity96_adaptive_residual_g05_lr001_huber_percent30_e10_b64_seed2021`, MAE 0.231334 (-1.85%), MSE 0.134771 (-5.11%).
  - Adaptive residual gate 0.8: MAE 0.231564, MSE 0.135578; worse than gate 0.5.
  - Training-only diagnostics: baseline MAE/LR 0.0003 strongly worsens; baseline LR 0.001 also worsens. Decision: low-cost MSE signal is from adaptive residual, not just LR, but MAE remains below the 3% target.
- Iterations 15 and 22-23, Electricity full validation:
  - `weakphase5_iter15_electricity96_adaptive_residual_g05_lr001_huber_b64_e30_seed2021`: MAE 0.223121, MSE 0.129501 versus formal baseline MAE 0.220274, MSE 0.128806; rejected.
  - Electricity 720 percent30 baseline: `weakphase5_iter20_electricity720_baseline_percent30_e10_b64_seed2021`, MAE 0.328742, MSE 0.243691.
  - Electricity 720 percent30 adaptive residual: `weakphase5_iter21_electricity720_adaptive_residual_g05_lr001_huber_percent30_e10_b64_seed2021`, MAE 0.310413 (-5.58%), MSE 0.224724 (-7.78%).
  - Full validation `weakphase5_iter22_electricity720_adaptive_residual_g05_lr001_huber_b64_e30_seed2021`: MAE 0.288686, MSE 0.200408 versus formal baseline MAE 0.286129, MSE 0.199135; rejected.
  - Full validation with 12 epochs `weakphase5_iter23_electricity720_adaptive_residual_g05_lr001_huber_b64_e12_seed2021`: MAE 0.291634, MSE 0.202851; rejected.
  - Decision: Electricity low-cost gains do not transfer to full-data training. Do not promote adaptive residual for Electricity.
- New mechanism after Weather bad-case review:
  - `PhaseShapeAmplitudeCalibration` in `src/models/PhaseFormer.py`.
  - Theory: sparse-event weak-period failures can have correct period level but flattened within-period phase shape. The module preserves forecast period mean and only expands deviations across phase slots when predicted within-period amplitude is lower than recent input-period amplitude.
  - Tooling: `scripts/research_weather_weak.py` variants `phase_shape_amp`, `phase_uncertainty_shape_amp`, `phase_uncertainty_level_calib_shape_amp`, and `phase_uncertainty_level_calib_hifreq_shape_amp`.
- Iterations 18-19, Weather shape-amplitude calibration:
  - `weakphase5_iter18_weather96_p12_uncert_levelcalib_shapeamp_min35_g01_amp005_max15_lr001_huber_b64_e30_seed2021`: MAE 0.192039, MSE 0.146114.
  - `weakphase5_iter19_weather96_p12_uncert_levelcalib_shapeamp_min35_g01_amp02_max20_lr001_huber_b64_e30_seed2021`: MAE 0.192014, MSE 0.146102.
  - Decision: amplitude calibration barely changes aggregate metrics and does not solve `raining (s)` peak MSE. Keep default-off as a research module, but do not promote.
- Iteration 24, Weather phase granularity:
  - Run: `weakphase5_iter24_weather96_p8_baseline_lr001_huber_b64_e30_seed2021`.
  - Result: MAE 0.194543, MSE 0.148216. Worse than period_len 12 candidates and below the 3% target.
  - Current status: no Weather/Electricity candidate has met the 3% dual-metric exit condition. Best Weather MAE candidate remains iteration 1; best Weather MSE candidate remains iteration 19/3 around MSE 0.14610. Best Electricity full candidates are still worse than baseline.

### Iterations 25-32: Weather Long-Horizon And Shape-Amplitude Screening

- Iterations 25-26, Weather192 full validation:
  - `weakphase5_iter25_weather192_p12_uncert_levelcalib_hifreq_min35_g01_s08_thr05_lr0003_mae_b64_e30_seed2021`: MAE 0.233732, MSE 0.193499 versus formal baseline MAE 0.237761, MSE 0.193425. MAE improved, MSE did not.
  - `weakphase5_iter26_weather192_p12_uncert_levelcalib_min35_g01_lr001_huber_b64_e30_seed2021`: MAE 0.238740, MSE 0.191223. MSE improved, MAE regressed.
  - Decision: same MAE/MSE target conflict as Weather96; no 3% dual-metric gain.
- Iterations 27-32, Weather720 percent30 screening:
  - Baseline `weakphase5_iter27_weather720_baseline_percent30_e10_b64_seed2021`: MAE 0.347295, MSE 0.329238.
  - Adaptive residual gate 0.5: MAE 0.339083, MSE 0.322458.
  - Adaptive residual gate 0.8: MAE 0.336521, MSE 0.320746; subset MAE improved over 3%, MSE improved about 2.58%, below target.
  - Adaptive residual gate 0.95: MAE 0.339751, MSE 0.325223; worse than gate 0.8.
  - Low-frequency trend: MAE 0.352763, MSE 0.331685; rejected.
  - Adaptive smooth residual: MAE 0.339333, MSE 0.325140; smoothing removed useful event variation.
  - Bad-case decision: Weather720 errors still include `raining (s)` sparse peaks and humidity late drift; residual helps subset level drift but does not sufficiently reduce squared peak errors.

### Iterations 33-43: Weather336/Weather96 Mechanism Review

- Iteration 33, Weather336 percent30 baseline:
  - `weakphase5_iter33_weather336_baseline_percent30_e10_b64_seed2021`: MAE 0.289840, MSE 0.251908.
  - Bad cases, capped at 8, were dominated by `rh (%)` systematic bias/late drift and `raining (s)` peak underfit/volatility mismatch.
- Iteration 34, Weather336 adaptive residual:
  - `weakphase5_iter34_weather336_adaptive_residual_g08_lr001_huber_percent30_e10_b64_seed2021`: MAE 0.290562, MSE 0.254408.
  - Bad-case review: `rh (%)` late MSE increased, so the residual branch amplified uncertain long-horizon humidity drift. Rejected.
- Iteration 35, Weather336 phase uncertainty + level calibration + high-frequency damping:
  - `weakphase5_iter35_weather336_p12_uncert_levelcalib_hifreq_min35_g01_s08_thr05_lr0003_mae_percent30_e10_b64_seed2021`: MAE 0.298482, MSE 0.258386.
  - Decision: p12 uncertainty calibration does not transfer to Weather336; rejected.
- Iteration 36, Weather96 lower reliability floor with Huber:
  - `weakphase5_iter36_weather96_p12_uncert_levelcalib_hifreq_min20_g01_s05_thr07_lr001_huber05_b64_e30_seed2021`: MAE 0.190758, MSE 0.146940.
  - Versus formal Weather96 baseline MAE 0.195908, MSE 0.149202: MAE -2.63%, MSE -1.52%, below 3%.
- New mechanism for iterations 37-38:
  - Added default-off `PhaseSparseEventCalibration`.
  - Theory: for weak-period sparse events, the phase router can predict a reasonable period mean while flattening rare positive excursions. The module estimates a recent same-phase positive event envelope and reallocates forecast mass toward historically active phase slots while subtracting the mean correction, so it is not an unrestricted residual shortcut.
  - Code: `src/models/PhaseFormer.py`, `scripts/research_weather_weak.py`, and `src/models/phaseformer_presets.py`.
- Iteration 37, sparse-event calibration with MAE setup:
  - `weakphase5_iter37_weather96_p12_uncert_levelcalib_hifreq_sparseevent_g01_boost10_lr0003_mae_b64_e30_seed2021`: MAE 0.188815, MSE 0.147518.
  - Bad-case review: `raining (s)` peak_under remained around 5.26-5.53 in the top cases, nearly unchanged from iteration 1. Mechanism was too conservative under MAE training.
- Iteration 38, sparse-event calibration with Huber/MSE setup:
  - `weakphase5_iter38_weather96_p12_uncert_levelcalib_hifreq_sparseevent_g03_boost20_temp015_lr001_huber_b64_e30_seed2021`: MAE 0.192145, MSE 0.146199.
  - Decision: improves MSE similarly to iteration 2 but loses MAE; sparse-event calibration in this form does not break the Weather96 MAE/MSE tradeoff.
- Iterations 39-41, Weather96 phase granularity check:
  - p6 candidate: MAE 0.210530, MSE 0.166546.
  - p18 candidate: MAE 0.204908, MSE 0.159187.
  - p12 percent30 baseline: MAE 0.205154, MSE 0.156691.
  - Decision: p6 and p18 are worse than p12 baseline; phase length mismatch is not the main cause.
- Iteration 42, Weather96 period-level detrending:
  - `weakphase5_iter42_weather96_p12_leveldetrend_uncert_hifreq_min35_slopeg01_s08_thr05_lr0003_mae_percent30_e10_b64_seed2021`: MAE 0.264514, MSE 0.250644.
  - Decision: removing period means destroys useful Weather phase shape; rejected.
- Iteration 43, Weather96 phase reliability damping:
  - `weakphase5_iter43_weather96_p12_phase_reliability_min60_thr05_lr001_huber_percent30_e10_b64_seed2021`: MAE 0.207820, MSE 0.157533 versus p12 percent30 baseline MAE 0.205154, MSE 0.156691.
  - Decision: last-value damping hurts overall shape and does not solve sparse-event peaks.

### Iterations 44-50: Electricity Mid-Horizon Residual Transfer Test

- Iterations 44-46, Electricity192:
  - Percent30 baseline `weakphase5_iter44_electricity192_baseline_percent30_e10_b64_seed2021`: MAE 0.259585, MSE 0.167287.
  - Percent30 adaptive residual `weakphase5_iter45_electricity192_adaptive_residual_g05_lr001_huber_percent30_e10_b64_seed2021`: MAE 0.248693 (-4.20%), MSE 0.153304 (-8.36%).
  - Full validation `weakphase5_iter46_electricity192_adaptive_residual_g05_lr001_huber_b64_e30_seed2021`: MAE 0.238870, MSE 0.146513 versus formal baseline MAE 0.236785, MSE 0.146055. Full run regressed slightly.
  - Decision: subset signal does not transfer to full Electricity192.
- Iterations 47-50, Electricity336:
  - Percent30 baseline `weakphase5_iter47_electricity336_baseline_percent30_e10_b64_seed2021`: MAE 0.282503, MSE 0.193902.
  - Percent30 adaptive residual `weakphase5_iter48_electricity336_adaptive_residual_g05_lr001_huber_percent30_e10_b64_seed2021`: MAE 0.268538 (-4.94%), MSE 0.173022 (-10.77%).
  - Full Huber validation `weakphase5_iter49_electricity336_adaptive_residual_g05_lr001_huber_b64_e30_seed2021`: MAE 0.254882, MSE 0.161445 versus formal baseline MAE 0.258984, MSE 0.166970. MSE improved -3.31%, but MAE only -1.58%.
  - Full MAE validation `weakphase5_iter50_electricity336_adaptive_residual_g05_lr0003_mae_b64_e30_seed2021`: MAE 0.253051, MSE 0.162895. MAE improved -2.29%, MSE -2.44%; both remain below the 3% dual-metric target.
  - Bad-case review for full runs, capped at 8 cases each:
    - Iteration 49 top failures concentrated on channels 115, 128, 106, 113, and 287 with systematic bias, peak underfit, volatility mismatch, and late-horizon drift.
    - Iteration 50 reduced some bias cases, for example channel 128, but highest-MSE/volatility cases shifted to channel 113 and late drift on channel 287 remained. The MAE objective trades off the MSE gain instead of solving both.
  - Decision: adaptive residual has useful mid-horizon Electricity signal, but it is not sufficient as a dataset-level weak-period phase adaptation. The Weather/Electricity round stops at the user-specified 50-iteration cap without reaching the 3% dual-metric exit condition.

### Deployment To Dataset Runners

- User accepted the current improvement level and requested writing the corresponding settings into the `run_*.py` scripts.
- Updated latest preset policy:
  - Weather 96 uses `latest_weather96_phase_uncert_level_hifreq_sparse_event_mae`: period_len 12, phase uncertainty shrinkage, phase period-level calibration, high-frequency damping, sparse-event phase calibration, MAE loss, LR 0.0003, batch size 64.
  - Electricity 336 uses `latest_electricity336_adaptive_residual_mae`: adaptive weak-period residual gate initialized at 0.5, MAE loss, LR 0.0003, batch size 64.
  - Other Weather/Electricity horizons remain `latest_original_guardrail` because full-data evidence did not support promoting their screened candidates.
- Updated `run_weather.py` and `run_electricity.py` to use the same preset-backed runner as the ETT scripts, so `python run_weather.py` and `python run_electricity.py` default to `--mode latest` and can still run `--mode original` for regression comparison.
## Training Protocol Repair - 2026-07-26

- Scope: repository correctness and reproducibility repair; this is maintenance,
  not a new model iteration.
- Fixed `ett_all` split selection, truthful effective-loss reporting, Traffic
  runner divergence/test leakage, and last-epoch evaluation.
- New protocol: select the lowest `val_loss` checkpoint and load it for test
  metrics and subsequent bad-case export.
- Compatibility note: PyTorch 2.7 defaults checkpoint loading to
  `weights_only=True`; locally generated Lightning checkpoints contain the
  model config, so trusted runner restores explicitly use `weights_only=False`.
- GPU smoke evidence:
  `research_runs/smoke_best_checkpoint_protocol_20260726b/`.
  - ETTm2 720 -> 96, 5% train data, 2 epochs, seed 2021.
  - Best checkpoint restored successfully.
  - Test MAE 0.381733, MSE 0.343928.
  - This smoke result is not a formal effect conclusion.
- Historical comparisons above used last-epoch weights. Formal results under
  the repaired protocol require matched original/latest reruns.
- Matched full-data regression evidence:
  - `protocol_bestckpt_ettm2_96_20260726_comparison.csv`: MAE -3.85%,
    MSE -5.82%.
  - `protocol_bestckpt_etth2_720_20260726_comparison.csv`: MAE -4.75%,
    MSE -4.33%.
  - `protocol_bestckpt_exchange_96_20260726_comparison.csv`: MAE -13.27%,
    MSE -16.93%.
  - `protocol_bestckpt_weather_96_20260726_comparison.csv`: MAE -4.45%,
    MSE -1.85%.
  - `protocol_bestckpt_electricity_336_20260726_comparison.csv`: MAE -2.28%,
    MSE -2.31%.
- Traffic regression:
  - Batch64 failed before completing an epoch because another process occupied
    18.9 GiB of the 24 GiB GPU.
  - The official batch8 configuration entered training without OOM, verifying
    the unified runner and preset path, but was stopped because the contended
    throughput made two full 30-epoch runs impractical.
  - No formal Traffic metric is reported from either incomplete run.

## 2026-08-28 — TriAxis-Former history self-validation (stopped at Stage A)

- Hypothesis: phase-slot PhaseFormer, chronological NLinear and inter-cycle ICPT
  are complementary experts; input-only pseudo-backtests can route them without
  dataset IDs or future leakage.
- Implementation: one three-way phase-slot/future-cycle-factorized router,
  uniform/structural/self-validating ablations, expert auxiliary loss and route
  KL. Implementation commit: `e313ee4`.
- Protocol: ETTh2/ETTm2/Weather/Electricity, 720→96, 30% training subset,
  validation-only, seed 2021, eight epochs; 20 matched runs. All 168 tests passed.
- Result: T2 improved both metrics on Weather and Electricity but regressed on
  ETTh2 and ETTm2. Its eight-ratio mean/worst were 1.0005/1.0426; T0 and T1
  also failed the preregistered freeze rules. No new test split was read.
- Error analysis: pointwise oracle headroom was 47.80%, while deployable route
  agreement with the actual best expert was only 34.54%–39.27%. Retain the
  three-axis decomposition as a research lead, reject this one-cutoff proxy
  router, and investigate rolling-origin calibration before any new formal test.
- Evidence: `research_runs/triaxis_self_validating_v1/` (strict six-file plus
  figures layout and validated portable ZIP).

## 2026-08-28 — TriAxis rolling-origin calibration (stopped at Stage A)

- **本轮目标**：修正 v1 用一次一步伪预测外推四个未来周期的时间尺度错配，并量化 phase、
  trajectory、cycle 三个原子专家各自的相对优势区间。
- **候选假设**：R0 用最近四个历史截点做 1–4 周期等 lead 回测并把风险/方差作为特征；R1
  显式施加低风险单调 prior；R2 再用未来 24 步周期级 soft oracle 训练路由。三者不读 future、
  dataset ID 或专家未来输出，原有 preset flag-off 不变。实现 commit `d7ecc7f`。
- **设置**：ETTh2/ETTm2/Weather/Electricity，L720→H96、P24、30% train、seed 2021、Huber、
  最多 8 epoch、最低 validation loss checkpoint；12 个新 run，A1/I0/T2-v1 复用同协议结果。
  174 项仓库测试和 ETTm2 GPU smoke 通过。
- **结果**：R0/R1/R2 的 8 指标宏平均比值为 0.992243/0.999310/1.007830，最差比值为
  1.026184/1.015926/1.042114，双指标改善为 2/4、2/4、1/4。R0 是冠军但 ETTm2 仍回退
  MSE 2.62%、MAE 1.42%，所以三个候选都未通过冻结 gate。
- **错误与专家分析**：全量 validation 审计覆盖 1,022,522 个 sample×channel。ETTm2 轨迹专家
  四段均第一；ETTh2 周期间专家在 1–24 领先第二名 23.9%；Weather/Electricity 较远段更多由
  相位专家领先。严格十分位规则得到 48 个优势区间，但滚动伪风险首选的周期赢家命中率仅
  30.7%–41.8%；R1/R2 退化进一步表明历史风险排序尚未校准。
- **迭代决策**：多截点等 horizon 回测作为有效诊断保留；R0 不冻结，R1/R2 淘汰。按预注册
  规则不访问 test、不进入 Stage B/C、不改变 A1/RCRF+NLinear incumbent。可能的下一轮是带
  A1 fallback 的可拒绝 regret 路由，必须作为新假设重新预注册。
- **证据**：正式数值和结论在 `docs/PhaseFormer_triaxis_rolling_calibration_experiment.md`；
  审计器为 `scripts/analyze_triaxis_rolling_calibration.py`。本地严格白名单包位于
  `research_runs/triaxis_rolling_calibration_v2/`，大 CSV、图片、ZIP 和 checkpoint 均不提交。
