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
- Result: pending.
