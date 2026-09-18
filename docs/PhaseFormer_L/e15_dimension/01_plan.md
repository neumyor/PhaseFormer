# E15 / minipaper §4.3 — dimension of the phase-complement subspace (plan)

Artifact: `scripts/phaseformer_L/e15_dimension.py` (new, single script).
Output root: one directory passed by `--output-root`.

## 1. What §4.3 asks for

`docs/PhaseFormer_L_minipaper.md:423-431`: a 28-row table (ETTh1, ETTh2, ETTm1, ETTm2, Weather,
Electricity, Traffic × H ∈ {96,192,336,720}) with, per setting:
`λ_1/Σλ`, `pred_dims_90`, `PR`, `b_1` best template (|cos|), `a_1` vs constant |cos|,
`used_var_share(1)` — plus three figures (scree, `b_1` vs lag, `a_1` vs horizon). §4.0
(`:345-346`) restricts §4.3 to train/validation: **the test split is never read**.

Expected magnitudes to check against: `λ_1/Σλ` ≈ 0.66–0.86, `pred_dims_90` = 2–4, `PR` 1.33–2.12,
`a_1` vs const 0.89–0.99, `b_1` last-24 mass 0.53–0.70, best template = exp kernel τ=6–72 with
|cos| 0.58–0.78 (`PhaseFormer_rank_capacity_and_data_property_report.md` §2.1/§2.6, §4.1 of the minipaper).

## 2. Definitions inherited (nothing redefined)

| quantity | formula | source |
|---|---|---|
| split borders, train-only standardization | 12/4/4×30-day (ETT h/m) or 70/10/20 (custom), mean/population-std from train | `scripts/analyze_optimal_lowrank_capture.py:97-122` |
| task pair | `Z = x_window − x_last` (720), `D = y_horizon − x_last` (H) | `analyze_optimal_lowrank_capture.py:125-139` |
| `Szz,Szy,Syy` | mean over all (window, channel) pairs; `n` counts pairs, not windows | `analyze_optimal_lowrank_capture.py:142-180` |
| RRR fit | `ols=(Szz+ridge I)^{-1}Szy)^T`, `S=Szy^T(Szz+ridge I)^{-1}Szy`, symmetrize, `eigh`, descending, clip at 0, `W_r=U_rU_rᵀ ols`, ridge 1e-8 | `analyze_optimal_lowrank_capture.py:183-211`, `:312` |
| `λ_1/Σλ` | `vals[0]/vals.sum()` | `scripts/describe_leading_direction.py:93` |
| `pred_dims_90` | `searchsorted(cumsum(λ)/Σλ, 0.90)+1` | `analyze_optimal_lowrank_capture.py:394` |
| `PR` | `(Σλ)²/Σλ²` | `analyze_optimal_lowrank_capture.py:397` |
| `b_1`,`a_1` | `a_1 = u_1` (leading eigenvector of `S`), `b_1 = u_1 ols`, both L2-normalized | `describe_leading_direction.py:49-59,88-90` |
| `a_1` vs const | `abs(u_1 · 1/√H)` (legacy `out_cos_const`) | `describe_leading_direction.py:171,177` |
| `used_var_share(1)` | `trace(BᵀB Szz)/trace(Szz)`, `B` = right singular vectors of `W_1` (rank-1 ⇒ = `lead_dir_var_share`) | `analyze_optimal_lowrank_capture.py:348-350` |
| tested rank grid | `TESTED_RANKS` | `analyze_optimal_lowrank_capture.py:75-80` |
| row-space spectra, `direction_stats`, `spectral_stats`, `tail_fit`, band shares, dictionary R², output profile | verbatim | `analyze_optimal_lowrank_capture.py:83-94,214-230`; `describe_leading_direction.py:62-70,120-181` |

**One added definition (documented):** `b_1` best template = `argmax |cos(b_1, exp(−lag/τ))|` over
τ ∈ {6,24,72,168}, `lag = L−1−i`. This family was validated by reproducing all seven published
values of `PhaseFormer_rank_capacity_and_data_property_report.md` §2.6(c) from the v2 moments
(0.58/0.58/0.67/0.67/0.66/0.74/0.78 with τ=6/24/6/6/6/24/72). The half-life convention
`exp(−lag·ln2/τ)` (as in `scripts/lowrank_checkpoint_core.py:164-169`) does **not** reproduce them.
A finer τ grid and the const/ramp/tail templates are also computed and recorded in the JSON summary.

`train_mse` uses the closed-form λ identity `MSE_r = (tr(Syy) − Σ_{i≤r} λ_i)/H` (`report` §1.5)
instead of a second pass over the train split; verified against the artifact.
`val_mse` uses the eigen-projection identity `W_r z = U_r(U_rᵀ ols z)` — algebraically the legacy
`d − z W_rᵀ`, but one validation pass instead of one per rank; verified equal to direct evaluation.

## 3. Split convention and the test-split guarantee

ETT h/m: `[0, 12·30·day)` train, `[12·30·day, 16·30·day)` validation; custom: first 70% train,
next 10% validation, last 20% test. Standardization uses train-split mean and population std
(ddof=0, ≡ sklearn `StandardScaler` as used by `src/dataset/data_loader.py`), zero-std → 1.
The CSV is parsed with `nrows = validation border`, so **rows at or after the test border are never
loaded at all** (only the date column is counted once, to locate the borders). The val segment is
used for `optimal_rank_capture.csv`'s `val_*` columns only; no §4.3 column depends on it.
The dataset registry (paths) is read from `src/dataset/data_info.py::DATASET_INFO`, extended to all
7 datasets via its `data` field; `--data-root` only replaces the leading `.../all_datasets/` part.

## 4. Streaming memory strategy (channel-independent)

Per setting: loop over (window block × channel block); each block materializes
`Z,D` of size `chunk × channels_in_block × (L+H)` float64, subtracts `x_last` **in place**, then
adds `ZᵀZ, ZᵀD, DᵀD` into the 720×720 / 720×H accumulators. Channels only ever enter as additive
terms, so results are independent of the blocking (verified: `--channel-block 1` and auto give
byte-identical CSVs). This is what makes Electricity (321 ch) and Traffic (862 ch) fit: nothing of
size `n_windows × H × channels` is ever allocated (the 13.9 GiB `(17344, 336, 321)` allocation is
structurally impossible here). Also: RRR maps are materialized only for the requested ranks (all
ranks would be 2.9 GiB at H=720). Peak memory ≈ `--mem-budget-mb` (default 512). `--max-channels`
caps channels for debugging only and changes every moment/metric.

## 5. CLI

`--datasets --horizons --output-root --seq-len --data-root --chunk --mem-budget-mb --channel-block
--ridge --max-channels --ranks --save-moments --no-val-mse --skip-figures --verify-existing
--reference-dir --quiet`.
`--verify-existing` defaults to the 7 settings already in `--reference-dir`.

## 6. Output schema (under `--output-root`)

* `dimension_table.csv` — 28 rows (7 datasets × 4 horizons):
  `dataset, horizon, source, lambda1_share_of_achievable, pred_dims_90, PR, b1_best_template,
  b1_best_template_abs_cos, a1_vs_const_abs_cos, used_var_share_r1`.
  `source = reused_v2_artifact` for the 7 existing settings (ETTh2-96/720, ETTm2-96/192,
  Weather-96/192, Electricity-336), `new_28_minus_7` for the other 21. The §4.3 "template (|cos|)"
  cell is split into two fields; no other column is added.
* `leading_direction.csv` — byte-compatible with `research_runs/lowrank_data_property_v2/leading_direction.csv` (23 cols).
* `optimal_rank_capture.csv` — same 16 cols as the v2 file (per-row `val_*` from the validation split).
* `figures/scree_lambda_spectrum.png`, `figures/b1_lag_profile.png`, `figures/a1_horizon_profile.png`.
* `dimension_summary.json` (also printed as the machine-readable JSON summary), `leading_directions.npz`,
  `moments_<dataset>_h<H>.npz` (legacy key layout) with `--save-moments`, `verify_existing.json` with `--verify-existing`.

## 7. Static-check checklist (next stage)

1. `PYTHONDONTWRITEBYTECODE=1 python3 -c "compile(open('scripts/phaseformer_L/e15_dimension.py').read(),'x','exec')"` passes.
2. `--help` exits 0 and lists all required flags (checked; matplotlib import happens after parsing).
3. `dimension_table.csv` has header exactly as §6 and **28** rows; `source` splits 7/21; `expected_28_rows` is true.
4. No column is a placeholder: each field is written in `main()` from computed values.
5. `grep -n "test" scripts/phaseformer_L/e15_dimension.py` shows only border arithmetic and `nrows` — the test split is never indexed.
6. `--verify-existing` reports `passed: true` with per-setting moments rel-diff < 1e-6 (expected ~0).
7. `λ` identity: `train_mse` matches the v2 artifact within its 1e-8 rounding (worst observed 4.7e-09, 46 cells).
8. `--channel-block 1` vs auto produce identical CSVs (blocking invariance).
9. The 7 reused rows reproduce the published §2.6(c) best-template values (checked in `--verify-existing`).
10. Figures exist, are non-empty PNGs, and the scree panel shows λ≈0.66–0.86 for i=1 with 2–4 directions at 90%.
11. Commands for the real run are recorded in the script docstring / the E15 report.

## 8. Verification already done on this machine (no server, no test data)

* All 23 columns × 7 rows of the v2 `leading_direction.csv` reproduced from the v2 moments (0 diffs).
* λ identity over 46 cells: worst abs diff 4.718e-09 (identical to the report's §1.5 figure).
* End-to-end run on synthetic CSVs in the registered layout: 28/28 rows, 3 figures, all outputs written.
* Legacy `analyze_optimal_lowrank_capture.py` re-run on that same synthetic data: moments **bit-identical**,
  all 16 columns of `optimal_rank_capture.csv` equal (worst 1e-8 = CSV rounding, on `train_mse` only),
  `leading_direction.csv` bit-identical to the legacy `describe_leading_direction.py` output.
* `--verify-existing` path exercised (moments diff 0.0, 111/111 metric comparisons pass).
