# strict-T28 master table run configs


Layout: `<Dataset>/<h<horizon>>/config.json` + `commands.sh` for every cell of the strict-T28 master table that has an on-disk run.

Collected:
- `ETTh2/h96` <- `confirm_etth2_h96_pctf_anchor_repair_strict_t28_p24_cp48_base_huber_lr0.001_pct100_e30_s2021_831747779dd5`
- `ETTh2/h192` <- `confirm_etth2_h192_pctf_anchor_repair_strict_t28_p24_cp48_base_huber_lr0.001_pct100_e30_s2021_b20f47c69597`
- `ETTh2/h336` <- `confirm_etth2_h336_pctf_anchor_repair_strict_t28_p24_cp48_base_huber_lr0.001_pct100_e30_s2021_b7f9cdf1191e`
- `ETTh2/h720` <- `confirm_etth2_h720_pctf_anchor_repair_strict_t28_p24_cp48_base_huber_lr0.001_pct100_e30_s2021_bccee3666b1c`
- `ETTm2/h96` <- `confirm_ettm2_h96_pctf_anchor_repair_strict_t28_p24_cp24_base_huber_lr0.001_pct100_e30_s2021_6816daf02adc`
- `ETTm2/h192` <- `confirm_ettm2_h192_pctf_anchor_repair_strict_t28_p24_cp24_base_huber_lr0.001_pct100_e30_s2021_bb087bb21343`
- `ETTm2/h336` <- `confirm_ettm2_h336_pctf_anchor_repair_strict_t28_p24_cp24_base_huber_lr0.001_pct100_e30_s2021_1ea01749c210`
- `ETTm2/h720` <- `confirm_ettm2_h720_pctf_anchor_repair_strict_t28_p24_cp24_base_huber_lr0.001_pct100_e30_s2021_504a7ddfc8a2`
- `Weather/h96` <- `confirm_weather_h96_pctf_anchor_repair_strict_t28_p24_cp24_base_mae_lr0.002_pct100_e30_s2021_b523c7f8481a`
- `Weather/h192` <- `confirm_weather_h192_pctf_anchor_repair_strict_t28_p24_cp24_base_mae_lr0.002_pct100_e30_s2021_31511dff7496`
- `Weather/h336` <- `confirm_weather_h336_pctf_anchor_repair_strict_t28_p24_cp24_base_mae_lr0.002_pct100_e30_s2021_05b7652b8fa8`
- `Weather/h720` <- `confirm_weather_h720_pctf_anchor_repair_strict_t28_p24_cp24_base_mae_lr0.002_pct100_e30_s2021_30bd2c30d7a3`

Not collected (no run data on this machine / not run):

- `ETTh1` (registered from external search results): u_lr020: cycle=24, caps 1.40/0.80/0.40, MAE, lr multiplier 0.20, ep50 (no run data on this machine)
- `ETTm1` (registered from external search results): w_aux01: cycle=24, caps 0.60/0.24/0.12, MAE, lr multiplier 0.20, shape/level/gate aux=0.01, ep50 (no run data on this machine)
- `Electricity` / `Traffic`: CANCELLED (not run).

Note: ETTh2/ETTm2 cells hold the seed-2021 representative of the 3-seed Stage D runs; seeds 2022/2023 runs remain in `research_runs/pctf_strict_t28_global_golden_v1/` and the manifest. The exact per-run invocation is preserved in each `commands.sh`.

