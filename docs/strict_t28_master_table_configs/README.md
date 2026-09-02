# strict-T28 master table run configs

Layout: `<Dataset>/<h<horizon>>/config.json` + `commands.sh` for every cell of
the strict-T28 master table. 20 settings registered (5 datasets x 4 horizons);
12 cells carry on-disk config/command files here.

Collected (config.json + commands.sh, 12 cells):

- `ETTh1/h96` `h192` `h336` `h720` — ETTh1 shared best (`u_lr020`): cycle=24,
  caps 1.40/0.80/0.40, MAE, lr multiplier 0.20, ep50, seed 2021. Copied from
  the other machine (`strict_t28_dataset_best/`), no run data on this machine.
- `ETTm1/h96` `h192` `h336` `h720` — ETTm1 shared best (`w_aux01`): cycle=24,
  caps 0.60/0.24/0.12, MAE, lr multiplier 0.20, shape/level/gate aux=0.01,
  ep50, seed 2021. Copied from the other machine.
- `ETTh2/h96` `h192` `h336` `h720` — seed-2021 representative of the 3-seed
  Stage D runs (C tier / huber / lr 0.001 / cycle 48).
- `ETTm2/h96` `h192` `h336` `h720` — seed-2021 representative of the 3-seed
  Stage D runs (C tier / huber / lr 0.001 / cycle 24).

Not collected (no run / cancelled):

- `Electricity` / `Traffic`: CANCELLED (not run).

Notes:

- ETTh1/ETTm1 files use the "best-config record" schema (`purpose` /
  `training` / `overrides` / `fixed_auxiliary_weights`); the other datasets
  use the runner's `config.json` schema (`hyperparams` / `config_hash`).
  Verified on this machine: every ETTh1/ETTm1 config matches its commands.sh,
  and the recorded `learning_rate` equals the runner's base LR x 0.2
  (ETTh1 H720 uses the long-horizon base 0.00015 -> 3e-05; all other cells
  0.001 -> 0.0002).
- ETTh2/ETTm2 cells hold the seed-2021 representative of the 3-seed Stage D
  runs; seeds 2022/2023 runs remain in
  `research_runs/pctf_strict_t28_global_golden_v1/` and the manifest. The
  exact per-run invocation is preserved in each `commands.sh`.
