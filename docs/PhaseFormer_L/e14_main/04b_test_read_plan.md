# E14 / minipaper §4.2 — stage B: the single test read (04b plan)

Artifact: `scripts/phaseformer_L/e14_read_test.py`. Output root: the stage-A root passed via `--output-root`
(e.g. `research_runs/phaseformer_L_e14_main_v1`). Stage B is the second half of §3's stage 4: it reads test for
every cell stage A trained and copies the registered read for every reused cell.

## 1. What §4.0 requires about the single test read

`docs/PhaseFormer_L_minipaper.md:340`: "每 checkpoint 只读一次 test".
`docs/PhaseFormer_L_execution_schedule.md:247`: "test 读取 | 每 checkpoint **只读一次**；`--evaluate-test` 仅
`--stage confirm` 允许". The read is a distinct *final* step after all training and audits (the E8 template's
framing, kept by schedule §3's stage-4 gate); `PhaseFormer_L_minipaper.md:345-346` draws the blind-test boundary.

Stage A (`scripts/phaseformer_L/e14_main_matrix.py`) never passes `--evaluate-test`, so every run it trains has
empty `test_mse`/`test_mae` in `metrics.csv` and the manifest records `reads_test: false`. Stage B is the only
consumer of the test split and enforces three consequences:

1. one read per **new** cell, after the checkpoint is frozen;
2. the read happens **only if** the validation MSE re-derived from the restored checkpoint reproduces the run's
   recorded `val_mse` within `VAL_REPRODUCE_TOL = 1e-3`; otherwise the cell is `rejected` with no test number
   (the read is not consumed, so a later corrected attempt is legitimate);
3. `reused` cells are never rebuilt and never re-read — their numbers are copied from the registration stage A
   audited (`EXTERNAL_TEST_EVIDENCE`, or the source run's own inline `metrics.csv`).

## 2. Inherited from the E8 template, and where it did not generalize

Template: `scripts/read_top2_direction_retention_test.py`. Reused in shape: `find_run` by reading
`config.json`; `build_model` via `make_exp_args` + `PhaseFormerPresetConfig`; best-checkpoint restore and its `unexpected_keys` guard; `evaluate_once`'s fused **and** branch-private MSE/MAE (`nlinear_mse`/`nlinear_mae`)
plus `gate_value`; the validation-reproduction gate with `VAL_REPRODUCE_WARN = 1e-4` / `VAL_REPRODUCE_TOL =
1e-3`; `resolve_checkpoint`'s relocatable-path fallbacks; `test_read_summary.json`; per-GPU subprocess dispatch.

Differences — (c)–(i) are places where the template did **not** generalize cleanly:

| # | E8 | E14 stage B |
|---|---|---|
| a | arm from `weak_residual_projection_arm` / mechanism, 4 arms | 6-arm fingerprint mirroring `ARMS` + `_arm_match()`; projection arms excluded and reported as near misses |
| b | `gate_value` = static `learned_residual_gate()` | `l_rcrf`/`a1` use RCRF fusion, which has **no** gate parameter (alpha is per-sample), so E8 would record `null`; E14 records the mean input-dependent weight with `gate_value_source` = `static_gate`/`input_dependent_mean`/`none` |
| c | test computed **before** the tolerance gate, so a `val_mismatch` still consumed a read | gate first: `rejected` cells never read test |
| d | idempotence only via `metrics.csv` columns its own read never writes → re-running E8 re-read test | per-cell artifact `test_read/<key>.json` + consumed-status whitelist: `read`/`val_drift`/`already_read` are never re-read |
| e | reuse index from E8's own `reuse_audit.json` | reuse read from the manifest's `source.test_evidence`; registered evidence rows are re-read and cross-checked against the manifest copy |
| f | omitted the runner's `set_float32_matmul_precision("medium")` and ETT `root_path → ETT-small` fallback | both mirrored: they change the numerics and which files are read |
| g | checkpoint fallback hardcoded `attempts/001` | globs `attempts/*/checkpoints/<name>` (E3 lineage keeps later attempts) |
| h | a run without recorded `val_mse` was accepted (gate silently skipped) | `rejected` — the checkpoint/protocol cannot be verified |
| i | `numpy`/`torch` imported at module top level | imported lazily inside the GPU paths, so `--dry-run` pre-flights without torch and writes nothing |

## 3. Arm fingerprint (mirrors `e14_main_matrix.ARMS` / `_arm_match()`, compared at every startup)

Common exclusions for all arms: `weak_residual_projection` set, `input_hypothesis` ∉ {none, ""},
`weak_period_residual_smooth_ratio != 0`, `init_checkpoint` set.

| arm | `mechanism` | `weak_period_residual_head_type` | rank / pool |
|---|---|---|---|
| `phase_only` | `no_residual` | — | — |
| `l_main` | `weak_residual` | `shared` | — |
| `l_q1_4` | `weak_residual` | `pooled_lowrank` | `rank = H//4`, `pool_factor = 1` |
| `l_q1_8` | `weak_residual` | `pooled_lowrank` | `rank = H//8`, `pool_factor = 1` |
| `l_rcrf` | `rcrf_nlinear_plain` | — | — |
| `a1` | `gold_combo_reliability_s2` | — | — |

Protocol fields re-checked per new cell (`_protocol_ok`): `lookback=720`, `loss=huber`, `max_epochs=30`, `percent=100`, `period=24`; a mismatch is `rejected` unless `--ignore-protocol-drift`.

## 4. Exact CLI

```bash
# stage-2 gate: manifest + fingerprint + run location + reuse evidence; no torch, writes nothing
python scripts/phaseformer_L/e14_read_test.py \
  --manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json \
  --output-root research_runs/phaseformer_L_e14_main_v1 --dry-run
# the single test read: one cell per GPU, one retry, 15 s poll
python scripts/phaseformer_L/e14_read_test.py \
  --manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json \
  --output-root research_runs/phaseformer_L_e14_main_v1 \
  --gpus 0,1,2,3,4,5,6,7 --retries 1 --poll-seconds 15 --num-workers 4
```

`--gpus` empty ⇒ sequential in-process. Single cell for debugging: `--cell arm:dataset:horizon:seed` (E8's
`dataset:horizon:seed:arm` also accepted). `--worker` is internal. JSON events: `launch`, `done`, `failed`,
`retry`, `planned`, `finished`.

## 5. Output schema (all under `--output-root`)

* `test_read_summary.json` — `protocol` (one read per new checkpoint; reused cells never re-read; the already-read
  guard; `val_gate` incl. "gate runs before the test read"; fixed protocol fields; arm fingerprint; `excluded_runs`), plus `manifest`, `run`, `fingerprint_check`, `counts`, `problems`,
  `warnings`, `failed_workers`, and one detail record per cell: `key, arm, dataset, horizon, seed, setting,
  status, reason, test_mse, test_mae, nlinear_mse, nlinear_mae, gate_value, gate_value_source, val_mse,
  recorded_val_mse, val_relative_difference, test_size, run_dir, config_hash, checkpoint, source, device,
  gpu_name, data_root, eval_count, val_size, protocol_failures, matched_run_dirs, near_miss_runs,
  warnings, read_at`.
* `results.csv` — one row per cell, exactly `arm,dataset,horizon,seed,setting,status,test_mse,test_mae,nlinear_mse,nlinear_mae,gate_value,val_mse,recorded_val_mse,val_relative_difference,test_size,run_dir,config_hash,source`;
  `setting = "<dataset>-<horizon>"`. For `reused` rows `val_mse` is blank (nothing was recomputed) and
  `recorded_val_mse` is the source's registered validation MSE; `source` is the evidence file or
  `<run_dir>/metrics.csv`, and `stage_b_single_test_read` for new cells.
* `test_read/<key>.json` (idempotence guard) and `_logs/test_read_<key>.log` per worker.

## 6. Static-check checklist

1. `PYTHONDONTWRITEBYTECODE=1 python3 -c "compile(open('scripts/phaseformer_L/e14_read_test.py').read(),'p','exec')"` → clean (not `py_compile`: it writes bytecode outside the workspace).
2. `python3 -m pyflakes scripts/phaseformer_L/e14_read_test.py` → clean.
3. `--dry-run` on the real manifest → `fingerprint_check.constants_equal = true`, `parity_cases = 20`, `parity_failures = []`.
4. `--dry-run` `by_arm`/`by_status` must match §2.2 (per arm `new + reused = settings × 3`; reused = 18/21/21/21/0/0) and stage A's `reuse_summary`.
5. `--dry-run` problems must be 0 and every `new` cell must resolve to a run with an existing checkpoint (`missing_run` ⇒ stage A is incomplete; do not proceed).
6. `--dry-run` must create no file under `--output-root` (compare `find <root> -type f | sort` before/after).
7. `--cell <tok> --dry-run` for one cell of each arm → `planned` for new, `reused` for reused.
8. Missing manifest / unknown cell token / unknown arm → non-zero exit with a readable reason, not a traceback.
9. After the real run: `results.csv` header equals §5's 18 columns exactly and there is one row per manifest cell.
10. Every `rejected` row has empty `test_mse`/`test_mae` and a `reason` naming the failing check.
11. Every `reused` row's `test_mse`/`test_mae` equals the registered value in its `source` (E8 rows: `phase_only`→`phase_only`, `l_main`→`direct_nlinear`).
12. New rows: `status ∈ {read, val_drift}`, `val_relative_difference ≤ 1e-3`, `test_size > 0`, `gate_value_source` set, `nlinear_*`/`gate_value` present for every arm except `phase_only` (no residual branch).
13. Re-running the same command reports `already_read_from_artifact` for consumed cells and touches no checkpoint again.
14. Confirm the `protocol` block and `run.gpus` before the §4.2 write-back; `research_runs/` is a synced tree and is never committed.

## 3. 上线前实测：两条分支都在真产物上跑通（2026-09-20 05:2x）

第 1 步（阶段 B 单次 test 读取）有**两条互不相同的分支**，本轮在 E14 仍在训练时把两条都先验了一遍
（`--dry-run` 不写任何产物，`wrote_outputs: false`）：

**(a) `new` 格分支**（411 格）——取一个**已完成**的真实 run 跑一遍：

```text
{"event": "fingerprint_check", "constants_equal": true, "differences": [], "parity_cases": 20, "parity_failures": []}
{"event": "planned", ..., "cells_selected": 1, "manifest_counts": {"new": 411, "reused": 81, "other": 0},
 "external_evidence_rows": 36, "external_evidence_rejected": 0}
E14PLAN {"key": "l_main__Traffic-96-s2021", "plan": "rebuild from config.json + restore best checkpoint,
 re-derive val_mse, then read test exactly once", "status": "planned", ...}
{"event": "finished", "dry_run": true, "cells": 1, "accepted": 1, "problems": 0, "wrote_outputs": false}
```

即：**协议指纹与 `e14_main_matrix.py` 完全一致**（`parity_cases: 20`、0 处差异）⇒
不会在 `preflight_new_cell` 处因 protocol drift 被拒；该格解析到正确的 run 与 checkpoint，
同时把同 setting 的另外 4 个臂（`no_residual` / `rcrf_nlinear_plain` / 两个 `pooled_lowrank`）
作为 **near-miss 带理由排除**——这正是"5 个臂共用一个 setting-seed、靠 config 指纹而不是目录名区分"的直接证据。

**(b) `reused` 格分支**（81 格）——用 `--cells-file`（行格式是 `arm:dataset:horizon:seed`，
**不是** manifest 的 `arm__Dataset-H-sSEED`）一次验完全部 81 格：

```text
reused cells whose source.test_evidence is empty: 0
{"event": "finished", "dry_run": true, "cells": 81, "accepted": 81, "problems": 0, "warnings": 0,
 "by_status": {"reused": 81}, "by_arm": {"l_main": 21, "l_q1_4": 21, "l_q1_8": 21, "phase_only": 18},
 "wrote_outputs": false}
```

即 81 格的登记证据**全部齐备**（`source.test_evidence` 无一为空；外部证据表 36 行、0 拒绝），
不存在"某个复用格没有 test 来源 ⇒ 第 1 步 fail-closed 卡住"的风险。
**至此第 1 步的两条分支都已在真产物上验证**；剩余不确定性只有**时长**（§9.1 估 ≈1.3 h，
依据是"单格 1–2 min × 411 / 8 卡"，且满规模解析已实测仅 23.75 s，见 §9.1.1a）。
