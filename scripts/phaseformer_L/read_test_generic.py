#!/usr/bin/env python3
"""Read the test split exactly once for a set of already-trained runs.

E14 got its own stage-B reader (`e14_read_test.py`), but E17 and E18 train cells
whose test metrics are still needed by the §4.5 and §4.6 tables, and neither
runner can produce them: both refuse `--evaluate-test` on purpose, marking the
column `pending_single_test_read`, and the "separate stage" they refer to was
never written.  This is that stage, kept generic so it works for any run this
line produces.

It takes a results CSV that names each run directory (E17 writes `run_dir` and,
for the frozen-subspace arms, `basis_file`), and for every row whose test metrics
are empty:

1. rebuild the model from the run's own ``config.json`` -- the config records the
   full override set, so no arm-specific fingerprint is needed here;
2. reinstall a frozen projection basis when the config says
   ``weak_residual_projection == "frozen_subspace"``, taking the basis from the
   ``basis_file`` column (the wrapper owns that path; the config only records
   that a projection was used);
3. recompute the validation MSE from the restored checkpoint and require it to
   match the recorded value (**this gate runs BEFORE the test read**, so a
   mismatched checkpoint consumes no test read and reports no test number);
4. evaluate the test split exactly once, recording the fused MSE/MAE, the
   residual branch's own MSE/MAE, and the fusion gate.

Re-running is idempotent: a row that already carries test metrics is copied
through untouched, and each freshly read cell leaves a per-cell JSON marker.

Usage::

    python scripts/phaseformer_L/read_test_generic.py \\
        --results research_runs/phaseformer_L_e17_conditional_v1/results.csv \\
        --gpus 0,1,2,3,4,5,6,7
    # writes <results>.with_test.csv next to the input
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VAL_REPRODUCE_WARN = 1e-4
VAL_REPRODUCE_TOL = 1e-3

TEST_COLUMNS = ("test_mse", "test_mae")
BRANCH_COLUMNS = ("nlinear_mse", "nlinear_mae")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="results CSV naming each cell's run_dir")
    parser.add_argument("--output", default="",
                        help="merged CSV; default <results>.with_test.csv")
    parser.add_argument("--marker-dir", default="",
                        help="per-cell JSON markers; default <results dir>/test_read")
    parser.add_argument("--gpus", default="", help="e.g. '0,1,2'; empty = CPU")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-tol", type=float, default=VAL_REPRODUCE_TOL)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0,
                        help="smoke only; a partial evaluation is flagged")
    parser.add_argument("--all-rows", action="store_true",
                        help="also re-read rows that already have test metrics")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", default="",
                        help="internal: process one row given as json")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# model reconstruction
# --------------------------------------------------------------------------

def build_model(config, checkpoint_path, basis_path, device):
    """Rebuild the trained model from its config and restore the checkpoint."""
    from src.dataset.data_factory import data_provider
    from src.models.PhaseFormer import PhaseFormer
    from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args

    hyper = dict(config["hyperparams"])
    exp_args = make_exp_args(
        config["dataset"], config["lookback"], config["horizon"], hyper,
        batch_size=config.get("batch_size") or None,
    )
    exp_args.dataset_args.percent = config.get("percent", 100)
    exp_args.dataset_args.num_workers = 4
    train_set, _ = data_provider(exp_args.dataset_args, "train")
    if hasattr(train_set, "data_stamp"):
        hyper["time_mark_dim"] = int(train_set.data_stamp.shape[-1])
    model = PhaseFormer(
        PhaseFormerPresetConfig(exp_args, config["lookback"], config["horizon"], hyper)
    )
    if hyper.get("weak_residual_projection") == "frozen_subspace":
        if not basis_path or not Path(basis_path).is_file():
            raise RuntimeError(
                "config requests a frozen projection basis but no readable "
                f"basis_file was supplied (got {basis_path!r})")
        basis = torch.as_tensor(np.load(basis_path), dtype=torch.float32)
        model.install_projection_basis(basis, source=str(basis_path))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    # The frozen-basis wrapper builds the head with the basis already installed,
    # so the checkpoint can carry that buffer; strict=False plus an explicit
    # unexpected-key check keeps a genuine mismatch loud.
    incompat = model.load_state_dict(state, strict=False)
    unexpected = [k for k in incompat.unexpected_keys
                  if "projection_basis" not in k]
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
    return model, exp_args


def evaluate_once(model, loader, split, target_var_index, max_batches=0):
    """Fused and branch-only MSE/MAE over one split, plus the gate."""
    model.eval()
    device = next(model.parameters()).device
    fused_sq = fused_abs = branch_sq = branch_abs = 0.0
    count = 0
    branch_count = 0
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            if max_batches and batches >= max_batches:
                break
            batches += 1
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(batch_x.float(), batch_x_mark.float(), dec,
                              batch_y_mark.float())
            pred = out[:, -model.pred_len:, :]
            true = batch_y.float()[:, -model.pred_len:, :]
            branch = model.last_residual_forecast
            if target_var_index != -1:
                true = true[:, :, target_var_index:target_var_index + 1]
                if branch is not None:
                    branch = branch[:, :, target_var_index:target_var_index + 1]
            err = pred - true
            fused_sq += float(err.pow(2).sum())
            fused_abs += float(err.abs().sum())
            count += err.numel()
            if branch is not None:
                berr = branch - true
                branch_sq += float(berr.pow(2).sum())
                branch_abs += float(berr.abs().sum())
                branch_count += berr.numel()
    gate = None
    getter = getattr(model, "learned_residual_gate", None)
    if callable(getter):
        gate = getter()
    result = {
        "split": split,
        "fused_mse": fused_sq / count if count else None,
        "fused_mae": fused_abs / count if count else None,
        "branch_mse": (branch_sq / branch_count) if branch_count else None,
        "branch_mae": (branch_abs / branch_count) if branch_count else None,
        "elements": count,
        "batches": batches,
        "gate_value": gate,
    }
    return result


def read_one(row: dict, args) -> dict:
    """Worker body: one cell.  Returns the per-cell record."""
    from src.dataset.data_factory import data_provider

    run_dir = ROOT / row["run_dir"] if not Path(row["run_dir"]).is_absolute() \
        else Path(row["run_dir"])
    record = {"cell": row.get("cell") or f"{row['arm']}__{row['dataset']}-"
                                        f"{row['horizon']}-s{row['seed']}",
              "run_dir": str(run_dir)}
    config = json.loads((run_dir / "config.json").read_text())
    metrics_path = run_dir / "metrics.csv"
    metrics = next(csv.DictReader(metrics_path.open(newline="")), {}) or {}
    recorded_val = metrics.get("val_mse", "")
    checkpoint = str(metrics.get("checkpoint", "")).strip()
    checkpoint_path = None
    for candidate in (ROOT / checkpoint, run_dir / checkpoint):
        if checkpoint and candidate.is_file():
            checkpoint_path = candidate
            break
    if checkpoint_path is None:
        for candidate in sorted(run_dir.glob("attempts/*/checkpoints/*.ckpt")):
            checkpoint_path = candidate
            break
    if checkpoint_path is None:
        record.update(status="missing_checkpoint")
        return record
    record["checkpoint"] = str(checkpoint_path.relative_to(ROOT))

    basis_path = str(row.get("basis_file", "") or "").strip()
    try:
        model, exp_args = build_model(config, checkpoint_path, basis_path, None)
    except Exception as error:  # noqa: BLE001 - reported per cell
        record.update(status="build_failed", error=repr(error))
        return record

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    target_var_index = int(getattr(model, "target_var_index", -1))
    val_set, val_loader = data_provider(exp_args.dataset_args, "val")

    # --- validation gate BEFORE any test read -----------------------------
    val = evaluate_once(model, val_loader, "val", target_var_index)
    record["recomputed_val_mse"] = val["fused_mse"]
    record["recorded_val_mse"] = float(recorded_val) if str(recorded_val).strip() else None
    if record["recorded_val_mse"] is not None and val["fused_mse"]:
        rel = abs(val["fused_mse"] - record["recorded_val_mse"]) / abs(
            record["recorded_val_mse"])
        record["val_relative_difference"] = rel
        if rel > args.val_tol:
            record.update(
                status="rejected",
                reason=(f"validation reproduction differs by {rel:.3e} > "
                        f"{args.val_tol:g}; the checkpoint or protocol does not "
                        "match, so the test split was NOT read"))
            return record
    else:
        record["val_relative_difference"] = None

    # --- the single test read --------------------------------------------
    test_set, test_loader = data_provider(exp_args.dataset_args, "test")
    test = evaluate_once(model, test_loader, "test", target_var_index,
                         max_batches=args.max_eval_batches)
    record.update({
        "status": "partial" if args.max_eval_batches else "read",
        "test_mse": test["fused_mse"],
        "test_mae": test["fused_mae"],
        "nlinear_mse": test["branch_mse"],
        "nlinear_mae": test["branch_mae"],
        "gate_value": test["gate_value"],
        "test_elements": test["elements"],
        "test_size": len(test_set),
        "test_read_once": True,
    })
    if args.max_eval_batches:
        record["partial_read_note"] = (
            f"--max-eval-batches {args.max_eval_batches} evaluated a subset; the "
            "number is NOT a full-split test result")
    return record


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def load_rows(path: Path, re_read: bool):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    todo, done = [], []
    for index, row in enumerate(rows):
        has_test = bool(str(row.get("test_mse", "")).strip())
        row["_index"] = index
        if has_test and not re_read:
            done.append(row)
        else:
            todo.append(row)
    return rows, fields, todo, done


def dispatch(todo, args, marker_dir: Path, results_path: Path):
    gpus = [g for g in str(args.gpus).split(",") if g.strip()]
    pending = list(todo)
    active: dict = {}
    attempts: dict = {}
    finished: dict = {}
    log_dir = marker_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or active:
        free = [g for g in gpus] if gpus else ["cpu"]
        free = [g for g in free if g not in active]
        while pending and free:
            slot = free.pop(0)
            row = pending.pop(0)
            key = f"{row['_index']:04d}"
            attempts[key] = attempts.get(key, 0) + 1
            marker = marker_dir / f"{key}.json"
            if marker.is_file():
                finished[key] = json.loads(marker.read_text())
                continue
            env = dict(os.environ)
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            payload = {k: v for k, v in row.items() if k != "_index"}
            log = open(log_dir / f"{key}.log", "w")
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker",
                 json.dumps(payload), "--results", str(results_path),
                 "--val-tol", str(args.val_tol),
                 "--max-eval-batches", str(args.max_eval_batches)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            print(json.dumps({"event": "launch", "cell": payload.get("cell"), "slot": slot}),
                  flush=True)
            active[slot] = (row, process, log, key)
        done_slots = []
        for slot, (row, process, log, key) in active.items():
            code = process.poll()
            if code is None:
                continue
            log.close()
            marker = marker_dir / f"{key}.json"
            if marker.is_file() and code == 0:
                finished[key] = json.loads(marker.read_text())
                done_slots.append(slot)
                print(json.dumps({"event": "done", "cell": row.get("cell")}), flush=True)
            elif attempts.get(key, 0) <= args.retries:
                pending.append(row)
                done_slots.append(slot)
                print(json.dumps({"event": "retry", "cell": row.get("cell"),
                                  "code": code}), flush=True)
            else:
                finished[key] = {"cell": row.get("cell"), "status": "worker_failed",
                                 "return_code": code}
                done_slots.append(slot)
                print(json.dumps({"event": "failed", "cell": row.get("cell"),
                                  "code": code}), flush=True)
        for slot in done_slots:
            del active[slot]
        if active:
            time.sleep(args.poll_seconds)
    return finished


def main() -> None:
    args = parse_args()
    if args.worker:
        payload = json.loads(args.worker)
        record = read_one(payload, args)
        print(json.dumps(record, ensure_ascii=False, default=str))

    else:
        results_path = Path(args.results)
        if not results_path.is_absolute():
            results_path = ROOT / results_path
        output = Path(args.output) if args.output else results_path.with_suffix(".with_test.csv")
        if not output.is_absolute():
            output = ROOT / output
        marker_dir = Path(args.marker_dir) if args.marker_dir else (
            results_path.parent / "test_read")
        if not marker_dir.is_absolute():
            marker_dir = ROOT / marker_dir

        rows, fields, todo, done = load_rows(results_path, args.all_rows)
        print(json.dumps({"event": "planned", "rows": len(rows),
                          "to_read": len(todo), "already_read": len(done),
                          "gpus": args.gpus or "cpu"}), flush=True)
        if args.dry_run:
            return

        finished = dispatch(todo, args, marker_dir, results_path)

        for column in TEST_COLUMNS + BRANCH_COLUMNS + (
                "gate_value", "test_read_status", "val_relative_difference"):
            if column not in fields:
                fields.append(column)
        rejected = []
        for row in rows:
            key = f"{row['_index']:04d}"
            record = finished.get(key)
            if not record:
                continue
            if record.get("status") in ("read", "partial"):
                row["test_mse"] = record.get("test_mse") or ""
                row["test_mae"] = record.get("test_mae") or ""
                row["nlinear_mse"] = record.get("nlinear_mse") or ""
                row["nlinear_mae"] = record.get("nlinear_mae") or ""
                row["gate_value"] = record.get("gate_value") or ""
                row["test_read_status"] = record.get("status")
                row["val_relative_difference"] = record.get(
                    "val_relative_difference", "")
            else:
                rejected.append({"cell": record.get("cell"),
                                 "status": record.get("status"),
                                 "reason": record.get("reason", record.get("error"))})
                row["test_read_status"] = record.get("status")
        for row in rows:
            row.pop("_index", None)
        marker_dir.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "event": "finished",
            "output_csv": str(output),
            "rows": len(rows),
            "read": sum(1 for r in rows if str(r.get("test_read_status")) in ("read", "partial")),
            "rejected": rejected,
            "protocol": {
                "single_read": "one test evaluation per newly read checkpoint",
                "gate_before_read": "validation reproduction is checked before the "
                                    "test split is touched",
                "idempotent": "rows with test metrics are copied through unless "
                              "--all-rows is given",
            },
        }
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
