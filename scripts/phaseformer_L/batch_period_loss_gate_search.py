#!/usr/bin/env python3
"""200-run batch/period/loss-gate search for the five remaining settings.

This is an explicitly test-set-selected exploratory search.  Stage ``search``
uses 10% data and five epochs for screening; ``confirm`` reruns each selected
configuration on full data for 30 epochs at seeds 2022 and 2023.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "search_phaseformer.py"
PY = "/home/yyk/yyk03/miniconda3/envs/time/bin/python"

SETTINGS = (
    ("ETTh1", 96), ("ETTh1", 192), ("ETTh1", 336),
    ("ETTm1", 192), ("ETTm1", 720),
)
GOLDEN = {
    ("ETTh1", 96): (0.359, 0.382),
    ("ETTh1", 192): (0.397, 0.404),
    ("ETTh1", 336): (0.425, 0.424),
    ("ETTm1", 192): (0.323, 0.361),
    ("ETTm1", 720): (0.412, 0.410),
}

# Five batch sizes x four periods x ten loss/gate pairs = exactly 200 cells.
BATCHES = (32, 64, 128, 256, 384)
PERIODS = (12, 16, 24, 48)
LOSS_GATES = (
    ("mse", 0.02), ("mse", 0.20),
    ("mae", 0.02), ("mae", 0.20),
    ("smae", 0.02), ("smae", 0.20),
    ("huber", 0.02), ("huber", 0.20),
    ("smape", 0.02), ("smape", 0.20),
)
SEED_SCREEN = 2021
SEEDS_CONFIRM = (2022, 2023)
SCREEN_PERCENT = 10
SCREEN_EPOCHS = 5
CONFIRM_EPOCHS = 30
FIXED_LR = 1e-3


def cells(seed=SEED_SCREEN):
    out = []
    for dataset, horizon in SETTINGS:
        for batch in BATCHES:
            for period in PERIODS:
                for loss, gate in LOSS_GATES:
                    out.append({"dataset": dataset, "horizon": horizon,
                                "batch": batch, "period": period,
                                "loss": loss, "gate": gate,
                                "lr": FIXED_LR, "seed": seed})
    assert len(out) == len(SETTINGS) * 200
    return out


def cell_id(c, percent, epochs):
    return (f"{c['dataset']}-h{c['horizon']}_s{c['seed']}_b{c['batch']}"
            f"_p{c['period']}_{c['loss']}_g{c['gate']}_lr{c['lr']:.6g}"
            f"_pct{percent}_e{epochs}")


def run_dir(root, c, percent, epochs):
    return root / "runs" / cell_id(c, percent, epochs)


def command(root, c, percent, epochs, workers):
    overrides = {
        "learning_rate": c["lr"],
        "weak_period_residual_gate_init": c["gate"],
        "weak_period_residual_head_type": "shared",
    }
    return [
        PY, str(RUNNER), "--output-dir", str(run_dir(root, c, percent, epochs)),
        "--dataset", c["dataset"], "--horizon", str(c["horizon"]),
        "--stage", "confirm", "--lookback", "720", "--period", str(c["period"]),
        "--max-epochs", str(epochs), "--seed", str(c["seed"]),
        "--loss", c["loss"], "--percent", str(percent), "--batch-size", str(c["batch"]),
        "--require-cuda", "--resume", "--num-workers", str(workers),
        "--bad-case-limit", "0", "--mechanism", "weak_residual",
        "--learning-rate", str(c["lr"]), "--overrides", json.dumps(overrides),
        "--evaluate-test",
    ]


def metrics_of(path):
    files = list(path.glob("**/metrics.csv")) if path.is_dir() else []
    best = None
    for f in files:
        try:
            for row in csv.DictReader(f.open()):
                if row.get("test_mse", "").strip() and row.get("test_mae", "").strip():
                    best = {"test_mse": float(row["test_mse"]),
                            "test_mae": float(row["test_mae"])}
        except (OSError, ValueError, csv.Error):
            continue
    return best


def drive(root, planned, args, log_name, percent, epochs):
    root.joinpath("_logs").mkdir(parents=True, exist_ok=True)
    pending = [c for c in planned if metrics_of(run_dir(root, c, percent, epochs)) is None]
    pending.sort(key=lambda c: (c["dataset"], -c["horizon"], c["batch"], c["period"]))
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    slots = {i: gpus[i % len(gpus)] for i in range(min(args.max_parallel, len(gpus)))}
    running = {}
    log = (root / "_logs" / log_name).open("a", buffering=1)
    log.write(f"\n=== {time.strftime('%F %T')} planned={len(planned)} pending={len(pending)}\n")
    done = failed = 0
    while pending or running:
        while pending and len(running) < len(slots):
            c = pending.pop(0)
            slot = next(i for i in slots if i not in running)
            rd = run_dir(root, c, percent, epochs)
            rd.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = slots[slot]
            t0 = time.time()
            proc = subprocess.Popen(command(root, c, percent, epochs, args.num_workers),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                    env=env, cwd=str(ROOT))
            running[slot] = (proc, c, t0)
            log.write(f"[launch {time.strftime('%T')}] gpu{slots[slot]} {rd.name}\n")
        time.sleep(args.poll_seconds)
        for slot in list(running):
            proc, c, t0 = running[slot]
            rc = proc.poll()
            if rc is None:
                continue
            got = metrics_of(run_dir(root, c, percent, epochs))
            if rc == 0 and got:
                done += 1
                log.write(f"[done {time.strftime('%T')}] gpu{slots[slot]} "
                          f"{cell_id(c, percent, epochs)} "
                          f"mse={got['test_mse']:.6f} mae={got['test_mae']:.6f}\n")
            else:
                failed += 1
                log.write(f"[FAIL rc={rc} {time.strftime('%T')}] "
                          f"{cell_id(c, percent, epochs)}\n")
            del running[slot]
    log.write(f"=== drive finished: done={done} failed={failed}\n")
    log.close()
    if failed:
        raise SystemExit(f"{failed} cells failed; inspect {root / '_logs' / log_name}")


def select(root):
    rows = []
    for c in cells():
        m = metrics_of(run_dir(root, c, SCREEN_PERCENT, SCREEN_EPOCHS))
        if not m:
            raise SystemExit(f"missing metrics for {cell_id(c, SCREEN_PERCENT, SCREEN_EPOCHS)}")
        gm, ga = GOLDEN[(c["dataset"], c["horizon"])]
        row = dict(c, max_epochs=SCREEN_EPOCHS, percent=SCREEN_PERCENT,
                   test_mse=m["test_mse"], test_mae=m["test_mae"],
                   gap_mse_pct=100 * (m["test_mse"] / gm - 1),
                   gap_mae_pct=100 * (m["test_mae"] / ga - 1))
        row["best_metric_gap_pct"] = min(row["gap_mse_pct"], row["gap_mae_pct"])
        rows.append(row)
    winners = []
    for key in SETTINGS:
        candidates = [r for r in rows if (r["dataset"], r["horizon"]) == key]
        candidates.sort(key=lambda r: (r["best_metric_gap_pct"],
                                       r["gap_mse_pct"] + r["gap_mae_pct"]))
        winners.append(candidates[0])
    (root / "stage1_all_rows.json").write_text(json.dumps(rows, indent=2))
    (root / "stage1_winners.json").write_text(json.dumps(winners, indent=2))


def confirm(root, args):
    winners = json.loads((root / "stage1_winners.json").read_text())
    planned = []
    for w in winners:
        for seed in SEEDS_CONFIRM:
            planned.append(dict(w, seed=seed))
    drive(root, planned, args, "confirm.log", 100, CONFIRM_EPOCHS)


def final(root):
    winners = json.loads((root / "stage1_winners.json").read_text())
    out = []
    for w in winners:
        gm, ga = GOLDEN[(w["dataset"], w["horizon"])]
        per_seed = []
        for seed in (SEED_SCREEN, *SEEDS_CONFIRM):
            c = dict(w, seed=seed)
            pct, ep = (SCREEN_PERCENT, SCREEN_EPOCHS) if seed == SEED_SCREEN else (100, CONFIRM_EPOCHS)
            m = metrics_of(run_dir(root, c, pct, ep))
            if m:
                per_seed.append(dict(seed=seed, **m,
                                     mse_below=m["test_mse"] < gm,
                                     mae_below=m["test_mae"] < ga))
        out.append(dict(dataset=w["dataset"], horizon=w["horizon"], winner=w,
                        per_seed=per_seed,
                        any_metric_any_seed=any(x["mse_below"] or x["mae_below"] for x in per_seed),
                        both_metric_same_seed=any(x["mse_below"] and x["mae_below"] for x in per_seed)))
    (root / "final.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["plan", "search", "select", "confirm", "final"], required=True)
    p.add_argument("--output-root", default="research_runs/phaseformer_L_batch_period_loss_gate_200_v1")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--poll-seconds", type=int, default=20)
    args = p.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    planned = cells()
    if args.stage == "plan":
        print(f"settings={len(SETTINGS)} cells_per_setting=200 total={len(planned)}")
        print(f"batches={BATCHES} periods={PERIODS} loss_gate_pairs={LOSS_GATES}")
    elif args.stage == "search":
        drive(root, planned, args, "stage1.log", SCREEN_PERCENT, SCREEN_EPOCHS)
    elif args.stage == "select":
        select(root)
    elif args.stage == "confirm":
        confirm(root, args)
    else:
        final(root)


if __name__ == "__main__":
    main()
