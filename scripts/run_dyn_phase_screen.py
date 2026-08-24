#!/usr/bin/env python3
"""Stage A screening driver for the weak-residual dynamic-phase plan.

Runs the search runner over the plan's core matrix — the 5 plan datasets x long
horizons {336, 720} x the cumulative ablation ladder plus the stage-1 residual
pair — at screening budget (30% data, 8 epochs), two jobs at a time on the free
GPUs. Deterministic run IDs make it resumable: already-completed configs are
skipped. On completion all metrics.csv rows are merged into a single summary.
"""

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = "/home/niuyiming/.conda/envs/py310/bin/python"
GPUS = [0, 1, 2, 3]

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "Electricity", "Traffic"]
HORIZONS = [336, 720]
# Cumulative ladder A->D (plan stage 10) + residual pair (plan stage 1)
# + next-stage paper plan mechanisms (velocity, circular bias, adaptive gate).
MECHANISMS = [
    "original",
    "phase_correction",
    "dyn_geo",
    "dyn_geo_rot",
    "dyn_stack",
    "residual_full",
    "no_residual",
    "phase_velocity",
    "phase_vel_geo",
    "residual_adaptive",
    "next_full",
]
OUTPUT_DIR = "research_runs/dyn_phase_screen"
STAGE = "mechanism_screen_1"
PERCENT = 30
MAX_EPOCHS = 8
SEED = 2021


def queue_jobs():
    jobs = []
    for ds in DATASETS:
        for h in HORIZONS:
            for m in MECHANISMS:
                jobs.append((ds, h, m))
    return jobs


def run_exists(ds, h, m):
    pattern = (
        f"runs/{STAGE}_{ds.lower()}_h{h}_{m}_p24_base_huber_*_pct{PERCENT}_"
        f"e{MAX_EPOCHS}_s{SEED}_*/metrics.csv"
    )
    return bool(list((REPO_ROOT / OUTPUT_DIR).glob(pattern)))


def launch(gpu, ds, h, m):
    log_dir = REPO_ROOT / OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ds}_h{h}_{m}_gpu{gpu}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [
        PY, "scripts/search_phaseformer.py",
        "--dataset", ds, "--horizon", str(h),
        "--stage", STAGE, "--mechanism", m, "--period", "24",
        "--percent", str(PERCENT), "--max-epochs", str(MAX_EPOCHS),
        "--seed", str(SEED), "--loss", "huber",
        "--output-dir", OUTPUT_DIR,
    ]
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                stdout=lf, stderr=subprocess.STDOUT)
    return proc, log_path


def merge_summary():
    out = Path(REPO_ROOT) / OUTPUT_DIR
    summary = out / "screen_summary.csv"
    rows = []
    for metrics in (out / "runs").glob("*/metrics.csv"):
        with metrics.open() as f:
            for row in csv.DictReader(f):
                rows.append(row)
    fields = [
        "run_id", "dataset", "horizon", "mechanism", "seed",
        "epochs_requested", "epochs_completed", "best_val_loss",
        "val_mae", "val_mse", "parameter_count", "elapsed_sec",
        "train_size", "val_size", "config_hash",
    ]
    rows.sort(key=lambda r: (r["dataset"], int(r["horizon"]), r["mechanism"]))
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Merged {len(rows)} rows -> {summary}", flush=True)


def main():
    queue = queue_jobs()
    skipped = [j for j in queue if run_exists(*j)]
    for j in skipped:
        print(f"SKIP (completed): {' '.join(str(x) for x in j)}", flush=True)
    queue = [j for j in queue if not run_exists(*j)]
    total = len(queue)
    done = 0
    procs = {}  # gpu -> (proc, label)
    while queue or procs:
        for gpu in GPUS:
            if gpu not in procs and queue:
                ds, h, m = queue.pop(0)
                proc, log_path = launch(gpu, ds, h, m)
                label = f"{ds} h{h} {m}"
                procs[gpu] = (proc, label)
                print(f"[{done + len(procs)}/{total}] LAUNCH gpu{gpu}: {label} -> {log_path.relative_to(REPO_ROOT)}", flush=True)
        finished = [g for g, (p, _) in procs.items() if p.poll() is not None]
        for gpu in finished:
            proc, label = procs.pop(gpu)
            done += 1
            print(f"[{done}/{total}] DONE gpu{gpu}: {label} rc={proc.returncode}", flush=True)
            if proc.returncode != 0:
                sys.stderr.write(f"FAILED: {label}\n")
        if procs:
            time.sleep(10)
    merge_summary()


if __name__ == "__main__":
    main()
