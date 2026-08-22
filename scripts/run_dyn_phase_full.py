#!/usr/bin/env python3
"""Full-budget confirm driver for the weak-residual dynamic-phase plan.

Runs benchmark_phaseformer_suite.py per (dataset, horizon) with the given modes
on the free GPUs, two jobs at a time. Run ids follow the suite's
{prefix}_{mode}_{scheme}_{dataset}_{horizon}_seed{seed} convention so
analyze_experiment.py's find_run_dir can locate them. Resumable: a run whose
metrics.csv already exists is skipped.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = "/home/niuyiming/.conda/envs/py310/bin/python"
GPUS = [3, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True, help="comma-separated")
    p.add_argument("--horizons", required=True, help="comma-separated")
    p.add_argument("--modes", required=True, help="comma-separated")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=2021)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    queue = [(ds, h) for ds in datasets for h in horizons]
    total = len(queue)
    done = 0
    procs = {}  # gpu -> (proc, label)
    while queue or procs:
        for gpu in GPUS:
            if gpu not in procs and queue:
                ds, h = queue.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                cmd = [
                    PY, "scripts/benchmark_phaseformer_suite.py",
                    "--datasets", ds, "--horizons", str(h),
                    "--modes", ",".join(modes),
                    "--lookback", "720", "--seed", str(args.seed),
                    "--output-dir", args.output_dir,
                    "--run-prefix", f"{args.run_prefix}_{ds}_{h}",
                ]
                if args.batch_size:
                    cmd += ["--batch-size", str(args.batch_size)]
                log_dir = REPO_ROOT / args.output_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{args.run_prefix}_{ds}_{h}_gpu{gpu}.log"
                with open(log_path, "w") as lf:
                    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                            stdout=lf, stderr=subprocess.STDOUT)
                procs[gpu] = (proc, f"{ds} h{h}")
                print(f"[{done + len(procs)}/{total}] LAUNCH gpu{gpu}: {ds} h{h} -> {log_path}", flush=True)
        finished = [g for g, (p, _) in procs.items() if p.poll() is not None]
        for gpu in finished:
            proc, label = procs.pop(gpu)
            done += 1
            print(f"[{done}/{total}] DONE gpu{gpu}: {label} rc={proc.returncode}", flush=True)
            if proc.returncode != 0:
                sys.stderr.write(f"FAILED: {label}\n")
        if procs:
            time.sleep(15)
    print(f"Full-budget confirm finished: {done} settings.", flush=True)


if __name__ == "__main__":
    main()
