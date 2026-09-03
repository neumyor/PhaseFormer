#!/usr/bin/env python3
"""Run matched D1/D2 remove-trained ETTm1-H192 models on one GPU."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_d1_d2_remove_screen import D1_PERIODS, D2_LENGTHS


MODELS = ("original", "weak_residual", "rcrf_nlinear_plain")


def jobs(output_dir: Path, max_epochs: int, workers: int, d1_sigma: float):
    for label, period in D1_PERIODS:
        for model in MODELS:
            yield label, model, [
                sys.executable, "scripts/search_phaseformer.py", "--dataset", "ETTm1", "--horizon", "192",
                "--stage", "input_components", "--mechanism", model, "--input-hypothesis", "d1",
                "--input-variant", "remove_full", "--input-d1-period", str(period), "--seed", "2021",
                "--input-d1-sigma", str(d1_sigma),
                "--max-epochs", str(max_epochs), "--percent", "100", "--num-workers", str(workers),
                "--require-cuda", "--resume", "--output-dir", str(output_dir),
            ]
    for length in D2_LENGTHS:
        label = f"D2-{length}"
        for model in MODELS:
            yield label, model, [
                sys.executable, "scripts/search_phaseformer.py", "--dataset", "ETTm1", "--horizon", "192",
                "--stage", "input_components", "--mechanism", model, "--input-hypothesis", "d2",
                "--input-variant", "remove_full", "--input-d2-recent-length", str(length), "--seed", "2021",
                "--max-epochs", str(max_epochs), "--percent", "100", "--num-workers", str(workers),
                "--require-cuda", "--resume", "--output-dir", str(output_dir),
            ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("research_runs/d1_d2_retrained_remove_scratch"))
    parser.add_argument("--control-dir", type=Path, default=Path("research_runs/d1_d2_retrained_remove_control"))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument("--d1-sigma", type=float, default=1.0 / 720.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    entries = list(jobs(args.output_dir, args.max_epochs, args.num_workers, args.d1_sigma))
    args.control_dir.mkdir(parents=True, exist_ok=True)
    (args.control_dir / "retrained_remove_protocol.json").write_text(json.dumps({
        "dataset": "ETTm1", "horizon": 192, "seed": 2021, "training": "remove input in both train and validation",
        "models": MODELS, "d1_periods": D1_PERIODS, "d1_sigma_frequency": args.d1_sigma,
        "d2_lengths": D2_LENGTHS,
        "job_count": len(entries), "max_epochs": args.max_epochs,
    }, indent=2) + "\n")
    for label, model, command in entries:
        print(label, model, subprocess.list2cmdline(command), flush=True)
    if not args.execute:
        return

    def run(entry):
        label, model, command = entry
        log = args.control_dir / "logs" / f"{label}_{model}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{label}/{model} failed; see {log}")
        return label, model

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs_per_gpu) as pool:
        futures = [pool.submit(run, entry) for entry in entries]
        for future in concurrent.futures.as_completed(futures):
            label, model = future.result()
            print(f"COMPLETED {label} {model}", flush=True)


if __name__ == "__main__":
    main()
