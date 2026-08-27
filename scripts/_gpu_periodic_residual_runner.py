#!/usr/bin/env python3
"""Run the pre-registered periodic-residual next-stage matrix across GPUs.

Builds the 36 formal commands (12 settings x 3 seeds, 8 modes each = 288 model
runs) from run_periodic_residual_next_stage.py and round-robins them over the
given GPUs. Each GPU runs its slice sequentially in a background bash process
with CUDA_VISIBLE_DEVICES pinned, so Lightning's devices=1 uses one GPU per
command. A `gpu<N>.done` marker is touched only if the whole slice finishes, so
the launcher can be polled and re-run (with --resume) after interruption.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import run_periodic_residual_next_stage as runner  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="comma-separated CUDA device indices")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("no GPUs given")

    ns = type("A", (), {})()
    ns.datasets = ",".join(runner.DATASETS)
    ns.horizons = ",".join(map(str, runner.HORIZONS))
    ns.seeds = ",".join(map(str, runner.SEEDS))
    ns.modes = ",".join(runner.MODES)
    ns.num_workers = args.num_workers
    ns.output_dir = runner.DEFAULT_OUTPUT
    ns.resume = args.resume
    ns.progress = args.progress
    commands, run_count = runner.build_commands(ns)
    print(f"generated {len(commands)} commands / {run_count} model runs over GPUs {gpus}",
          flush=True)

    per_gpu = {gpu: [] for gpu in gpus}
    for index, command in enumerate(commands):
        per_gpu[gpus[index % len(gpus)]].append(command)

    marker_dir = REPO_ROOT / "research_runs" / "periodic_residual_parallel_launch"
    marker_dir.mkdir(parents=True, exist_ok=True)
    processes = {}
    for gpu, slice_commands in per_gpu.items():
        script = marker_dir / f"gpu{gpu}.sh"
        lines = ["#!/usr/bin/env bash", "set -e", f"export CUDA_VISIBLE_DEVICES={gpu}"]
        for command in slice_commands:
            quoted = [f'"{c}"' if " " in c else c for c in command]
            lines.append(" ".join(quoted))
        lines.append(f"touch {marker_dir / f'gpu{gpu}.done'}")
        script.write_text("\n".join(lines) + "\n")
        proc = subprocess.Popen(["bash", str(script)], cwd=REPO_ROOT)
        processes[gpu] = proc
        print(f"GPU {gpu}: {len(slice_commands)} commands, pid {proc.pid}", flush=True)

    failed = False
    for gpu, proc in processes.items():
        returncode = proc.wait()
        print(f"GPU {gpu}: finished with rc={returncode}", flush=True)
        failed = failed or returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
