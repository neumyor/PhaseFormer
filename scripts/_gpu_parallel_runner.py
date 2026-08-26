#!/usr/bin/env python3
"""Run a list of search/benchmark commands across GPUs with round-robin split.

Usage:
    python scripts/_gpu_parallel_runner.py --stage architecture_screen --gpus 0,1 [--num-workers 4]

Loads the ICPT runner's command generator, splits the generated commands evenly
across the given GPUs and runs each GPU's slice sequentially in its own
background process.  A JSON marker file is written per GPU when its slice
finishes so the launcher can be polled.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import run_intercycle_patch_experiment as runner  # noqa: E402


def build_commands(stage, settings, num_workers):
    if stage == "architecture_screen":
        allowed = runner.ANCHOR_SETTINGS
        modes = runner.STAGE_A_MODES
        output_dir = runner.SCREEN_OUTPUT
    elif stage == "pe_screen":
        allowed = runner.ANCHOR_SETTINGS
        modes = runner.PE_INDEX_MODES + (runner.PE_CALENDAR_MODE,)
        output_dir = runner.SCREEN_OUTPUT
    else:
        raise ValueError(f"unsupported parallel stage: {stage}")
    settings = settings or ",".join(f"{d}:{h}" for d, h in allowed)
    commands = []
    for dataset, horizon in runner.parse_settings(settings, allowed):
        for mode in modes:
            commands.append(
                runner._screen_command(
                    type("A", (), {"num_workers": num_workers, "output_dir": output_dir})(),
                    dataset, horizon, mode,
                )
            )
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("architecture_screen", "pe_screen"))
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--settings")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    commands = build_commands(args.stage, args.settings, args.num_workers)
    print(f"generated {len(commands)} commands over GPUs {gpus}", flush=True)
    # Round-robin partition.
    per_gpu = {gpu: [] for gpu in gpus}
    for index, command in enumerate(commands):
        per_gpu[gpus[index % len(gpus)]].append(command)
    processes = {}
    marker_dir = REPO_ROOT / "research_runs" / f"icpt_parallel_{args.stage}"
    marker_dir.mkdir(parents=True, exist_ok=True)
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
