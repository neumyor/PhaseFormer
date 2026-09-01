#!/usr/bin/env python3
"""Weather H96/H192 large-scale search driver for pctf_anchor_repair_strict_t28.

Goal: single seed, H96 AND H192 both MSE and MAE at least 0.5% better than
Golden (0.148/0.195 and 0.193/0.237).  Every candidate runs as a frozen
confirm run with --evaluate-test so test metrics are the comparison source.

Hierarchy:
  Layer 1  protocol screen (LR x epochs x loss) at tier X, lookback 720
  Layer 2  mechanism extreme scan (tier x gate, then warmup / lr-scale)
  Layer 3  refinement (LR local / lookback / anchor-composer balance)
Usage:
  python scripts/search_weather_t28.py --layer 1 --gpus 0,1,2,3 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MECHANISM = "pctf_anchor_repair_strict_t28"
OUTPUT = "/home/niuyiming/PhaseFormer/research_runs/pctf_weather_search_v1"
SEED = 2021
HORIZONS = (96, 192)

# trust-region tier: correction / deformation / global-level
TIERS = {
    "W": (0.60, 0.24, 0.12),
    "X": (0.75, 0.30, 0.15),
    "Y": (1.00, 0.40, 0.20),
    "Z": (1.25, 0.50, 0.25),
}


def tier_overrides(tier, gate=0.05, warmup=0, anchor_scale=1.0,
                   composer_scale=1.0):
    c, d, g = TIERS[tier]
    return {
        "anchored_pctf_correction_max": c,
        "anchored_pctf_deformation_max": d,
        "anchored_pctf_global_level_max": g,
        "anchored_pctf_gate_aux_weight": gate,
        "anchored_pctf_correction_warmup_epochs": warmup,
        "anchored_pctf_anchor_lr_scale": anchor_scale,
        "anchored_pctf_composer_lr_scale": composer_scale,
    }


def cmd(num_workers, **kw):
    base = {
        "dataset": "Weather", "horizon": kw["horizon"],
        "stage": "confirm", "mechanism": MECHANISM,
        "period": 24, "lookback": kw.get("lookback", 720),
        "cycle_period": 24, "percent": 100,
        "max_epochs": kw["epochs"], "seed": SEED,
        "loss": kw["loss"], "learning_rate": kw["lr"],
        "capacity": "base", "num_workers": num_workers,
        "bad_case_limit": 0,
        "overrides": json.dumps(
            tier_overrides(kw["tier"], kw.get("gate", 0.05),
                           kw.get("warmup", 0),
                           kw.get("anchor_scale", 1.0),
                           kw.get("composer_scale", 1.0)),
            sort_keys=True),
    }
    command = [sys.executable, "scripts/search_phaseformer.py"]
    for flag, value in [
        ("--dataset", base["dataset"]), ("--horizon", str(base["horizon"])),
        ("--stage", base["stage"]), ("--mechanism", base["mechanism"]),
        ("--period", str(base["period"])),
        ("--lookback", str(base["lookback"])),
        ("--cycle-period", str(base["cycle_period"])),
        ("--percent", str(base["percent"])),
        ("--max-epochs", str(base["max_epochs"])),
        ("--seed", str(base["seed"])), ("--loss", base["loss"]),
        ("--learning-rate", repr(base["learning_rate"])),
        ("--capacity", base["capacity"]),
        ("--num-workers", str(base["num_workers"])),
        ("--bad-case-limit", str(base["bad_case_limit"])),
        ("--overrides", base["overrides"]),
        ("--output-dir", OUTPUT), ("--require-cuda", ""),
        ("--resume", ""), ("--evaluate-test", ""),
    ]:
        if value == "":
            command.append(flag)
        else:
            command.extend((flag, value))
    return command


def layer1(num_workers):
    """12 configs x 2 horizons = 24 runs. tier X, lookback 720."""
    out = []
    for lr in (1e-3, 2e-3, 3e-3):
        for epochs in (30, 60):
            for loss in ("huber", "mae"):
                for h in HORIZONS:
                    out.append(cmd(num_workers, horizon=h, lr=lr,
                                   epochs=epochs, loss=loss, tier="X"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True, choices=(1, 2, 3))
    p.add_argument("--gpus")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    if args.layer == 1:
        commands = layer1(args.num_workers)
    else:
        raise SystemExit("layers 2/3 configs are filled after layer 1 results")

    if args.dry_run:
        print(f"commands={len(commands)}")
        for command in commands:
            print("  " + " ".join(command))
        return 0
    if not args.gpus:
        for command in commands:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        return 0

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    per_gpu = {g: [] for g in gpus}
    for i, command in enumerate(commands):
        per_gpu[gpus[i % len(gpus)]].append(command)
    marker = Path(OUTPUT) / f"driver_layer{args.layer}"
    marker.mkdir(parents=True, exist_ok=True)
    procs = {}
    for gpu, slice_commands in per_gpu.items():
        script = marker / f"gpu{gpu}.sh"
        lines = ["#!/usr/bin/env bash", "set -e",
                 f"export CUDA_VISIBLE_DEVICES={gpu}"]
        for command in slice_commands:
            lines.append(" ".join(shlex.quote(part) for part in command))
        lines.append(f"touch {marker / f'gpu{gpu}.done'}")
        script.write_text("\n".join(lines) + "\n")
        proc = subprocess.Popen(["bash", str(script)], cwd=REPO_ROOT)
        procs[gpu] = proc
        print(f"GPU {gpu}: {len(slice_commands)} commands, pid {proc.pid}",
              flush=True)
    failed = False
    for gpu, proc in procs.items():
        rc = proc.wait()
        print(f"GPU {gpu}: finished rc={rc}", flush=True)
        failed = failed or rc != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
