#!/usr/bin/env python3
"""Dispatch the two-stage pooled low-rank NLinear screen on available GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_results(output_root, dataset, horizon, seed):
    rows = []
    for path in sorted((ROOT / output_root).glob("*/result.json")):
        row = json.loads(path.read_text())
        if (
            row["dataset"] == dataset
            and int(row["horizon"]) == horizon
            and int(row["seed"]) == seed
        ):
            row["_path"] = str(path)
            rows.append(row)
    return rows


def rank_configs(lookback, horizon, pool_factors, ranks):
    configs = []
    for pool_factor in pool_factors:
        pooled_len = math.ceil(lookback / pool_factor)
        max_rank = min(pooled_len, horizon)
        for rank in ranks:
            if rank > max_rank:
                continue
            configs.append(
                {
                    "config_id": f"rank_p{pool_factor}_r{rank}_s000",
                    "pool_factor": pool_factor,
                    "rank": rank,
                    "smooth_ratio": 0.0,
                }
            )
    return configs


def select_low_rank_configs(rows, count):
    candidates = []
    for row in rows:
        if row["smooth_ratio"] != 0.0:
            continue
        max_rank = min(int(row["pooled_len"]), int(row["horizon"]))
        if int(row["rank"]) >= max_rank:
            continue
        score = (
            float(row["val_mse"]) / float(row["val_phase_mse"])
            + float(row["val_mae"]) / float(row["val_phase_mae"])
        )
        candidates.append((score, row))
    candidates.sort(key=lambda item: (item[0], item[1]["rank_ratio"]))
    selected = []
    seen = set()
    for score, row in candidates:
        key = (int(row["pool_factor"]), int(row["rank"]))
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "config_id": f"smooth_p{row['pool_factor']}_r{row['rank']}",
                "pool_factor": int(row["pool_factor"]),
                "rank": int(row["rank"]),
                "validation_score": score,
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"needed {count} low-rank configurations, found only {len(selected)}"
        )
    return selected


def build_command(args, config):
    command = [
        sys.executable,
        str(ROOT / "scripts/run_pooled_lowrank_nlinear.py"),
        "--phase-config",
        args.phase_config,
        "--phase-checkpoint",
        args.phase_checkpoint,
        "--config-id",
        config["config_id"],
        "--pool-factor",
        str(config["pool_factor"]),
        "--rank",
        str(config["rank"]),
        "--smooth-ratio",
        str(config["smooth_ratio"]),
        "--smooth-window",
        str(args.smooth_window),
        "--seed",
        str(args.seed),
        "--max-epochs",
        str(args.max_epochs),
        "--num-workers",
        str(args.num_workers),
        "--output-root",
        args.output_root,
        "--evaluate-test",
        "--require-cuda",
    ]
    return command


def dispatch(args, configs):
    pending = list(configs)
    active = {}
    while pending or active:
        while pending and len(active) < len(args.gpus):
            gpu = next(gpu for gpu in args.gpus if gpu not in active)
            config = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = build_command(args, config)
            print(
                f"launch gpu={gpu} config={config['config_id']}: "
                + " ".join(command),
                flush=True,
            )
            active[gpu] = (
                config,
                subprocess.Popen(command, cwd=ROOT, env=env),
            )
        finished = []
        for gpu, (config, process) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            if return_code != 0:
                raise RuntimeError(
                    f"{config['config_id']} on GPU {gpu} failed with {return_code}"
                )
            print(f"finished gpu={gpu} config={config['config_id']}", flush=True)
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if active:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-config", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=["ETTh1", "ETTm1"])
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--lookback", type=int, default=720)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--stage", choices=["rank", "smooth"], required=True)
    parser.add_argument("--output-root", default="research_runs/pooled_lowrank_nlinear_scratch")
    parser.add_argument("--pool-factors", default="1,2,4,8")
    parser.add_argument("--ranks", default="4,8,16,32,64,96")
    parser.add_argument("--smooth-ratios", default="0.10,0.25,0.50,0.75")
    parser.add_argument("--selected-count", type=int, default=3)
    parser.add_argument("--smooth-window", type=int, default=24)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", default="0")
    args = parser.parse_args()
    args.gpus = [int(item) for item in args.gpus.split(",") if item]
    if not args.gpus:
        parser.error("--gpus must contain at least one GPU")

    if args.stage == "rank":
        pool_factors = [int(item) for item in args.pool_factors.split(",") if item]
        ranks = [int(item) for item in args.ranks.split(",") if item]
        configs = rank_configs(args.lookback, args.horizon, pool_factors, ranks)
    else:
        rows = load_results(args.output_root, args.dataset, args.horizon, args.seed)
        selected = select_low_rank_configs(rows, args.selected_count)
        ratios = [float(item) for item in args.smooth_ratios.split(",") if item]
        configs = [
            {
                "config_id": f"{base['config_id']}_s{round(ratio * 100):03d}",
                "pool_factor": base["pool_factor"],
                "rank": base["rank"],
                "smooth_ratio": ratio,
            }
            for base in selected
            for ratio in ratios
        ]
        selection_path = ROOT / args.output_root / f"{args.dataset}_h{args.horizon}_s{args.seed}_smooth_selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "selection_source": "validation",
                    "selected_no_smoothing_configs": selected,
                    "smooth_ratios": ratios,
                },
                indent=2,
            )
            + "\n"
        )
        print(selection_path)
    dispatch(args, configs)


if __name__ == "__main__":
    main()
