#!/usr/bin/env python3
"""Run only the audited missing rank-sweep cells, one process at a time."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / "scripts" / "search_phaseformer.py"
OUTPUT = "research_runs/rank_sweep_2_multiseed_stage1_20260914_v4"

FROZEN = {
    ("Weather", 96): (0.2, 0.0003),
    ("Weather", 192): (0.5, 0.001),
    ("Electricity", 336): (0.5, 0.001),
}

JOBS = [
    ("Weather", 96, 2022, "rank", 3),
    ("Weather", 192, 2023, "rank", 24),
    ("Electricity", 336, 2022, "direct", None),
    ("Electricity", 336, 2022, "rank", 84),
    ("Electricity", 336, 2022, "rank", 42),
    ("Electricity", 336, 2022, "rank", 21),
    ("Electricity", 336, 2022, "rank", 10),
    ("Electricity", 336, 2023, "direct", None),
    ("Electricity", 336, 2023, "rank", 84),
    ("Electricity", 336, 2023, "rank", 42),
    ("Electricity", 336, 2023, "rank", 21),
    ("Electricity", 336, 2023, "rank", 10),
]


def command(job: tuple) -> list[str]:
    dataset, horizon, seed, kind, rank = job
    gate, learning_rate = FROZEN[(dataset, horizon)]
    overrides = {
        "weak_period_residual_gate_init": gate,
        "learning_rate": learning_rate,
    }
    if kind == "direct":
        overrides["weak_period_residual_head_type"] = "shared"
    else:
        overrides.update(
            {
                "weak_period_residual_head_type": "pooled_lowrank",
                "weak_period_residual_pool_factor": 1,
                "weak_period_residual_rank": rank,
                "weak_period_residual_smooth_ratio": 0.0,
                "weak_period_residual_smooth_window": 24,
            }
        )
    return [
        sys.executable,
        str(SEARCH),
        "--dataset",
        dataset,
        "--horizon",
        str(horizon),
        "--stage",
        "confirm",
        "--mechanism",
        "weak_residual",
        "--lookback",
        "720",
        "--period",
        "24",
        "--max-epochs",
        "30",
        "--seed",
        str(seed),
        "--loss",
        "huber",
        "--output-dir",
        OUTPUT,
        "--num-workers",
        "4",
        "--bad-case-limit",
        "0",
        "--require-cuda",
        "--resume",
        "--evaluate-test",
        "--overrides",
        json.dumps(overrides, sort_keys=True),
    ]


def main() -> None:
    manifest = ROOT / OUTPUT / "explicit_repair_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "protocol": "explicit audited missing cells; serial; no launcher retries",
                "jobs": [command(job) for job in JOBS],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for index, job in enumerate(JOBS, 1):
        print(json.dumps({"event": "start", "index": index, "total": len(JOBS), "job": job}), flush=True)
        result = subprocess.run(command(job), cwd=ROOT)
        print(json.dumps({"event": "exit", "index": index, "code": result.returncode}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
