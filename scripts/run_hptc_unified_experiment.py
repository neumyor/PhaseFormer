#!/usr/bin/env python3
"""Run the preregistered HPTC unified-model validation search."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SEARCH = REPO / "scripts" / "search_phaseformer.py"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
MODES = (
    "hptc_fixed_b10",
    "hptc_rolling_b10",
    "hptc_rolling_b25",
    "hptc_rolling_b50",
    "hptc_rolling_b25_r05",
)


def command(python, output, dataset, horizon, mode):
    return [
        python, str(SEARCH), "--dataset", dataset, "--horizon", str(horizon),
        "--stage", "mechanism_screen_2", "--mechanism", mode,
        "--lookback", "720", "--period", "24", "--percent", "30",
        "--max-epochs", "8", "--seed", "2021", "--loss", "huber",
        "--num-workers", "4", "--bad-case-limit", "0",
        "--output-dir", str(output), "--resume",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("h96", "h192", "all"), default="h96")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "research_runs" / "hptc_unified_v1_scratch",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    horizons = (96, 192) if args.stage == "all" else (
        96 if args.stage == "h96" else 192,
    )
    commands = [
        command(sys.executable, args.output_dir, dataset, horizon, mode)
        for horizon in horizons for dataset in args.datasets for mode in args.modes
    ]
    for cmd in commands:
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO, check=True)
    print(f"commands={len(commands)}", flush=True)


if __name__ == "__main__":
    main()
