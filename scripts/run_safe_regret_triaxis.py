#!/usr/bin/env python3
"""Run the preregistered Safe-Regret TriAxis validation matrix.

The script deliberately never passes ``--evaluate-test``.  It trains a fresh
A1 anchor per dataset/horizon, loads that exact checkpoint into every safe
candidate, and freezes one unified H96 winner before H192 is launched.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "search_phaseformer.py"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
REFERENCES = (
    "gold_combo_reliability_s2",
    "rcrf_icpt_none",
    "triaxis_rolling_features",
)
CANDIDATES = (
    "safe_triaxis_anchor",
    "safe_triaxis_regret",
    "safe_triaxis_guarded",
    "safe_triaxis_monotone",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_command(args, output_dir, dataset, horizon, mechanism, checkpoint=""):
    stage = "mechanism_screen_2" if horizon == 96 else "mechanism_full8"
    command = [
        sys.executable,
        str(RUNNER),
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", stage,
        "--mechanism", mechanism,
        "--lookback", "720",
        "--period", "24",
        "--percent", "30",
        "--max-epochs", "8",
        "--seed", "2021",
        "--loss", "huber",
        "--num-workers", str(args.num_workers),
        "--bad-case-limit", "0",
        "--output-dir", str(output_dir),
        "--resume",
    ]
    if checkpoint:
        command.extend(("--init-checkpoint", checkpoint))
    print("RUN", " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def one_metrics(output_dir, dataset, horizon, mechanism):
    stage = "mechanism_screen_2" if horizon == 96 else "mechanism_full8"
    pattern = f"{stage}_{dataset.lower()}_h{horizon}_{mechanism}_*/metrics.csv"
    matches = sorted((Path(output_dir) / "runs").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one metrics file for {dataset}/{horizon}/{mechanism}: {matches}")
    with matches[0].open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return row


def checkpoint_for(output_dir, dataset, horizon):
    row = one_metrics(
        output_dir, dataset, horizon, "gold_combo_reliability_s2"
    )
    path = Path(row["checkpoint"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def run_stage(args, output_dir, horizon, mechanisms):
    for dataset in DATASETS:
        for mechanism in REFERENCES:
            run_command(args, output_dir, dataset, horizon, mechanism)
        if args.dry_run:
            checkpoint = "<A1-checkpoint>"
        else:
            checkpoint = checkpoint_for(output_dir, dataset, horizon)
        for mechanism in mechanisms:
            run_command(
                args, output_dir, dataset, horizon, mechanism, checkpoint
            )


def score_candidates(output_dir, horizons, candidates):
    rows = []
    decisions = []
    for candidate in candidates:
        ratios = []
        for horizon in horizons:
            for dataset in DATASETS:
                reference_rows = [
                    one_metrics(output_dir, dataset, horizon, mechanism)
                    for mechanism in REFERENCES
                ]
                candidate_row = one_metrics(
                    output_dir, dataset, horizon, candidate
                )
                for metric in ("val_mse", "val_mae"):
                    reference = min(float(row[metric]) for row in reference_rows)
                    value = float(candidate_row[metric])
                    ratio = value / reference
                    ratios.append(ratio)
                    rows.append({
                        "dataset": dataset,
                        "horizon": horizon,
                        "metric": metric,
                        "candidate": candidate,
                        "candidate_value": value,
                        "reference_envelope": reference,
                        "ratio": ratio,
                    })
        decisions.append({
            "candidate": candidate,
            "macro_ratio": sum(ratios) / len(ratios),
            "worst_ratio": max(ratios),
            "all_cells_below_one": all(ratio < 1.0 for ratio in ratios),
            "strict_pass": all(ratio < 1.0 for ratio in ratios)
            and sum(ratios) / len(ratios) <= 0.995,
        })
    decisions.sort(key=lambda item: (item["macro_ratio"], item["worst_ratio"]))
    return rows, decisions


def write_score(output_dir, name, rows, decision):
    path = Path(output_dir) / f"{name}_ratios.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(Path(output_dir) / f"{name}_decision.json", decision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "b", "all"), default="all")
    parser.add_argument(
        "--output-dir",
        default="research_runs/safe_regret_triaxis_v1_scratch",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "matrix_manifest.json", {
        "created_at": utc_now(),
        "datasets": DATASETS,
        "references": REFERENCES,
        "candidates": CANDIDATES,
        "lookback": 720,
        "horizons": (96, 192),
        "percent": 30,
        "epochs": 8,
        "seed": 2021,
        "loss": "huber",
        "test_accessed": False,
    })

    winner = args.candidate
    if args.stage in ("a", "all"):
        run_stage(args, output_dir, 96, CANDIDATES)
        if args.dry_run:
            return
        rows, decisions = score_candidates(output_dir, (96,), CANDIDATES)
        winner = decisions[0]["candidate"]
        write_score(output_dir, "stage_a", rows, {
            "selected_candidate": winner,
            "strict_gate_passed": decisions[0]["strict_pass"],
            "ranking": decisions,
            "selection_rule": "minimum macro ratio, then minimum worst ratio",
        })

    if args.stage in ("b", "all"):
        if winner is None:
            decision_path = output_dir / "stage_a_decision.json"
            if not decision_path.is_file():
                parser.error("Stage B requires --candidate or Stage-A decision")
            winner = json.loads(decision_path.read_text())["selected_candidate"]
        run_stage(args, output_dir, 192, (winner,))
        if args.dry_run:
            return
        rows, decisions = score_candidates(output_dir, (96, 192), (winner,))
        write_score(output_dir, "final", rows, {
            "selected_candidate": winner,
            "validation_gate_passed": decisions[0]["strict_pass"],
            "result": decisions[0],
            "test_accessed": False,
        })


if __name__ == "__main__":
    main()
