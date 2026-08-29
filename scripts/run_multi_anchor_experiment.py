#!/usr/bin/env python3
"""Run the preregistered, validation-only multi-anchor experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUNNER = REPO_ROOT / "scripts" / "search_phaseformer.py"
ROUTER_RUNNER = REPO_ROOT / "scripts" / "search_multi_anchor.py"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
PILOT_DATASETS = ("ETTh1", "ETTm2", "Weather")
ANCHORS = {
    "a1": "gold_combo_reliability_s2",
    "i0": "rcrf_icpt_none",
    "r0": "triaxis_rolling_features",
}
CANDIDATES = (
    "multi_anchor_global_hard",
    "multi_anchor_structural_hard",
    "multi_anchor_guarded_hard",
    "multi_anchor_structural_soft",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def matching_run(root, dataset, horizon, mechanism, percent, epochs):
    matches = []
    for config_path in (Path(root) / "runs").glob("*/config.json"):
        spec = json.loads(config_path.read_text())
        wanted = (
            spec.get("dataset") == dataset
            and int(spec.get("horizon", -1)) == horizon
            and spec.get("mechanism") == mechanism
            and int(spec.get("percent", -1)) == percent
            and int(spec.get("max_epochs", -1)) == epochs
            and int(spec.get("seed", -1)) == 2021
            and spec.get("loss") == "huber"
            and (config_path.parent / "metrics.csv").is_file()
        )
        if wanted:
            matches.append(config_path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one run: {dataset}/H{horizon}/{mechanism}/pct{percent}/e{epochs}: {matches}"
        )
    return matches[0]


def run_logged(command, log_path, dry_run):
    print("RUN", " ".join(map(str, command)), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
    if completed.returncode:
        tail = "\n".join(log_path.read_text().splitlines()[-50:])
        raise RuntimeError(f"run failed: {log_path}\n{tail}")
    print("DONE", log_path.stem, flush=True)


def run_shadow(args, dataset, horizon):
    stage = "mechanism_screen_2" if horizon == 96 else "mechanism_full8"
    for mechanism in ANCHORS.values():
        command = [
            sys.executable, str(REFERENCE_RUNNER),
            "--dataset", dataset,
            "--horizon", str(horizon),
            "--stage", stage,
            "--mechanism", mechanism,
            "--lookback", "720",
            "--period", "24",
            "--percent", "24",
            "--max-epochs", "8",
            "--seed", "2021",
            "--loss", "huber",
            "--num-workers", str(args.num_workers),
            "--bad-case-limit", "0",
            "--output-dir", str(args.output_dir),
            "--resume",
        ]
        run_logged(
            command,
            args.output_dir / "driver_logs" / f"shadow_{dataset}_h{horizon}_{mechanism}.log",
            args.dry_run,
        )


def run_candidates(args, dataset, horizon, candidates):
    shadow_paths = {
        name: matching_run(args.output_dir, dataset, horizon, mechanism, 24, 8)
        for name, mechanism in ANCHORS.items()
    }
    full_paths = {
        name: matching_run(args.reference_dir, dataset, horizon, mechanism, 30, 8)
        for name, mechanism in ANCHORS.items()
    }
    stage = "pilot" if dataset in PILOT_DATASETS and horizon == 96 else (
        "stage_a" if horizon == 96 else "stage_b"
    )
    for candidate in candidates:
        command = [
            sys.executable, str(ROUTER_RUNNER),
            "--dataset", dataset,
            "--horizon", str(horizon),
            "--stage", stage,
            "--mechanism", candidate,
            "--lookback", "720",
            "--period", "24",
            "--shadow-percent", "24",
            "--full-percent", "30",
            "--anchor-epochs", "8",
            "--max-epochs", "20",
            "--seed", "2021",
            "--num-workers", str(args.num_workers),
            "--output-dir", str(args.output_dir),
            "--resume",
        ]
        for bank, paths in (("shadow", shadow_paths), ("full", full_paths)):
            for anchor, path in paths.items():
                command.extend((f"--{bank}-{anchor}", str(path)))
        run_logged(
            command,
            args.output_dir / "driver_logs" / f"router_{dataset}_h{horizon}_{candidate}.log",
            args.dry_run,
        )


def candidate_row(root, dataset, horizon, candidate):
    matches = []
    for config_path in (Path(root) / "runs").glob("*/config.json"):
        spec = json.loads(config_path.read_text())
        if (
            spec.get("protocol_version") == "multi-anchor-selector-v1"
            and spec.get("dataset") == dataset
            and int(spec.get("horizon", -1)) == horizon
            and spec.get("mechanism") == candidate
            and (config_path.parent / "metrics.csv").is_file()
        ):
            matches.append(config_path.parent / "metrics.csv")
    if len(matches) != 1:
        raise RuntimeError(f"expected one candidate row: {dataset}/H{horizon}/{candidate}: {matches}")
    with matches[0].open(newline="") as handle:
        return next(csv.DictReader(handle))


def reference_row(root, dataset, horizon, mechanism):
    run = matching_run(root, dataset, horizon, mechanism, 30, 8)
    with (run / "metrics.csv").open(newline="") as handle:
        return next(csv.DictReader(handle))


def score(args, datasets, horizons, candidates):
    cells = []
    ranking = []
    for candidate in candidates:
        ratios = []
        for horizon in horizons:
            for dataset in datasets:
                refs = [
                    reference_row(args.reference_dir, dataset, horizon, mechanism)
                    for mechanism in ANCHORS.values()
                ]
                row = candidate_row(args.output_dir, dataset, horizon, candidate)
                for metric in ("val_mse", "val_mae"):
                    reference = min(float(item[metric]) for item in refs)
                    value = float(row[metric])
                    ratio = value / reference
                    ratios.append(ratio)
                    cells.append({
                        "dataset": dataset,
                        "horizon": horizon,
                        "metric": metric,
                        "candidate": candidate,
                        "candidate_value": value,
                        "reference_envelope": reference,
                        "ratio": ratio,
                    })
        ranking.append({
            "candidate": candidate,
            "macro_ratio": sum(ratios) / len(ratios),
            "worst_ratio": max(ratios),
            "all_below_one": all(value < 1.0 for value in ratios),
            "strict_pass": all(value < 1.0 for value in ratios)
            and sum(ratios) / len(ratios) <= 0.995,
        })
    ranking.sort(key=lambda item: (item["macro_ratio"], item["worst_ratio"]))
    return cells, ranking


def save_score(root, name, cells, decision):
    with (Path(root) / f"{name}_ratios.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    write_json(Path(root) / f"{name}_decision.json", decision)


def run_settings(args, datasets, horizon, candidates):
    for dataset in datasets:
        run_shadow(args, dataset, horizon)
        if not args.dry_run:
            run_candidates(args, dataset, horizon, candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "a", "b", "all"), default="all")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "research_runs" / "multi_anchor_selector_v1_scratch",
    )
    parser.add_argument(
        "--reference-dir", type=Path,
        default=REPO_ROOT / "research_runs" / "safe_regret_triaxis_v1_scratch",
    )
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "matrix_manifest.json", {
        "created_at": utc_now(),
        "experiment": "multi_anchor_selector_v1",
        "datasets": DATASETS,
        "pilot_datasets": PILOT_DATASETS,
        "horizons": (96, 192),
        "anchors": ANCHORS,
        "candidates": CANDIDATES,
        "lookback": 720,
        "shadow_percent": 24,
        "full_percent": 30,
        "anchor_epochs": 8,
        "router_epochs": 20,
        "seed": 2021,
        "test_accessed": False,
    })

    winner = args.candidate
    if args.stage in ("pilot", "all"):
        run_settings(args, PILOT_DATASETS, 96, CANDIDATES)
        if args.dry_run:
            return
        cells, ranking = score(args, PILOT_DATASETS, (96,), CANDIDATES)
        pilot_pass = ranking[0]["macro_ratio"] < 1.01 and ranking[0]["worst_ratio"] < 1.03
        save_score(args.output_dir, "pilot", cells, {
            "selected_candidate": ranking[0]["candidate"],
            "promotion_passed": pilot_pass,
            "ranking": ranking,
            "rule": "macro < 1.01 and worst < 1.03",
        })
        if not pilot_pass:
            print("STOP pilot promotion gate failed", flush=True)
            return

    if args.stage in ("a", "all"):
        run_settings(
            args,
            tuple(dataset for dataset in DATASETS if dataset not in PILOT_DATASETS),
            96,
            CANDIDATES,
        )
        if args.dry_run:
            return
        cells, ranking = score(args, DATASETS, (96,), CANDIDATES)
        winner = ranking[0]["candidate"]
        save_score(args.output_dir, "stage_a", cells, {
            "selected_candidate": winner,
            "promotion_passed": ranking[0]["strict_pass"],
            "ranking": ranking,
            "rule": "all 12 ratios < 1 and macro <= 0.995",
        })
        if not ranking[0]["strict_pass"]:
            print("STOP Stage-A strict gate failed", flush=True)
            return

    if args.stage in ("b", "all"):
        if winner is None:
            decision_path = args.output_dir / "stage_a_decision.json"
            if not decision_path.is_file():
                parser.error("Stage B needs --candidate or stage_a_decision.json")
            winner = json.loads(decision_path.read_text())["selected_candidate"]
        run_settings(args, DATASETS, 192, (winner,))
        if args.dry_run:
            return
        cells, ranking = score(args, DATASETS, (96, 192), (winner,))
        save_score(args.output_dir, "final", cells, {
            "selected_candidate": winner,
            "validation_gate_passed": ranking[0]["strict_pass"],
            "result": ranking[0],
            "test_accessed": False,
        })


if __name__ == "__main__":
    main()
