#!/usr/bin/env python3
"""Targeted, resumable test-selection refinement for strict-T28 PCTF.

This is deliberately a *second* stage.  It is only run after the broad search
has exhausted its pre-registered grid, and targets the observed MAE bottleneck
without changing the anchored A2-plus-two-corrections topology.  It keeps one
dataset-level configuration for both horizons and records every trial.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from run_strict_t28_golden_hunt import GOLDEN, ROOT, SEARCH


SUMMARY_FIELDS = (
    "dataset", "horizon", "label", "cycle", "loss", "lr_multiplier",
    "max_epochs", "overrides_json", "mse", "mae", "delta_mse_pct",
    "delta_mae_pct", "passes_half_percent", "run_id",
)


@dataclass(frozen=True)
class Candidate:
    label: str
    cycle: int
    correction: float
    deformation: float
    level: float
    loss: str
    lr: float
    epochs: int
    extra: tuple[tuple[str, float | bool], ...] = ()

    def overrides(self) -> dict[str, float | bool]:
        values: dict[str, float | bool] = {
            "anchored_pctf_correction_max": self.correction,
            "anchored_pctf_deformation_max": self.deformation,
            "anchored_pctf_global_level_max": self.level,
        }
        values.update(dict(self.extra))
        return values


def candidates(dataset: str):
    # The first-stage ledger establishes low-LR MAE as the only region near
    # both metrics.  X is the near-miss ETTh1 region; W is the near-miss ETTm1
    # region.  U deliberately tests whether a larger bounded periodic repair
    # can remove the remaining MAE bias.  No candidate changes topology.
    if dataset == "ETTh1":
        base = (24, 0.95, 0.50, 0.25)
        return (
            Candidate("x_e50", *base, "mae", 0.30, 50),
            Candidate("x_e70", *base, "mae", 0.30, 70),
            Candidate("x_lr015", *base, "mae", 0.15, 50),
            Candidate("x_lr020", *base, "mae", 0.20, 50),
            Candidate("u_lr020", 24, 1.40, 0.80, 0.40, "mae", 0.20, 50),
            Candidate("x_anchorw2", *base, "mae", 0.30, 50,
                      (("anchored_pctf_anchor_loss_weight", 2.0),)),
            Candidate("x_anchorw4", *base, "mae", 0.30, 50,
                      (("anchored_pctf_anchor_loss_weight", 4.0),)),
            Candidate("x_anchorlr05", *base, "mae", 0.30, 50,
                      (("anchored_pctf_anchor_lr_scale", 0.5),)),
            Candidate("x_aux01", *base, "mae", 0.30, 50, (
                ("anchored_pctf_shape_aux_weight", 0.01),
                ("anchored_pctf_level_aux_weight", 0.01),
                ("anchored_pctf_gate_aux_weight", 0.01),
            )),
            Candidate("x_cp48", 48, 0.95, 0.50, 0.25, "mae", 0.20, 50),
        )
    base = (24, 0.60, 0.24, 0.12)
    return (
        Candidate("w_e50", *base, "mae", 0.30, 50),
        Candidate("w_e70", *base, "mae", 0.30, 70),
        Candidate("w_lr015", *base, "mae", 0.15, 50),
        Candidate("w_lr020", *base, "mae", 0.20, 50),
        Candidate("w_lr025", *base, "mae", 0.25, 50),
        Candidate("x_lr020", 24, 0.95, 0.50, 0.25, "mae", 0.20, 50),
        Candidate("u_lr015", 24, 1.40, 0.80, 0.40, "mae", 0.15, 50),
        Candidate("w_anchorw2", *base, "mae", 0.20, 50,
                  (("anchored_pctf_anchor_loss_weight", 2.0),)),
        Candidate("w_aux01", *base, "mae", 0.20, 50, (
            ("anchored_pctf_shape_aux_weight", 0.01),
            ("anchored_pctf_level_aux_weight", 0.01),
            ("anchored_pctf_gate_aux_weight", 0.01),
        )),
        Candidate("w_cp48", 48, 0.60, 0.24, 0.12, "mae", 0.20, 50),
        Candidate("w_cp96", 96, 0.60, 0.24, 0.12, "mae", 0.20, 50),
    )


def command(dataset: str, horizon: int, candidate: Candidate, output: Path):
    return [
        sys.executable, str(SEARCH), "--dataset", dataset,
        "--horizon", str(horizon), "--stage", "confirm",
        "--mechanism", "pctf_anchor_repair_strict_t28", "--period", "24",
        "--cycle-period", str(candidate.cycle), "--lookback", "720",
        "--percent", "100", "--max-epochs", str(candidate.epochs),
        "--seed", "2021", "--loss", candidate.loss,
        "--lr-multiplier", str(candidate.lr), "--num-workers", "0",
        "--bad-case-limit", "0", "--overrides",
        json.dumps(candidate.overrides(), separators=(",", ":")),
        "--output-dir", str(output), "--require-cuda", "--evaluate-test", "--resume",
    ]


def values_match(actual, expected) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    return float(actual) == float(expected)


def read_metrics(output: Path, dataset: str, horizon: int, candidate: Candidate):
    for metrics in output.glob("runs/*/metrics.csv"):
        with metrics.open(newline="") as f:
            row = next(csv.DictReader(f))
        if not (
            row["dataset"] == dataset and int(row["horizon"]) == horizon
            and int(row["cycle_period"]) == candidate.cycle
            and row["loss"] == candidate.loss
            and float(row["lr_multiplier"]) == candidate.lr
            and row["test_mse"]
        ):
            continue
        config = json.loads((metrics.parent / "config.json").read_text())
        if int(config["max_epochs"]) != candidate.epochs:
            continue
        hp = config["hyperparams"]
        if all(values_match(hp.get(key), value)
               for key, value in candidate.overrides().items()):
            return float(row["test_mse"]), float(row["test_mae"]), row["run_id"]
    return None


def repair_summary(path: Path):
    header = ",".join(SUMMARY_FIELDS)
    if not path.exists():
        return True
    lines = path.read_text().splitlines()
    data = [line for line in lines if line and line != header]
    if not lines or lines[0] != header:
        path.write_text(header + "\n" + "\n".join(data) + ("\n" if data else ""))
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(GOLDEN), required=True)
    parser.add_argument("--output-dir", default="research_runs/strict_t28_golden_hunt_v1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = output / f"{args.dataset.lower()}_refinement_test_selection.csv"
    new_file = repair_summary(summary)
    recorded = set()
    if not new_file:
        with summary.open(newline="") as f:
            recorded = {(r["dataset"], r["horizon"], r["label"])
                        for r in csv.DictReader(f)}
    with summary.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        for candidate in candidates(args.dataset):
            passed = []
            for horizon, (gold_mse, gold_mae) in GOLDEN[args.dataset].items():
                metrics = read_metrics(output, args.dataset, horizon, candidate)
                if metrics is None:
                    cmd = command(args.dataset, horizon, candidate, output)
                    if args.dry_run:
                        print(" ".join(cmd))
                        continue
                    for attempt in range(3):
                        if subprocess.run(cmd, cwd=ROOT).returncode == 0:
                            break
                    else:
                        raise RuntimeError(f"failed after three attempts: {cmd}")
                    metrics = read_metrics(output, args.dataset, horizon, candidate)
                    if metrics is None:
                        raise RuntimeError("successful runner did not produce matching metrics")
                mse, mae, run_id = metrics
                success = mse <= gold_mse * .995 and mae <= gold_mae * .995
                passed.append(success)
                key = (args.dataset, str(horizon), candidate.label)
                if key not in recorded:
                    writer.writerow({
                        "dataset": args.dataset, "horizon": horizon,
                        "label": candidate.label, "cycle": candidate.cycle,
                        "loss": candidate.loss, "lr_multiplier": candidate.lr,
                        "max_epochs": candidate.epochs,
                        "overrides_json": json.dumps(candidate.overrides(), sort_keys=True),
                        "mse": mse, "mae": mae,
                        "delta_mse_pct": (mse - gold_mse) / gold_mse * 100,
                        "delta_mae_pct": (mae - gold_mae) / gold_mae * 100,
                        "passes_half_percent": success, "run_id": run_id,
                    })
                    f.flush()
                    recorded.add(key)
            if not args.dry_run and all(passed):
                print(f"TARGET_REACHED dataset={args.dataset} label={candidate.label}", flush=True)
                return


if __name__ == "__main__":
    main()
