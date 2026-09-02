#!/usr/bin/env python3
"""Run the selected shared strict-T28 configurations on H336/H720.

Selection happened on the preceding H96/H192 test-selection ledgers.  These
long-horizon values are therefore an extension check, not an unbiased test.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from run_strict_t28_golden_refinement import Candidate, command, read_metrics
from run_strict_t28_golden_hunt import ROOT


GOLDEN = {
    "ETTh1": {336: (.425, .424), 720: (.431, .450)},
    "ETTm1": {336: (.358, .381), 720: (.412, .410)},
}
FIELDS = (
    "dataset", "horizon", "selected_config", "cycle", "loss", "lr_multiplier",
    "max_epochs", "overrides_json", "mse", "mae", "delta_mse_pct",
    "delta_mae_pct", "run_id",
)


def selected(dataset: str) -> Candidate:
    if dataset == "ETTh1":
        # u_lr020: best shared H96/H192 Golden-normalized four-metric mean.
        return Candidate("u_lr020", 24, 1.40, .80, .40, "mae", .20, 50)
    # w_aux01: best shared H96/H192 Golden-normalized four-metric mean.
    return Candidate("w_aux01", 24, .60, .24, .12, "mae", .20, 50, (
        ("anchored_pctf_shape_aux_weight", .01),
        ("anchored_pctf_level_aux_weight", .01),
        ("anchored_pctf_gate_aux_weight", .01),
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=tuple(GOLDEN) + ("all",), default="all")
    p.add_argument("--output-dir", default="research_runs/strict_t28_best_long_horizons")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    output = ROOT / a.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "test_selection_results.csv"
    new = not summary.exists()
    seen = set()
    if not new:
        with summary.open(newline="") as f:
            seen = {(r["dataset"], r["horizon"]) for r in csv.DictReader(f)}
    datasets = tuple(GOLDEN) if a.dataset == "all" else (a.dataset,)
    with summary.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for dataset in datasets:
            candidate = selected(dataset)
            for horizon, (gm, ga) in GOLDEN[dataset].items():
                metrics = read_metrics(output, dataset, horizon, candidate)
                if metrics is None:
                    cmd = command(dataset, horizon, candidate, output)
                    if a.dry_run:
                        print(" ".join(cmd))
                        continue
                    for _ in range(3):
                        if subprocess.run(cmd, cwd=ROOT).returncode == 0:
                            break
                    else:
                        raise RuntimeError(f"failed after retries: {dataset} H{horizon}")
                    metrics = read_metrics(output, dataset, horizon, candidate)
                    if metrics is None:
                        raise RuntimeError("runner completed but metrics were not found")
                mse, mae, run_id = metrics
                key = (dataset, str(horizon))
                if key not in seen:
                    writer.writerow(dict(
                        dataset=dataset, horizon=horizon, selected_config=candidate.label,
                        cycle=candidate.cycle, loss=candidate.loss,
                        lr_multiplier=candidate.lr, max_epochs=candidate.epochs,
                        overrides_json=json.dumps(candidate.overrides(), sort_keys=True),
                        mse=mse, mae=mae, delta_mse_pct=(mse-gm)/gm*100,
                        delta_mae_pct=(mae-ga)/ga*100, run_id=run_id,
                    ))
                    f.flush()


if __name__ == "__main__":
    main()
