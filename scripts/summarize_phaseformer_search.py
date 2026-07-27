#!/usr/bin/env python3
"""Build unified leaderboard and validation Pareto tables from search artifacts."""
import argparse
import csv
from pathlib import Path


def read_rows(root):
    rows = []
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        with path.open() as f:
            found = list(csv.DictReader(f))
        if found:
            rows.append(found[0])
    return rows


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def write(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="research_runs/search_v1")
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root)
    baselines = {}
    for row in rows:
        if row["stage"] == "baseline" and row["mechanism"] == "original":
            baselines[(row["dataset"], row["horizon"], row["seed"])] = row
    leaderboard = []
    for row in rows:
        item = dict(row)
        baseline = baselines.get((row["dataset"], row["horizon"], row["seed"]))
        if baseline:
            mae_imp = (f(baseline, "val_mae") - f(row, "val_mae")) / f(baseline, "val_mae") * 100
            mse_imp = (f(baseline, "val_mse") - f(row, "val_mse")) / f(baseline, "val_mse") * 100
            item["val_mae_improvement_pct"] = mae_imp
            item["val_mse_improvement_pct"] = mse_imp
            item["score"] = 0.5 * mae_imp + 0.5 * mse_imp
            item["eliminated_regression"] = str(mae_imp < -0.5 or mse_imp < -0.5).lower()
        else:
            item.update({
                "val_mae_improvement_pct": "", "val_mse_improvement_pct": "",
                "score": "", "eliminated_regression": "",
            })
        leaderboard.append(item)
    leaderboard.sort(key=lambda x: (x["dataset"], int(x["horizon"]), x["stage"], -f(x, "score")))

    grouped = {}
    for row in leaderboard:
        grouped.setdefault((row["dataset"], row["horizon"], row["stage"]), []).append(row)
    pareto = []
    for _, group in grouped.items():
        for row in group:
            dominated = any(
                f(other, "val_mae") <= f(row, "val_mae")
                and f(other, "val_mse") <= f(row, "val_mse")
                and (f(other, "val_mae") < f(row, "val_mae") or f(other, "val_mse") < f(row, "val_mse"))
                for other in group if other is not row
            )
            if not dominated:
                item = dict(row)
                item["pareto"] = "true"
                pareto.append(item)
    write(root / "leaderboard.csv", leaderboard)
    write(root / "pareto.csv", pareto)
    print(f"runs={len(rows)} leaderboard={root/'leaderboard.csv'} pareto={root/'pareto.csv'}")


if __name__ == "__main__":
    main()
