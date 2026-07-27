#!/usr/bin/env python3
"""Build unified leaderboard and validation Pareto tables from search artifacts."""
import argparse
import csv
import json
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


def select_reference(row, rows, formal):
    stage = row["stage"]
    candidates = [x for x in rows if x["dataset"] == row["dataset"] and x["horizon"] == row["horizon"] and x["seed"] == row["seed"]]
    if stage == "period_screen":
        matches = [x for x in candidates if x["stage"] == stage and x["mechanism"] == "original" and x["period"] == "24"]
        if matches:
            return matches[0], "same_budget_original_p24"
    if stage.startswith("mechanism_screen") or stage == "mechanism_full8":
        matches = [x for x in candidates if x["stage"] == stage and x["mechanism"] == "original" and x["period"] == row["period"]]
        if matches:
            return matches[0], "same_budget_original"
    if stage in ("hp_low", "hp_mid"):
        matches = [x for x in candidates if x["stage"] == stage and x["mechanism"] == row["mechanism"] and x["period"] == row["period"] and x["capacity"] == "base" and x["loss"] == "huber" and abs(f(x, "lr_multiplier") - 1.0) < 1e-12]
        if matches:
            return matches[0], "same_budget_huber_1x_base"
    if formal:
        return formal, "formal_original_baseline"
    return None, "missing"


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
        formal = baselines.get((row["dataset"], row["horizon"], row["seed"]))
        reference, reference_kind = select_reference(row, rows, formal)
        item["reference_kind"] = reference_kind
        item["reference_run_id"] = reference["run_id"] if reference else ""
        if reference:
            mae_imp = (f(reference, "val_mae") - f(row, "val_mae")) / f(reference, "val_mae") * 100
            mse_imp = (f(reference, "val_mse") - f(row, "val_mse")) / f(reference, "val_mse") * 100
            item["val_mae_improvement_pct"] = mae_imp
            item["val_mse_improvement_pct"] = mse_imp
            item["score"] = 0.5 * mae_imp + 0.5 * mse_imp
            item["eliminated_regression"] = str(mae_imp < -0.5 or mse_imp < -0.5).lower()
        else:
            item.update({"val_mae_improvement_pct": "", "val_mse_improvement_pct": "", "score": "", "eliminated_regression": ""})
        if formal:
            item["formal_val_mae_improvement_pct"] = (f(formal, "val_mae") - f(row, "val_mae")) / f(formal, "val_mae") * 100
            item["formal_val_mse_improvement_pct"] = (f(formal, "val_mse") - f(row, "val_mse")) / f(formal, "val_mse") * 100
        else:
            item["formal_val_mae_improvement_pct"] = ""
            item["formal_val_mse_improvement_pct"] = ""
        leaderboard.append(item)
    leaderboard.sort(key=lambda x: (x["dataset"], int(x["horizon"]), x["stage"], -f(x, "score")))

    grouped = {}
    for row in leaderboard:
        grouped.setdefault((row["dataset"], row["horizon"], row["stage"]), []).append(row)
    pareto = []
    for _, group in grouped.items():
        for row in group:
            dominated = any(
                f(other, "val_mae") <= f(row, "val_mae") and f(other, "val_mse") <= f(row, "val_mse")
                and (f(other, "val_mae") < f(row, "val_mae") or f(other, "val_mse") < f(row, "val_mse"))
                for other in group if other is not row
            )
            if not dominated:
                item = dict(row); item["pareto"] = "true"; pareto.append(item)
    write(root / "leaderboard.csv", leaderboard)
    write(root / "pareto.csv", pareto)

    failures = []
    for status_path in sorted((root / "runs").glob("*/status.json")):
        try:
            status = json.loads(status_path.read_text())
        except Exception:
            continue
        if status.get("status") == "failed":
            failures.append({"run_id": status_path.parent.name, "failed_at": status.get("failed_at", ""), "error": status.get("error", "")})
    if failures:
        write(root / "failures.csv", failures)
    print(f"runs={len(rows)} failures={len(failures)} leaderboard={root/'leaderboard.csv'} pareto={root/'pareto.csv'}")


if __name__ == "__main__":
    main()
