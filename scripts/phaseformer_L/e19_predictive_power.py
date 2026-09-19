#!/usr/bin/env python3
"""E19 stage 2: minipaper §4.7 predictive-power table.

Correlates the three train-set level statistics produced by E19 stage 1
(``cycle_level_std``, ``last_cycle_shift``, ``tau_hat_steps``) against, per
setting:

* ``delta_mse_pct`` -- PhaseFormer-L's test MSE relative to the matched
  ``phase_only`` rerun (negative = the corrector helps), three-seed mean;
* ``gate_value``    -- the learned fusion gate of the PhaseFormer-L branch.

and reports Spearman rho for each of the six pairs, plus the diagnostic
prediction column ``s = 1[tau_hat_steps > nu_star]`` with its per-setting
hit/miss verdict.

Inputs are produced elsewhere and are never recomputed here:

* ``--stats``  : E19 stage 1 ``level_statistics.csv`` (28 rows, train-only);
* ``--results``: E14's test-read ``results.csv`` (one row per arm x setting x
  seed, from ``scripts/phaseformer_L/e14_read_test.py``).

This script reads no dataset and never touches the test split itself; it only
aggregates numbers another stage already recorded.

The frozen threshold (minipaper §3.4.3, decided 2026-09-18 from train-set
statistics only) is ``nu_star = 57.35`` steps, the midpoint of the separable
interval (51.11, 63.58) of the settings whose corrector sign is known from the
§4.1 pilot.  It is a constant here on purpose: the threshold must not be
re-fitted against the results this table reports.

Usage::

    python scripts/phaseformer_L/e19_predictive_power.py \
        --stats research_runs/phaseformer_L_e19_predictive_v1/level_statistics.csv \
        --results research_runs/phaseformer_L_e14_main_v1/results.csv \
        --output-root research_runs/phaseformer_L_e19_predictive_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]

# Frozen by minipaper §3.4.3 (2026-09-18).  Do not re-fit here.
NU = "tau_hat_steps"
NU_STAR = 57.35
NU_SEPARABLE_INTERVAL = (51.113, 63.578)

STATISTICS = ("cycle_level_std", "last_cycle_shift", "tau_hat_steps")
CORRECTOR_ARM = "l_main"      # PhaseFormer-L
BASELINE_ARM = "phase_only"
MIN_SEEDS_PER_SETTING = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--traffic",
        choices=["include", "exclude"],
        default="include",
        help="Traffic is an exploratory appendix; 'both' is reported regardless",
    )
    return parser.parse_args()


def read_stats(path: Path) -> dict:
    index = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], int(row["horizon"]))
            index[key] = {name: float(row[name]) for name in STATISTICS}
            index[key]["n_windows"] = int(row["n_windows"])
    return index


def read_results(path: Path) -> dict:
    """(arm, dataset, horizon) -> {'mse': [...], 'gate': [...], 'seeds': [...]}."""
    index: dict = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            arm = row.get("arm", "").strip()
            if arm not in (CORRECTOR_ARM, BASELINE_ARM):
                continue
            mse = str(row.get("test_mse", "")).strip()
            if not mse:
                continue
            try:
                mse_value = float(mse)
            except ValueError:
                continue
            key = (arm, row["dataset"], int(row["horizon"]))
            entry = index.setdefault(key, {"mse": [], "gate": [], "seeds": []})
            entry["mse"].append(mse_value)
            entry["seeds"].append(int(row["seed"]))
            gate = str(row.get("gate_value", "")).strip()
            if gate:
                try:
                    entry["gate"].append(float(gate))
                except ValueError:
                    pass
    return index


def spearman(x: list[float], y: list[float]) -> dict:
    n = len(x)
    if n < 3:
        return {"n": n, "rho": None, "p_value": None}
    result = scipy_stats.spearmanr(np.asarray(x, float), np.asarray(y, float))
    return {
        "n": n,
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def main() -> None:
    args = parse_args()
    out_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() \
        else Path(args.output_root)
    stats_index = read_stats(Path(args.stats))
    results_index = read_results(Path(args.results))

    rows = []
    incomplete = []
    for (dataset, horizon), stat in sorted(stats_index.items()):
        corr = results_index.get((CORRECTOR_ARM, dataset, horizon))
        base = results_index.get((BASELINE_ARM, dataset, horizon))
        if not corr or not base:
            incomplete.append({
                "setting": f"{dataset}-{horizon}",
                "reason": "missing " + ", ".join(
                    name for name, value in ((CORRECTOR_ARM, corr), (BASELINE_ARM, base))
                    if not value),
            })
            continue
        n_seeds = min(len(corr["mse"]), len(base["mse"]))
        if n_seeds < MIN_SEEDS_PER_SETTING:
            incomplete.append({
                "setting": f"{dataset}-{horizon}",
                "reason": f"only {n_seeds} completed seed(s) for both arms",
            })
            continue
        mse_corr = float(np.mean(corr["mse"]))
        mse_base = float(np.mean(base["mse"]))
        gate = float(np.mean(corr["gate"])) if corr["gate"] else None
        row = {
            "dataset": dataset,
            "horizon": int(horizon),
            "setting": f"{dataset}-{horizon}",
            "is_traffic_appendix": dataset == "Traffic",
            "n_seeds": n_seeds,
            "n_windows": stat["n_windows"],
            "phase_only_mse": round(mse_base, 6),
            "phaseformer_l_mse": round(mse_corr, 6),
            "delta_mse_pct": round(100.0 * (mse_corr - mse_base) / mse_base, 4),
            "gate_value": None if gate is None else round(gate, 6),
            "diagnostic_s": int(stat[NU] > NU_STAR),
        }
        for name in STATISTICS:
            row[name] = round(stat[name], 6)
        # Pre-registered prediction: s=1 means "this dataset needs the level
        # channel", so a hit is (s == 1 and the corrector helps) or
        # (s == 0 and it does not help).
        row["corrector_helps"] = int(row["delta_mse_pct"] < 0.0)
        row["s_prediction_hit"] = int(bool(row["diagnostic_s"])
                                      == bool(row["corrector_helps"]))
        rows.append(row)

    if not rows:
        raise SystemExit("no complete settings; nothing to correlate")

    def correlate(subset, label):
        block = {"scope": label, "n_settings": len(subset), "spearman": {}}
        for name in STATISTICS:
            x = [r[name] for r in subset]
            block["spearman"][f"{name}_vs_delta_mse_pct"] = spearman(
                x, [r["delta_mse_pct"] for r in subset])
            with_gate = [r for r in subset if r["gate_value"] is not None]
            block["spearman"][f"{name}_vs_gate_value"] = spearman(
                [r[name] for r in with_gate], [r["gate_value"] for r in with_gate])
        return block

    main_rows = [r for r in rows if not r["is_traffic_appendix"]]
    traffic_rows = [r for r in rows if r["is_traffic_appendix"]]
    table = correlate(rows, "all_28_settings")
    sensitivity = correlate(main_rows, "main_24_settings")

    hits = [r for r in rows if r["s_prediction_hit"]]
    misses = [r for r in rows
              if not r["s_prediction_hit"]]
    per_dataset = {}
    for row in rows:
        entry = per_dataset.setdefault(row["dataset"], {
            "settings": 0, "hits": 0, "delta_mse_pct": [], "tau_hat_steps": None,
            "diagnostic_s": row["diagnostic_s"],
        })
        entry["settings"] += 1
        entry["hits"] += row["s_prediction_hit"]
        entry["delta_mse_pct"].append(row["delta_mse_pct"])
        entry["tau_hat_steps"] = row["tau_hat_steps"]
    for entry in per_dataset.values():
        entry["mean_delta_mse_pct"] = round(
            float(np.mean(entry["delta_mse_pct"])), 4)
        entry["corrector_helps"] = bool(entry["mean_delta_mse_pct"] < 0)
        entry["all_settings_hit"] = entry["hits"] == entry["settings"]
        entry.pop("delta_mse_pct")

    summary = {
        "experiment": "E19 stage 2 (minipaper §4.7 predictive power)",
        "reads_test": False,
        "inputs": {"stats": str(args.stats), "results": str(args.results)},
        "frozen_threshold": {
            "nu": NU,
            "nu_star": NU_STAR,
            "separable_interval": list(NU_SEPARABLE_INTERVAL),
            "fitted_on": "5 datasets with known corrector sign from minipaper §4.1 "
                         "(train-set statistics only)",
            "note": "constant, deliberately not re-fitted against these results",
        },
        "settings_with_data": len(rows),
        "settings_without_data": incomplete,
        "predictive_power": table,
        "sensitivity_main_24": sensitivity,
        "diagnostic_accuracy": {
            "settings": len(rows),
            "hits": len(hits),
            "misses": len(misses),
            "hit_rate": round(len(hits) / len(rows), 4),
            "missed_settings": [r["setting"] for r in misses],
            "per_dataset": per_dataset,
        },
    }

    out_root.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "horizon", "setting", "is_traffic_appendix", "n_seeds",
              "n_windows", "phase_only_mse", "phaseformer_l_mse", "delta_mse_pct",
              "gate_value", "diagnostic_s", "corrector_helps", "s_prediction_hit",
              *STATISTICS]
    with (out_root / "predictive_power.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "predictive_power_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "event": "finished",
        "settings_with_data": len(rows),
        "settings_without_data": len(incomplete),
        "rho_delta_mse": {k: v["rho"] for k, v in table["spearman"].items()
                          if "delta_mse" in k},
        "diagnostic_hit_rate": summary["diagnostic_accuracy"]["hit_rate"],
        "missed": summary["diagnostic_accuracy"]["missed_settings"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
