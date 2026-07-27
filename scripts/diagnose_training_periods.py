#!/usr/bin/env python3
"""Training-split-only period diagnostics required by EXPERIMENT_SEARCH_PLAN.md."""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.data_info import DATASET_INFO

CANDIDATES = {
    "ETTh1": [12, 24, 48], "ETTh2": [12, 24, 48],
    "ETTm1": [24, 48, 96], "ETTm2": [24, 48, 96],
    "Weather": [12, 24, 48], "Electricity": [12, 24, 48], "Traffic": [12, 24, 48],
    "Exchange": [7, 14, 30],
}


def training_length(dataset, total):
    if dataset in ("ETTh1", "ETTh2"):
        return 12 * 30 * 24
    if dataset in ("ETTm1", "ETTm2"):
        return 12 * 30 * 24 * 4
    return int(total * 0.7)


def lag_correlation(values, lag):
    left, right = values[:-lag], values[lag:]
    left = left - np.nanmean(left, axis=0, keepdims=True)
    right = right - np.nanmean(right, axis=0, keepdims=True)
    numerator = np.nansum(left * right, axis=0)
    denominator = np.sqrt(np.nansum(left * left, axis=0) * np.nansum(right * right, axis=0))
    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return float(np.nanmean(corr)), float(np.nanmedian(corr)), float(np.mean(corr > 0.3))


def main():
    root = Path("research_runs/search_v1/diagnostics")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, periods in CANDIDATES.items():
        info = DATASET_INFO[dataset]
        data_root = Path(info["root_path"])
        if not data_root.exists() and dataset.startswith("ETT"):
            data_root = Path("resources/all_datasets/ETT-small")
        path = data_root / info["data_path"]
        frame = pd.read_csv(path)
        train_rows = training_length(dataset, len(frame))
        values = frame.iloc[:train_rows, 1:].to_numpy(dtype=np.float64)
        for period in periods:
            mean_corr, median_corr, strong_fraction = lag_correlation(values, period)
            rows.append({
                "dataset": dataset, "source": str(path), "split": "train",
                "train_rows": train_rows, "variables": values.shape[1], "period": period,
                "mean_lag_correlation": mean_corr, "median_lag_correlation": median_corr,
                "fraction_variables_corr_gt_0_3": strong_fraction,
            })
    with (root / "training_periods.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root / "metadata.json").write_text(json.dumps({
        "protocol": "training-split-only", "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "per-variable Pearson lag correlation; aggregate mean/median and fraction > 0.3",
        "candidate_periods": CANDIDATES,
    }, indent=2) + "\n")
    print(root / "training_periods.csv")


if __name__ == "__main__":
    main()
