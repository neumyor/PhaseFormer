#!/usr/bin/env python3
"""Select one direction-1 neighborhood width per dataset from test metrics.

This script deliberately performs test-set selection.  It must only be used for
the exploratory protocol documented in
``docs/PhaseFormer_direction1_neighborhood_experiment_plan.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_top2_direction_retention import row_from_metrics  # noqa: E402
from scripts.run_direction1_neighborhood_matrix import (  # noqa: E402
    ALL_SETTINGS,
    SWEEP_WIDTHS,
    cone_arm,
)

MSE_TIE_TOLERANCE_PCT = 0.10


def load_rows(root: Path) -> dict:
    rows = {}
    for path in sorted((root / "runs").glob("*/metrics.csv")):
        row = row_from_metrics(path)
        if row is None:
            continue
        key = (row["dataset"], row["horizon"], row["seed"], row["arm"])
        rows[key] = row
    return rows


def relative_pct(candidate, reference):
    if candidate is None or reference in (None, 0):
        return None
    return (candidate / reference - 1.0) * 100.0


def dataset_settings() -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for dataset, horizon in ALL_SETTINGS:
        grouped.setdefault(dataset, []).append(horizon)
    return grouped


def evaluate_dataset(dataset: str, horizons: list[int], rows: dict) -> dict:
    candidates = []
    complete = True
    for width in SWEEP_WIDTHS:
        setting_rows = []
        for horizon in horizons:
            direct = rows.get((dataset, horizon, 2021, "direct_nlinear"), {})
            cone = rows.get((dataset, horizon, 2021, cone_arm(width)), {})
            rrr2 = rows.get((dataset, horizon, 2021, "rrr_direction_1_2"), {})
            values = (
                direct.get("test_mse"),
                direct.get("test_mae"),
                cone.get("test_mse"),
                cone.get("test_mae"),
                rrr2.get("test_mse"),
                rrr2.get("test_mae"),
            )
            complete = complete and all(
                value is not None and math.isfinite(float(value))
                for value in values
            )
            setting_rows.append(
                {
                    "setting": f"{dataset}-{horizon}",
                    "direct_mse": direct.get("test_mse"),
                    "direct_mae": direct.get("test_mae"),
                    "rrr2_mse": rrr2.get("test_mse"),
                    "rrr2_mae": rrr2.get("test_mae"),
                    "cone_mse": cone.get("test_mse"),
                    "cone_mae": cone.get("test_mae"),
                    "delta_mse_vs_direct_pct": relative_pct(
                        cone.get("test_mse"), direct.get("test_mse")
                    ),
                    "delta_mae_vs_direct_pct": relative_pct(
                        cone.get("test_mae"), direct.get("test_mae")
                    ),
                    "delta_mse_vs_rrr2_pct": relative_pct(
                        cone.get("test_mse"), rrr2.get("test_mse")
                    ),
                    "delta_mae_vs_rrr2_pct": relative_pct(
                        cone.get("test_mae"), rrr2.get("test_mae")
                    ),
                }
            )
        mse_values = [
            row["delta_mse_vs_direct_pct"]
            for row in setting_rows
            if row["delta_mse_vs_direct_pct"] is not None
        ]
        mae_values = [
            row["delta_mae_vs_direct_pct"]
            for row in setting_rows
            if row["delta_mae_vs_direct_pct"] is not None
        ]
        candidates.append(
            {
                "width": width,
                "macro_delta_mse_vs_direct_pct": (
                    sum(mse_values) / len(mse_values) if mse_values else None
                ),
                "macro_delta_mae_vs_direct_pct": (
                    sum(mae_values) / len(mae_values) if mae_values else None
                ),
                "settings": setting_rows,
            }
        )

    if not complete:
        return {
            "dataset": dataset,
            "complete": False,
            "selected_width": None,
            "candidates": candidates,
        }

    best_mse = min(item["macro_delta_mse_vs_direct_pct"] for item in candidates)
    near_best = [
        item
        for item in candidates
        if item["macro_delta_mse_vs_direct_pct"]
        <= best_mse + MSE_TIE_TOLERANCE_PCT
    ]
    selected = min(
        near_best,
        key=lambda item: (
            item["macro_delta_mae_vs_direct_pct"],
            item["width"],
        ),
    )
    return {
        "dataset": dataset,
        "complete": True,
        "selected_width": selected["width"],
        "best_macro_delta_mse_vs_direct_pct": best_mse,
        "selection_pool": [item["width"] for item in near_best],
        "candidates": candidates,
    }


def fmt(value, digits=3, suffix=""):
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def render_markdown(evaluations: list[dict]) -> str:
    lines = [
        "# Direction-1 neighborhood test-set selection",
        "",
        "> This is explicit test-set selection and is not an unbiased generalization estimate.",
        "",
        "| Dataset | k | macro ΔMSE vs direct | macro ΔMAE vs direct | selected |",
        "|---|---:|---:|---:|---|",
    ]
    for evaluation in evaluations:
        selected = evaluation["selected_width"]
        for candidate in evaluation["candidates"]:
            lines.append(
                f"| {evaluation['dataset']} | {candidate['width']} | "
                f"{fmt(candidate['macro_delta_mse_vs_direct_pct'], 2, '%')} | "
                f"{fmt(candidate['macro_delta_mae_vs_direct_pct'], 2, '%')} | "
                f"{'YES' if candidate['width'] == selected else ''} |"
            )
    lines.extend(
        [
            "",
            "Selection rule: minimize dataset-level macro test MSE versus direct;",
            f"widths within {MSE_TIE_TOLERANCE_PCT:.2f} percentage points use macro "
            "test MAE, then the smaller width, as tie-breakers.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="research_runs/direction1_neighborhood_v1"
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    rows = load_rows(root)
    evaluations = [
        evaluate_dataset(dataset, horizons, rows)
        for dataset, horizons in dataset_settings().items()
    ]
    complete = all(item["complete"] for item in evaluations)
    selected = {
        item["dataset"]: item["selected_width"]
        for item in evaluations
        if item["selected_width"] is not None
    }
    payload = {
        "protocol": "direction1-bootstrap-neighborhood-test-selection-v1",
        "test_set_selection": True,
        "selection_scope": "one shared width per dataset",
        "seed_used_for_selection": 2021,
        "candidate_widths": list(SWEEP_WIDTHS),
        "mse_tie_tolerance_percentage_points": MSE_TIE_TOLERANCE_PCT,
        "status": "selected" if complete else "incomplete",
        "selected_width_by_dataset": selected,
        "evaluations": evaluations,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "test_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (root / "test_selection.md").write_text(render_markdown(evaluations))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not complete:
        raise SystemExit("test sweep is incomplete")


if __name__ == "__main__":
    main()
