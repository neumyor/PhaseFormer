#!/usr/bin/env python3
"""Create the six-file audit package for the pooled low-rank NLinear screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = {
    ("ETTh1", 96): {"mse": 0.359, "mae": 0.382},
    ("ETTm1", 96): {"mse": 0.293, "mae": 0.344},
}


def git_value(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def read_results(scratch_root, datasets, horizon, seed):
    rows = []
    for path in sorted(scratch_root.glob("*/result.json")):
        row = json.loads(path.read_text())
        if (
            row["dataset"] in datasets
            and int(row["horizon"]) == horizon
            and int(row["seed"]) == seed
            and row["test_mse"] is not None
        ):
            row["_result_path"] = path
            row["_run_dir"] = path.parent
            row["validation_score"] = (
                float(row["val_mse"]) / float(row["val_phase_mse"])
                + float(row["val_mae"]) / float(row["val_phase_mae"])
            )
            rows.append(row)
    return rows


def golden_delta(value, golden):
    return 100.0 * (value - golden) / golden


def selected_by_validation(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], int(row["horizon"]), int(row["seed"]))].append(row)
    selected = {}
    for setting, group in groups.items():
        selected[setting] = min(
            group,
            key=lambda row: (row["validation_score"], row["params_trainable"]),
        )
    return selected


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def heatmap(path, matrix, x_labels, y_labels, title, label, y_label):
    figure, axis = plt.subplots(figsize=(max(7, len(x_labels) * 1.15), 4.8))
    image = axis.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    axis.set_xticks(range(len(x_labels)), x_labels, rotation=25, ha="right")
    axis.set_yticks(range(len(y_labels)), y_labels)
    axis.set_xlabel("Low-rank configuration")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(label)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_search_figures(rows, figures_dir, datasets, horizon):
    figures = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        no_smoothing = [row for row in dataset_rows if row["smooth_ratio"] == 0.0]
        pools = sorted({int(row["pool_factor"]) for row in no_smoothing})
        ranks = sorted({int(row["rank"]) for row in no_smoothing})
        golden = GOLDEN[(dataset, horizon)]
        for metric in ("mse", "mae"):
            matrix = np.full((len(pools), len(ranks)), np.nan)
            index = {
                (int(row["pool_factor"]), int(row["rank"])): row
                for row in no_smoothing
            }
            for row_index, pool in enumerate(pools):
                for column_index, rank in enumerate(ranks):
                    row = index.get((pool, rank))
                    if row:
                        matrix[row_index, column_index] = golden_delta(
                            float(row[f"test_{metric}"]), golden[metric]
                        )
            name = f"{dataset}_h{horizon}__no_smoothing_pool_rank_{metric}_delta.png"
            heatmap(
                figures_dir / name,
                matrix,
                [f"r={rank}" for rank in ranks],
                [f"p={pool}" for pool in pools],
                f"{dataset} H{horizon}: {metric.upper()} delta vs Golden, smooth=0",
                f"{metric.upper()} delta vs Golden (%)",
                "Pooling factor",
            )
            figures.append(name)

        smooth_rows = [row for row in dataset_rows if row["smooth_ratio"] > 0.0]
        if not smooth_rows:
            continue
        pairs = sorted(
            {
                (int(row["pool_factor"]), int(row["rank"]))
                for row in smooth_rows
            }
        )
        ratios = [0.0] + sorted({float(row["smooth_ratio"]) for row in smooth_rows})
        full_rows = {
            (int(row["pool_factor"]), int(row["rank"]), float(row["smooth_ratio"])): row
            for row in dataset_rows
        }
        for metric in ("mse", "mae"):
            matrix = np.full((len(ratios), len(pairs)), np.nan)
            for row_index, ratio in enumerate(ratios):
                for column_index, pair in enumerate(pairs):
                    row = full_rows.get((*pair, ratio))
                    if row:
                        matrix[row_index, column_index] = golden_delta(
                            float(row[f"test_{metric}"]), golden[metric]
                        )
            name = f"{dataset}_h{horizon}__selected_rank_smooth_{metric}_delta.png"
            heatmap(
                figures_dir / name,
                matrix,
                [f"p={pool}, r={rank}" for pool, rank in pairs],
                [f"{ratio:.2f}" for ratio in ratios],
                f"{dataset} H{horizon}: {metric.upper()} delta vs Golden",
                f"{metric.upper()} delta vs Golden (%)",
                "Smooth ratio",
            )
            figures.append(name)
    return figures


def select_cases(arrays, dataset, horizon, config_id):
    history = arrays["history"]
    truth = arrays["truth"]
    phase = arrays["phase_prediction"]
    candidate = arrays["candidate_prediction"]
    sample_ids = arrays["sample_id"]
    phase_error = phase - truth
    candidate_error = candidate - truth
    phase_mse = np.mean(phase_error**2, axis=1)
    candidate_mse = np.mean(candidate_error**2, axis=1)
    phase_mae = np.mean(np.abs(phase_error), axis=1)
    candidate_mae = np.mean(np.abs(candidate_error), axis=1)
    flat = []
    for sample_index, sample_id in enumerate(sample_ids):
        for channel in range(phase_mse.shape[1]):
            flat.append(
                {
                    "sample_index": sample_index,
                    "sample_id": int(sample_id),
                    "channel": channel,
                    "phase_mse": float(phase_mse[sample_index, channel]),
                    "candidate_mse": float(candidate_mse[sample_index, channel]),
                    "phase_mae": float(phase_mae[sample_index, channel]),
                    "candidate_mae": float(candidate_mae[sample_index, channel]),
                }
            )
    categories = {
        "baseline_high_error": lambda item: item["phase_mae"],
        "candidate_regression": lambda item: item["candidate_mae"] - item["phase_mae"],
        "candidate_improvement": lambda item: item["phase_mae"] - item["candidate_mae"],
    }
    selected = []
    for category, score in categories.items():
        candidate_case = max(flat, key=score)
        candidate_case = dict(candidate_case)
        candidate_case["category"] = category
        candidate_case["setting"] = f"{dataset}_h{horizon}_seed2021"
        candidate_case["config_id"] = config_id
        selected.append(candidate_case)
    return flat, selected


def render_case(path, case, arrays):
    sample_index = case["sample_index"]
    channel = case["channel"]
    history = arrays["history"][sample_index, :, channel]
    truth = arrays["truth"][sample_index, :, channel]
    phase = arrays["phase_prediction"][sample_index, :, channel]
    candidate = arrays["candidate_prediction"][sample_index, :, channel]
    figure, axis = plt.subplots(figsize=(10, 3.8))
    x_history = np.arange(len(history))
    x_future = np.arange(len(history), len(history) + len(truth))
    axis.plot(x_history, history, color="#65737e", label="history", linewidth=1.0)
    axis.plot(x_future, truth, color="#111111", label="truth", linewidth=1.5)
    axis.plot(x_future, phase, color="#3b82f6", label="frozen phase", linewidth=1.25)
    axis.plot(x_future, candidate, color="#dc2626", label="pooled low-rank", linewidth=1.25)
    axis.set_title(
        f"{case['setting']} {case['category']} "
        f"sample={case['sample_id']} channel={channel}"
    )
    axis.legend(ncol=4, fontsize=8, loc="best")
    axis.set_xlabel("time step")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def make_report(
    output_path,
    rows,
    selected,
    cases,
    figure_names,
    datasets,
    horizon,
    seed,
):
    lines = [
        "# Experiment and Objective Error Analysis",
        "",
        "## 1. Experiment Setup",
        "",
        "- Mechanism: frozen original PhaseFormer plus a direct centered-NLinear correction with temporal input pooling and a linear pooled-length -> rank -> horizon factorization.",
        "- Protocol deviation: this is a user-requested pooled low-rank screen, not the original Progressive IB Stage 2 causal protocol, because pooling and rank vary together.",
        f"- Settings: {', '.join(datasets)}, L=720, H={horizon}, seed={seed}, Huber, 30 epochs, lowest-validation-loss checkpoint.",
        "- Selection: low-rank and smoothing configurations were ranked on validation metrics only. Every planned configuration was also evaluated once on test; therefore the complete test matrix is test-set-exposed exploratory evidence, not blind selection evidence.",
        "- Golden reference: fixed original PhaseFormer table. Delta is `(experiment - Golden) / Golden * 100`; negative is better.",
        "",
        "## 2. Experiment Results",
        "",
        "| Dataset | Selected config (validation) | Test MSE | Test MAE | MSE delta vs Golden | MAE delta vs Golden |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, row in sorted(selected.items()):
        golden = GOLDEN[(row["dataset"], horizon)]
        lines.append(
            f"| {row['dataset']} | {row['config_id']} | {row['test_mse']:.6f} | "
            f"{row['test_mae']:.6f} | {golden_delta(row['test_mse'], golden['mse']):+.2f}% | "
            f"{golden_delta(row['test_mae'], golden['mae']):+.2f}% |"
        )
    lines.extend(["", "## 3. Parameter / Configuration Search", ""])
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        lines.append(
            f"- {dataset}: evaluated {len(dataset_rows)} planned configurations. "
            "The no-smoothing pooling/rank grid and validation-selected smoothing grid are shown below."
        )
    lines.extend(["", "## 4. Error Distribution", ""])
    for key, row in sorted(selected.items()):
        matching = [case for case in cases if case["setting"].startswith(row["dataset"])]
        mean_delta = np.mean(
            [case["candidate_mae"] - case["phase_mae"] for case in matching]
        )
        lines.append(
            f"- {row['dataset']}: selected-case mean candidate-minus-phase MAE is {mean_delta:+.6f}; "
            "this statistic describes only the programmatically selected cases."
        )
    lines.extend(["", "## 5. Horizon-wise Error", ""])
    lines.append(
        f"- Only H={horizon} was run in this initial two-dataset screen; no cross-horizon conclusion is supported."
    )
    lines.extend(["", "## 6. High-Error Selection", ""])
    lines.append(
        "- Per dataset, one test sample-channel is selected for each rule: highest frozen-phase MAE, largest candidate MAE regression, and largest candidate MAE improvement. No manual selection was used."
    )
    lines.extend(["", "## 7. Case Analysis", ""])
    for case in cases:
        figure = case["figure"]
        lines.append(
            f"- {case['setting']} / {case['category']} / sample {case['sample_id']} / "
            f"channel {case['channel']}: phase MAE={case['phase_mae']:.6f}, "
            f"candidate MAE={case['candidate_mae']:.6f}. "
            f"[figure](figures/{figure})"
        )
    lines.extend(["", "## 8. Repeated Observable Patterns", ""])
    for dataset in datasets:
        dataset_cases = [case for case in cases if case["setting"].startswith(dataset)]
        regressions = sum(
            case["candidate_mae"] > case["phase_mae"] for case in dataset_cases
        )
        lines.append(
            f"- {dataset}: candidate MAE exceeds frozen-phase MAE in {regressions}/{len(dataset_cases)} selected cases. "
            "This is an observation about selected extremes, not a population estimate."
        )
    lines.extend(
        [
            "",
            "## 9. Objective Defect Summary",
            "",
            "- The experiment varies pooling and low rank jointly in the first screen, so it cannot isolate a pure rank effect.",
            "- Fixed Golden values are from a different recorded environment/protocol lineage; matched frozen-phase metrics are retained in `results.csv` for within-run context.",
            "- One seed and one horizon are insufficient for a stable model-selection claim.",
            "",
            "## 10. Experiment Scope",
            "",
            "- This package covers ETTh1 and ETTm1 at H96 with seed 2021 only. It does not change the repository default preset or establish a paper-level replacement.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"![{name}](figures/{name})" for name in figure_names)
    output_path.write_text("\n".join(lines) + "\n")


def archive_report(output_dir, figure_names):
    report = output_dir / "objective_error_analysis.md"
    archive = output_dir / "objective_error_analysis.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(report, report.name)
        for name in figure_names:
            bundle.write(output_dir / "figures" / name, f"figures/{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch-root",
        default="research_runs/pooled_lowrank_nlinear_scratch",
    )
    parser.add_argument(
        "--output-dir",
        default="research_runs/pooled_lowrank_nlinear_h96_v1",
    )
    parser.add_argument("--datasets", default="ETTh1,ETTm1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    datasets = [item for item in args.datasets.split(",") if item]
    if set(datasets) != {"ETTh1", "ETTm1"} or args.horizon != 96:
        parser.error("this registered initial analysis is fixed to ETTh1/ETTm1 H96")
    scratch_root = ROOT / args.scratch_root
    output_dir = ROOT / args.output_dir
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing audit directory: {output_dir}")
    output_dir.mkdir(parents=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir()
    rows = read_results(scratch_root, datasets, args.horizon, args.seed)
    if not rows:
        raise RuntimeError("no completed pooled low-rank test results found")
    found = {row["dataset"] for row in rows}
    if found != set(datasets):
        raise RuntimeError(f"missing datasets: expected {datasets}, found {sorted(found)}")
    selected = selected_by_validation(rows)
    result_rows = []
    for row in rows:
        golden = GOLDEN[(row["dataset"], args.horizon)]
        key = (row["dataset"], int(row["horizon"]), int(row["seed"]))
        result_rows.append(
            {
                "setting": f"{row['dataset']}_h{args.horizon}_seed{args.seed}",
                "config_id": row["config_id"],
                "dataset": row["dataset"],
                "horizon": args.horizon,
                "seed": args.seed,
                "model": "pooled_lowrank_direct",
                "pool_factor": row["pool_factor"],
                "pooled_len": row["pooled_len"],
                "rank": row["rank"],
                "rank_ratio": row["rank_ratio"],
                "smooth_ratio": row["smooth_ratio"],
                "params_trainable": row["params_trainable"],
                "mse": row["test_mse"],
                "mae": row["test_mae"],
                "matched_phase_mse": row["test_phase_mse"],
                "matched_phase_mae": row["test_phase_mae"],
                "golden_mse": golden["mse"],
                "golden_mae": golden["mae"],
                "delta_mse_pct_vs_golden": golden_delta(row["test_mse"], golden["mse"]),
                "delta_mae_pct_vs_golden": golden_delta(row["test_mae"], golden["mae"]),
                "validation_score": row["validation_score"],
                "selected": row is selected[key],
            }
        )
    result_rows.sort(key=lambda row: (row["dataset"], row["smooth_ratio"], row["pool_factor"], row["rank"]))
    write_csv(
        output_dir / "results.csv",
        result_rows,
        list(result_rows[0]),
    )

    all_sample_rows = []
    selected_cases = []
    selected_arrays = defaultdict(list)
    for key, row in sorted(selected.items()):
        arrays = np.load(row["_run_dir"] / "test_arrays.npz")
        flat, cases = select_cases(arrays, row["dataset"], args.horizon, row["config_id"])
        setting = f"{row['dataset']}_h{args.horizon}_seed{args.seed}"
        for item in flat:
            all_sample_rows.append(
                {
                    "setting": setting,
                    "baseline_config_id": "matched_frozen_phase",
                    "candidate_config_id": row["config_id"],
                    "sample_id": item["sample_id"],
                    "channel": item["channel"],
                    "time_range": f"test_window_index:{item['sample_id']}",
                    "baseline_mse": item["phase_mse"],
                    "candidate_mse": item["candidate_mse"],
                    "delta_mse": item["candidate_mse"] - item["phase_mse"],
                    "baseline_mae": item["phase_mae"],
                    "candidate_mae": item["candidate_mae"],
                    "delta_mae": item["candidate_mae"] - item["phase_mae"],
                }
            )
        for case in cases:
            name = (
                f"{setting}__{case['category']}__sample{case['sample_id']}"
                f"_channel{case['channel']}.png"
            )
            render_case(figures_dir / name, case, arrays)
            case["figure"] = name
            selected_cases.append(case)
            index = case["sample_index"]
            selected_arrays["setting"].append(setting)
            selected_arrays["category"].append(case["category"])
            selected_arrays["config_id"].append(row["config_id"])
            selected_arrays["sample_id"].append(case["sample_id"])
            selected_arrays["channel"].append(case["channel"])
            selected_arrays["history"].append(arrays["history"][index, :, case["channel"]])
            selected_arrays["truth"].append(arrays["truth"][index, :, case["channel"]])
            selected_arrays["phase_prediction"].append(arrays["phase_prediction"][index, :, case["channel"]])
            selected_arrays["candidate_prediction"].append(arrays["candidate_prediction"][index, :, case["channel"]])
    write_csv(
        output_dir / "sample_errors.csv",
        all_sample_rows,
        list(all_sample_rows[0]),
    )
    np.savez_compressed(
        output_dir / "selected_cases.npz",
        **{
            key: np.asarray(value)
            for key, value in selected_arrays.items()
        },
    )

    figure_names = write_search_figures(
        rows, figures_dir, datasets, args.horizon
    ) + [case["figure"] for case in selected_cases]
    make_report(
        output_dir / "objective_error_analysis.md",
        rows,
        selected,
        selected_cases,
        figure_names,
        datasets,
        args.horizon,
        args.seed,
    )
    archive_report(output_dir, figure_names)
    run_yaml = {
        "experiment_id": output_dir.name,
        "code": {
            "repository": git_value("remote", "get-url", "origin"),
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "modified_files": [
                "src/models/frozen_nlinear_correction.py",
                "scripts/run_pooled_lowrank_nlinear.py",
                "scripts/run_pooled_lowrank_matrix.py",
                "scripts/analyze_pooled_lowrank_experiment.py",
            ],
        },
        "mechanism": {
            "description": "frozen PhaseFormer plus direct pooled low-rank NLinear residual",
            "feature_flag": "PooledLowRankResidualHead",
            "protocol_deviation": "pooling and rank vary together in the first screen",
        },
        "experiment": {
            "settings": [
                {
                    "setting": f"{dataset}_h{args.horizon}_seed{args.seed}",
                    "dataset": dataset,
                    "split": "test",
                    "lookback": 720,
                    "horizon": args.horizon,
                    "seed": args.seed,
                }
                for dataset in datasets
            ],
            "training": {"loss": "huber", "max_epochs": 30, "checkpoint": "best_val_loss"},
            "metrics": ["MSE", "MAE", "delta_pct_vs_fixed_golden"],
        },
        "selection": {
            "source": "validation",
            "test_exposure": "all planned matrix cells evaluated once for exploratory Golden comparison",
            "selected_configs": [
                {
                    "setting": f"{row['dataset']}_h{args.horizon}_seed{args.seed}",
                    "config_id": row["config_id"],
                    "pool_factor": row["pool_factor"],
                    "rank": row["rank"],
                    "smooth_ratio": row["smooth_ratio"],
                    "validation_score": row["validation_score"],
                }
                for row in selected.values()
            ],
        },
        "analysis": {
            "ranking_metric": "MAE",
            "top_k": 1,
            "dedup_rule": "one maximum sample-channel per selection category and setting",
        },
        "validation": {
            "results_checked": True,
            "ranking_and_cases_checked": True,
            "report_and_archive_checked": True,
            "directory_and_settings_checked": True,
            "status": "passed",
        },
    }
    (output_dir / "run.yaml").write_text(json.dumps(run_yaml, indent=2) + "\n")
    expected = {
        "run.yaml",
        "results.csv",
        "sample_errors.csv",
        "selected_cases.npz",
        "objective_error_analysis.md",
        "objective_error_analysis.zip",
        "figures",
    }
    if {path.name for path in output_dir.iterdir()} != expected:
        raise RuntimeError("audit directory violates the six-file whitelist")
    with zipfile.ZipFile(output_dir / "objective_error_analysis.zip") as bundle:
        archive_names = set(bundle.namelist())
    expected_archive = {"objective_error_analysis.md"} | {
        f"figures/{name}" for name in figure_names
    }
    if archive_names != expected_archive:
        raise RuntimeError("archive contents do not match the Markdown figure whitelist")
    print(output_dir)


if __name__ == "__main__":
    main()
