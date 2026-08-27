#!/usr/bin/env python3
"""Run the ICPT inter-cycle patch residual experiment (plan §11 interface).

Stage A (architecture_screen): validation-only comparison of A2 (current RCRF +
NLinear), A3 (RepeatLastCycle), A4 (CycleNet-style) and A5 (ICPT-none) on four
anchor settings.

Stage B (pe_screen): validation-only screen of P1-P9 on the same four settings;
A5/P0 and A2 are reused from Stage A.  `freeze` then applies the pre-registered
ratio rule to freeze one index-PE (plus an optional calendar candidate) without
ever reading test.

Stage C (full): after freeze, six settings x three seeds x A1/A2/A5/A6 (+A7 if
the calendar candidate was eligible) with full data, best-validation checkpoint
restore and a single test read.

Stage D (ablation): B1-B5 mechanism ablations on ETTh2-720 and Electricity-336.

All validation-only stages run through search_phaseformer.py; all full-budget
stages run through benchmark_phaseformer_suite.py.  Every stage supports
--resume.  The frozen index-PE is exported via the ICPT_FROZEN_PE environment
variable so the B-mode presets resolve it inside subprocess runners.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Stage A/B anchor settings (plan §7).
ANCHOR_SETTINGS = (("ETTh2", 720), ("ETTm2", 96), ("Electricity", 336), ("Weather", 336))
# Stage C confirmation settings (plan §7).
FULL_SETTINGS = (
    ("ETTh1", 96), ("ETTh2", 720), ("ETTm2", 96),
    ("Weather", 336), ("Electricity", 336), ("Traffic", 96),
)
ABLATION_SETTINGS = (("ETTh2", 720), ("Electricity", 336))
SEEDS = (2021, 2022, 2023)

A2 = "gold_combo_reliability_s2"
A3 = "rcrf_repeat_last_cycle"
A4 = "rcrf_cycle_net"
A5 = "rcrf_icpt_none"
A1 = "original"
STAGE_A_MODES = (A2, A3, A4, A5)

# P1-P8 index-PE candidates (ranked together); P9 calendar is scored separately.
PE_INDEX_MODES = (
    "rcrf_icpt_sincos",
    "rcrf_icpt_learned_abs",
    "rcrf_icpt_time2vec",
    "rcrf_icpt_rope",
    "rcrf_icpt_relative",
    "rcrf_icpt_alibi",
    "rcrf_icpt_lff",
    "rcrf_icpt_sincos_relative",
)
PE_CALENDAR_MODE = "rcrf_icpt_calendar"

B_MODES = ("icpt_only", "icpt_fixed_fusion", "icpt_patch16",
           "icpt_no_anchor", "icpt_no_attention")

# loss / lr / batch per (dataset, horizon) from the plan tables.
SETTING_TRAIN = {
    ("ETTh1", 96): {"loss": "mae", "lr": 0.0003, "batch": 256},
    ("ETTh2", 720): {"loss": "huber", "lr": 0.001, "batch": 256},
    ("ETTm2", 96): {"loss": "mae", "lr": 0.0003, "batch": 256},
    ("Weather", 336): {"loss": "mae", "lr": 0.0003, "batch": 256},
    ("Electricity", 336): {"loss": "mae", "lr": 0.0003, "batch": 16},
    ("Traffic", 96): {"loss": "mae", "lr": 0.0003, "batch": 8},
}
GOLDEN = {
    ("ETTh1", 96): (0.359, 0.382),
    ("ETTh2", 720): (0.402, 0.436),
    ("ETTm2", 96): (0.163, 0.256),
    ("Weather", 336): (0.242, 0.278),
    ("Electricity", 336): (0.165, 0.257),
    ("Traffic", 96): (0.361, 0.238),
}

SCREEN_OUTPUT = "research_runs/phaseformer_icpt_pe_screen"
FULL_OUTPUT = "research_runs/phaseformer_icpt_pe_full"
ABLATION_OUTPUT = "research_runs/phaseformer_icpt_pe_ablation"
FREEZE_RECORD = REPO_ROOT / SCREEN_OUTPUT / "freeze_record.json"


def parse_settings(value, allowed):
    result = []
    for item in value.split(","):
        dataset, horizon = item.strip().split(":", 1)
        setting = (dataset, int(horizon))
        if setting not in allowed:
            raise ValueError(f"unsupported setting: {setting}")
        result.append(setting)
    return result


def _screen_command(args, dataset, horizon, mode):
    train = SETTING_TRAIN[(dataset, horizon)]
    return [
        sys.executable,
        "scripts/search_phaseformer.py",
        "--dataset", dataset,
        "--horizon", str(horizon),
        "--stage", "mechanism_screen_2",
        "--mechanism", mode,
        "--period", "24",
        "--lookback", "720",
        "--percent", "30",
        "--max-epochs", "8",
        "--seed", "2021",
        "--loss", train["loss"],
        "--learning-rate", str(train["lr"]),
        "--batch-size", str(train["batch"]),
        "--num-workers", str(args.num_workers),
        "--output-dir", args.output_dir,
        "--resume",
    ]


def architecture_screen_commands(args):
    commands = []
    for dataset, horizon in parse_settings(args.settings, ANCHOR_SETTINGS):
        for mode in STAGE_A_MODES:
            commands.append(_screen_command(args, dataset, horizon, mode))
    return commands


def pe_screen_commands(args):
    commands = []
    modes = PE_INDEX_MODES + (PE_CALENDAR_MODE,)
    for dataset, horizon in parse_settings(args.settings, ANCHOR_SETTINGS):
        for mode in modes:
            commands.append(_screen_command(args, dataset, horizon, mode))
    return commands


def read_rows(output_dir):
    rows, seen = [], set()
    root = REPO_ROOT / output_dir
    for pattern in ("runs/*/metrics.csv", "*/metrics.csv"):
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            with path.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_screen(output_dir):
    keyed = {
        (row["dataset"], int(row["horizon"]), row["mechanism"]): row
        for row in read_rows(output_dir)
        if row.get("stage") == "mechanism_screen_2"
    }
    summary = []
    for dataset, horizon in ANCHOR_SETTINGS:
        base = keyed.get((dataset, horizon, A2))
        p0 = keyed.get((dataset, horizon, A5))
        if base is None or p0 is None:
            continue
        base_mse, base_mae = float(base["val_mse"]), float(base["val_mae"])
        p0_mse, p0_mae = float(p0["val_mse"]), float(p0["val_mae"])
        for mode in STAGE_A_MODES:
            row = keyed.get((dataset, horizon, mode))
            if row is None:
                continue
            mse, mae = float(row["val_mse"]), float(row["val_mae"])
            summary.append({
                "dataset": dataset, "horizon": horizon, "mode": mode,
                "val_mse": f"{mse:.8f}", "val_mae": f"{mae:.8f}",
                "mse_ratio_vs_a2": f"{mse / base_mse:.8f}",
                "mae_ratio_vs_a2": f"{mae / base_mae:.8f}",
                "parameter_count": row.get("parameter_count", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
                "config_hash": row.get("config_hash", ""),
                "run_id": row.get("run_id", ""),
            })
        # P1-P9 entries (ratios relative to P0 = A5).
        for mode in PE_INDEX_MODES + (PE_CALENDAR_MODE,):
            row = keyed.get((dataset, horizon, mode))
            if row is None:
                continue
            mse, mae = float(row["val_mse"]), float(row["val_mae"])
            summary.append({
                "dataset": dataset, "horizon": horizon, "mode": mode,
                "val_mse": f"{mse:.8f}", "val_mae": f"{mae:.8f}",
                "mse_ratio_vs_a2": f"{mse / base_mse:.8f}",
                "mae_ratio_vs_a2": f"{mae / base_mae:.8f}",
                "mse_ratio_vs_p0": f"{mse / p0_mse:.8f}",
                "mae_ratio_vs_p0": f"{mae / p0_mae:.8f}",
                "parameter_count": row.get("parameter_count", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
                "config_hash": row.get("config_hash", ""),
                "run_id": row.get("run_id", ""),
            })
    write_rows(REPO_ROOT / output_dir / "screen_summary.csv", summary)
    return summary


def _pe_scores(summary, modes):
    scores = {}
    for mode in modes:
        rows = [row for row in summary if row["mode"] == mode]
        if len(rows) != len(ANCHOR_SETTINGS):
            continue
        ratios = [
            float(row["mse_ratio_vs_p0"])
            for row in rows
            if row.get("mse_ratio_vs_p0") not in (None, "")
        ] + [
            float(row["mae_ratio_vs_p0"])
            for row in rows
            if row.get("mae_ratio_vs_p0") not in (None, "")
        ]
        if len(ratios) != 2 * len(ANCHOR_SETTINGS):
            continue
        scores[mode] = {
            "mean_ratio": sum(ratios) / len(ratios),
            "worst_ratio": max(ratios),
            "parameter_count": max(int(row["parameter_count"]) for row in rows),
            "elapsed_sec": sum(float(row["elapsed_sec"]) for row in rows),
            "eligible": sum(ratios) / len(ratios) < 1.0 and max(ratios) <= 1.01,
        }
    return scores


def _stage_a_gate(summary):
    """Stage A gate: A5 vs A2, 8 validation ratios mean < 1 and worst <= 1.01,
    OR >= 3/4 settings improve both metrics with <= 0.5% regression elsewhere.
    """
    ratios, both_improve, worst_regress = [], 0, 0.0
    for row in summary:
        if row["mode"] != A5:
            continue
        mr, ar = float(row["mse_ratio_vs_a2"]), float(row["mae_ratio_vs_a2"])
        ratios += [mr, ar]
        if mr < 1.0 and ar < 1.0:
            both_improve += 1
        worst_regress = max(worst_regress, mr - 1.0, ar - 1.0)
    if not ratios:
        return False
    mean = sum(ratios) / len(ratios)
    cond_a = mean < 1.0 and max(ratios) <= 1.01
    cond_b = both_improve >= 3 and worst_regress <= 0.005
    return cond_a or cond_b


def freeze(output_dir):
    summary = summarize_screen(output_dir)
    passed = _stage_a_gate(summary)
    if not passed:
        record = {
            "stage_a_passed": False,
            "stage_a_gate": "A5 vs A2 8-ratio mean<1 & worst<=1.01, or >=3/4 settings "
                            "both-metric improve with <=0.5% regression",
            "note": "ICPT main line stopped; no PE freeze performed.",
        }
        FREEZE_RECORD.parent.mkdir(parents=True, exist_ok=True)
        FREEZE_RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(record, indent=2))
        return 2

    index_scores = _pe_scores(summary, PE_INDEX_MODES)
    calendar_scores = _pe_scores(summary, (PE_CALENDAR_MODE,))
    eligible = [(mode, info) for mode, info in index_scores.items() if info["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda item: (
            item[1]["mean_ratio"],
            item[1]["worst_ratio"],
            item[1]["parameter_count"],
            item[1]["elapsed_sec"],
        ),
    )
    frozen = ranked[0][0] if ranked else None
    calendar = calendar_scores.get(PE_CALENDAR_MODE)
    calendar_eligible = bool(calendar and calendar["eligible"])
    record = {
        "stage_a_passed": True,
        "frozen_index_pe": frozen,
        "calendar_eligible": calendar_eligible,
        "selection_source": "validation_only",
        "test_read_before_freeze": False,
        "eligibility_rule": "relative to P0=ICPT-none: mean of 8 ratios <1 and "
                            "worst <=1.01; rank by (mean, worst, params, runtime)",
        "index_pe_scores": index_scores,
        "calendar_pe_scores": calendar_scores,
        "index_pe_ranking": [mode for mode, _ in ranked],
        "note": "No post-screen hyperparameter changes are allowed before Stage C.",
    }
    FREEZE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_RECORD.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if frozen else 2


def _read_freeze():
    if not FREEZE_RECORD.exists():
        raise RuntimeError("freeze record is missing; run --stage freeze first")
    freeze = json.loads(FREEZE_RECORD.read_text())
    if not freeze.get("stage_a_passed"):
        raise RuntimeError("Stage A did not pass; ICPT main line stopped")
    return freeze


def full_commands(args):
    freeze = _read_freeze()
    frozen = freeze.get("frozen_index_pe")
    if not frozen:
        raise RuntimeError("Stage B found no eligible index-PE candidate")
    os.environ["ICPT_FROZEN_PE"] = frozen.split("rcrf_icpt_", 1)[1]
    modes = [A1, A2, A5, frozen]
    if freeze.get("calendar_eligible"):
        modes.append(PE_CALENDAR_MODE)
    commands = []
    for dataset, horizon in parse_settings(args.settings, FULL_SETTINGS):
        train = SETTING_TRAIN[(dataset, horizon)]
        for seed in args.seeds:
            commands.append([
                sys.executable,
                "scripts/benchmark_phaseformer_suite.py",
                "--datasets", dataset,
                "--horizons", str(horizon),
                "--modes", ",".join(modes),
                "--lookback", "720",
                "--seed", str(seed),
                "--loss", train["loss"],
                "--learning-rate", str(train["lr"]),
                "--batch-size", str(train["batch"]),
                "--num-workers", str(args.num_workers),
                "--bad-case-limit", "8",
                "--output-dir", args.output_dir,
                "--run-prefix", f"icpt_full_{dataset}_{horizon}",
                "--resume",
            ])
    return commands


def ablation_commands(args):
    freeze = _read_freeze()
    frozen = freeze.get("frozen_index_pe")
    if not frozen:
        raise RuntimeError("Stage B found no eligible index-PE candidate")
    os.environ["ICPT_FROZEN_PE"] = frozen.split("rcrf_icpt_", 1)[1]
    commands = []
    for dataset, horizon in parse_settings(args.settings, ABLATION_SETTINGS):
        train = SETTING_TRAIN[(dataset, horizon)]
        commands.append([
            sys.executable,
            "scripts/benchmark_phaseformer_suite.py",
            "--datasets", dataset,
            "--horizons", str(horizon),
            "--modes", ",".join(B_MODES),
            "--lookback", "720",
            "--seed", "2021",
            "--loss", train["loss"],
            "--learning-rate", str(train["lr"]),
            "--batch-size", str(train["batch"]),
            "--num-workers", str(args.num_workers),
            "--bad-case-limit", "8",
            "--output-dir", args.output_dir,
            "--run-prefix", f"icpt_ablation_{dataset}_{horizon}",
            "--resume",
        ])
    return commands


def summarize_full(output_dir, out_path):
    freeze = _read_freeze()
    frozen = freeze.get("frozen_index_pe")
    calendar = freeze.get("calendar_eligible")
    modes = {A1, A2, A5, frozen}
    if calendar:
        modes.add(PE_CALENDAR_MODE)
    rows = read_rows(output_dir)
    summary = []
    for row in rows:
        mode = row.get("mode")
        if mode not in modes:
            continue
        dataset, horizon = row["dataset"], int(row["horizon"])
        if (dataset, horizon) not in FULL_SETTINGS:
            continue
        mse, mae = float(row["test_mse"]), float(row["test_mae"])
        golden_mse, golden_mae = GOLDEN[(dataset, horizon)]
        summary.append({
            "dataset": dataset, "horizon": horizon,
            "seed": int(row["seed"]), "mode": mode,
            "test_mse": f"{mse:.8f}", "test_mae": f"{mae:.8f}",
            "golden_mse": golden_mse, "golden_mae": golden_mae,
            "delta_mse_pct_vs_golden": f"{(golden_mse - mse) / golden_mse * 100:.4f}",
            "delta_mae_pct_vs_golden": f"{(golden_mae - mae) / golden_mae * 100:.4f}",
            "epochs_completed": row.get("epochs_completed", ""),
            "elapsed_sec": row.get("elapsed_sec", ""),
            "run_id": row.get("run_id", ""),
        })
    summary.sort(key=lambda r: (r["dataset"], r["horizon"], r["seed"], r["mode"]))
    write_rows(out_path, summary)
    return summary


def summarize_ablation(output_dir, out_path):
    rows = read_rows(output_dir)
    summary = []
    for row in rows:
        if row.get("mode") not in B_MODES:
            continue
        dataset, horizon = row["dataset"], int(row["horizon"])
        if (dataset, horizon) not in ABLATION_SETTINGS:
            continue
        summary.append({
            "dataset": dataset, "horizon": horizon,
            "seed": int(row["seed"]), "mode": row["mode"],
            "test_mse": row["test_mse"], "test_mae": row["test_mae"],
            "epochs_completed": row.get("epochs_completed", ""),
            "elapsed_sec": row.get("elapsed_sec", ""),
            "run_id": row.get("run_id", ""),
        })
    summary.sort(key=lambda r: (r["dataset"], r["horizon"], r["mode"], r["seed"]))
    write_rows(out_path, summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=("architecture_screen", "pe_screen", "freeze", "full", "ablation"),
    )
    parser.add_argument("--settings")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--output-dir")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.seeds = tuple(int(item) for item in args.seeds.split(",") if item)
    if not args.output_dir:
        if args.stage in ("architecture_screen", "pe_screen", "freeze"):
            args.output_dir = SCREEN_OUTPUT
        elif args.stage == "full":
            args.output_dir = FULL_OUTPUT
        else:
            args.output_dir = ABLATION_OUTPUT
    if not args.settings:
        if args.stage in ("architecture_screen", "pe_screen"):
            args.settings = ",".join(f"{d}:{h}" for d, h in ANCHOR_SETTINGS)
        elif args.stage == "full":
            args.settings = ",".join(f"{d}:{h}" for d, h in FULL_SETTINGS)
        else:
            args.settings = ",".join(f"{d}:{h}" for d, h in ABLATION_SETTINGS)
    return args


def main():
    args = parse_args()
    if args.stage == "architecture_screen":
        commands = architecture_screen_commands(args)
    elif args.stage == "pe_screen":
        commands = pe_screen_commands(args)
    elif args.stage == "full":
        commands = full_commands(args)
    elif args.stage == "ablation":
        commands = ablation_commands(args)
    else:
        commands = []
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    if args.dry_run:
        return 0
    if args.stage in ("architecture_screen", "pe_screen"):
        summarize_screen(args.output_dir)
        return 0
    if args.stage == "freeze":
        return freeze(args.output_dir)
    if args.stage == "full":
        summarize_full(args.output_dir, REPO_ROOT / FULL_OUTPUT / "full_summary.csv")
        return 0
    summarize_ablation(args.output_dir, REPO_ROOT / ABLATION_OUTPUT / "ablation_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
