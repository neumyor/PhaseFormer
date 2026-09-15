#!/usr/bin/env python3
"""Non-invasive sample-level test export for the structured low-rank plan.

Loads finished Round 1 runs, rebuilds each candidate's model from its own
``config.json``, restores its validation-selected checkpoint, and evaluates the
test split once per run.  Nothing is trained and no metric is recomputed for
selection: the script only records what each finished run predicts.

Outputs (plan sections 12 and 15): a long-form ``sample_errors.csv`` with one row
per ``setting x candidate x sample x channel`` and a ``selected_cases.npz`` with
the programmatically selected cases.  Selection follows the plan's six required
buckets, each capped at ``--per-bucket`` (8 by default) with consecutive-window
deduplication.

Example::

    python scripts/export_structured_lowrank_cases.py \
        --scratch-root research_runs/structured_lowrank_round1_scratch \
        --direct-root research_runs/rank_sweep_2_stage1 \
        --output-dir research_runs/structured_lowrank_round1_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.data_factory import data_provider  # noqa: E402
from src.models.PhaseFormer import PhaseFormer  # noqa: E402
from src.models.phaseformer_presets import (  # noqa: E402
    PhaseFormerPresetConfig,
    make_exp_args,
)

# Candidate label -> the override key that identifies it in a run config.
CANDIDATE_MARKERS = {
    "A_period_lowrank": ("residual_period_rank", 4),
    "B_segment_basis": ("residual_basis_count", 4),
    "C_level_shape": ("residual_level_mode", "dense"),
    "D_recent_sparse": ("residual_recent_taps", 7),
    "E_separable": ("residual_separable_components", 1),
}


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def discover_runs(scratch_root):
    """Map candidate label -> list of (config_path, hyperparams)."""

    found = {}
    for config_path in sorted(Path(scratch_root).glob("*/runs/*/config.json")):
        config = read_json(config_path)
        hyper = config["hyperparams"]
        head_type = hyper.get("weak_period_residual_head_type", "shared")
        label = None
        if head_type == "time_axis_matched_lowrank":
            label = "matched_time_axis_lowrank"
        else:
            for name, (key, value) in CANDIDATE_MARKERS.items():
                if hyper.get(key) == value and head_type.startswith("structured_"):
                    label = name
                    break
            if label is None and head_type == "structured_period_lowrank":
                label = f"A_period_lowrank_r{hyper.get('residual_period_rank')}"
        if label is None:
            continue
        metrics_path = config_path.with_name("metrics.csv")
        if not metrics_path.is_file():
            continue
        found.setdefault(label, []).append((config_path, config, hyper))
    return found


def best_run_per_setting(entries):
    """Keep the run with the lowest validation loss for each setting."""

    best = {}
    for config_path, config, hyper in entries:
        metrics_path = config_path.with_name("metrics.csv")
        with metrics_path.open() as handle:
            metric = next(csv.DictReader(handle))
        if not metric.get("test_mse"):
            continue
        key = (config["dataset"], int(config["horizon"]))
        score = float(metric["val_mse"]) + float(metric["val_mae"])
        if key not in best or score < best[key][0]:
            best[key] = (score, config_path, config, hyper, metric)
    return best


def build_model(config, hyper, num_workers):
    dataset = config["dataset"]
    lookback = int(config["lookback"])
    horizon = int(config["horizon"])
    exp_args = make_exp_args(
        dataset, lookback, horizon, hyper, batch_size=config.get("batch_size")
    )
    exp_args.dataset_args.num_workers = num_workers
    exp_args.training_args.num_workers = num_workers
    configured_root = Path(exp_args.dataset_args.root_path)
    fallback = ROOT / "resources" / "all_datasets" / "ETT-small"
    if not configured_root.exists() and dataset.startswith("ETT") and fallback.exists():
        exp_args.dataset_args.root_path = str(fallback)
    train_set, _ = data_provider(exp_args.dataset_args, "train")
    if hasattr(train_set, "data_stamp"):
        hyper["time_mark_dim"] = int(train_set.data_stamp.shape[-1])
    model = PhaseFormer(
        PhaseFormerPresetConfig(exp_args, lookback, horizon, hyper)
    )
    return exp_args, model


def load_checkpoint(model, run_dir):
    candidates = sorted(Path(run_dir).glob("attempts/*/checkpoints/best.ckpt"))
    if not candidates:
        candidates = sorted(Path(run_dir).glob("checkpoints/best.ckpt"))
    if not candidates:
        raise SystemExit(f"no best checkpoint under {run_dir}")
    payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state)
    return candidates[0]


def evaluate_test(model, exp_args, device):
    _, loader = data_provider(exp_args.dataset_args, "test")
    model.eval().to(device)
    preds, truths, origins = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = [item.to(device) if torch.is_tensor(item) else item for item in batch]
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(
                batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float()
            )
            preds.append(out[:, -model.pred_len :, :].detach().cpu())
            truths.append(batch_y.float()[:, -model.pred_len :, :].detach().cpu())
    return torch.cat(preds), torch.cat(truths)


def resolve_run_dir(raw):
    """Turn the recorded (possibly relative) run directory into a path."""

    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def reused_run_dirs(scratch_root, dataset, horizon, seed):
    """Locate the reused direct and generic baselines for one setting."""

    matches = sorted(
        Path(scratch_root).glob(f"*{dataset}_h{horizon}_s{seed}_results.csv")
    ) or sorted(Path(scratch_root).glob(f"*{dataset.lower()}_h{horizon}_s{seed}_results.csv"))
    if not matches:
        raise SystemExit(f"no reused result csv for {dataset}-{horizon}")
    with matches[0].open() as handle:
        rows = list(csv.DictReader(handle))
    wanted = {}
    for row in rows:
        config_id = row["config_id"]
        if config_id == "direct_nlinear":
            wanted["direct_nlinear"] = row
        elif config_id.startswith("pool") and "_q" in config_id:
            if abs(float(config_id.split("_q")[1].split("_")[0]) - 0.125) < 1e-9:
                wanted["generic_q1/8"] = row
    resolved = {}
    for name, row in wanted.items():
        run_dir = resolve_run_dir(row["run_dir"])
        resolved[name] = {
            "run_dir": run_dir,
            "test_mse": float(row["test_mse"]),
            "test_mae": float(row["test_mae"]),
        }
    return resolved


def evaluate_run(run_dir, fallback_config, num_workers, device, seed):
    """Rebuild one run's model from its own config and read the test split once."""

    config_path = Path(run_dir) / "config.json"
    if config_path.is_file():
        config = read_json(config_path)
    else:
        config = fallback_config
    hyper = dict(config["hyperparams"])
    exp_args, model = build_model(config, hyper, num_workers)
    checkpoint = load_checkpoint(model, run_dir)
    preds, truths = evaluate_test(model, exp_args, device)
    return preds, truths, checkpoint


def per_sample_metrics(preds, truths):
    err = preds - truths
    return err.pow(2).mean(dim=1).numpy(), err.abs().mean(dim=1).numpy()


def pick_cases(candidate_mse, candidate_mae, baseline_mse, baseline_mae, generic_mse, limit):
    """Programmatic case selection covering the plan's required buckets."""

    flat_candidate_mse = candidate_mse.reshape(-1)
    flat_candidate_mae = candidate_mae.reshape(-1)
    flat_baseline_mse = baseline_mse.reshape(-1)
    flat_baseline_mae = baseline_mae.reshape(-1)
    flat_generic_mse = generic_mse.reshape(-1)
    channels = candidate_mse.shape[1]

    def index_of(flat):
        return int(flat // channels), int(flat % channels)

    both_improve = (flat_candidate_mse < flat_baseline_mse) & (
        flat_candidate_mae < flat_baseline_mae
    )
    both_regress = (flat_candidate_mse > flat_baseline_mse) & (
        flat_candidate_mae > flat_baseline_mae
    )
    buckets = {}
    order = np.argsort(-(flat_baseline_mse - flat_candidate_mse))
    buckets["dual_improvement"] = [index_of(i) for i in order if both_improve[i]][:limit]
    order = np.argsort(-(flat_candidate_mse - flat_baseline_mse))
    buckets["dual_regression"] = [index_of(i) for i in order if both_regress[i]][:limit]
    order = np.argsort(-flat_baseline_mse)
    buckets["baseline_high_error"] = [index_of(i) for i in order][:limit]
    divergence = np.abs(flat_candidate_mse - flat_generic_mse)
    order = np.argsort(-divergence)
    buckets["generic_structured_divergence"] = [index_of(i) for i in order][:limit]
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", required=True, help="Round 1 scratch root")
    parser.add_argument(
        "--reuse-root",
        required=True,
        help="root holding the reused phase-a results csv and run directories",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-bucket", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-samples", type=int, default=0, help="smoke limit")
    parser.add_argument("--only-setting", help="restrict to one dataset:horizon")
    parser.add_argument(
        "--require-metrics", help="csv of candidate,setting pairs to export"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    discovered = discover_runs(args.scratch_root)
    if not discovered:
        raise SystemExit(f"no finished runs found under {args.scratch_root}")

    # candidate label -> setting -> best run description
    best_by_label = {}
    settings = set()
    for label, entries in discovered.items():
        best = best_run_per_setting(entries)
        best_by_label[label] = best
        settings.update(best.keys())

    rows = []
    case_store = {}
    selections = {}
    for dataset, horizon in sorted(settings):
        if args.only_setting:
            wanted = args.only_setting.split(":")
            if dataset != wanted[0] or str(horizon) != wanted[1]:
                continue
        setting = f"{dataset}_h{horizon}_seed{args.seed}"
        reused = reused_run_dirs(args.reuse_root, dataset, horizon, args.seed)
        predictions = {}
        fallback = next(iter(best_by_label.values()))[(dataset, horizon)][2]
        truths = None
        for name in ("direct_nlinear", "generic_q1/8"):
            record = reused[name]
            preds, truths, checkpoint = evaluate_run(
                record["run_dir"], fallback, args.num_workers, device, args.seed
            )
            predictions[name] = preds
            print(
                f"{setting} {name} (reused) test {record['test_mse']:.6f} "
                f"({checkpoint.name})",
                flush=True,
            )
        baseline_preds = predictions["direct_nlinear"]
        generic_preds = predictions["generic_q1/8"]
        baseline_mse, baseline_mae = per_sample_metrics(baseline_preds, truths)
        generic_mse, generic_mae = per_sample_metrics(generic_preds, truths)

        for label, best in sorted(best_by_label.items()):
            if (dataset, horizon) not in best:
                continue
            _, config_path, config, hyper, metric = best[(dataset, horizon)]
            run_dir = config_path.parent
            preds, truths, checkpoint = evaluate_run(
                run_dir, config, args.num_workers, device, args.seed
            )
            cand_mse, cand_mae = per_sample_metrics(preds, truths)
            print(
                f"{setting} {label} test {metric['test_mse']} "
                f"(direct {baseline_mse.mean():.6f})",
                flush=True,
            )
            for sample in range(cand_mse.shape[0]):
                for channel in range(cand_mse.shape[1]):
                    rows.append(
                        {
                            "setting": setting,
                            "baseline_config_id": "direct_nlinear",
                            "candidate_config_id": label,
                            "sample_id": sample,
                            "channel": channel,
                            "time_range": f"{sample * baseline_preds.shape[1]}"
                            f"..{(sample + 1) * baseline_preds.shape[1] - 1}",
                            "baseline_mse": float(baseline_mse[sample, channel]),
                            "candidate_mse": float(cand_mse[sample, channel]),
                            "delta_mse": float(
                                cand_mse[sample, channel] - baseline_mse[sample, channel]
                            ),
                            "baseline_mae": float(baseline_mae[sample, channel]),
                            "candidate_mae": float(cand_mae[sample, channel]),
                            "delta_mae": float(
                                cand_mae[sample, channel] - baseline_mae[sample, channel]
                            ),
                        }
                    )
            if args.max_samples:
                preds = preds[: args.max_samples]
                truths = truths[: args.max_samples]
            case_store[f"{setting}|{label}|truth"] = truths.numpy()
            case_store[f"{setting}|{label}|pred"] = preds.numpy()
            case_store[f"{setting}|{label}|baseline_pred"] = baseline_preds.numpy()
            case_store[f"{setting}|{label}|generic_pred"] = generic_preds.numpy()
            selections[f"{setting}|{label}"] = pick_cases(
                cand_mse, cand_mae, baseline_mse, baseline_mae, generic_mse, args.per_bucket
            )

    fields = [
        "setting",
        "baseline_config_id",
        "candidate_config_id",
        "sample_id",
        "channel",
        "time_range",
        "baseline_mse",
        "candidate_mse",
        "delta_mse",
        "baseline_mae",
        "candidate_mae",
        "delta_mae",
    ]
    with (output / "sample_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "selected_cases.npz", **case_store)
    with (output / "case_selection.json").open("w") as handle:
        json.dump(selections, handle, indent=2, sort_keys=True)
    print(f"wrote {output/'sample_errors.csv'} ({len(rows)} rows)")
    print(f"wrote {output/'selected_cases.npz'}")
    print(f"wrote {output/'case_selection.json'}")


if __name__ == "__main__":
    main()
