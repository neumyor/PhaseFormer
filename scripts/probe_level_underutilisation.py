#!/usr/bin/env python3
"""Does a frozen PhaseFormer leave predictable level information unused?

The paired comparisons show that adding a temporal branch improves the fused
forecast on ETTh2/ETTm2/Weather, and sample-level diagnostics show the gain
concentrates in windows whose cross-cycle level moves.  Neither establishes the
stronger claim that the frozen model *underuses* level information, because the
original model also consumes level information.

This script tests it directly and cheaply:

1. freeze a trained **phase-only** checkpoint (no residual branch);
2. fit, on the **training split only**, an affine map from the window's
   cross-cycle *level trajectory* to the frozen model's remaining error;
3. choose the ridge penalty on the **validation split**;
4. evaluate on the **test split**, overall and grouped by how much the level
   moved, against controls that share the parameter count but not the level
   semantics.

If level information predicts the frozen model's error and the reduction
concentrates in the expected windows -- while the controls do not reproduce it --
then the frozen model uses the information without fully exploiting it.  That
statement is about these models and these datasets, not about what the
architecture could represent in principle.

Fitting and penalty selection touch only train and validation.  The test split
is read once per pass, for evaluation.  The *checkpoints* are not blind; the
report's disclosure section states that limitation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lowrank_checkpoint_model import (  # noqa: E402
    build_loaders,
    build_model,
    load_checkpoint_into,
)

#: Dominant physical cycle in steps.  ETTm1/ETTm2 sample every 15 minutes.
DATASET_PERIOD = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96, "Weather": 24}

LOOKBACK = 720


def cycle_means(xn: np.ndarray, period: int, count: int) -> np.ndarray:
    """Mean of each of the ``count`` most recent non-overlapping cycles.

    ``xn`` is (N, L, C) and the result is (N, C, count); cycle 0 is the most
    recent.  When ``L`` is not a multiple of ``period`` the leading remainder is
    left unused rather than folded into a partial cycle.
    """
    length = xn.shape[1]
    # (N, C, count): channels stay the middle axis so a feature vector is one
    # channel's level trajectory, which is what the per-channel ridge couples.
    features = np.empty((xn.shape[0], xn.shape[2], count), dtype=np.float64)
    features[:, :, 0] = xn[:, length - period:, :].mean(axis=1)
    for index in range(1, count):
        stop = length - index * period
        start = stop - period
        features[:, :, index] = (
            features[:, :, index - 1] if start < 0
            else xn[:, start:stop, :].mean(axis=1)
        )
    return features


def with_intercept(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones(features.shape[:2] + (1,))], axis=2)


def ridge_fit(s_ff: np.ndarray, s_yf: np.ndarray, penalty: float) -> np.ndarray:
    """``W`` minimising ``sum ||y - W f||^2 + penalty ||W||^2``; ``s_yf`` is (H, F)."""
    size = s_ff.shape[0]
    return s_yf @ np.linalg.solve(s_ff + penalty * np.eye(size), np.eye(size))


def moments(features: np.ndarray, targets: np.ndarray):
    """Second-moment statistics summed over (sample, channel).

    ``features`` is (N, C, F) and ``targets`` is (N, H, C) -- the layout the
    metrics use -- so the target is moved to (N, C, H) before flattening; the
    ridge problem couples one channel's features to that same channel's horizon.
    """
    flat_f = features.reshape(-1, features.shape[-1])
    flat_y = np.moveaxis(targets, 2, 1).reshape(-1, targets.shape[1])
    return flat_f.T @ flat_f, flat_y.T @ flat_f, flat_f.shape[0]


def calibrate_scales(model, loader, device, max_batches: int = 8) -> np.ndarray:
    """Per-channel error scale, from the median window standard deviation.

    The branch's own form multiplies its map by the per-window RevIN scale, so a
    single shared map can serve channels of very different magnitude.  Using the
    raw per-window scale is unusable here -- a nearly constant window has
    sigma ~ 3e-3 and the quotient reaches 1e3 -- so the per-channel *median*
    over a handful of training batches is used instead.  It is a fixed,
    train-derived constant, not a per-sample quantity.
    """
    deviations = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = [t.to(device) if torch.is_tensor(t) else t for t in batch]
            xn = batch[0].double().cpu().numpy()
            deviations.append(xn.std(axis=1))            # (N, C)
    stacked = np.concatenate(deviations, axis=0)
    return np.median(stacked, axis=0)                    # (C,)


def forward_split(model, loader, device, period, cycle_count,
                  projections: dict[str, np.ndarray] | None = None,
                  scales: np.ndarray | None = None,
                  max_batches: int = 0):
    """Stream one split.

    Always returns the level and time-shuffled feature sets, the normalized
    target, the RevIN scale and the window moments.  When ``projections`` is
    given, each named ``(L, cycle_count)`` matrix is also applied to every
    window, which is what the parameter-matched control needs.
    """
    level_parts, shuffled_parts, target_parts = [], [], []
    projected_parts = {name: [] for name in (projections or {})}
    window_moment = np.zeros((LOOKBACK, LOOKBACK))
    window_sum = np.zeros(LOOKBACK)
    samples = 0
    permutation = np.random.default_rng(0).permutation(LOOKBACK)
    statistics = {name: None for name in ("level", "time_shuffled")}

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = [t.to(device) if torch.is_tensor(t) else t for t in batch]
            x, y, x_mark, y_mark = batch
            dec = model._build_decoder_input(y.float())
            out, _, _ = model(x.float(), x_mark.float(), dec, y_mark.float())
            horizon = out.shape[1]
            truth = y.float()[:, -horizon:, :]

            xn = x.double().cpu().numpy()
            # The target is the frozen model's error in *value* space.  Dividing
            # by the per-window RevIN scale would be the branch's own coordinate
            # system, but windows that are nearly constant have sigma ~ 0.003 and
            # the quotient reaches 1e3, which makes the regression fit noise.
            # The features are already normalized, so a linear map from them to
            # the value-space error is scale-free without that division.
            residual = truth.double().cpu().numpy() - out.double().cpu().numpy()
            if scales is not None:
                residual = residual / scales

            level = cycle_means(xn, period, cycle_count)
            shuffled = cycle_means(xn[:, permutation, :], period, cycle_count)

            for name, features in (("level", level), ("time_shuffled", shuffled)):
                moment_f, moment_y, _ = moments(with_intercept(features), residual)
                statistics[name] = (
                    (moment_f, moment_y) if statistics[name] is None
                    else (statistics[name][0] + moment_f, statistics[name][1] + moment_y)
                )
            for name, matrix in (projections or {}).items():
                projected_parts[name].append(
                    np.einsum("nlc,lf->ncf", xn, matrix)
                )

            level_parts.append(level)
            shuffled_parts.append(shuffled)
            target_parts.append(residual)
            samples += xn.shape[0]

            flat_xn = xn.reshape(-1, LOOKBACK)
            window_moment += flat_xn.T @ flat_xn
            window_sum += flat_xn.sum(axis=0)
            del xn, out, truth, residual, level, shuffled, flat_xn

    return {
        "statistics": statistics,
        "samples": samples,
        "window_moment": window_moment,
        "window_sum": window_sum,
        "level": np.concatenate(level_parts, axis=0),
        "time_shuffled": np.concatenate(shuffled_parts, axis=0),
        "target": np.concatenate(target_parts, axis=0),
        "projected": {name: np.concatenate(parts, axis=0)
                      for name, parts in projected_parts.items()},
    }


def score(features: np.ndarray, target: np.ndarray, weight: np.ndarray,
          scales: np.ndarray | None = None) -> dict:
    """Value-space metrics for one correction map, plus per-window squared errors.

    ``target`` is the frozen model's error divided by the per-channel scale, so
    multiplying back by ``scales`` restores value-space units and makes
    ``mse_before`` comparable with the checkpoint's own recorded test metric.
    """
    unit = 1.0 if scales is None else scales
    # Output subscripts are (n, h, c): the correction writes a horizon shape
    # per channel, matching the (N, H, C) layout the metrics use.
    correction = np.einsum("ncf,hf->nhc", features, weight)
    error_before = target * unit
    error_after = (target - correction) * unit
    return {
        "mse_before": float(np.mean(error_before ** 2)),
        "mse_after": float(np.mean(error_after ** 2)),
        "mae_before": float(np.mean(np.abs(error_before))),
        "mae_after": float(np.mean(np.abs(error_after))),
        "per_window_before": np.mean(error_before ** 2, axis=(1, 2)),
        "per_window_after": np.mean(error_after ** 2, axis=(1, 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="research_runs/phaseformer_L_e14_main_v1/results.csv")
    parser.add_argument("--output-dir", default="research_runs/level_underutilisation_v1")
    parser.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Weather")
    parser.add_argument("--seeds", default="2021,2022,2023")
    parser.add_argument("--penalties", default="0.003,0.01,0.03,0.1,0.3,1,3,10,30")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    if not (repo_root / "src").is_dir():
        repo_root = Path.cwd().resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    datasets = {v for v in args.datasets.split(",") if v}
    seeds = {int(v) for v in args.seeds.split(",") if v}
    penalties = [float(v) for v in args.penalties.split(",") if v]

    rows = [
        r for r in csv.DictReader((repo_root / args.results).open())
        # "reused" cells carry test metrics copied from an earlier run rather
        # than re-read in E14; their checkpoints are present and usable, and
        # they are exactly the six exploratory settings per dataset.
        if r["arm"] == "phase_only" and r["status"] in ("read", "reused")
        and r["dataset"] in datasets and int(r["seed"]) in seeds
    ]
    rows.sort(key=lambda r: (r["dataset"], int(r["horizon"]), int(r["seed"])))
    rows = [r for index, r in enumerate(rows) if index % args.shard_count == args.shard_index]
    print(f"shard {args.shard_index}/{args.shard_count}: {len(rows)} cells on {device}", flush=True)

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    groups: list[dict] = []

    for position, row in enumerate(rows, start=1):
        dataset, horizon, seed = row["dataset"], int(row["horizon"]), int(row["seed"])
        setting = row["setting"]
        run_dir = repo_root / row["run_dir"]
        checkpoints = sorted((run_dir / "attempts").glob("*/checkpoints/best.ckpt"))
        if not checkpoints:
            print(f"  [skip] no checkpoint for {setting} seed={seed}", flush=True)
            continue

        config = json.loads((run_dir / "config.json").read_text())
        hyperparams = dict(config["hyperparams"])
        batch_size = int(config.get("batch_size") or 256)
        period = DATASET_PERIOD[dataset]
        cycle_count = LOOKBACK // period

        started = time.time()
        exp_args, handles = build_loaders(
            dataset, LOOKBACK, horizon, hyperparams, batch_size, repo_root,
            splits=("train", "val", "test"),
        )
        model = build_model(exp_args, LOOKBACK, horizon, hyperparams)
        load_checkpoint_into(model, checkpoints[0])
        model.to(device)
        model.eval()

        scales = calibrate_scales(model, handles["train"][1], device)
        first = {split: forward_split(model, handles[split][1], device, period, cycle_count,
                                      scales=scales, max_batches=args.max_batches)
                 for split in ("train", "val", "test")}

        # Two parameter-matched generic paths, both of dimension `cycle_count`
        # and hence the same parameter count as the level map (plus intercept):
        # a fixed random projection, and the principal axes of the normalized
        # window fitted on train.
        rng = np.random.default_rng(seed)
        random_projection = rng.standard_normal((LOOKBACK, cycle_count)) / np.sqrt(LOOKBACK)
        principal = _principal_axes(first["train"]["window_moment"],
                                    first["train"]["window_sum"],
                                    first["train"]["samples"], cycle_count)
        projections = {"random_projection": random_projection, "pca_matched": principal}
        second = {split: forward_split(model, handles[split][1], device, period, cycle_count,
                                       projections=projections, scales=scales,
                                       max_batches=args.max_batches)
                  for split in ("train", "val", "test")}
        for split in ("train", "val", "test"):
            for name, value in second[split]["projected"].items():
                first[split]["projected"][name] = value
        del model, handles, second

        # --- penalty selection on validation, never on test -----------------
        best_penalty, best_val, level_weight = None, np.inf, None
        for penalty in penalties:
            weight = ridge_fit(*first["train"]["statistics"]["level"], penalty)
            candidate = score(with_intercept(first["val"]["level"]),
                              first["val"]["target"], weight, scales)
            if candidate["mse_after"] < best_val:
                best_val, best_penalty, level_weight = candidate["mse_after"], penalty, weight

        train, test = first["train"], first["test"]
        result = score(with_intercept(test["level"]), test["target"], level_weight, scales)

        control_results: dict[str, dict] = {}
        shuffled_train = train["level"][np.random.default_rng(seed).permutation(train["level"].shape[0])]
        weight = ridge_fit(*moments(with_intercept(shuffled_train), train["target"])[:2], best_penalty)
        control_results["shuffled_level"] = score(
            with_intercept(test["level"]), test["target"], weight, scales)

        weight = ridge_fit(*train["statistics"]["time_shuffled"], best_penalty)
        control_results["time_shuffled_level"] = score(
            with_intercept(test["time_shuffled"]), test["target"], weight, scales)

        for name in ("random_projection", "pca_matched"):
            weight = ridge_fit(*moments(with_intercept(train["projected"][name]),
                                        train["target"])[:2], best_penalty)
            control_results[name] = score(
                with_intercept(test["projected"][name]), test["target"], weight, scales)

        entry = {
            "setting": setting, "dataset": dataset, "horizon": horizon, "seed": seed,
            "lookback": LOOKBACK, "period": period, "cycle_features": cycle_count,
            "parameter_count": int(level_weight.shape[0] * level_weight.shape[1]),
            "train_samples": train["samples"], "val_samples": first["val"]["samples"],
            "test_samples": test["samples"],
            "selected_penalty": best_penalty, "val_mse_after": best_val,
            "channel_scales": ";".join(f"{v:.6g}" for v in scales),
            "test_mse_phase_only": result["mse_before"],
            "test_mse_level_correction": result["mse_after"],
            "test_delta_mse": result["mse_after"] - result["mse_before"],
            "test_mae_phase_only": result["mae_before"],
            "test_mae_level_correction": result["mae_after"],
            "test_delta_mae": result["mae_after"] - result["mae_before"],
            "recorded_test_mse": float(row["test_mse"]),
            "recorded_test_mae": float(row["test_mae"]),
        }
        for name, value in control_results.items():
            entry[f"control_{name}_mse"] = value["mse_after"]
            entry[f"control_{name}_delta_mse"] = value["mse_after"] - value["mse_before"]

        level = test["level"]                       # (N, C, K)
        level_std = level.std(axis=2).mean(axis=1)
        prior = (level[:, :, 1:].mean(axis=2) if level.shape[2] > 1
                 else level[:, :, 0])
        last_offset = np.abs(level[:, :, 0] - prior).mean(axis=1)
        gain = result["per_window_before"] - result["per_window_after"]

        for name, statistic in (("level_std", level_std), ("last_cycle_offset", last_offset)):
            for quartile, index in enumerate(np.array_split(np.argsort(statistic), 4), start=1):
                groups.append({
                    "setting": setting, "dataset": dataset, "horizon": horizon, "seed": seed,
                    "grouping": name, "quartile": quartile, "windows": int(index.size),
                    "mean_statistic": float(statistic[index].mean()),
                    "mse_before": float(result["per_window_before"][index].mean()),
                    "mse_after": float(result["per_window_after"][index].mean()),
                    "delta_mse": float((result["per_window_after"][index]
                                        - result["per_window_before"][index]).mean()),
                })
            entry[f"spearman_{name}_vs_gain"] = float(np.corrcoef(
                statistic.argsort().argsort().astype(float), gain)[0, 1])

        summary.append(entry)
        print(
            f"  [{position}/{len(rows)}] {setting} seed={seed} lam={best_penalty:g} "
            f"dMSE={entry['test_delta_mse']:+.6f} "
            f"({entry['test_delta_mse'] / result['mse_before']:+.2%}) "
            f"shuf={entry['control_shuffled_level_delta_mse']:+.6f} "
            f"tshuf={entry['control_time_shuffled_level_delta_mse']:+.6f} "
            f"pca={entry['control_pca_matched_delta_mse']:+.6f} "
            f"({time.time() - started:.0f}s)", flush=True,
        )

    write_csv(summary, output_dir / f"level_underutilisation_cells_shard{args.shard_index}.csv")
    write_csv(groups, output_dir / f"level_underutilisation_groups_shard{args.shard_index}.csv")
    print(f"shard {args.shard_index} done", flush=True)


def _principal_axes(moment: np.ndarray, total: np.ndarray, count: int, axes: int) -> np.ndarray:
    centre = total / max(count, 1)
    centred = moment / max(count, 1) - np.outer(centre, centre)
    values, vectors = np.linalg.eigh(0.5 * (centred + centred.T))
    return vectors[:, np.argsort(values)[::-1][:axes]]


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
