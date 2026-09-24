#!/usr/bin/env python3
"""Does a frozen PhaseFormer leave predictable level information unused?

The paired comparisons show that adding a temporal branch improves the fused
forecast on ETTh2/ETTm2/Weather, and sample-level diagnostics show the gain
concentrates in windows whose cross-cycle level moves.  Neither establishes the
stronger claim that the frozen model *underuses* level information, because the
original model also consumes level information.

This script tests that claim in two parts, both on data the fit never saw.

**Part A -- is the frozen model's error level-predictable out of sample?**
A ridge map from the window's cross-cycle level trajectory to the frozen
model's error *magnitude* is fitted on the training split, its penalty chosen on
validation, and its out-of-sample rank correlation reported on test.  Predicting
the scalar error rather than the full error vector is deliberate: it is far
better powered, and it answers the question actually asked -- whether level
information tells you *where* the frozen model will be wrong.

An earlier version of this probe instead fitted a map to the whole error vector
and applied it as a correction.  That map has only `horizon x (cycles + 1)`
parameters and cannot beat a no-op at any penalty on these settings, so it
measures the weakness of the correction form, not the model's use of level
information.  It is retained only as a reported negative control.

**Part B -- does the trained correction's benefit concentrate where the level
moves?** The paired PhaseFormer-L checkpoint for the same setting and seed is run
on the same windows, and its per-window gain over the frozen model is grouped by
how far the level moved.

Controls share the parameter count but not the level semantics: shuffled level
features, cycle means of a time-permuted window, a fixed random projection, and
the principal axes of the normalized window.

Fitting and penalty selection touch only train and validation; the test split is
read only for evaluation.  The checkpoints themselves are not blind -- see the
report's disclosure section.
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


# ---------------------------------------------------------------------------
# Level features
# ---------------------------------------------------------------------------


def cycle_means(xn: np.ndarray, period: int, count: int) -> np.ndarray:
    """Mean of each of the ``count`` most recent non-overlapping cycles.

    ``xn`` is (N, L, C) and the result is (N, C, count); cycle 0 is the most
    recent.  When ``L`` is not a multiple of ``period`` the leading remainder is
    left unused rather than folded into a partial cycle.
    """
    length = xn.shape[1]
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


def level_statistics(level: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window level-movement descriptors, averaged over channels."""
    trajectory = level.mean(axis=1)                     # (N, K)
    latest = trajectory[:, 0]
    prior = trajectory[:, 1:].mean(axis=1) if trajectory.shape[1] > 1 else latest
    return {
        "level_std": trajectory.std(axis=1),
        "last_cycle_offset": np.abs(latest - prior),
        "level_range": np.abs(trajectory[:, 0] - trajectory[:, -1]),
    }


def with_intercept(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones(features.shape[:2] + (1,))], axis=2)


# ---------------------------------------------------------------------------
# Ridge on a scalar target
# ---------------------------------------------------------------------------


def standardise(train: np.ndarray, others: dict[str, np.ndarray]):
    """Feature mean and standard deviation from train, applied everywhere.

    Without this the penalty is meaningless: the raw level features have a Gram
    matrix whose largest eigenvalue is ~1e6, so a grid reaching only 30 applies
    essentially no shrinkage and the fit generalises at chance.
    """
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    return mean, scale, {name: (value - mean) / scale for name, value in others.items()}


def ridge_scalar(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    gram = x.T @ x
    return np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), x.T @ y)


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a.argsort().argsort().astype(float),
                             b.argsort().argsort().astype(float))[0, 1])


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------


def calibrate_scales(model, loader, device, max_batches: int = 8) -> np.ndarray:
    """Per-channel error scale: the median window standard deviation on train."""
    deviations = []
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            xn = batch[0].to(device).double().cpu().numpy()
            deviations.append(xn.std(axis=1))
    return np.median(np.concatenate(deviations, axis=0), axis=0)


def run_split(model, loader, device, period, cycle_count,
              projections: dict[str, np.ndarray] | None = None,
              max_batches: int = 0):
    """Per-window quantities for one split under one checkpoint."""
    level_parts, shuffled_parts, projected_parts, error_parts = [], [], {}, []
    projected_parts = {name: [] for name in (projections or {})}
    window_moment = np.zeros((LOOKBACK, LOOKBACK))
    samples = 0
    permutation = np.random.default_rng(0).permutation(LOOKBACK)

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            batch = [t.to(device) if torch.is_tensor(t) else t for t in batch]
            x, y, x_mark, y_mark = batch
            dec = model._build_decoder_input(y.float())
            out, _, _ = model(x.float(), x_mark.float(), dec, y_mark.float())
            horizon = out.shape[1]
            truth = y.float()[:, -horizon:, :]

            xn = x.double().cpu().numpy()
            error = (truth.double().cpu().numpy() - out.double().cpu().numpy())
            error_parts.append(np.mean(error ** 2, axis=(1, 2)))   # per window

            level_parts.append(cycle_means(xn, period, cycle_count))
            shuffled_parts.append(cycle_means(xn[:, permutation, :], period, cycle_count))
            for name, matrix in (projections or {}).items():
                projected_parts[name].append(np.einsum("nlc,lf->ncf", xn, matrix))
            flat_xn = xn.reshape(-1, LOOKBACK)
            window_moment += flat_xn.T @ flat_xn
            samples += xn.shape[0]
            del xn, out, truth, error, flat_xn

    return {
        "samples": samples,
        "level": np.concatenate(level_parts, axis=0),
        "time_shuffled": np.concatenate(shuffled_parts, axis=0),
        "per_window_error": np.concatenate(error_parts, axis=0),
        "window_moment": window_moment,
        "projected": {name: np.concatenate(parts, axis=0)
                      for name, parts in projected_parts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="research_runs/phaseformer_L_e14_main_v1/results.csv")
    parser.add_argument("--output-dir", default="research_runs/level_underutilisation_v1")
    parser.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Weather")
    parser.add_argument("--seeds", default="2021,2022,2023")
    parser.add_argument("--penalties", default="0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000")
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

    table = list(csv.DictReader((repo_root / args.results).open()))
    index = {(r["setting"], r["seed"], r["arm"]): r for r in table}
    rows = [
        r for r in table
        if r["arm"] == "phase_only" and r["status"] in ("read", "reused")
        and r["dataset"] in datasets and int(r["seed"]) in seeds
    ]
    rows.sort(key=lambda r: (r["dataset"], int(r["horizon"]), int(r["seed"])))
    rows = [r for i, r in enumerate(rows) if i % args.shard_count == args.shard_index]
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
            print(f"  [skip] no phase checkpoint for {setting} seed={seed}", flush=True)
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
        phase_model = build_model(exp_args, LOOKBACK, horizon, hyperparams)
        load_checkpoint_into(phase_model, checkpoints[0])
        phase_model.to(device).eval()

        rng = np.random.default_rng(seed)
        random_projection = rng.standard_normal((LOOKBACK, cycle_count)) / np.sqrt(LOOKBACK)
        first = {split: run_split(phase_model, handles[split][1], device, period, cycle_count,
                                  max_batches=args.max_batches)
                 for split in ("train", "val", "test")}
        principal = _principal_axes(first["train"]["window_moment"], cycle_count)
        projections = {"random_projection": random_projection, "pca_matched": principal}
        second = {split: run_split(phase_model, handles[split][1], device, period, cycle_count,
                                   projections=projections, max_batches=args.max_batches)
                  for split in ("train", "val", "test")}
        for split in ("train", "val", "test"):
            for name, value in second[split]["projected"].items():
                first[split]["projected"][name] = value
        del phase_model, second

        # --- Part A: is the frozen error level-predictable out of sample? ----
        def design(split: str, name: str) -> np.ndarray:
            if name == "level":
                return first[split]["level"]
            if name == "time_shuffled_level":
                return first[split]["time_shuffled"]
            return first[split]["projected"][name]

        variants = ["level", "shuffled_level", "time_shuffled_level",
                    "random_projection", "pca_matched"]
        predictability: dict[str, dict] = {}
        for name in variants:
            source = "level" if name == "shuffled_level" else name
            flat = {split: with_intercept(design(split, source)).reshape(
                -1, with_intercept(design(split, source)).shape[-1])
                for split in ("train", "val", "test")}
            targets = {split: np.repeat(first[split]["per_window_error"],
                                        first[split]["level"].shape[1])
                       for split in ("train", "val", "test")}
            if name == "shuffled_level":
                permuted = rng.permutation(flat["train"].shape[0])
                flat["train"] = flat["train"][permuted]
                targets["train"] = targets["train"][permuted]

            mean, scale, standardised = standardise(flat["train"], {
                split: flat[split] for split in ("val", "test")})
            standardised["train"] = (flat["train"] - mean) / scale

            best, best_val = None, np.inf
            for penalty in penalties:
                weight = ridge_scalar(standardised["train"], targets["train"], penalty)
                residual = standardised["val"] @ weight - targets["val"]
                if float(np.mean(residual ** 2)) < best_val:
                    best_val, best = float(np.mean(residual ** 2)), weight
            prediction = standardised["test"] @ best
            observed = targets["test"]
            variance = float(np.var(observed))
            predictability[name] = {
                "spearman": rank_correlation(prediction, observed),
                "r2": (1.0 - float(np.mean((prediction - observed) ** 2)) / variance
                       if variance > 0 else float("nan")),
                "val_mse": best_val,
            }

        # --- Part B: does the trained correction help most where level moves? -
        correction_row = index.get((setting, str(seed), "l_main"))
        gain_by_window = None
        if correction_row is not None:
            correction_dir = repo_root / correction_row["run_dir"]
            correction_checkpoints = sorted(
                (correction_dir / "attempts").glob("*/checkpoints/best.ckpt"))
            if correction_checkpoints:
                correction_config = json.loads((correction_dir / "config.json").read_text())
                correction_args, correction_handles = build_loaders(
                    dataset, LOOKBACK, horizon, dict(correction_config["hyperparams"]),
                    int(correction_config.get("batch_size") or 256), repo_root,
                    splits=("test",),
                )
                correction_model = build_model(correction_args, LOOKBACK, horizon,
                                               dict(correction_config["hyperparams"]))
                load_checkpoint_into(correction_model, correction_checkpoints[0])
                correction_model.to(device).eval()
                corrected = run_split(correction_model, correction_handles["test"][1],
                                      device, period, cycle_count,
                                      max_batches=args.max_batches)
                gain_by_window = first["test"]["per_window_error"] - corrected["per_window_error"]
                del correction_model, correction_handles

        statistics = level_statistics(first["test"]["level"])
        if gain_by_window is not None:
            for name, statistic in statistics.items():
                for quartile, members in enumerate(np.array_split(np.argsort(statistic), 4), start=1):
                    groups.append({
                        "setting": setting, "dataset": dataset, "horizon": horizon,
                        "seed": seed, "grouping": name, "quartile": quartile,
                        "windows": int(members.size),
                        "mean_statistic": float(statistic[members].mean()),
                        "phase_mse": float(first["test"]["per_window_error"][members].mean()),
                        "fused_mse": float((first["test"]["per_window_error"][members]
                                            - gain_by_window[members]).mean()),
                        "gain": float(gain_by_window[members].mean()),
                    })

        entry = {
            "setting": setting, "dataset": dataset, "horizon": horizon, "seed": seed,
            "lookback": LOOKBACK, "period": period, "cycle_features": cycle_count,
            "train_windows": int(first["train"]["samples"]),
            "val_windows": int(first["val"]["samples"]),
            "test_windows": int(first["test"]["samples"]),
            "test_phase_mse": float(first["test"]["per_window_error"].mean()),
            "recorded_test_mse": float(row["test_mse"]),
        }
        for name, value in predictability.items():
            entry[f"predict_{name}_spearman"] = value["spearman"]
            entry[f"predict_{name}_r2"] = value["r2"]
        if gain_by_window is not None:
            entry["mean_gain_phase_to_fused"] = float(gain_by_window.mean())
            for name, statistic in statistics.items():
                entry[f"spearman_{name}_vs_gain"] = rank_correlation(statistic, gain_by_window)

        summary.append(entry)
        print(
            f"  [{position}/{len(rows)}] {setting} seed={seed} "
            f"rho(level->err)={entry['predict_level_spearman']:+.3f} "
            f"rho(shuf)={entry['predict_shuffled_level_spearman']:+.3f} "
            f"rho(pca)={entry['predict_pca_matched_spearman']:+.3f} "
            f"gain={entry.get('mean_gain_phase_to_fused', float('nan')):+.5f} "
            f"({time.time() - started:.0f}s)", flush=True,
        )

    write_csv(summary, output_dir / f"level_probe_cells_shard{args.shard_index}.csv")
    write_csv(groups, output_dir / f"level_probe_groups_shard{args.shard_index}.csv")
    print(f"shard {args.shard_index} done", flush=True)


def _principal_axes(moment: np.ndarray, axes: int) -> np.ndarray:
    """Top principal axes of the normalized window, from its second moment.

    These are the leading temporal patterns of the input window, with no
    level-specific construction, so projecting onto them gives a generic
    temporal path of the same dimension and parameter count as the level map.
    """
    values, vectors = np.linalg.eigh(0.5 * (moment + moment.T))
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
