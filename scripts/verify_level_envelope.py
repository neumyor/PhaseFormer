#!/usr/bin/env python3
"""Measure the Theorem 2 level envelope on trained checkpoints (minipaper 0930).

Theorem 2 of ``docs/minipaper_0930.md`` bounds the level of every normalized
phase-path forecast by an interval read from the checkpoint's weights,

    I_phi = (c0 - C1, c0 + C1),   C_phi = |c0| + C1,

with ``A = W_d diag(gamma_2)``, ``q' = W_d beta_2 + c_d``, ``c0 = 1'q'/K'`` and
``C1 = sqrt(D) ||J(gamma_2 * W_d' 1)|| / K'`` taken from the last routing unit's
``norm2`` and the linear predictor.  This script measures, per checkpoint and
per channel-window in RevIN coordinates (``d`` is the level demand, the mean of
the normalized target):

* the envelope constants, and the utilisation ``max |l_p - c0| / C1`` over
  phases and windows (the theorem requires < 1, so this is a direct check);
* the fraction of windows whose demand lies outside ``I_phi`` and the
  envelope-floor and level shares of the phase-only MSE, via split (7);
* prediction P5: whether the paired PhaseFormer-L gain concentrates on the
  out-of-envelope windows, split into its level and shape parts;
* the L model's own phase path (its envelope and level tracking) and its
  branch level, to read how the fused model divides the level.

Validation is the primary split.  Several checkpoints were reused from earlier
test-read campaigns, so test numbers are conditional on that selection.
Nothing here trains a model.
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

LOOKBACK = 720

#: Flags that would insert something between the final LN_2 and the output, or
#: rescale the RevIN coordinates.  Any of them on voids the premise of Theorem 2.
PREMISE_FLAGS = (
    "use_latent_long_residual", "use_layerwise_latent_residual",
    "use_harmonic_modulation", "use_trajectory_decoder", "use_phase_local_trend",
    "use_phase_period_level_calibration", "use_phase_sparse_event_calibration",
    "use_phase_cycle_fusion", "use_triaxis_fusion", "use_time_mark_adjustment",
    "use_additive_output_residual", "use_topology_output_convex_residual",
    "use_phase_noise_hifreq_damping", "use_anchored_phase_cycle_fusion",
    "use_safe_triaxis", "use_rcrf_fusion", "use_dual_reliability_fusion",
)


# ---------------------------------------------------------------------------
# Envelope constants
# ---------------------------------------------------------------------------


def envelope_constants(model) -> dict:
    """``c0``, ``C1`` and ``C_phi`` of equation (19), from the weights."""
    norm2 = model.routing_layers[-1].interact.norm2
    decoder = model.predictor.decoder
    if not isinstance(decoder, torch.nn.Linear):
        raise ValueError("Theorem 2 needs a linear predictor")
    gamma = norm2.weight.detach().double().cpu()
    beta = norm2.bias.detach().double().cpu()
    weight = decoder.weight.detach().double().cpu()        # (K', D)
    bias = decoder.bias.detach().double().cpu()            # (K',)
    k_out, dim = weight.shape
    q = weight @ beta + bias
    v = gamma * weight.sum(dim=0)                           # A' 1
    centred = v - v.mean()
    c0 = float(q.mean())
    c1 = float(np.sqrt(dim) * centred.norm() / k_out)
    return {
        "c0": c0, "C1": c1, "C_phi": abs(c0) + c1,
        "lower": c0 - c1, "upper": c0 + c1,
        "latent_dim": dim, "k_out": k_out, "ln_eps": float(norm2.eps),
        "gamma_rms": float(gamma.pow(2).mean().sqrt()),
    }


def premise_violations(model) -> list[str]:
    active = [name for name in PREMISE_FLAGS if bool(getattr(model, name, False))]
    if getattr(model.predictor, "use_mlp", False):
        active.append("predictor_use_mlp")
    if not getattr(model, "use_revin", True):
        active.append("no_revin")
    elif getattr(model.revin, "affine", False):
        active.append("revin_affine")
    return active


# ---------------------------------------------------------------------------
# Forward pass with the phase path exposed
# ---------------------------------------------------------------------------


class Capture:
    """Forward hooks on the predictor (phase path) and the residual branch."""

    def __init__(self, model):
        self.phase = None
        self.residual = None
        self.handles = [model.predictor.register_forward_hook(self._phase)]
        branch = getattr(model, "weak_period_residual", None)
        if branch is not None:
            self.handles.append(branch.register_forward_hook(self._residual))

    def _phase(self, module, inputs, output):
        self.phase = output.detach()

    def _residual(self, module, inputs, output):
        self.residual = output.detach() if torch.is_tensor(output) else None

    def remove(self):
        for handle in self.handles:
            handle.remove()


def run_split(model, loader, device, horizon, envelope, max_batches=0):
    """Per channel-window quantities for one split, in RevIN coordinates.

    Returns flat float64 arrays of length ``windows * channels``.
    """
    capture = Capture(model)
    parts: dict[str, list] = {key: [] for key in (
        "sigma2", "demand", "err", "lev", "lev_phase", "phase_util",
        "lev_branch", "gate")}
    reconstruction = 0.0
    channels = 0
    c0, c1 = envelope["c0"], envelope["C1"]
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            x, y, x_mark, y_mark = [t.to(device) for t in batch[:4]]
            dec = model._build_decoder_input(y.float())
            out, _, _ = model(x.float(), x_mark.float(), dec, y_mark.float())
            truth = y.float()[:, -horizon:, :].double()
            channels = int(truth.shape[-1])
            xd = x.double()
            mu = xd.mean(dim=1, keepdim=True)
            sigma = (xd.var(dim=1, keepdim=True, unbiased=False) + model.revin.eps).sqrt()
            target = (truth - mu) / sigma                               # (B, H, C)
            fused = (out.double() - mu) / sigma                         # (B, H, C)

            steps = capture.phase.double()                              # (B, C, P, K')
            phase = steps.permute(0, 1, 3, 2).reshape(steps.shape[0], steps.shape[1], -1)
            phase = phase[..., :horizon].permute(0, 2, 1)               # (B, H, C)
            per_phase_level = steps.mean(dim=-1)                        # (B, C, P)
            util = ((per_phase_level - c0).abs() / c1).amax(dim=-1)     # (B, C)

            gate = model.last_weak_residual_alpha
            if gate is not None and capture.residual is not None:
                g = gate.double().expand_as(fused)
                branch = capture.residual.double()
                recon = (1.0 - g) * phase + g * branch
                parts["lev_branch"].append(branch.mean(dim=1).flatten().cpu())
                parts["gate"].append(g.mean(dim=1).flatten().cpu())
            else:
                recon = phase
            reconstruction = max(reconstruction, float((recon - fused).abs().max()))

            parts["sigma2"].append((sigma[:, 0, :] ** 2).flatten().cpu())
            parts["demand"].append(target.mean(dim=1).flatten().cpu())
            parts["err"].append(((fused - target) ** 2).mean(dim=1).flatten().cpu())
            parts["lev"].append(fused.mean(dim=1).flatten().cpu())
            parts["lev_phase"].append(phase.mean(dim=1).flatten().cpu())
            parts["phase_util"].append(util.flatten().cpu())
    capture.remove()
    arrays = {key: torch.cat(value).numpy() for key, value in parts.items() if value}
    arrays["reconstruction_max_abs"] = reconstruction
    arrays["channels"] = channels
    return arrays


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def distance_to_interval(value, lower, upper):
    return np.maximum(lower - value, 0.0) + np.maximum(value - upper, 0.0)


def rank_correlation(a, b) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def level_tracking(lev, demand) -> float:
    """Fraction of level-demand energy the forecast level explains."""
    energy = float(np.mean(demand ** 2))
    return 1.0 - float(np.mean((lev - demand) ** 2)) / energy if energy > 0 else float("nan")


def summarise(phase, fused, envelope, fused_envelope):
    """Cell-level numbers plus grouped P5 rows for one split."""
    sigma2, demand = phase["sigma2"], phase["demand"]
    lower, upper = envelope["lower"], envelope["upper"]
    dist = distance_to_interval(demand, lower, upper)
    outside = dist > 0
    level_p = (phase["lev"] - demand) ** 2
    mse_p = float(np.mean(sigma2 * phase["err"]))
    cell = {
        "windows": int(demand.size),
        "phase_mse": mse_p,
        "phase_util_max": float(phase["phase_util"].max()),
        "phase_util_p99": float(np.quantile(phase["phase_util"], 0.99)),
        "phase_util_median": float(np.median(phase["phase_util"])),
        "phase_reconstruction_max_abs": phase["reconstruction_max_abs"],
        "abs_d_median": float(np.median(np.abs(demand))),
        "abs_d_p90": float(np.quantile(np.abs(demand), 0.9)),
        "abs_d_p99": float(np.quantile(np.abs(demand), 0.99)),
        "frac_outside": float(outside.mean()),
        "frac_abs_d_gt_Cphi": float((np.abs(demand) > envelope["C_phi"]).mean()),
        "level_share": float(np.mean(sigma2 * level_p)) / mse_p,
        "floor_share": float(np.mean(sigma2 * dist ** 2)) / mse_p,
        "level_share_revin": float(np.mean(level_p) / np.mean(phase["err"])),
        "floor_share_revin": float(np.mean(dist ** 2) / np.mean(phase["err"])),
        "phase_level_tracking": level_tracking(phase["lev"], demand),
        "phase_level_spearman": rank_correlation(phase["lev"], demand),
    }
    groups = []
    if fused is None:
        return cell, groups

    if not np.allclose(fused["demand"], demand, atol=1e-6):
        raise RuntimeError("phase and fused loaders disagree on the windows")
    level_l = (fused["lev"] - demand) ** 2
    gain = sigma2 * (phase["err"] - fused["err"])
    level_gain = sigma2 * (level_p - level_l)
    shape_gain = gain - level_gain
    total_gain = float(gain.sum())
    total_level_gain = float(level_gain.sum())
    fused_dist = distance_to_interval(
        demand, fused_envelope["lower"], fused_envelope["upper"])
    cell.update({
        "fused_mse": float(np.mean(sigma2 * fused["err"])),
        "fused_reconstruction_max_abs": fused["reconstruction_max_abs"],
        "gain": total_gain / demand.size,
        "level_gain": total_level_gain / demand.size,
        "shape_gain": float(shape_gain.sum()) / demand.size,
        "outside_gain_share": float(gain[outside].sum()) / total_gain if total_gain else float("nan"),
        "outside_level_gain_share": (float(level_gain[outside].sum()) / total_level_gain
                                     if total_level_gain else float("nan")),
        "outside_gain_per_window": float(gain[outside].mean()) if outside.any() else float("nan"),
        "inside_gain_per_window": float(gain[~outside].mean()) if (~outside).any() else float("nan"),
        "spearman_dist_vs_gain": rank_correlation(dist, gain),
        "spearman_absd_vs_gain": rank_correlation(np.abs(demand), gain),
        "fused_level_tracking": level_tracking(fused["lev"], demand),
        "fused_level_share": float(np.mean(sigma2 * level_l)) / float(np.mean(sigma2 * fused["err"])),
        "fused_phase_util_max": float(fused["phase_util"].max()),
        "fused_phase_level_tracking": level_tracking(fused["lev_phase"], demand),
        "fused_phase_level_spearman": rank_correlation(fused["lev_phase"], demand),
        "fused_phase_frac_outside": float((fused_dist > 0).mean()),
        "fused_phase_abs_level_mean": float(np.mean(np.abs(fused["lev_phase"]))),
        "phase_abs_level_mean": float(np.mean(np.abs(phase["lev"]))),
    })
    if "lev_branch" in fused:
        branch = fused["lev_branch"]
        gate = fused["gate"]
        cell.update({
            "gate_mean": float(gate.mean()),
            "branch_level_tracking": level_tracking(branch, demand),
            "branch_frac_outside_own_envelope": float(
                (distance_to_interval(branch, fused_envelope["lower"],
                                      fused_envelope["upper"]) > 0).mean()),
            # Share of the fused level carried by the branch, (g lev_R) / lev_L,
            # measured as an energy ratio so that it is sign-free.
            "branch_level_energy_share": float(
                np.mean((gate * branch) ** 2) /
                max(np.mean((gate * branch) ** 2) + np.mean(((1 - gate) * fused["lev_phase"]) ** 2),
                    1e-300)),
        })

    def group_row(grouping, label, members):
        return {
            "grouping": grouping, "group": label, "windows": int(members.sum()),
            "window_frac": float(members.mean()),
            "abs_d_mean": float(np.abs(demand[members]).mean()),
            "phase_mse": float(np.mean(sigma2[members] * phase["err"][members])),
            "fused_mse": float(np.mean(sigma2[members] * fused["err"][members])),
            "gain_per_window": float(gain[members].mean()),
            "level_gain_per_window": float(level_gain[members].mean()),
            "shape_gain_per_window": float(shape_gain[members].mean()),
            "gain_share": float(gain[members].sum()) / total_gain if total_gain else float("nan"),
            "level_gain_share": (float(level_gain[members].sum()) / total_level_gain
                                 if total_level_gain else float("nan")),
            "phase_level_share": float(np.sum(sigma2[members] * level_p[members]) /
                                       np.sum(sigma2[members] * phase["err"][members])),
            "floor_per_window": float(np.mean(sigma2[members] * dist[members] ** 2)),
        }

    groups.append(group_row("envelope", "inside", ~outside))
    if outside.any():
        groups.append(group_row("envelope", "outside", outside))
    order = np.argsort(np.abs(demand), kind="stable")
    for quartile, members_index in enumerate(np.array_split(order, 4), start=1):
        members = np.zeros(demand.size, dtype=bool)
        members[members_index] = True
        groups.append(group_row("abs_d_quartile", str(quartile), members))
    top = np.zeros(demand.size, dtype=bool)
    top[order[-max(1, demand.size // 10):]] = True
    groups.append(group_row("abs_d_top_decile", "top10", top))
    return cell, groups


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load(row, dataset, horizon, repo_root, device, splits):
    run_dir = repo_root / row["run_dir"]
    checkpoints = sorted((run_dir / "attempts").glob("*/checkpoints/best.ckpt"))
    if not checkpoints:
        return None
    config = json.loads((run_dir / "config.json").read_text())
    hyperparams = dict(config["hyperparams"])
    exp_args, handles = build_loaders(
        dataset, LOOKBACK, horizon, hyperparams,
        int(config.get("batch_size") or 256), repo_root, splits=splits)
    model = build_model(exp_args, LOOKBACK, horizon, hyperparams)
    report = load_checkpoint_into(model, checkpoints[0])
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError(f"state mismatch for {checkpoints[0]}: {report}")
    model.to(device).eval()
    return model, handles, checkpoints[0], report["epoch"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="research_runs/phaseformer_L_e14_main_v1/results.csv")
    parser.add_argument("--output-dir", default="research_runs/level_envelope_v1")
    parser.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,Traffic")
    parser.add_argument("--seeds", default="2021,2022,2023")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--save-windows-max-channels", type=int, default=21,
                        help="save per-window npz only for datasets with at most this many channels")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets = {v for v in args.datasets.split(",") if v}
    seeds = {int(v) for v in args.seeds.split(",") if v}
    horizons = {int(v) for v in args.horizons.split(",") if v}
    splits = tuple(v for v in args.splits.split(",") if v)

    table = list(csv.DictReader((repo_root / args.results).open()))
    index = {(r["setting"], r["seed"], r["arm"]): r for r in table}
    rows = [
        r for r in table
        if r["arm"] == "phase_only" and r["status"] in ("read", "reused")
        and r["dataset"] in datasets and int(r["seed"]) in seeds
        and int(r["horizon"]) in horizons
    ]
    rows.sort(key=lambda r: (r["dataset"], int(r["horizon"]), int(r["seed"])))
    rows = [r for i, r in enumerate(rows) if i % args.shard_count == args.shard_index]
    print(f"shard {args.shard_index}/{args.shard_count}: {len(rows)} cells on {device}", flush=True)

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cells, groups = [], []

    for position, row in enumerate(rows, start=1):
        dataset, horizon, seed = row["dataset"], int(row["horizon"]), int(row["seed"])
        setting = row["setting"]
        started = time.time()
        loaded = load(row, dataset, horizon, repo_root, device, splits)
        if loaded is None:
            print(f"  [skip] no phase checkpoint for {setting} seed={seed}", flush=True)
            continue
        phase_model, phase_handles, phase_ckpt, phase_epoch = loaded
        envelope = envelope_constants(phase_model)
        violations = premise_violations(phase_model)
        period = int(phase_model.period_len)
        phase_runs = {split: run_split(phase_model, phase_handles[split][1], device,
                                       horizon, envelope, args.max_batches)
                      for split in splits}
        del phase_model, phase_handles

        fused_row = index.get((setting, str(seed), "l_main"))
        fused_runs, fused_envelope, fused_ckpt, fused_violations = {}, None, "", []
        if fused_row is not None:
            loaded = load(fused_row, dataset, horizon, repo_root, device, splits)
            if loaded is not None:
                fused_model, fused_handles, fused_ckpt, _ = loaded
                fused_envelope = envelope_constants(fused_model)
                fused_violations = premise_violations(fused_model)
                fused_runs = {split: run_split(fused_model, fused_handles[split][1], device,
                                               horizon, fused_envelope, args.max_batches)
                              for split in splits}
                del fused_model, fused_handles

        for split in splits:
            cell, split_groups = summarise(phase_runs[split], fused_runs.get(split),
                                           envelope, fused_envelope)
            base = {
                "setting": setting, "dataset": dataset, "horizon": horizon, "seed": seed,
                "split": split, "period": period,
                "phase_status": row["status"],
                "fused_status": fused_row["status"] if fused_row else "",
                "phase_recorded_test_mse": float(row["test_mse"]),
                "fused_recorded_test_mse": float(fused_row["test_mse"]) if fused_row else float("nan"),
                "phase_recorded_val_mse": float(row["val_mse"]) if row.get("val_mse") else float("nan"),
                "fused_recorded_val_mse": (float(fused_row["val_mse"])
                                           if fused_row and fused_row.get("val_mse") else float("nan")),
                "c0": envelope["c0"], "C1": envelope["C1"], "C_phi": envelope["C_phi"],
                "latent_dim": envelope["latent_dim"], "k_out": envelope["k_out"],
                "gamma_rms": envelope["gamma_rms"],
                "fused_c0": fused_envelope["c0"] if fused_envelope else float("nan"),
                "fused_C1": fused_envelope["C1"] if fused_envelope else float("nan"),
                "fused_C_phi": fused_envelope["C_phi"] if fused_envelope else float("nan"),
                "phase_premise_violations": ";".join(violations),
                "fused_premise_violations": ";".join(fused_violations),
                "phase_checkpoint": str(phase_ckpt.relative_to(repo_root)),
                "fused_checkpoint": str(fused_ckpt.relative_to(repo_root)) if fused_ckpt else "",
            }
            cells.append({**base, **cell})
            for group in split_groups:
                groups.append({"setting": setting, "dataset": dataset, "horizon": horizon,
                               "seed": seed, "split": split, **group})
            if phase_runs[split]["channels"] <= args.save_windows_max_channels:
                payload = {f"phase_{k}": v.astype(np.float32)
                           for k, v in phase_runs[split].items() if isinstance(v, np.ndarray)}
                if split in fused_runs:
                    payload.update({f"fused_{k}": v.astype(np.float32)
                                    for k, v in fused_runs[split].items()
                                    if isinstance(v, np.ndarray)})
                window_dir = output_dir / "windows"
                window_dir.mkdir(exist_ok=True)
                np.savez_compressed(window_dir / f"{setting}_s{seed}_{split}.npz", **payload)
        primary = [c for c in cells if c["setting"] == setting and c["seed"] == seed][0]
        print(
            f"  [{position}/{len(rows)}] {setting} seed={seed} {primary['split']}: "
            f"c0={envelope['c0']:+.3f} C1={envelope['C1']:.3f} "
            f"util={primary['phase_util_max']:.3f} out={primary['frac_outside']:.4f} "
            f"level_share={primary['level_share']:.3f} floor_share={primary['floor_share']:.4f} "
            f"gain={primary.get('gain', float('nan')):+.5f} "
            f"out_gain_share={primary.get('outside_gain_share', float('nan')):.3f} "
            f"({time.time() - started:.0f}s)", flush=True)
        write_csv(cells, output_dir / f"envelope_cells_shard{args.shard_index}.csv")
        write_csv(groups, output_dir / f"envelope_groups_shard{args.shard_index}.csv")
    print(f"shard {args.shard_index} done", flush=True)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
