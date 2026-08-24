#!/usr/bin/env python3
"""Analysis experiments for the Pure Phase Modeling plan (plan section 5).

Recomputes each mode's test predictions from best.ckpt and produces:

1. Phase trajectory visualization: per-slot mean cumulative displacement for
   the phase_velocity (near-constant drift) vs phase_deformation / pure_full
   (nonlinear deformation), answering "constant drift vs nonlinear".
2. Phase deformation field: per-slot mean advance rate, stretch factor - 1, and
   cumulative displacement for the deformation models -> learned shift /
   stretch / compression.
3. Frequency-phase consistency: per-period circular argmax distance (peak shift
   error) and within-3-step agreement for original / phase_deformation /
   pure_full, using the same peak_metrics as analyze_peak_shift.
4. Trajectory smoothness: mean |y_{k+1} - y_k| over the horizon for every mode,
   comparing the low-order trajectory decoder vs the free per-cycle predictor.
5. Multi-scale zeta: the multiscale fusion gate vector's open fraction / mean
   magnitude (whether the long-period branch is actually used).

Outputs CSVs under <output>/ and figures under <output>/figures/.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_experiment import (  # noqa: E402
    build_model,
    data_provider,
    find_run_dir,
    load_checkpoint,
    parse_setting,
)
from scripts.analyze_peak_shift import peak_metrics  # noqa: E402

DEFAULT_SETTINGS = (
    "ETTh2_h336_seed2021,ETTm1_h336_seed2021,Electricity_h336_seed2021,"
    "Traffic_h336_seed2021,ETTh1_h336_seed2021,"
    "ETTh2_h720_seed2021,ETTm1_h720_seed2021,Electricity_h720_seed2021,"
    "Traffic_h720_seed2021,ETTh1_h720_seed2026"
)
DEFAULT_MODES = (
    "original,phase_velocity,multiscale_phase,phase_deformation,phase_geo,"
    "phase_graph,predictor_mlp,trajectory_decoder,pure_full"
)


def collect_diagnostics(model, exp_args, horizon, device):
    """One test-set forward pass, capturing per-module diagnostics.

    Returns (pred, truth, diag) where diag holds (concatenated over batches):
      - velocity_delta (N, C, L) cumulative displacement (PhaseVelocity)
      - deform_rate / deform_stretch / deform_delta (N, C, L) (PhaseDeformation)
      - graph_message (list of per-batch mean |message|) (PhaseGraph)
      - multiscale_abs_long (list of per-batch mean |Z_long|) (MultiScalePhase)
      - traj_smoothness (list of per-batch mean |dy|) (TrajectoryDecoder)
    """
    import torch

    _, test_loader = data_provider(exp_args.dataset_args, "test")
    preds, truths = [], []
    vel_delta, def_rate, def_stretch, def_delta = [], [], [], []
    graph_msg, ms_abs_long, traj_smooth = [], [], []
    has_vel = hasattr(model, "phase_velocity")
    has_def = hasattr(model, "phase_deformation")
    has_graph = hasattr(model, "phase_graph")
    has_ms = hasattr(model, "multiscale_phase")
    has_traj = hasattr(model, "trajectory_decoder")
    with torch.inference_mode():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(
                batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float()
            )
            preds.append(out[:, -horizon:, :].float().cpu().numpy())
            truths.append(batch_y.float()[:, -horizon:, :].cpu().numpy())
            if has_vel:
                vel_delta.append(model.phase_velocity.last_delta.cpu().numpy())
            if has_def:
                def_rate.append(model.phase_deformation.last_rate.cpu().numpy())
                def_stretch.append(model.phase_deformation.last_stretch.cpu().numpy())
                def_delta.append(model.phase_deformation.last_delta.cpu().numpy())
            if has_graph:
                graph_msg.append(model.phase_graph.last_mean_message)
            if has_ms:
                ms_abs_long.append(model.multiscale_phase.last_mean_abs_long)
            if has_traj:
                traj_smooth.append(model.trajectory_decoder.last_smoothness)
    pred = np.concatenate(preds, axis=0)
    truth = np.concatenate(truths, axis=0)
    diag = {}
    if has_vel:
        diag["velocity_delta"] = np.concatenate(vel_delta, axis=0)
    if has_def:
        diag["deform_rate"] = np.concatenate(def_rate, axis=0)
        diag["deform_stretch"] = np.concatenate(def_stretch, axis=0)
        diag["deform_delta"] = np.concatenate(def_delta, axis=0)
    if has_graph:
        diag["graph_message"] = float(np.mean(graph_msg))
    if has_ms:
        diag["multiscale_abs_long"] = float(np.mean(ms_abs_long))
    if has_traj:
        diag["traj_smoothness"] = float(np.mean(traj_smooth))
    return pred, truth, diag


def slot_profile(arr):
    """Mean over (sample, channel) of a (N, C, L) field -> (L,)."""
    return arr.mean(axis=(0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="research_runs/dyn_phase_full")
    p.add_argument("--run-prefix", default="dynphase")
    p.add_argument("--modes", default=DEFAULT_MODES)
    p.add_argument("--settings", default=DEFAULT_SETTINGS)
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", default="research_runs/pure_phase_analysis")
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    out_dir = Path(args.output)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    peak_rows, smooth_rows, zeta_rows, deform_rows = [], [], [], []

    for setting in settings:
        ds, horizon, seed = parse_setting(setting)
        traj_figures = []  # (mode, (L,) displacement, label)
        def_fields = {}   # mode -> (rate, stretch_minus1, delta) slot profiles
        zeta_vecs = {}    # mode -> (D,) zeta vector
        smooth_vals = {}  # mode -> smoothness
        for mode in modes:
            try:
                run_dir = find_run_dir(args.run_dir, args.run_prefix, mode, ds, horizon, seed)
            except FileNotFoundError:
                print(f"SKIP (no run): {setting} {mode}", flush=True)
                continue
            hp = json.loads((run_dir / "config.json").read_text())["hyperparams"]
            model, exp_args = build_model(ds, horizon, args.lookback, hp, args.device)
            load_checkpoint(model, run_dir / "checkpoints" / "best.ckpt", args.device)
            pred, truth, diag = collect_diagnostics(model, exp_args, horizon, args.device)

            err = pred - truth
            mse = float(np.square(err).mean())
            mae = float(np.abs(err).mean())
            smooth = float(np.abs(pred[..., 1:] - pred[..., :-1]).mean())
            smooth_vals[mode] = smooth
            pk = peak_metrics(pred, truth)
            peak_rows.append(dict(
                setting=setting, mode=mode, dataset=ds, horizon=horizon, seed=seed,
                mse=mse, mae=mae,
                peak_shift_err=pk["peak_shift_err"], peak_within3=pk["peak_within3"],
                peak_amp_err=pk["peak_amp_err"],
            ))
            smooth_rows.append(dict(
                setting=setting, mode=mode, dataset=ds, horizon=horizon, seed=seed,
                smoothness=smooth,
                traj_diag=diag.get("traj_smoothness", ""),
            ))
            print(
                f"{setting:26s} {mode:20s} mse={mse:.5f} mae={mae:.5f} "
                f"smooth={smooth:.5f} peak_shift={pk['peak_shift_err']:.3f} "
                f"within3={pk['peak_within3']:.3f}",
                flush=True,
            )

            # --- 1. Phase trajectory (cumulative displacement) ---
            if "velocity_delta" in diag:
                traj_figures.append(
                    (mode, slot_profile(diag["velocity_delta"]), "phase_velocity")
                )
            if "deform_delta" in diag:
                traj_figures.append(
                    (mode, slot_profile(diag["deform_delta"]), mode)
                )

            # --- 2. Deformation field (per-slot rate / stretch / displacement) ---
            if "deform_rate" in diag:
                def_fields[mode] = (
                    slot_profile(diag["deform_rate"]),
                    slot_profile(diag["deform_stretch"] - 1.0),
                    slot_profile(diag["deform_delta"]),
                )
                for slot in range(diag["deform_delta"].shape[-1]):
                    deform_rows.append(dict(
                        setting=setting, mode=mode, dataset=ds, horizon=horizon,
                        seed=seed, slot=slot,
                        mean_rate=float(def_fields[mode][0][slot]),
                        mean_stretch_minus1=float(def_fields[mode][1][slot]),
                        mean_delta=float(def_fields[mode][2][slot]),
                    ))

            # --- 5. Multi-scale zeta vector ---
            if hasattr(model, "multiscale_phase"):
                z = model.multiscale_phase.zeta.detach().cpu().numpy()  # (D,)
                zeta_vecs[mode] = z
                zeta_rows.append(dict(
                    setting=setting, mode=mode, dataset=ds, horizon=horizon, seed=seed,
                    zeta_dim=int(z.size),
                    zeta_mean_abs=float(np.abs(z).mean()),
                    zeta_std=float(z.std()),
                    zeta_max_abs=float(np.abs(z).max()),
                    zeta_open_frac=float((np.abs(z) > 1e-4).mean()),
                    zeta_nnz=float((np.abs(z) > 1e-4).sum()),
                    multiscale_abs_long=diag.get("multiscale_abs_long", ""),
                ))

        # --- Figures per setting ---
        L = None
        if traj_figures:
            L = len(traj_figures[0][1])
        fig, ax = plt.subplots(figsize=(8, 4))
        slots = np.arange(L) if L else np.arange(0)
        for mode, prof, label in traj_figures:
            ax.plot(slots, prof, "-o", ms=3, label=label)
        ax.axhline(0, color="k", ls="--", label="static baseline (0)")
        ax.set_title(f"{setting}: phase trajectory (cumulative displacement)")
        ax.set_xlabel("phase slot"); ax.set_ylabel("mean displacement")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"{setting}__phase_trajectory.png", dpi=120)
        plt.close(fig)

        if def_fields:
            names = list(def_fields.keys())
            fig, axs = plt.subplots(1, 3, figsize=(14, 4))
            for name in names:
                rate, stretch_m1, delta = def_fields[name]
                axs[0].plot(np.arange(len(rate)), rate, "-o", ms=3, label=name)
                axs[1].plot(np.arange(len(stretch_m1)), stretch_m1, "-o", ms=3, label=name)
                axs[2].plot(np.arange(len(delta)), delta, "-o", ms=3, label=name)
            axs[0].set_title("mean advance rate"); axs[1].set_title("mean stretch - 1")
            axs[2].set_title("mean cumulative displacement")
            for ax in axs:
                ax.set_xlabel("phase slot"); ax.legend()
            fig.suptitle(f"{setting}: phase deformation field", y=1.02)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{setting}__deformation_field.png", dpi=120)
            plt.close(fig)

        if smooth_vals:
            names = list(smooth_vals.keys())
            vals = [smooth_vals[n] for n in names]
            fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(names)), 4))
            ax.bar(np.arange(len(names)), vals)
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_title(f"{setting}: trajectory smoothness (mean |y_{{k+1}} - y_k|)")
            ax.set_ylabel("mean |dy|")
            fig.tight_layout()
            fig.savefig(fig_dir / f"{setting}__trajectory_smoothness.png", dpi=120)
            plt.close(fig)

        if zeta_vecs:
            names = list(zeta_vecs.keys())
            fig, ax = plt.subplots(figsize=(8, 4))
            for name in names:
                ax.plot(np.arange(len(zeta_vecs[name])), zeta_vecs[name], "-o", ms=2, label=name)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{setting}: multiscale fusion gate zeta")
            ax.set_xlabel("latent dim"); ax.set_ylabel("zeta")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / f"{setting}__multiscale_phase_zeta.png", dpi=120)
            plt.close(fig)

        # Peak-shift comparison for the plan's core trio.
        trio = [m for m in ("original", "phase_deformation", "pure_full") if m in smooth_vals]
        if len(trio) >= 2:
            pk_rows = [r for r in peak_rows
                       if r["setting"] == setting and r["mode"] in trio]
            by_mode = {r["mode"]: r for r in pk_rows}
            names = list(by_mode.keys())
            shift = [by_mode[n]["peak_shift_err"] for n in names]
            within = [by_mode[n]["peak_within3"] for n in names]
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].bar(np.arange(len(names)), shift)
            axs[0].set_xticks(np.arange(len(names))); axs[0].set_xticklabels(names)
            axs[0].set_title("peak shift error (circular argmax distance)")
            axs[1].bar(np.arange(len(names)), within)
            axs[1].set_xticks(np.arange(len(names))); axs[1].set_xticklabels(names)
            axs[1].set_title("peak within 3 steps fraction")
            fig.suptitle(f"{setting}: frequency-phase consistency", y=1.02)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{setting}__peak_shift_comparison.png", dpi=120)
            plt.close(fig)

    # --- Write CSVs ---
    def write_csv(rows, name):
        if rows:
            with (out_dir / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {len(rows)} rows -> {out_dir / name}", flush=True)

    write_csv(peak_rows, "frequency_phase_consistency.csv")
    write_csv(smooth_rows, "trajectory_smoothness.csv")
    write_csv(zeta_rows, "zeta_analysis.csv")
    write_csv(deform_rows, "deformation_field.csv")
    print(f"figures -> {fig_dir}", flush=True)


if __name__ == "__main__":
    main()
