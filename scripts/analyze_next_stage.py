#!/usr/bin/env python3
"""Analysis experiments for the next-stage paper plan (section 7 of the plan).

1. Phase trajectory visualization: mean cumulative phase displacement per slot
   (from the PhaseVelocity module's last_delta capture) vs the static baseline
   (zero trajectory), for the velocity mechanism on each setting.
2. Residual gate alpha visualization: per-(sample, channel) alpha from the
   AdaptiveResidualGate module, correlated with a trend-strength proxy of the
   input history (|last - first| / recent volatility), testing the plan's
   "strong trend -> higher residual weight" hypothesis.
3. Error decomposition: phase-only vs residual-only vs fused forecast errors,
   obtained by monkeypatching the gate to constant 0/1 and re-running the
   forward, plus the fraction of cells where the phase branch is closer.

Outputs a CSV of per-cell decomposition + gate/trend data, and figures under
<output>/figures/<setting>__<figure>.png.
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
    find_run_dir,
    load_checkpoint,
    parse_setting,
)

PERIOD = 24


def gate_eval(model, exp_args, horizon, device, gate_value):
    """Recompute test predictions with the adaptive gate fixed to a constant.

    Returns (pred, truth, history) arrays. gate_value None -> normal (adaptive)
    forward; 0.0 -> phase-only forecast; 1.0 -> residual-only forecast.
    """
    import torch
    import torch.nn as nn

    from scripts.analyze_experiment import evaluate_test

    if gate_value is not None and hasattr(model, "adaptive_residual_gate"):
        C = model.enc_in

        class ConstGate(nn.Module):
            def forward(self, Z, x_in):
                B = Z.shape[0]
                return torch.full((B, 1, C), gate_value, device=Z.device)

        model.adaptive_residual_gate = ConstGate()
    pred, truth, hist, _ = evaluate_test(model, exp_args, horizon, device)
    return pred, truth, hist


def collect_diagnostics(model, exp_args, horizon, device, setting):
    """Recompute predictions while accumulating per-cell diagnostics.

    Returns (pred, truth, history, diag) where diag holds:
      - phase_velocity trajectories (mean |delta|, and per-slot mean delta)
      - adaptive gate alpha (B, C) per cell
    """
    import torch

    from scripts.analyze_experiment import data_provider

    has_vel = hasattr(model, "phase_velocity")
    has_gate = hasattr(model, "adaptive_residual_gate")
    test_set, test_loader = data_provider(exp_args.dataset_args, "test")
    preds, truths, histories = [], [], []
    alphas, deltas, trends = [], [], []
    with torch.inference_mode():
        for batch in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [
                b.to(device) if torch.is_tensor(b) else b for b in batch
            ]
            dec = model._build_decoder_input(batch_y.float())
            out, _, _ = model(
                batch_x.float(), batch_x_mark.float(), dec, batch_y_mark.float()
            )
            pred = out[:, -horizon:, :].float().cpu().numpy()
            true = batch_y.float()[:, -horizon:, :].cpu().numpy()
            hist = batch_x.float()[:, :, :].cpu().numpy()
            preds.append(pred)
            truths.append(true)
            histories.append(hist)
            if has_gate:
                a = model.adaptive_residual_gate.last_alpha.cpu().numpy()  # (B,1,C)
                alphas.append(a[:, 0, :])
            if has_vel:
                d = model.phase_velocity.last_delta.cpu().numpy()  # (B,C,L)
                deltas.append(d)
            # Trend-strength proxy over the last 2 periods of history.
            if has_gate:
                h = hist  # (B, seq, C)
                last2 = h[:, -2 * PERIOD:, :]
                trend = np.abs(last2[:, -1, :] - last2[:, 0, :])
                trends.append(trend)
    pred = np.concatenate(preds, axis=0)
    truth = np.concatenate(truths, axis=0)
    history = np.concatenate(histories, axis=0)
    diag = {}
    if has_gate:
        diag["alpha"] = np.concatenate(alphas, axis=0)  # (N, C)
        diag["trend"] = np.concatenate(trends, axis=0)  # (N, C)
    if has_vel:
        delta_arr = np.concatenate(deltas, axis=0)  # (N, C, L)
        diag["trajectory"] = delta_arr.mean(axis=(0, 1))  # (L,)
        diag["trajectory_std"] = delta_arr.std(axis=(0, 1))  # (L,)
        diag["mean_abs_delta"] = float(np.abs(delta_arr).mean())
    return pred, truth, history, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--modes", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--lookback", type=int, default=720)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    out_dir = Path(args.output)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    decom_rows = []
    gate_rows = []
    traj_data = {}

    for setting in settings:
        ds, horizon, seed = parse_setting(setting)
        for mode in modes:
            run_dir = find_run_dir(args.run_dir, args.run_prefix, mode, ds, horizon, seed)
            hp = json.loads((run_dir / "config.json").read_text())["hyperparams"]
            model, exp_args = build_model(ds, horizon, args.lookback, hp, args.device)
            load_checkpoint(model, run_dir / "checkpoints" / "best.ckpt", args.device)
            pred, truth, history, diag = collect_diagnostics(model, exp_args, horizon, args.device, setting)

            # --- Error decomposition (only for the adaptive-gate model) ---
            if hasattr(model, "adaptive_residual_gate"):
                pred_phase, _, _ = gate_eval(model, exp_args, horizon, args.device, 0.0)
                pred_resid, _, _ = gate_eval(model, exp_args, horizon, args.device, 1.0)
                err_phase = np.abs(pred_phase - truth).mean(axis=1)  # (N, C)
                err_resid = np.abs(pred_resid - truth).mean(axis=1)
                err_fused = np.abs(pred - truth).mean(axis=1)
                phase_better = (err_phase < err_resid)
                # alpha agrees with the better branch when alpha > 0.5 and residual
                # better, or alpha < 0.5 and phase better.
                alpha = diag["alpha"]
                agree = (alpha > 0.5) == (~phase_better)
                decom_rows.append(dict(
                    setting=setting, mode=mode, dataset=ds, horizon=horizon, seed=seed,
                    phase_mae=float(err_phase.mean()),
                    resid_mae=float(err_resid.mean()),
                    fused_mae=float(err_fused.mean()),
                    phase_better_frac=float(phase_better.mean()),
                    alpha_mean=float(alpha.mean()),
                    alpha_agree_frac=float(agree.mean()),
                    n_cells=int(alpha.size),
                ))
                for n in range(0, alpha.shape[0], max(1, alpha.shape[0] // 200)):
                    for c in range(alpha.shape[1]):
                        gate_rows.append(dict(
                            setting=setting, mode=mode, dataset=ds, horizon=horizon,
                            seed=seed, sample=n, channel=c,
                            alpha=float(alpha[n, c]),
                            trend=float(diag["trend"][n, c]),
                            resid_better=bool(phase_better[n, c]),
                        ))
                # Figure: gate alpha distribution + alpha vs trend-strength.
                fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
                ax[0].hist(alpha.ravel(), bins=40)
                ax[0].set_title(f"{setting} {mode}: gate alpha histogram")
                ax[0].set_xlabel("alpha"); ax[0].set_ylabel("cells")
                tr = diag["trend"].ravel()
                al = alpha.ravel()
                nbins = 30
                order = np.argsort(tr)
                bin_ids = np.clip(
                    (tr[order] - tr.min()) / max(tr.max() - tr.min(), 1e-9) * (nbins - 1),
                    0, nbins - 1,
                ).astype(int)
                binned_mean = np.zeros(nbins)
                binned_cnt = np.zeros(nbins)
                np.add.at(binned_cnt, bin_ids, 1)
                for b in range(nbins):
                    sel = bin_ids == b
                    binned_mean[b] = al[order][sel].mean() if sel.sum() else np.nan
                bin_center = np.linspace(tr.min(), tr.max(), nbins)
                ax[1].plot(bin_center, binned_mean, "-o", ms=3)
                ax[1].set_title("mean alpha vs trend-strength (history |d|)")
                ax[1].set_xlabel("trend-strength |last-first|"); ax[1].set_ylabel("mean alpha")
                fig.tight_layout()
                fig.savefig(fig_dir / f"{setting}__{mode}_gate_alpha.png", dpi=120)
                plt.close(fig)

            # --- Phase trajectory figure ---
            if "trajectory" in diag:
                traj_data[setting] = diag
                fig, ax = plt.subplots(figsize=(8, 4))
                slots = np.arange(len(diag["trajectory"]))
                ax.plot(slots, diag["trajectory"], "-o", label=f"{mode} (dynamic)")
                ax.fill_between(
                    slots,
                    diag["trajectory"] - diag["trajectory_std"],
                    diag["trajectory"] + diag["trajectory_std"],
                    alpha=0.2,
                )
                ax.axhline(0, color="k", ls="--", label="static baseline (0)")
                ax.set_title(f"{setting} {mode}: phase trajectory (cumulative displacement)")
                ax.set_xlabel("phase slot"); ax.set_ylabel("mean displacement")
                ax.legend()
                fig.tight_layout()
                fig.savefig(fig_dir / f"{setting}__{mode}_phase_trajectory.png", dpi=120)
                plt.close(fig)

    # Write CSVs
    if decom_rows:
        with (out_dir / "error_decomposition.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(decom_rows[0].keys()))
            w.writeheader(); w.writerows(decom_rows)
    if gate_rows:
        with (out_dir / "gate_alpha.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
            w.writeheader(); w.writerows(gate_rows)
    print(f"wrote decomposition/gate CSVs + figures to {out_dir}", flush=True)
    for setting, d in traj_data.items():
        print(
            f"trajectory {setting}: mean_abs_delta={d['mean_abs_delta']:.4f} "
            f"slots={list(np.round(d['trajectory'], 3))}",
            flush=True,
        )
    for r in decom_rows:
        print(
            f"decomp {r['setting']} {r['mode']}: phase={r['phase_mae']:.5f} "
            f"resid={r['resid_mae']:.5f} fused={r['fused_mae']:.5f} "
            f"phase_better={r['phase_better_frac']:.3f} alpha={r['alpha_mean']:.3f} "
            f"agree={r['alpha_agree_frac']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
