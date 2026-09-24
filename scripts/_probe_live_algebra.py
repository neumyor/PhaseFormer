#!/usr/bin/env python3
"""Live forward-pass check of the branch algebra, on one cell.

Answers, from the model's own tensors rather than from the cache:
  1. is ``fused == (1-g) * phase + g * branch`` with the captured phase?
  2. does the cached ``phase`` equal the fused phase component?
  3. does the closed-form ``I_i`` decomposition satisfy the additivity identity
     ``sum_i I_i == MSE(y_hat_0) - MSE(y_hat)``?

Read-only: loads frozen checkpoints, reads the validation split, never test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "src").is_dir():
    # The probe is copied to /tmp when it runs on the remote host; fall back to
    # the current working directory, which REMOTE_SERVER.md requires to be the
    # repository root because data_provider resolves paths relative to it.
    REPO_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lowrank_checkpoint_model import (  # noqa: E402
    build_loaders,
    build_model,
    load_checkpoint_into,
    instrument_model,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--features-dir", default="research_runs/lowrank_checkpoint_information_v1/features")
    ap.add_argument("--inventory", default="research_runs/lowrank_checkpoint_information_v1/checkpoint_inventory.csv")
    ap.add_argument("--max-batches", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    repo_root = REPO_ROOT
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = list(csv.DictReader(open(repo_root / args.inventory)))
    row = next(
        r for r in rows
        if r["setting"] == args.setting and int(r["seed"]) == args.seed
        and r["cell"] == args.cell
    )
    run_dir = repo_root / row["selected_run_dir"]
    config = json.loads((run_dir / "config.json").read_text())
    hyperparams = dict(config["hyperparams"])
    batch_size = int(config.get("batch_size") or hyperparams.get("batch_size") or 256)
    ckpt = repo_root / row["checkpoint_path"]
    print(f"cell   : {args.setting} seed={args.seed} {args.cell} rank={row['rank']}")
    print(f"ckpt   : {ckpt}")
    print(f"device : {device}")

    exp_args, handles = build_loaders(
        row["dataset"], 720, int(row["horizon"]), hyperparams, batch_size, repo_root,
        splits=("val",),
    )
    model = build_model(exp_args, 720, int(row["horizon"]), hyperparams)
    info = load_checkpoint_into(model, ckpt)
    print(f"load   : missing={len(info['missing_keys'])} unexpected={len(info['unexpected_keys'])}")
    model.eval()
    model.to(device)
    val_set, val_loader = handles["val"]

    head = model.weak_period_residual
    dw = head.decoder.weight.detach().double().cpu().numpy()
    db = head.decoder.bias.detach().double().cpu().numpy()
    ew = head.encoder.weight.detach().double().cpu().numpy()
    eb = head.encoder.bias.detach().double().cpu().numpy()
    W = dw @ ew

    zs, hs, gs, ss, mus, phs, rss, fus, tgt, xln = [], [], [], [], [], [], [], [], [], []
    with torch.inference_mode():
        for i, batch in enumerate(val_loader):
            if i >= args.max_batches:
                break
            batch = [t.to(device) if torch.is_tensor(t) else t for t in batch]
            x, y, x_mark, y_mark = batch
            dec = model._build_decoder_input(y.float())
            with instrument_model(model, None) as inst:
                out, _, _ = inst(x.float(), x_mark.float(), dec, y_mark.float())
                rec = inst.last_lowrank_records
                phs.append(inst.last_phase_forecast.double().cpu().numpy())
                rss.append(inst.last_residual_forecast.double().cpu().numpy())
                fus.append(out.double().cpu().numpy())
                tgt.append(y.float()[:, -int(row["horizon"]):, :].double().cpu().numpy())
                hs.append(rec["hidden"].double().cpu().numpy())
                zs.append(rec["z"].permute(0, 2, 1).double().cpu().numpy())
                mu_, sg_ = rec["stats"]
                mus.append(mu_.double().cpu().numpy())
                ss.append(sg_.double().cpu().numpy())
                gate = rec["gate"]
                gs.append(gate.reshape(1, 1, -1).expand(x.shape[0], 1, gate.shape[-1]).double().cpu().numpy())
                xln.append(rec["anchor64"].double().cpu().numpy())

    cat = lambda a: np.concatenate(a, axis=0)  # noqa: E731
    phase, residual = cat(phs), cat(rss)
    fused, target = cat(fus), cat(tgt)
    hidden, z = cat(hs), cat(zs)
    mu, sigma, gate, x_true = cat(mus), cat(ss), cat(gs), cat(xln)
    print(f"captured: phase{phase.shape} hidden{hidden.shape} z{z.shape} gate{gate.shape}")

    g = gate  # (N,1,C)
    print("\n--- 1. fusion identity with the model's own tensors ---")
    rec_fused = (1.0 - g) * phase + g * residual
    print(f"  |fused - ((1-g)phase + g*residual)|max = {np.abs(rec_fused - fused).max():.3e}")
    phase_implied = (fused - g * residual) / (1.0 - g)
    print(f"  |last_phase_forecast - implied phase|max   = {np.abs(phase - phase_implied).max():.3e}")

    print("\n--- 2. branch closed forms vs the model's own residual ---")
    # true anchor in value space, straight from the model
    last_abs_true = sigma * x_true + mu
    dec_h = np.einsum("ncr,hr->nhc", hidden, dw)
    variants = {
        "last_abs + sigma*(dw@h + db)": last_abs_true + sigma * (dec_h + db[None, :, None]),
        "last_abs + sigma*(dw@h + Wb + db)": last_abs_true + sigma * (
            dec_h + (dw @ eb)[None, :, None] + db[None, :, None]),
    }
    for label, branch in variants.items():
        f = (1.0 - g) * phase + g * branch
        print(f"  {label:38s} |fused-recon|max={np.abs(f - fused).max():.3e} "
              f"|branch-residual|max={np.abs(branch - residual).max():.3e}")

    print("\n--- 3. cached vs live, same cell ---")
    cache_path = repo_root / args.features_dir / f"{args.setting}_seed{args.seed}_{args.cell.replace('/', '-')}.npz"
    if cache_path.is_file():
        d = np.load(cache_path)
        for key, live in (("hidden", hidden), ("z", z), ("sigma", sigma),
                          ("mu", mu), ("gate", gate), ("phase", phase),
                          ("target", target), ("fused", fused)):
            c = d[key].astype(np.float64)
            shape_note = "" if c.shape == live.shape else f" SHAPE cached{c.shape} live{live.shape}"
            err = np.abs(c - live).max() if c.shape == live.shape else float("nan")
            print(f"  {key:10s} |cached-live|max={err:.3e}{shape_note}")
    else:
        print(f"  (no cache at {cache_path})")

    print("\n--- 4. mode decomposition + additivity ---")
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    r = s.size
    a = np.einsum("rl,nlc->nrc", vt, z)  # (r, N, C) mode activations
    # branch_abs = last_abs + sigma * (W z + c) ; split W z into modes
    c_const = (dw @ eb) + db  # the cache-consistent affine constant
    WZ = np.einsum("hr,rnc->nhc", u * s[None, :], a)  # = W z, (N,H,C)
    branch_full = last_abs_true + sigma * (
        np.einsum("ncr,hr->nhc", hidden, dw) + (dw @ eb)[None, :, None] + db[None, :, None])
    fused_full = (1.0 - g) * phase + g * branch_full

    branch_zero = last_abs_true + sigma * ((dw @ eb)[None, :, None] + db[None, :, None])
    fused_zero = (1.0 - g) * phase + g * branch_zero

    e0 = target - fused_zero
    # d_i[n,h,c] = g * sigma * s_i * u_i[h] * a[i,n,c]
    I = np.zeros(r)
    for i in range(r):
        di = g * sigma * s[i] * u[None, :, i][:, None, :] * a[i][:, None, None]
        I[i] = float(np.mean(2.0 * e0 * di - di ** 2))

    mse_0 = float(np.mean((fused_zero - target) ** 2))
    mse_full = float(np.mean((fused_full - target) ** 2))
    print(f"  rank r={r}   MSE(W=0)={mse_0:.9f}   MSE(W)={mse_full:.9f}")
    print(f"  sum(I_i)          = {I.sum():.12f}")
    print(f"  MSE0 - MSE_full   = {mse_0 - mse_full:.12f}")
    print(f"  additivity resid  = {abs(I.sum() - (mse_0 - mse_full)):.3e}")
    print(f"  recon vs recorded fused MSE: {mse_full:.9f} vs {float(np.mean((fused-target)**2)):.9f}")
    print(f"  top-5 I_i = {np.round(np.sort(I)[::-1][:5], 8)}")
    print(f"  singular s[:5] = {np.round(s[:5], 5)}")


if __name__ == "__main__":
    main()
