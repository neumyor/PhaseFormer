#!/usr/bin/env python3
"""Settle the branch algebra against the cached features, empirically.

Loads one cached cell and tests which closed-form reconstruction of the
residual branch reproduces the model's own `fused` output.  Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    path = Path(sys.argv[1])
    d = np.load(path)
    hidden = d["hidden"].astype(np.float64)      # (N, C, r)
    z = d["z"].astype(np.float64)                # (N, L, C)
    sigma = d["sigma"].astype(np.float64)        # (N, 1, C)
    mu = d["mu"].astype(np.float64)              # (N, 1, C)
    gate = d["gate"].astype(np.float64)          # (N, 1, C)
    phase = d["phase"].astype(np.float64)        # (N, H, C)
    target = d["target"].astype(np.float64)
    fused = d["fused"].astype(np.float64)
    x_last = d["x_last_norm"].astype(np.float64)  # (N, 1, C)
    ew = d["encoder_weight"].astype(np.float64)  # (r, L)
    eb = d["encoder_bias"].astype(np.float64)    # (r,)
    dw = d["decoder_weight"].astype(np.float64)  # (H, r)
    db = d["decoder_bias"].astype(np.float64)    # (H,)

    print(f"file: {path.name}")
    print(f"  shapes hidden={hidden.shape} z={z.shape} phase={phase.shape} "
          f"gate={gate.shape} ew={ew.shape} dw={dw.shape}")
    print(f"  gate: mean={gate.mean():.6f} std={gate.std():.8f} "
          f"min={gate.min():.6f} max={gate.max():.6f}")

    # --- Q1: does the cached `hidden` include the encoder bias? -----------
    enc_no_bias = np.einsum("rl,nlc->ncr", ew, z)
    enc_with_bias = enc_no_bias + eb[None, None, :]
    for label, h in (("encoder(z)", enc_no_bias),
                     ("encoder(z)+bias", enc_with_bias)):
        err = np.abs(h - hidden).max()
        print(f"  hidden vs {label:18s}: max_abs={err:.6e}")

    last_abs = sigma * x_last + mu
    dec_hidden = np.einsum("ncr,hr->nhc", hidden, dw)
    mapped_eb = dw @ eb

    # The cache's ``x_last_norm`` is the horizon-mean of
    # ``residual_norm - dw@h - dw@eb - db``, i.e. the true anchor minus
    # ``mean_h(dw@eb)``.  Recover the true anchor and check the branch output
    # against the fusion inverted from the model's own ``fused``.
    deficit = float(mapped_eb.mean())
    x_last_true = x_last + deficit
    last_abs_true = sigma * x_last_true + mu
    branch_inverted = (fused - (1.0 - gate) * phase) / np.maximum(gate, 1e-12)

    recovered = last_abs_true + sigma * (dec_hidden + db[None, :, None])
    print(f"  mean_h(dw@eb) = {deficit:+.6e}   |dw@eb| range "
          f"[{mapped_eb.min():+.4e}, {mapped_eb.max():+.4e}]")
    print(f"  branch(recovered anchor) vs inverted-from-fused: "
          f"max_abs={np.abs(recovered - branch_inverted).max():.3e}")

    candidates = {
        "sigma*(dec(h)+db)": last_abs + sigma * (dec_hidden + db[None, :, None]),
        "sigma*(dec(h)+Wb+db)": last_abs + sigma * (
            dec_hidden + mapped_eb[None, :, None] + db[None, :, None]),
        "RECOVERED true anchor": recovered,
    }
    print("  --- reconstruction of fused = (1-g)*phase + g*branch_abs ---")
    for label, branch_abs in candidates.items():
        fused_calc = (1.0 - gate) * phase + gate * branch_abs
        err = np.abs(fused_calc - fused).max()
        scale = np.abs(fused).max()
        print(f"    {label:24s} max_abs={err:.6e}  (|fused|max={scale:.4f})")

    # --- Q2: what does the model's own residual look like? ----------------
    # fusion is exact, so branch_abs = (fused - (1-g)*phase) / g
    branch_from_fused = (fused - (1.0 - gate) * phase) / np.maximum(gate, 1e-12)
    for label, branch_abs in candidates.items():
        err = np.abs(branch_abs - branch_from_fused).max()
        print(f"  branch vs fused-inverted {label:24s}: max_abs={err:.6e}")

    print(f"  fused MSE = {np.mean((fused - target) ** 2):.6f}")
    print(f"  phase MSE = {np.mean((phase - target) ** 2):.6f}")
    print(f"  |fused|max = {np.abs(fused).max():.4f}  "
          f"|phase|max = {np.abs(phase).max():.4f}")


if __name__ == "__main__":
    main()
