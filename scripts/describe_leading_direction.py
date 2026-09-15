#!/usr/bin/env python3
"""Describe the leading (rank-1) predictive direction of the NLinear task.

The rank-1 optimum of ``min_{rank(W)<=1} E||D - WZ||^2`` factorizes as
``W_1 = a_1 b_1^T``: ``b_1`` is the input functional (what the branch reads out of
the centered window) and ``a_1`` is the output direction (the horizon shape it
writes).  Both are computed here from the saved second moments produced by
``scripts/analyze_optimal_lowrank_capture.py --save-moments``.

The script answers two separate questions, with deliberately different kinds of
evidence:

  1. Is this direction *dominant*?  Read ``lambda_1 / sum(lambda)``: by the RRR
     identity the rank-1 map removes exactly ``lambda_1`` of the achievable MSE
     reduction over the persistence anchor.  This is an exact statement about the
     global optimum, not a description.
  2. Is it a *low-frequency recent-level / local-trend* direction?  This is a
     descriptive claim and is measured three ways:
       * lag profile: where ``b_1`` puts its mass, and how well {const, ramp}
         explains it restricted to the recent tail;
       * spectrum: energy share at periods >= 24 h / >= 72 h and the centroid
         period, so "low frequency" is stated in hours, not by a half-spectrum
         convention;
       * output shape: ``a_1`` vs a flat level shift and vs a ramp, sampled across
         the horizon, so the reader can see whether the map predicts "the level
         moves and stays" or "the level drifts".

Usage:

    python scripts/describe_leading_direction.py \
        --moments-dir research_runs/lowrank_data_property_v2 \
        --output-csv research_runs/lowrank_data_property_v2/leading_direction.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

RIDGE = 1e-8
SEQ_LEN = 720
STEP_HOURS = 1.0  # all sweep settings are hourly or finer in index terms; ETTm2 is 15 min


def leading_direction(szz: np.ndarray, szy: np.ndarray):
    szz_r = szz + RIDGE * np.eye(szz.shape[0])
    ols = np.linalg.solve(szz_r, szy).T  # H x L
    s_mat = szy.T @ np.linalg.solve(szz_r, szy)
    s_mat = 0.5 * (s_mat + s_mat.T)
    vals, vecs = np.linalg.eigh(s_mat)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    u1 = vecs[:, 0]
    b1 = u1 @ ols  # input functional, length L
    return u1, b1, vals


def tail_fit(vector: np.ndarray, window: int) -> float:
    """R^2 of fitting {constant, linear ramp} on the most recent ``window`` steps."""
    seg = vector[-window:]
    t = np.arange(window, dtype=float)
    design = np.stack([np.ones(window), t], axis=1)
    coef, *_ = np.linalg.lstsq(design, seg, rcond=None)
    resid = seg - design @ coef
    denom = float((seg**2).sum())
    return float(1.0 - (resid**2).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments-dir", default="research_runs/lowrank_data_property_v2")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.moments_dir, "moments_*.npz"))):
        data = np.load(path)
        szz, szy, syy = data["szz"], data["szy"], data["syy"]
        H = szy.shape[1]
        L = szy.shape[0]
        stem = os.path.basename(path).replace("moments_", "").replace(".npz", "")
        dataset, hs = stem.rsplit("_h", 1)

        u1, b1, vals = leading_direction(szz, szy)
        b1 = b1 / (np.linalg.norm(b1) + 1e-12)
        u1 = u1 / (np.linalg.norm(u1) + 1e-12)

        # --- dominance (exact) -------------------------------------------------
        lam1_share = float(vals[0] / vals.sum())
        trace_syy = float(np.trace(syy))
        gain_r1 = float(vals[0]) / trace_syy  # share of the persistence MSE removed

        # --- lag profile of b1 ---
        energy = b1**2
        total = float(energy.sum())
        mass_last = {k: float(energy[-k:].sum() / total) for k in (24, 72, 168)}
        # smallest trailing window holding 80% of the mass
        cumulative = np.cumsum(energy[::-1]) / total
        win80 = int(np.searchsorted(cumulative, 0.80) + 1)
        rows.append(
            dict(
                dataset=dataset,
                horizon=int(hs),
                rrr_rank=1,
                lambda1_share_of_achievable=round(lam1_share, 4),
                r1_gain_vs_persistence_pct=round(100 * gain_r1, 2),
                mass_last24=round(mass_last[24], 3),
                mass_last72=round(mass_last[72], 3),
                mass_last168=round(mass_last[168], 3),
                window_holding_80pct_steps=win80,
                tail168_r2_vs_const_ramp=round(tail_fit(b1, 168), 3),
                tail24_r2_vs_const_ramp=round(tail_fit(b1, 24), 3),
            )
        )

        # --- spectrum of b1 (in hours, index-based) ---
        spec = np.abs(np.fft.rfft(b1)) ** 2
        freqs = np.arange(len(spec), dtype=float)  # cycles per L steps
        # DC is excluded from the centroid (its "period" is infinite); the mean
        # frequency is expressed as steps-per-cycle for readability.
        mean_freq = float((freqs[1:] * spec[1:]).sum() / spec[1:].sum())
        centroid_period = 1.0 / mean_freq if mean_freq > 0 else float("inf")
        periods = np.divide(
            L, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0
        )
        # Band profile in window-relative cycles: band k covers periods
        # L/(k+1)..L/k steps.  "steps" = dataset sampling interval (1 h for
        # ETTh2/Weather/Electricity, 15 min for ETTm2).
        band_edges = [(0, 0), (1, 5), (6, 29), (30, 59), (60, 179), (180, 360)]
        band_shares = []
        for lo, hi in band_edges:
            band_shares.append(round(float(spec[lo : hi + 1].sum() / spec.sum()), 3))
        rows[-1]["band_shares_dc_1to5_6to29_30to59_60to179_180to360"] = band_shares
        low24 = float(spec[periods >= 24].sum() / spec.sum())
        low72 = float(spec[periods >= 72].sum() / spec.sum())
        sub_daily = float(spec[periods < 24].sum() / spec.sum())
        rows[-1].update(
            centroid_period_steps=round(centroid_period, 1),
            energy_period_ge24=round(low24, 3),
            energy_period_ge72=round(low72, 3),
            energy_period_lt24=round(sub_daily, 3),
            cos_const=round(abs(float(b1 @ (np.ones(L) / np.sqrt(L)))), 3),
            cos_ramp=round(
                abs(float(b1 @ ((np.arange(L) - (L - 1) / 2) / np.linalg.norm(np.arange(L) - (L - 1) / 2)))),
                3,
            ),
        )

        # --- how much of b1 is "constant plateau + linear slope" ---
        # Regression onto {const, ramp, tail-mean(24), tail-mean(168)} gives an
        # honest "explained variance" for the level/trend label, unlike a single
        # |cos| against one template.
        dict_cols = [np.ones(L) / np.sqrt(L)]
        ramp_full = np.arange(L) - (L - 1) / 2
        dict_cols.append(ramp_full / np.linalg.norm(ramp_full))
        for window in (24, 168):
            col = np.zeros(L)
            col[-window:] = 1.0 / np.sqrt(window)
            dict_cols.append(col)
        design_dict = np.stack(dict_cols, axis=1)
        coef_d, *_ = np.linalg.lstsq(design_dict, b1, rcond=None)
        resid_d = b1 - design_dict @ coef_d
        dict_r2 = float(1.0 - (resid_d**2).sum() / float((b1**2).sum()))
        rows[-1]["level_trend_dictionary_r2"] = round(dict_r2, 3)

        # --- output direction a1 = u1 ---
        ones = np.ones(H) / np.sqrt(H)
        ramp = np.arange(H) - (H - 1) / 2
        ramp = ramp / np.linalg.norm(ramp)
        profile_idx = [0, H // 4, H // 2, 3 * H // 4, H - 1]
        peak = float(np.abs(u1).max()) + 1e-12
        rows[-1].update(
            out_cos_const=round(abs(float(u1 @ ones)), 3),
            out_cos_ramp=round(abs(float(u1 @ ramp)), 3),
            out_profile= " / ".join(f"{u1[i] / peak:+.2f}" for i in profile_idx),
            out_sign_consistency=round(float(np.mean(np.sign(u1) == np.sign(u1[0]))), 3),
        )
        print(f"[done] {dataset}-{H}: lambda1 share {lam1_share:.3f}, "
              f"mass last24/72/168 = {mass_last[24]:.2f}/{mass_last[72]:.2f}/{mass_last[168]:.2f}, "
              f"centroid period {centroid_period:.0f} steps, >=24h energy {low24:.2f}", flush=True)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[written] {args.output_csv}")


if __name__ == "__main__":
    main()
