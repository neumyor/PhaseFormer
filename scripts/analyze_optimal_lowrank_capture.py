#!/usr/bin/env python3
"""Optimal-rank analysis of the NLinear weak-period residual branch.

Question this script answers (training-free, CPU-only, no checkpoints needed):

    Can a rank bottleneck on ``weak_period_residual`` preserve the branch's
    performance, and if so, which property of the data makes that possible?

It trains nothing.  Instead it solves the branch's own regression task exactly,
on the same data the branch sees:

    input   Z = x_window - x_last                             [720]
    target  D = y_horizon - x_last                            [H]

in train-split-standardized units, using the repository's own split borders and
the same channel-shared linear map as ``WeakPeriodResidualHead`` /
``PooledLowRankWeakPeriodResidualHead``.

The per-window RevIN normalization cancels exactly for this branch, so no
approximation is involved:  the head computes delta_n = W (x_n - x_last_n) with
x_n = (x - mu)/sigma, and the model then denormalizes by multiplying with the
same sigma, giving  sigma * delta_n = W (x - x_last)  in raw (scaled) units.
Working in raw deltas therefore reproduces the model's own objective weighting
(per-channel scale preserved) instead of equal-weighting channels that happen to
have a near-constant window.

For each rank r it computes the *globally optimal* rank-r linear map
(reduced-rank regression)

    W_r = Syy^{1/2} U_r D_r V_r^T Szz^{-1/2},
    U D V^T = SVD( Syy^{-1/2} Szy^T Szz^{-1/2} ),

with Szz/Szy/Syy the second moments of (Z, D) on the training split.  ``W_r`` is
therefore the best any rank-r NLinear branch could be on that distribution, and
its held-out MSE bounds what rank compression can achieve: if ``W_r`` already
matches the full-rank optimum, compression cannot be the cause of any loss.

Per setting and rank the script reports:
  * ``val_mse``          held-out MSE in RevIN-normalized units
  * ``capture_val_pct``  (persistence - mse_r) / (persistence - mse_fullrank)*100
  * ``used_var_share``   share of the centered-window variance the rank-r row
                         space actually reads
  * ``rowspace_high_freq_frac`` / ``rowspace_centroid``  spectral content of it

Usage (dataset layout follows the repository ``resources/all_datasets/``):

    python scripts/analyze_optimal_lowrank_capture.py \
        --seq-len 720 --per-channel \
        --settings ETTh2:96,ETTh2:720,ETTm2:96,ETTm2:192,Weather:96,Weather:192,Electricity:336 \
        --output-dir research_runs/lowrank_data_property_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATASETS = {
    "ETTh2": ("ETT", "ETTh2.csv", "ett_hour"),
    "ETTm2": ("ETT", "ETTm2.csv", "ett_minute"),
    "Weather": ("weather", "weather.csv", "custom"),
    "Electricity": ("electricity", "electricity.csv", "custom"),
}

# Ranks actually tested by the conditioned rank sweep (r = q * H).  Ranks 1/2/3
# are added so the capture curve is anchored at the low end; extra ranks cost
# almost nothing because the second-moment accumulation dominates the runtime.
TESTED_RANKS = {
    96: [1, 2, 3, 6, 12, 24, 96],
    192: [1, 2, 3, 6, 12, 24, 48, 192],
    336: [1, 2, 3, 10, 21, 42, 84, 336],
    720: [1, 2, 3, 5, 10, 22, 45, 90, 180, 720],
}


def spectral_stats(vector: np.ndarray) -> tuple[float, float]:
    """(centroid normalized by spectrum length, high-frequency energy share).

    Mirrors ``scripts/analyze_weak_residual_lowrank_basis_fft.py``: the top half
    of the rfft spectrum counts as high frequency.
    """
    spectrum = np.abs(np.fft.rfft(vector))
    total = float(spectrum.sum()) + 1e-12
    freqs = np.arange(len(spectrum))
    centroid = float((freqs * spectrum).sum() / total) / max(1, len(freqs))
    cutoff = len(spectrum) // 2
    return centroid, float(spectrum[cutoff:].sum() / total)


def load_split(dataset: str, seq_len: int, data_root: Path):
    """Reproduce the repository's train/val splits and train-split scaling."""
    subdir, filename, kind = DATASETS[dataset]
    df_raw = pd.read_csv(os.path.join(data_root, subdir, filename))
    raw = df_raw[df_raw.columns[1:]].values.astype(np.float64)

    if kind in ("ett_hour", "ett_minute"):
        day = 24 if kind == "ett_hour" else 24 * 4
        base = 12 * 30 * day
        border1s = [0, base - seq_len, base + 4 * 30 * day - seq_len]
        border2s = [base, base + 4 * 30 * day, base + 8 * 30 * day]
    else:  # custom: 70 / 10 / 20
        num_train = int(len(raw) * 0.7)
        num_test = int(len(raw) * 0.2)
        num_vali = len(raw) - num_train - num_test
        border1s = [0, num_train - seq_len, len(raw) - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, len(raw)]

    train_raw = raw[border1s[0] : border2s[0]]
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    data = (raw - mean) / std
    train_seg = data[border1s[0] : border2s[0]]
    val_seg = data[border1s[1] : border2s[1]]
    return train_seg, val_seg


def iter_batches(seg: np.ndarray, seq_len: int, pred_len: int, chunk: int):
    """Yield (Z, D) chunks of RevIN-normalized window/deviation pairs."""
    total = len(seg) - seq_len - pred_len + 1
    if total <= 0:
        return
    x_all = sliding_window_view(seg, seq_len, axis=0)[:total]
    y_all = sliding_window_view(seg[seq_len:], pred_len, axis=0)[:total]
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        x = np.asarray(x_all[start:stop], dtype=np.float64)
        y = np.asarray(y_all[start:stop], dtype=np.float64)
        last = x[:, :, -1:]
        z = (x - last).reshape(-1, seq_len)
        d = (y - last).reshape(-1, pred_len)
        yield z, d


def accumulate(seg, seq_len, pred_len, chunk, per_channel=False):
    """Second moments of (Z, D) plus the rank-0 (persistence) error."""
    n_ch = seg.shape[1]
    s_zz = np.zeros((seq_len, seq_len))
    s_zy = np.zeros((seq_len, pred_len))
    s_yy = np.zeros((pred_len, pred_len))
    s_zz_ch = np.zeros((n_ch, seq_len, seq_len)) if per_channel else None
    s_zy_ch = np.zeros((n_ch, seq_len, pred_len)) if per_channel else None
    s_yy_ch = np.zeros((n_ch, pred_len, pred_len)) if per_channel else None
    sq_ch = np.zeros(n_ch) if per_channel else None
    count = 0
    for z, d in iter_batches(seg, seq_len, pred_len, chunk):
        s_zz += z.T @ z
        s_zy += z.T @ d
        s_yy += d.T @ d
        count += len(z)
        if per_channel:
            zz = z.reshape(-1, n_ch, seq_len)
            dd = d.reshape(-1, n_ch, pred_len)
            sq_ch += (dd**2).sum(axis=(0, 2))
            for c in range(n_ch):
                zc, dc = zz[:, c, :], dd[:, c, :]
                s_zz_ch[c] += zc.T @ zc
                s_zy_ch[c] += zc.T @ dc
                s_yy_ch[c] += dc.T @ dc
    n = max(count, 1)
    out = {
        "n": count,
        "szz": s_zz / n,
        "szy": s_zy / n,
        "syy": s_yy / n,
        "persistence_mse": float(np.trace(s_yy) / (n * pred_len)),
    }
    if per_channel:
        out["szz_ch"] = s_zz_ch / n
        out["szy_ch"] = s_zy_ch / n
        out["syy_ch"] = s_yy_ch / n
        out["persistence_mse_ch"] = sq_ch / (n * pred_len)
    return out


def fit_rrr(szz, szy, syy, ranks, ridge):
    """Exact minimizers of  E||D - W Z||^2  subject to rank(W) <= r.

    Derivation (plain-MSE reduced-rank regression, no Syy weighting):
      E||D - ABZ||^2 with W = AB; given A the optimal B is
      B = (A^T A)^-1 A^T G Szz^-1 with G = E[D Z^T] = Szy^T, which reduces the
      objective to  tr(Syy) - tr(P_A S),  S = G Szz^-1 G^T = Szy^T Szz^-1 Szy.
      Maximizing tr(P_A S) gives P_A = top-r eigenprojector of S, hence

          W_r = U_r U_r^T Szy^T Szz^-1,   U_r = top-r eigenvectors of S.

    The eigenvalue lambda_i of S is exactly the MSE reduction bought by the i-th
    RRR direction, so its spectrum is a directly interpretable "predictive
    structure" spectrum of the data.
    """
    szz_r = szz + ridge * np.eye(szz.shape[0])
    ols = np.linalg.solve(szz_r, szy).T  # H x L, full-rank optimum
    s_mat = szy.T @ np.linalg.solve(szz_r, szy)  # H x H
    s_mat = 0.5 * (s_mat + s_mat.T)
    vals, vecs = np.linalg.eigh(s_mat)  # ascending
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]

    maps = {}
    for r in sorted(set(list(ranks) + [len(vals)])):
        rr = min(r, len(vals))
        proj = vecs[:, :rr] @ vecs[:, :rr].T
        maps[r] = proj @ ols
    return maps, vals, ols


def direction_stats(vec: np.ndarray, szz: np.ndarray) -> dict:
    """Interpretable description of one input-side direction of a rank-r map."""
    v = vec / (np.linalg.norm(vec) + 1e-12)
    L = len(v)
    idx = np.arange(L)
    ramp = idx - idx.mean()
    ramp = ramp / np.linalg.norm(ramp)
    ones = np.ones(L) / np.sqrt(L)
    tail24 = np.zeros(L)
    tail24[-24:] = 1.0
    tail24 = tail24 / np.linalg.norm(tail24)
    return {
        "cos_ones": round(abs(float(v @ ones)), 3),
        "cos_ramp": round(abs(float(v @ ramp)), 3),
        "cos_last24": round(abs(float(v @ tail24)), 3),
        "var_read_share": round(float((v @ szz @ v) / np.trace(szz)), 6),
    }


def eval_mse(seg, seq_len, pred_len, chunk, maps):
    """Held-out MSE (normalized units) for every candidate map + persistence."""
    se = {r: 0.0 for r in maps}
    pers = 0.0
    tot = 0
    for z, d in iter_batches(seg, seq_len, pred_len, chunk):
        tot += len(z)
        pers += float((d * d).sum())
        for r, w in maps.items():
            resid = d - z @ w.T
            se[r] += float((resid * resid).sum())
    scale = max(tot * pred_len, 1)
    out = {r: v / scale for r, v in se.items()}
    out["persistence"] = pers / scale
    return out, tot


def per_channel_capture(train_seg, val_seg, seq_len, horizon, ranks, chunk, ridge):
    """Channel-by-channel capture, to test whether low rank is intrinsic."""
    stats = accumulate(train_seg, seq_len, horizon, chunk, per_channel=True)
    n_ch = train_seg.shape[1]
    caps: dict[int, list[float]] = {r: [] for r in ranks}
    for c in range(n_ch):
        cmaps, _, _ = fit_rrr(
            stats["szz_ch"][c], stats["szy_ch"][c], stats["syy_ch"][c], ranks, ridge
        )
        # Evaluate channel c only.
        se = {r: 0.0 for r in cmaps}
        pers = 0.0
        tot = 0
        for z, d in iter_batches(val_seg, seq_len, horizon, chunk):
            zz = z.reshape(-1, n_ch, seq_len)[:, c, :]
            dd = d.reshape(-1, n_ch, horizon)[:, c, :]
            pers += float((dd * dd).sum())
            tot += len(zz)
            for r, w in cmaps.items():
                resid = dd - zz @ w.T
                se[r] += float((resid * resid).sum())
        scale = max(tot * horizon, 1)
        base = pers / scale
        full = se[horizon] / scale
        denom = base - full
        if denom <= 0:
            continue
        for r in ranks:
            caps[r].append(100 * (base - se[r] / scale) / denom)
    return {r: (float(np.mean(v)) if v else float("nan")) for r, v in caps.items()}


def _write_outputs(out_dir: Path, rows: list[dict], summaries: list[dict]) -> None:
    """Write (or rewrite) both result tables; called after every setting."""
    if rows:
        with (out_dir / "optimal_rank_capture.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if summaries:
        with (out_dir / "optimal_rank_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        (out_dir / "optimal_rank_summary.json").write_text(
            json.dumps(summaries, indent=2)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument(
        "--settings",
        default=(
            "ETTh2:96,ETTh2:720,ETTm2:96,ETTm2:192,"
            "Weather:96,Weather:192,Electricity:336"
        ),
    )
    parser.add_argument("--data-root", default="./resources/all_datasets")
    parser.add_argument("--output-dir", default="research_runs/lowrank_data_property_v1")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument(
        "--save-moments",
        action="store_true",
        help="dump per-setting second moments so ranks can be re-fitted offline",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    seq_len = args.seq_len
    rows: list[dict] = []
    summaries: list[dict] = []

    for token in args.settings.split(","):
        dataset, horizon_s = token.split(":")
        horizon = int(horizon_s)
        train_seg, val_seg = load_split(dataset, seq_len, data_root)
        n_ch = train_seg.shape[1]
        stats = accumulate(train_seg, seq_len, horizon, args.chunk)
        ranks = TESTED_RANKS[horizon]
        maps, eigvals, _ = fit_rrr(
            stats["szz"], stats["szy"], stats["syy"], ranks, args.ridge
        )
        train_mse, n_train = eval_mse(
            train_seg, seq_len, horizon, args.chunk, maps
        )
        val_mse, n_val = eval_mse(val_seg, seq_len, horizon, args.chunk, maps)

        full = horizon
        base_val = val_mse["persistence"]
        denom = base_val - val_mse[full]
        for r in sorted(maps):
            w = maps[r]
            _, _, vwt = np.linalg.svd(w, full_matrices=False)
            basis = vwt[: min(r, vwt.shape[0]), :]
            used_share = float(np.trace((basis.T @ basis) @ stats["szz"]) / np.trace(stats["szz"]))
            cents, highs = zip(*[spectral_stats(v) for v in basis]) if len(basis) else ((0.0,), (0.0,))
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "rank": r,
                    "rank_frac_of_H": round(r / horizon, 6),
                    "params_vs_fullrank_pct": round(
                        100 * (seq_len * r + r * horizon) / (seq_len * horizon), 3
                    ),
                    "train_mse": round(train_mse[r], 8),
                    "val_mse": round(val_mse[r], 8),
                    "gain_vs_persistence_val_pct": round(
                        100 * (base_val - val_mse[r]) / base_val, 4
                    ),
                    "capture_val_pct": round(100 * (base_val - val_mse[r]) / denom, 4)
                    if denom > 0
                    else float("nan"),
                    "used_var_share": round(used_share, 6),
                    "rowspace_high_freq_frac": round(float(np.mean(highs)), 4),
                    "rowspace_centroid": round(float(np.mean(cents)), 4),
                    "lead_dir_cos_ones": direction_stats(basis[0], stats["szz"])["cos_ones"],
                    "lead_dir_cos_ramp": direction_stats(basis[0], stats["szz"])["cos_ramp"],
                    "lead_dir_cos_last24": direction_stats(basis[0], stats["szz"])["cos_last24"],
                    "lead_dir_var_share": direction_stats(basis[0], stats["szz"])["var_read_share"],
                }
            )

        svals = np.linalg.svd(maps[full], compute_uv=False)
        energy = np.cumsum(svals**2) / np.sum(svals**2)
        pred_energy = np.cumsum(eigvals) / np.sum(eigvals)
        summary = {
            "dataset": dataset,
            "horizon": horizon,
            "channels": n_ch,
            "n_train_windows": n_train,
            "n_val_windows": n_val,
            "persistence_val_mse_norm": round(base_val, 8),
            "fullrank_val_mse_norm": round(val_mse[full], 8),
            "fullrank_val_capture_pct": round(100 * denom / base_val, 4),
            "ols_map_rank90": int(np.searchsorted(energy, 0.90) + 1),
            "ols_map_rank95": int(np.searchsorted(energy, 0.95) + 1),
            "ols_map_rank99": int(np.searchsorted(energy, 0.99) + 1),
            "pred_dims_for_90pct": int(np.searchsorted(pred_energy, 0.90) + 1),
            "pred_dims_for_95pct": int(np.searchsorted(pred_energy, 0.95) + 1),
            "pred_dims_for_99pct": int(np.searchsorted(pred_energy, 0.99) + 1),
            "pred_spectrum_participation_ratio": round(
                float(eigvals.sum() ** 2 / np.sum(eigvals**2)), 2
            ),
            "pred_eigvals_top5": [round(float(x), 6) for x in eigvals[:5]],
            "capture_at_tested_ranks_pct": {
                str(r): round(100 * (base_val - val_mse[r]) / denom, 3) for r in ranks
            },
        }
        if args.save_moments:
            np.savez_compressed(
                out_dir / f"moments_{dataset}_h{horizon}.npz",
                szz=stats["szz"],
                szy=stats["szy"],
                syy=stats["syy"],
                persistence_mse=stats["persistence_mse"],
                train_seg_shape=np.array(train_seg.shape),
                val_seg_shape=np.array(val_seg.shape),
            )
        if args.per_channel and n_ch <= 32:
            pc = per_channel_capture(
                train_seg, val_seg, seq_len, horizon, ranks, args.chunk, args.ridge
            )
            summary["per_channel_capture_pct"] = {str(r): round(v, 2) for r, v in pc.items()}
        summaries.append(summary)
        _write_outputs(out_dir, rows, summaries)
        print(
            f"[done] {dataset}-{horizon}: persistence={base_val:.6f} "
            f"fullrank_val={val_mse[full]:.6f} "
            f"capture={json.dumps(summary['capture_at_tested_ranks_pct'])}",
            flush=True,
        )

    _write_outputs(out_dir, rows, summaries)
    print(f"[written] {out_dir}/optimal_rank_capture.csv and optimal_rank_summary.csv")


if __name__ == "__main__":
    main()
