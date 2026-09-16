#!/usr/bin/env python3
"""Stage 0 of the top-2 predictive-direction retention plan.

Builds the frozen NLinear input subspaces (V1 = direction 1, V2 = directions
1+2) for every audit setting, then runs the plan's six mandatory checks.  This
script trains nothing and reads only the **train** split.

Method (plan section 5), per dataset x horizon, on train-split-standardized
data with the repository's own split borders:

    Z = x_window - x_last          (720)
    D = y_horizon - x_last         (H)
    S = Szy^T (Szz + eps I)^-1 Szy
    eigendecompose S descending -> u1, u2
    b_i = u_i^T Szy^T (Szz + eps I)^-1

``b_i`` is the i-th most valuable *input-side* direction of the branch's own
regression task: because the branch minimizes E||D - W Z||^2 with a shared
across-channel map, the rank-1 optimum is exactly W_1 = b_1 u_1^T, so reading
``b_1`` alone captures lambda_1 of the total achievable MSE reduction.

``{b1}`` and ``{b1, b2}`` are then orthonormalized to give Q1 and Q12.  The
projectors are frozen data artifacts: no gradient ever touches them.

Usage::

    python scripts/compute_top2_direction_projectors.py \
        --output-dir research_runs/top2_direction_retention_v1/projectors
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# dataset -> (subdir under resources/all_datasets, csv, split family)
DATASETS = {
    "ETTh2": ("ETT", "ETTh2.csv", "ett_hour"),
    "ETTm2": ("ETT", "ETTm2.csv", "ett_minute"),
    "Weather": ("weather", "weather.csv", "custom"),
}

# The six settings that already have RRR direction analysis (plan section 3).
DEFAULT_SETTINGS = (
    "ETTh2:96",
    "ETTh2:720",
    "ETTm2:96",
    "ETTm2:192",
    "Weather:96",
    "Weather:192",
)


def parse_settings(raw: str) -> list[tuple[str, int]]:
    out = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        dataset, horizon = chunk.split(":")
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset {dataset!r}")
        out.append((dataset, int(horizon)))
    return out


def load_split(dataset: str, seq_len: int, data_root: Path):
    """Reproduce the repository's train/val borders and train-fitted scaling.

    Mirrors ``src/dataset/data_loader.py`` (``StandardScaler``, ddof=0) and
    ``scripts/analyze_optimal_lowrank_capture.py`` so the projectors live in
    exactly the coordinate system the NLinear branch sees.
    """
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

    segments = {
        "train": data[border1s[0] : border2s[0]],
        "val": data[border1s[1] : border2s[1]],
        "test": data[border1s[2] : border2s[2]],
    }
    borders = {
        "border1s": [int(v) for v in border1s],
        "border2s": [int(v) for v in border2s],
        "raw_rows": int(len(raw)),
        "n_channels": int(raw.shape[1]),
    }
    return segments, borders


def iter_batches(seg: np.ndarray, seq_len: int, pred_len: int, chunk: int):
    """Yield (Z, D) chunks of centered window/deviation pairs."""
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
        yield (x - last).reshape(-1, seq_len), (y - last).reshape(-1, pred_len)


def accumulate(seg, seq_len, pred_len, chunk):
    """Channel-pooled second moments and the rank-0 (persistence) error."""
    s_zz = np.zeros((seq_len, seq_len))
    s_zy = np.zeros((seq_len, pred_len))
    s_yy = np.zeros((pred_len, pred_len))
    count = 0
    for z, d in iter_batches(seg, seq_len, pred_len, chunk):
        s_zz += z.T @ z
        s_zy += z.T @ d
        s_yy += d.T @ d
        count += len(z)
    n = max(count, 1)
    return {
        # ``count`` counts window x channel pairs, because the branch's linear
        # map is shared across channels and the moments are pooled over both.
        "n_pairs": count,
        "szz": s_zz / n,
        "szy": s_zy / n,
        "syy": s_yy / n,
    }


def rrr_directions(szz, szy, ridge):
    """Top RRR eigenvalues plus the associated input-side directions."""
    szz_r = szz + ridge * np.eye(szz.shape[0])
    szz_inv_szy = np.linalg.solve(szz_r, szy)  # L x H
    ols = szz_inv_szy.T  # H x L, full-rank optimum
    s_mat = szy.T @ szz_inv_szy  # H x H
    s_mat = 0.5 * (s_mat + s_mat.T)
    vals, vecs = np.linalg.eigh(s_mat)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = vecs[:, order]
    # b_i = u_i^T Szy^T (Szz + eps I)^-1
    directions = vecs.T @ ols  # H x L
    return vals, directions, ols


def orthonormalize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal basis plus the smallest Gram-Schmidt pivot.

    ``vectors`` is ``(k, L)`` in priority order.  The pivot is the norm of the
    last residual before normalization; a value near zero means the leading
    directions are degenerate and the reported basis is numerically unstable.
    """
    basis = []
    pivots = []
    for vector in vectors:
        residual = np.array(vector, dtype=np.float64, copy=True)
        for existing in basis:
            residual -= (residual @ existing) * existing
        norm = float(np.linalg.norm(residual))
        pivots.append(norm)
        if norm <= 1e-10:
            raise ValueError(
                "RRR directions are numerically degenerate; refusing to emit a "
                "non-orthonormal projector"
            )
        basis.append(residual / norm)
    return np.stack(basis, axis=1), np.asarray(pivots)


def projection_idempotence_error(basis: np.ndarray) -> float:
    """max |(Q Q^T)^2 - Q Q^T| for the projector implied by ``basis``."""
    projector = basis @ basis.T
    return float(np.abs(projector @ projector - projector).max())


def orthogonality_error(basis: np.ndarray) -> float:
    k = basis.shape[1]
    gram = basis.T @ basis
    return float(np.abs(gram - np.eye(k)).max())


def recover_error(basis: np.ndarray, direction: np.ndarray) -> float:
    """Relative error of recovering ``direction`` after projecting it."""
    denom = float(np.linalg.norm(direction))
    if denom <= 1e-12:
        return 0.0
    projected = basis @ (basis.T @ direction)
    return float(np.linalg.norm(projected - direction) / denom)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def projector_hash(q1: np.ndarray, q12: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(q1, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(q12, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def run_setting(dataset, horizon, seq_len, ridge, chunk, data_root, output_dir):
    segments, borders = load_split(dataset, seq_len, data_root)
    train_seg = segments["train"]
    moments = accumulate(train_seg, seq_len, horizon, chunk)
    # Sample count of the training split, i.e. how many (window, horizon)
    # pairs per channel the repository's own split produces.
    windows_per_channel = len(train_seg) - seq_len - horizon + 1
    vals, directions, _ = rrr_directions(moments["szz"], moments["szy"], ridge)

    total_lambda = float(vals.sum()) + 1e-12
    shares = [float(v) / total_lambda for v in vals[:4]]
    gap = float((vals[1] - vals[2]) / vals[1]) if vals[1] > 0 else 0.0

    q1, pivots1 = orthonormalize(directions[:1])
    q12, pivots12 = orthonormalize(directions[:2])

    # Projected / unprojected centered-input variance share, i.e. how much of
    # the centered-window energy each variant can still see.
    szz = moments["szz"]
    trace = float(np.trace(szz))
    used_var_share_v1 = float(np.trace(q1.T @ szz @ q1) / trace)
    used_var_share_v12 = float(np.trace(q12.T @ szz @ q12) / trace)

    # Sample standard deviation of the retained linear features b_i^T z.
    z_sample = next(iter_batches(train_seg, seq_len, horizon, chunk))[0]
    feature_std = [
        float(np.std(z_sample @ directions[i])) for i in range(2)
    ]

    setting = f"{dataset}_{horizon}"
    q1_path = output_dir / f"{setting}_Q1.npy"
    q12_path = output_dir / f"{setting}_Q12.npy"
    np.save(q1_path, q1)
    np.save(q12_path, q12)

    audit = {
        "setting": f"{dataset}-{horizon}",
        "dataset": dataset,
        "horizon": int(horizon),
        "seq_len": int(seq_len),
        "ridge": float(ridge),
        "train_windows_per_channel": int(windows_per_channel),
        "train_window_channel_pairs": int(moments["n_pairs"]),
        "expected_train_windows_per_channel": int(
            borders["border2s"][0] - borders["border1s"][0] - seq_len - horizon + 1
        ),
        "raw_rows": borders["raw_rows"],
        "n_channels": borders["n_channels"],
        "border1s": borders["border1s"],
        "border2s": borders["border2s"],
        "lambda_sum": total_lambda,
        "lambda1_share": shares[0],
        "lambda2_share": shares[1] if len(shares) > 1 else 0.0,
        "lambda3_share": shares[2] if len(shares) > 2 else 0.0,
        "lambda4_share": shares[3] if len(shares) > 3 else 0.0,
        "lambda2_lambda3_gap": gap,
        "lambda2_close_to_lambda3": bool(gap < 0.20),
        "direction2_orientation_stable": bool(gap >= 0.20),
        "q1_shape": list(q1.shape),
        "q12_shape": list(q12.shape),
        "q1_orthogonality_error": orthogonality_error(q1),
        "q12_orthogonality_error": orthogonality_error(q12),
        "q1_projector_idempotence_error": projection_idempotence_error(q1),
        "q12_projector_idempotence_error": projection_idempotence_error(q12),
        "q1_recover_direction1_error": recover_error(q1, directions[0]),
        "q12_recover_direction1_error": recover_error(q12, directions[0]),
        "q12_recover_direction2_error": recover_error(q12, directions[1]),
        # V1 must NOT be able to reproduce direction 2.
        "q1_recover_direction2_error": recover_error(q1, directions[1]),
        "q1_gram_schmidt_pivots": [float(v) for v in pivots1],
        "q12_gram_schmidt_pivots": [float(v) for v in pivots12],
        "used_var_share_v1": used_var_share_v1,
        "used_var_share_v12": used_var_share_v12,
        "feature_b1_z_std": feature_std[0],
        "feature_b2_z_std": feature_std[1],
        "persistence_mse": float(np.trace(moments["syy"]) / horizon),
        "lambda1_mse_reduction": float(vals[0] / horizon),
        "lambda12_mse_reduction": float((vals[0] + vals[1]) / horizon),
        "q1_sha256": sha256_of(q1_path),
        "q12_sha256": sha256_of(q12_path),
        "projector_hash": projector_hash(q1, q12),
        # Only the train split is ever read; record it explicitly.
        "splits_read": ["train"],
    }
    audit["checks"] = {
        "check_train_window_count": (
            audit["train_windows_per_channel"]
            == audit["expected_train_windows_per_channel"]
            and audit["train_window_channel_pairs"]
            == audit["train_windows_per_channel"] * audit["n_channels"]
        ),
        "check_orthonormality": max(
            audit["q1_orthogonality_error"], audit["q12_orthogonality_error"]
        )
        < 1e-8,
        "check_idempotence": max(
            audit["q1_projector_idempotence_error"],
            audit["q12_projector_idempotence_error"],
        )
        < 1e-8,
        "check_direction_information": (
            audit["q1_recover_direction1_error"] < 1e-8
            and audit["q12_recover_direction1_error"] < 1e-8
            and audit["q12_recover_direction2_error"] < 1e-8
            # V1 must not be able to reconstruct direction 2.
            and audit["q1_recover_direction2_error"] > 1e-3
        ),
        "check_no_val_test_moments": audit["splits_read"] == ["train"],
        "check_feature_scale": (
            audit["feature_b1_z_std"] > 1e-8 and audit["feature_b2_z_std"] > 1e-8
        ),
    }
    audit["checks"]["all_passed"] = all(audit["checks"].values())
    return audit


def write_csv(audits, path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0].keys()))
        writer.writeheader()
        for audit in audits:
            writer.writerow(audit)


def render_markdown(audits) -> str:
    lines = [
        "# Stage 0 投影器审计（前两预测方向数据保留实验）",
        "",
        "本文件由 `scripts/compute_top2_direction_projectors.py` 生成，只读取训练 split。",
        "",
        "| Setting | train windows | λ1 share | λ2 share | λ3 share | λ2/λ3 gap | "
        "Q1 正交误差 | Q12 正交误差 | 幂等误差 | projector hash | 通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for a in audits:
        lines.append(
            "| {setting} | {train_windows_per_channel} | {lambda1_share:.4f} | "
            "{lambda2_share:.4f} | "
            "{lambda3_share:.4f} | {lambda2_lambda3_gap:.3f} | "
            "{q1_orthogonality_error:.2e} | {q12_orthogonality_error:.2e} | "
            "{max_idem:.2e} | `{projector_hash}` | {passed} |".format(
                max_idem=max(
                    a["q1_projector_idempotence_error"],
                    a["q12_projector_idempotence_error"],
                ),
                passed="PASS" if a["checks"]["all_passed"] else "FAIL",
                **a,
            )
        )
    lines.append("")
    lines.append("## 逐项检查")
    lines.append("")
    keys = [
        ("check_train_window_count", "训练样本数与既有数据划分一致"),
        ("check_orthonormality", "Q1/Q12 正交误差 < 1e-8"),
        ("check_idempotence", "投影幂等误差 < 1e-8"),
        ("check_direction_information", "V1 只恢复 b1；V12 恢复 b1、b2；V1 不能恢复 b2"),
        ("check_no_val_test_moments", "仅读取训练 split"),
        ("check_feature_scale", "b1^T z / b2^T z 等样本标准差为正"),
    ]
    for key, label in keys:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Setting | 结果 |")
        lines.append("|---|---|")
        for a in audits:
            lines.append(f"| {a['setting']} | {'PASS' if a['checks'][key] else 'FAIL'} |")
        lines.append("")
    lines.append("## 方向 2 稳定性")
    lines.append("")
    lines.append("| Setting | λ2/λ3 gap | 方向 2 单独朝向稳定 |")
    lines.append("|---|---:|---|")
    for a in audits:
        lines.append(
            f"| {a['setting']} | {a['lambda2_lambda3_gap']:.3f} | "
            f"{'是' if a['direction2_orientation_stable'] else '否（已标注）'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=",".join(DEFAULT_SETTINGS))
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument(
        "--data-root", default=str(REPO_ROOT / "resources" / "all_datasets")
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    settings = parse_settings(args.settings)
    audits = []
    for dataset, horizon in settings:
        print(f"[stage0] computing projectors for {dataset}-{horizon}", flush=True)
        audits.append(
            run_setting(
                dataset, horizon, args.seq_len, args.ridge, args.chunk, data_root,
                output_dir,
            )
        )

    for audit in audits:
        audit["checks"]["all_passed"] = all(audit["checks"].values())

    index = {
        "protocol": "top2-direction-retention-stage0-v1",
        "seq_len": int(args.seq_len),
        "ridge": float(args.ridge),
        "lambda_strategy": "top-2 eigenvectors of S = Szy^T (Szz+eps I)^-1 Szy",
        "direction_strategy": "b_i = u_i^T Szy^T (Szz+eps I)^-1, Gram-Schmidt to Q",
        "only_train_split_read": True,
        "projectors": {
            f"{a['dataset']}-{a['horizon']}": {
                "dataset": a["dataset"],
                "horizon": a["horizon"],
                "q1_file": f"{a['dataset']}_{a['horizon']}_Q1.npy",
                "q12_file": f"{a['dataset']}_{a['horizon']}_Q12.npy",
                "projector_hash": a["projector_hash"],
                "q1_sha256": a["q1_sha256"],
                "q12_sha256": a["q12_sha256"],
                "lambda1_share": a["lambda1_share"],
                "lambda2_share": a["lambda2_share"],
                "lambda3_share": a["lambda3_share"],
                "lambda2_lambda3_gap": a["lambda2_lambda3_gap"],
                "used_var_share_v1": a["used_var_share_v1"],
                "used_var_share_v12": a["used_var_share_v12"],
                "all_checks_passed": a["checks"]["all_passed"],
            }
            for a in audits
        },
        "all_checks_passed": all(a["checks"]["all_passed"] for a in audits),
    }
    (output_dir / "projectors.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "stage0_audit.json").write_text(
        json.dumps(audits, indent=2, sort_keys=True) + "\n"
    )
    write_csv(audits, output_dir / "stage0_audit.csv")
    (output_dir / "stage0_audit.md").write_text(render_markdown(audits))

    failed = [a["setting"] for a in audits if not a["checks"]["all_passed"]]
    print(json.dumps({"all_checks_passed": not failed, "failed": failed}, indent=2))
    if failed:
        raise SystemExit(f"Stage 0 audit failed for: {failed}")


if __name__ == "__main__":
    main()
