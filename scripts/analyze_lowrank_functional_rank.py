#!/usr/bin/env python3
"""Functional rank, per-mode contribution and mode pruning of a trained low-rank
temporal head.

Implements the parts of ``low_rank_checkpoint_analysis_experiment_plan.md`` that
the earlier checkpoint-information analysis did not cover: section 6.4
(activation energy), section 8 (per-mode fused-MSE contribution), section 9
(functional rank), section 11 (mode pruning) and the mode-level half of
section 13.

Algebra
-------
With the branch's private centered input ``z``, its per-channel gate ``g`` and
the exact RevIN statistics ``(mu, sigma)``:

    branch_abs = last_abs + sigma * (W z + c),   c = decoder @ encoder_bias + decoder_bias
    fused      = (1 - g) * phase + g * branch_abs
    y_hat      = y_hat_0 + sum_i d_i,
    d_i        = g * sigma * s_i * u_i * (v_i^T z)

where ``W = u diag(s) v^T`` is the canonical SVD of the effective map and
``y_hat_0`` is the same model with ``W`` removed.  Because the left singular
vectors ``u_i`` are orthonormal in the horizon space and the gate is a
per-channel scalar, the cross terms vanish:

    mean(d_i * d_j) = 0  for i != j.

Every subset MSE therefore collapses to a sum of per-mode scalars:

    MSE(S) = MSE(y_hat_0) - sum_{i in S} I_i,     I_i = 2*mean(e_0 d_i) - mean(d_i^2)

so the plan's additivity identity holds by construction, and the cumulative
reconstruction curve, the pruning sweep and the random control band are all
exact scalar arithmetic rather than repeated tensor passes.  The identity is
still verified numerically per cell (``additivity_residual``).

Nothing is trained, no checkpoint is modified, and the test split is never read.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def scoped_write(rows: list[dict], path: Path, key: tuple[str, ...]) -> None:
    """Merge ``rows`` into ``path``, replacing only the keys this shard owns.

    Every shard writes the same tables, so a plain overwrite would keep only the
    last shard's rows.  The merge key is explicit per table because a cell has
    many modes and many curve points.

    The read-merge-write cycle is guarded by an exclusive ``flock``.  Without it
    two shards can both read the table before either writes, and the second
    write then drops the first shard's rows -- a race this pipeline has already
    lost rows to once, which is why the merge is not merely "write my own rows".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            existing: dict[tuple, dict] = {}
            if path.is_file():
                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        existing[tuple(row[column] for column in key)] = row
            for row in rows:
                existing[tuple(str(row[column]) for column in key)] = row
            fieldnames = (
                list(rows[0].keys()) if rows
                else (list(next(iter(existing.values())).keys()) if existing else [])
            )
            temporary = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing.values():
                    writer.writerow({column: row.get(column, "") for column in fieldnames})
            temporary.replace(path)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def canonical_modes(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``W = u diag(s) v^T`` with a deterministic sign convention.

    Singular-vector signs are arbitrary; fix them so a mode read from two seeds
    is comparable without a global flip (plan section 5.2).
    """
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    for index in range(u.shape[1]):
        total = float(u[:, index].sum())
        largest = int(np.argmax(np.abs(u[:, index])))
        flip = total < 0.0 if abs(total) > 1e-12 else u[largest, index] < 0.0
        if flip:
            u[:, index] *= -1.0
            vt[index, :] *= -1.0
    return u, s, vt


def analyse_cell(
    payload: dict,
    setting: str, dataset: str, horizon: int, seed: int, cell: str, rank_nominal: int,
    *, random_repeats: int, rng: np.random.Generator, top_modes: int,
):
    hidden = payload["hidden"]           # (N, C, r)
    z = payload["z"]                     # (N, L, C)
    sigma = payload["sigma"]             # (N, 1, C)
    mu = payload["mu"]                   # (N, 1, C)
    gate = payload["gate"]               # (N, 1, C)
    phase = payload["phase"]             # (N, H, C)
    target = payload["target"]           # (N, H, C)
    fused = payload["fused"]             # (N, H, C)
    x_last = payload["x_last_norm"]      # (N, 1, C)
    ew = payload["encoder_weight"]
    eb = payload["encoder_bias"]
    dw = payload["decoder_weight"]
    db = payload["decoder_bias"]

    n, channels, rank = hidden.shape
    assert phase.shape[1] == horizon, f"horizon {phase.shape[1]} != {horizon}"

    W = dw @ ew
    mapped_eb = dw @ eb
    c_const = mapped_eb + db

    # The cache holds the horizon-mean of the anchor deflated by the mapped
    # encoder bias; add it back to recover the model's own anchor.  Verified
    # against forward passes at ~1e-6 (scripts/_probe_cache_algebra.py).
    x_last_true = x_last + float(mapped_eb.mean())
    last_abs = sigma * x_last_true + mu

    u, s, vt = canonical_modes(W)
    # ``W = decoder @ encoder`` has rank at most ``rank`` (the encoder's output
    # width), so ``svd`` returns ``min(H, L)`` values of which only the leading
    # ``rank`` are structural; the rest are numerical zeros and must be dropped
    # or the mode tables would claim far more modes than the head can express.
    structural_rank = int(min(rank, s.size))
    u = np.ascontiguousarray(u[:, :structural_rank])
    s = s[:structural_rank]
    vt = np.ascontiguousarray(vt[:structural_rank, :])
    numerical_rank = int(np.sum(s > s.max() * 1e-10)) if s.size else 0
    rank = structural_rank          # every downstream loop is over real modes
    A = np.einsum("rl,nlc->ncr", vt, z, optimize=True)   # (N, C, r) mode activations

    branch_0 = last_abs + sigma * c_const[None, :, None]
    w_z = np.einsum("ncr,hr->nhc", hidden, dw, optimize=True) - mapped_eb[None, :, None]
    branch_w = branch_0 + sigma * w_z
    fused_0 = (1.0 - gate) * phase + gate * branch_0
    fused_w = (1.0 - gate) * phase + gate * branch_w

    e0 = target - fused_0
    b0 = branch_0 - target
    mse_0 = float(np.mean(e0 ** 2))
    mse_w = float(np.mean((target - fused_w) ** 2))
    mse_recorded = float(np.mean((target - fused) ** 2))
    reconstruction_gap = float(np.abs(fused_w - fused).max())
    bmse_0 = float(np.mean(b0 ** 2))
    bmse_w = float(np.mean((branch_w - target) ** 2))

    # Per-mode scalars.  ``d_i[n,h,c] = Pf[n,c,i] * s_i * u_i[h]`` factorises over
    # ``(n,c)`` and ``h``, and ``u`` has orthonormal columns, so both moments
    # reduce to (N, C, r) contractions instead of one (N, H, C) tensor per mode.
    # The direct broadcast form is also catastrophically slow on this platform
    # (an (N,H,C) times (N,1,1) strided view took ~20 s for 1.9M elements), so
    # this is a correctness-neutral speed fix as well.
    E_fused = np.tensordot(e0, u, axes=([1], [0]))       # (N, C, r)
    E_branch = np.tensordot(b0, u, axes=([1], [0]))
    # The fused contribution carries the gate as well as the RevIN scale;
    # the branch contribution carries only the scale.
    gs2 = (gate * sigma)[:, 0, :]                        # (N, C)
    sig2 = sigma[:, 0, :]
    P_fused = gs2[:, :, None] * A
    P_branch = sig2[:, :, None] * A
    T_fused = np.mean(P_fused * E_fused, axis=(0, 1))
    U_fused = np.mean(P_fused ** 2, axis=(0, 1))
    T_branch = np.mean(P_branch * E_branch, axis=(0, 1))
    U_branch = np.mean(P_branch ** 2, axis=(0, 1))
    # Both moments carry the 1/H from the horizon average: ``sum_h`` is inside
    # the (n, c) reduction, so the mean over (n, h, c) is ``mean_{n,c} / H``.
    I = (2.0 * s * T_fused - (s ** 2) * U_fused) / horizon
    I_branch = (2.0 * s * T_branch - (s ** 2) * U_branch) / horizon

    act_sq = np.mean(A ** 2, axis=(0, 1))
    act_mean = np.mean(A, axis=(0, 1))
    act_std = np.std(A, axis=(0, 1))
    del A, P_fused, P_branch, E_fused, E_branch

    additivity_residual = float(abs(I.sum() - (mse_0 - mse_w)))
    branch_additivity_residual = float(abs(I_branch.sum() - (bmse_0 - bmse_w)))

    weight_energy = s ** 2
    activation_energy = s ** 2 * act_sq
    weight_share = weight_energy / weight_energy.sum()
    activation_share = activation_energy / activation_energy.sum()
    contribution_share = I / I.sum() if abs(I.sum()) > 1e-15 else np.zeros(rank)

    # Any subset's MSE is mse_0 minus the sum of its members' I_i.
    def mse_of_subset(members) -> float:
        return mse_0 - float(I[list(members)].sum())

    orderings = {
        "singular": np.argsort(-s, kind="stable"),
        "weight_energy": np.argsort(-weight_energy, kind="stable"),
        "activation_energy": np.argsort(-activation_energy, kind="stable"),
        "contribution": np.argsort(-I, kind="stable"),
    }

    curve_rows: list[dict] = []
    functional_rank: dict[str, int] = {}
    denominator = mse_0 - mse_w
    for name, order in orderings.items():
        mse_curve = np.array([mse_of_subset(order[:k]) for k in range(1, rank + 1)])
        recovery = (
            (mse_0 - mse_curve) / denominator if abs(denominator) > 1e-15
            else np.zeros(rank)
        )
        for k in range(1, rank + 1):
            curve_rows.append({
                "setting": setting, "dataset": dataset, "horizon": horizon,
                "seed": seed, "cell": cell, "rank": rank_nominal,
                "ordering": name, "k": k,
                "mse": mse_curve[k - 1], "recovery": recovery[k - 1],
            })
        for level, label in ((0.90, "r90"), (0.95, "r95"), (0.99, "r99")):
            reached = np.nonzero(recovery >= level - 1e-12)[0]
            functional_rank[f"{label}_{name}"] = int(reached[0] + 1) if reached.size else rank

    # --- zero-shot pruning -------------------------------------------------
    pruning_rows: list[dict] = []
    fractions = (0.10, 0.25, 0.50)
    for name, order in orderings.items():
        for fraction in fractions:
            drop = int(round(fraction * rank))
            if drop <= 0 or drop >= rank:
                continue
            kept = order[: rank - drop]
            subset = set(kept.tolist())
            pruning_rows.append({
                "setting": setting, "dataset": dataset, "horizon": horizon,
                "seed": seed, "cell": cell, "rank": rank_nominal,
                "criterion": name, "fraction": fraction, "dropped": drop,
                "kept_fused_mse": mse_of_subset(sorted(subset)),
                "delta_fused_mse_vs_full": mse_of_subset(sorted(subset)) - mse_w,
                "random_repeats": 0,
            })
    negative = np.nonzero(I < 0)[0]
    if negative.size:
        pruning_rows.append({
            "setting": setting, "dataset": dataset, "horizon": horizon,
            "seed": seed, "cell": cell, "rank": rank_nominal,
            "criterion": "negative_contribution", "fraction": "",
            "dropped": int(negative.size),
            "kept_fused_mse": mse_of_subset(sorted(set(range(rank)) - set(negative.tolist()))),
            "delta_fused_mse_vs_full": mse_of_subset(
                sorted(set(range(rank)) - set(negative.tolist()))) - mse_w,
            "random_repeats": 0,
        })
    for fraction in fractions:
        drop = int(round(fraction * rank))
        if drop <= 0 or drop >= rank:
            continue
        values = np.empty(random_repeats)
        for repeat in range(random_repeats):
            kept = rng.choice(rank, size=rank - drop, replace=False)
            values[repeat] = mse_0 - I[kept].sum()
        pruning_rows.append({
            "setting": setting, "dataset": dataset, "horizon": horizon,
            "seed": seed, "cell": cell, "rank": rank_nominal,
            "criterion": "random", "fraction": fraction, "dropped": drop,
            "kept_fused_mse": float(values.mean()),
            "delta_fused_mse_vs_full": float(values.mean()) - mse_w,
            "random_repeats": random_repeats,
            "random_std": float(values.std()),
            "random_p05": float(np.percentile(values, 5)),
            "random_p95": float(np.percentile(values, 95)),
        })

    order_by_contribution = np.argsort(-I, kind="stable")
    position_of = np.empty(rank, dtype=int)
    for position, index in enumerate(order_by_contribution, start=1):
        position_of[index] = position
    mode_rows: list[dict] = []
    for i in range(min(top_modes, rank)):
        mode_rows.append({
            "setting": setting, "dataset": dataset, "horizon": horizon,
            "seed": seed, "cell": cell, "rank": rank_nominal, "mode_index": i,
            "singular_value": s[i],
            "weight_energy_share": weight_share[i],
            "activation_energy": activation_energy[i],
            "activation_energy_share": activation_share[i],
            "mean_activation_squared": act_sq[i],
            "activation_mean": act_mean[i],
            "activation_std": act_std[i],
            "fused_mse_contribution": I[i],
            "fused_contribution_share": contribution_share[i],
            "branch_mse_contribution": I_branch[i],
            "contribution_rank_position": int(position_of[i]),
        })

    cell_row = {
        "setting": setting, "dataset": dataset, "horizon": horizon,
        "seed": seed, "cell": cell, "rank": rank_nominal,
        "numerical_rank": numerical_rank,
        "samples": n, "channels": channels,
        "mse_base_w0": mse_0, "mse_full_lowrank": mse_w,
        "mse_recorded_fused": mse_recorded,
        "reconstruction_gap_max_abs": reconstruction_gap,
        "additivity_residual": additivity_residual,
        "branch_additivity_residual": branch_additivity_residual,
        "branch_mse_full": bmse_w, "branch_mse_w0": bmse_0,
        "phase_only_mse": float(np.mean((phase - target) ** 2)),
        "total_improvement": mse_0 - mse_w,
        "sum_positive_contribution": float(I[I > 0].sum()),
        "sum_negative_contribution": float(I[I < 0].sum()),
        "n_negative_contribution": int(np.sum(I < 0)),
        "contribution_share_of_top1": float(I[order_by_contribution[:1]].sum() / I.sum()),
        "contribution_share_of_top5": float(I[order_by_contribution[:5]].sum() / I.sum()),
        "singular_share_of_top1": float(weight_share[0]),
    }
    cell_row.update(functional_rank)
    return cell_row, mode_rows, curve_rows, pruning_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="research_runs/lowrank_checkpoint_information_v1/features")
    parser.add_argument("--inventory", default="research_runs/lowrank_checkpoint_information_v1/checkpoint_inventory.csv")
    parser.add_argument("--output-dir", default="research_runs/lowrank_functional_rank_v1")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--cells", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--top-modes", type=int, default=16)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20240924)
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    if not (repo_root / "src").is_dir():
        repo_root = Path.cwd().resolve()

    inventory = list(csv.DictReader((repo_root / args.inventory).open()))
    wanted_settings = {value for value in args.settings.split(",") if value}
    wanted_seeds = {int(value) for value in args.seeds.split(",") if value}
    wanted_cells = {value for value in args.cells.split(",") if value}
    rows = [
        row for row in inventory
        if row["is_diagnostic_only"] in ("False", "0")
        and (not wanted_settings or row["setting"] in wanted_settings)
        and (not wanted_seeds or int(row["seed"]) in wanted_seeds)
        and (not wanted_cells or row["cell"] in wanted_cells)
    ]
    rows.sort(key=lambda row: (row["setting"], int(row["seed"]), row["cell"]))
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    print(f"shard {args.shard_index}/{args.shard_count}: {len(rows)} cells", flush=True)

    features_dir = repo_root / args.features_dir
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + args.shard_index)

    for position, row in enumerate(rows, start=1):
        setting, dataset = row["setting"], row["dataset"]
        horizon, seed = int(row["horizon"]), int(row["seed"])
        cell, rank = row["cell"], int(row["rank"])
        cache = features_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz"
        if not cache.is_file():
            print(f"  [skip] missing cache {cache.name}", flush=True)
            continue
        started = time.time()
        payload = {
            key: value.astype(np.float64) for key, value in np.load(cache).items()
        }
        cell_row, mode_rows, curve_rows, pruning_rows = analyse_cell(
            payload, setting, dataset, horizon, seed, cell, rank,
            random_repeats=args.random_repeats, rng=rng, top_modes=args.top_modes,
        )
        scoped_write([cell_row], output_dir / "functional_rank_cells.csv",
                     ("setting", "seed", "cell"))
        scoped_write(mode_rows, output_dir / "mode_contributions.csv",
                     ("setting", "seed", "cell", "mode_index"))
        scoped_write(curve_rows, output_dir / "functional_rank_curves.csv",
                     ("setting", "seed", "cell", "ordering", "k"))
        scoped_write(pruning_rows, output_dir / "mode_pruning.csv",
                     ("setting", "seed", "cell", "criterion", "fraction"))
        print(
            f"  [{position}/{len(rows)}] {setting} seed={seed} {cell} r={rank} "
            f"nrank={cell_row['numerical_rank']} "
            f"add={cell_row['additivity_residual']:.2e} "
            f"gap={cell_row['reconstruction_gap_max_abs']:.2e} "
            f"neg={cell_row['n_negative_contribution']} "
            f"r95={cell_row['r95_contribution']}/{rank} "
            f"({time.time() - started:.1f}s)",
            flush=True,
        )
        del payload

    print(f"done shard {args.shard_index}", flush=True)


if __name__ == "__main__":
    main()
