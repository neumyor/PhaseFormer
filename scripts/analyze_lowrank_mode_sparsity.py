#!/usr/bin/env python3
"""Sections 12 and 13 of the low-rank checkpoint analysis plan.

Section 12 asks whether a single canonical mode still needs its full dense
kernel: the input direction ``v_i`` is a 720-step vector, and the mode may read
only a handful of lags or a few named semantics.  Section 13 asks whether the
modes themselves are stable across seeds, which decides whether naming one by
index is legitimate at all.

Both work from the exported canonical mode tensors, so nothing is retrained and
no checkpoint is touched.  Every sparsified mode is scored twice: by how well it
reproduces the original direction (R^2, atoms kept) and by what it costs the
fused forecast.  Because the mode output directions stay orthonormal under an
input-side sparsification, the fused MSE is again a sum of per-mode scalars, so
the performance column is exact rather than a re-fit.

Only the validation split is read.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lowrank_checkpoint_core import build_groups, input_templates, output_templates  # noqa: E402


def soft_threshold(vector: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(vector) * np.maximum(np.abs(vector) - threshold, 0.0)


def group_soft_threshold(vector: np.ndarray, blocks: list[slice], threshold: float) -> np.ndarray:
    out = np.zeros_like(vector)
    for block in blocks:
        segment = vector[block]
        norm = float(np.linalg.norm(segment))
        if norm > threshold:
            out[block] = segment * (1.0 - threshold / norm)
    return out


def fista(
    target: np.ndarray,
    prox,
    lipschitz: float,
    iterations: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    """Minimize ``0.5*||x - target||^2 + g(x)`` by accelerated proximal gradient."""
    x = target.copy()
    y = x.copy()
    t = 1.0
    step = 1.0 / lipschitz
    for _ in range(iterations):
        previous = x
        x = prox(y - step * (y - target), step)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x + ((t - 1.0) / t_next) * (x - previous)
        t = t_next
        if np.linalg.norm(x - previous) <= tol * max(1.0, np.linalg.norm(x)):
            break
    return x


_FACTORISATION_CACHE: dict = {}


def fused_lasso(target: np.ndarray, lambda_tv: float, iterations: int = 200, rho: float = 100.0) -> np.ndarray:
    """Prox of ``lambda_tv * ||D x||_1`` for the chain graph, by ADMM.

    Solves ``min_x 0.5||x - target||^2 + lambda_tv ||D x||_1`` with the split
    ``z = D x``.  The ``x``-update is a tridiagonal solve of ``I + rho D^T D``,
    which is precomputed once.  The dual projected-gradient alternative converges
    too slowly here: ``D D^T`` has spectral norm 4 on a path, so its step sits at
    the stability boundary and large penalties need thousands of iterations.
    """
    size = target.size
    if size < 2 or lambda_tv <= 0.0:
        return target.copy()
    difference = np.zeros((size - 1, size))
    rows = np.arange(size - 1)
    difference[rows, rows] = -1.0
    difference[rows, rows + 1] = 1.0
    solve = _FACTORISATION_CACHE.get((size, rho))
    if solve is None:
        system = np.eye(size) + rho * (difference.T @ difference)
        solve = np.linalg.solve(system, np.eye(size))
        _FACTORISATION_CACHE[(size, rho)] = solve

    x = target.copy()
    z = difference @ x
    dual = np.zeros(size - 1)
    for _ in range(iterations):
        right = target + rho * (difference.T @ (z - dual))
        x = solve @ right
        dx = difference @ x
        z = soft_threshold(dx + dual, lambda_tv / rho)
        dual = dual + dx - z
    return x


def lag_sparsifiers(seq_len: int, block_sizes: tuple[int, ...] = (24, 96)) -> dict:
    """Named sparsification operators for an input direction of length ``seq_len``."""
    operators = {}
    for fraction in (0.90, 0.95, 0.99):
        keep = max(1, int(round((1.0 - fraction) * seq_len)))

        def hard(vector: np.ndarray, keep=keep) -> np.ndarray:
            out = np.zeros_like(vector)
            index = np.argsort(-np.abs(vector), kind="stable")[:keep]
            out[index] = vector[index]
            return out

        operators[f"hard_{int(fraction * 100)}"] = hard
    for block in block_sizes:
        blocks = [slice(start, min(start + block, seq_len)) for start in range(0, seq_len, block)]

        def grouped(vector: np.ndarray, blocks=blocks) -> np.ndarray:
            best = None
            for threshold in np.linspace(0.0, float(np.linalg.norm(vector)), 200):
                candidate = group_soft_threshold(vector, blocks, float(threshold))
                residual = float(np.linalg.norm(vector - candidate))
                if best is None or residual < best[0]:
                    best = (residual, candidate)
            return best[1]

        operators[f"group_lasso_{block}"] = grouped
    # A monotone objective is needed to pick lambda, so TV is swept the same way.
    def tv(vector: np.ndarray, lambdas=(0.05, 0.2, 0.5)) -> np.ndarray:
        best = None
        for lam in lambdas:
            candidate = fused_lasso(vector, float(lam))
            residual = float(np.linalg.norm(vector - candidate))
            if best is None or residual < best[0]:
                best = (residual, candidate)
        return best[1]

    operators["fused_lasso_tv"] = tv
    return operators


def sparse_regression(dictionary: np.ndarray, target: np.ndarray, l1_ratios=(0.01, 0.05, 0.2, 0.5)):
    """Non-negative-free Lasso of ``target`` on ``dictionary`` by FISTA.

    ``dictionary`` has orthonormal columns, so the Lipschitz constant of the data
    term is 1 and no ``sklearn`` dependency is needed.
    """
    best = None
    for ratio in l1_ratios:
        lam = float(ratio)
        prox = lambda value, step, lam=lam: soft_threshold(value, step * lam)  # noqa: E731
        coefficients = fista(target, prox, lipschitz=1.0)
        residual = float(np.linalg.norm(target - dictionary @ coefficients))
        if best is None or residual < best[0]:
            best = (residual, coefficients)
    return best[1]


def match_modes(modes_a: dict, modes_b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal one-to-one matching of canonical modes between two seeds.

    The score is the plan's section 13.2 ``|v_i . v_j| * |u_i . u_j|``: a mode is
    the same direction only if *both* what it reads and what it writes agree, so
    an input-only or output-only match must not count.  Maximizing the total is a
    linear assignment problem.
    """
    u_a, vt_a = modes_a["u"], modes_a["vt"]
    u_b, vt_b = modes_b["u"], modes_b["vt"]
    input_cos = np.abs(vt_a @ vt_b.T)
    output_cos = np.abs(u_a.T @ u_b)
    score = input_cos * output_cos
    size = min(score.shape)
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415
        rows, columns = linear_sum_assignment(-score)
    except ImportError:  # pragma: no cover - scipy is present in the run env
        rows = np.arange(size)
        columns = np.argmax(score, axis=1)
    pairs = [(int(r), int(c)) for r, c in zip(rows, columns) if r < score.shape[0] and c < score.shape[1]]
    return (
        np.array([input_cos[r, c] for r, c in pairs]),
        np.array([output_cos[r, c] for r, c in pairs]),
        np.array(pairs),
    )


def seed_stability(
    modes_dir: Path, inventory_rows: list[dict], output_dir: Path,
) -> None:
    """Section 13: are the canonical modes stable across seeds?"""
    grouped: dict[tuple[str, str], dict[int, Path]] = {}
    for row in inventory_rows:
        key = (row["setting"], row["cell"])
        grouped.setdefault(key, {})[int(row["seed"])] = (
            modes_dir / f"{row['setting']}_seed{row['seed']}_{row['cell'].replace('/', '-')}.npz"
        )

    rows: list[dict] = []
    for (setting, cell), seeds in sorted(grouped.items()):
        available = sorted(seed for seed, path in seeds.items() if path.is_file())
        for first, second in zip(available, available[1:]):
            a = {k: v for k, v in np.load(seeds[first]).items()}
            b = {k: v for k, v in np.load(seeds[second]).items()}
            if a["u"].shape != b["u"].shape:
                continue
            in_cos, out_cos, pairs = match_modes(a, b)
            contribution_a = a["contribution"][pairs[:, 0]]
            contribution_b = b["contribution"][pairs[:, 1]]
            ranks_a = np.argsort(np.argsort(contribution_a)).astype(float)
            ranks_b = np.argsort(np.argsort(contribution_b)).astype(float)
            correlation = (
                float(np.corrcoef(ranks_a, ranks_b)[0, 1])
                if contribution_a.size > 2 and ranks_a.std() > 0 and ranks_b.std() > 0
                else float("nan")
            )
            rows.append({
                "setting": setting, "cell": cell, "rank": int(a["rank"]),
                "seed_a": first, "seed_b": second, "modes_matched": int(pairs.shape[0]),
                "input_cosine_mean": float(in_cos.mean()),
                "input_cosine_min": float(in_cos.min()),
                "output_cosine_mean": float(out_cos.mean()),
                "output_cosine_min": float(out_cos.min()),
                "fraction_matched_above_0p7": float(np.mean((in_cos > 0.7) & (out_cos > 0.7))),
                "matched_contribution_rank_correlation": correlation,
            })
        print(f"  {setting} {cell}: {len(available)} seeds", flush=True)

    path = output_dir / "seed_mode_stability.csv"
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="research_runs/lowrank_checkpoint_information_v1/features")
    parser.add_argument("--modes-dir", default="research_runs/lowrank_functional_rank_v1/modes")
    parser.add_argument("--inventory", default="research_runs/lowrank_checkpoint_information_v1/checkpoint_inventory.csv")
    parser.add_argument("--output-dir", default="research_runs/lowrank_functional_rank_v1")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--top-modes", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    features_dir = repo_root / args.features_dir
    modes_dir = repo_root / args.modes_dir
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = list(csv.DictReader((repo_root / args.inventory).open()))
    wanted_settings = {v for v in args.settings.split(",") if v}
    wanted_seeds = {int(v) for v in args.seeds.split(",") if v}
    rows = [
        row for row in inventory
        if row["is_diagnostic_only"] == "False"
        and row["dataset"] != "Electricity"
        and (not wanted_settings or row["setting"] in wanted_settings)
        and (not wanted_seeds or int(row["seed"]) in wanted_seeds)
    ]
    rows.sort(key=lambda row: (row["setting"], int(row["seed"]), row["cell"]))
    all_rows = list(rows)          # unsharded, for the cross-seed table
    rows = [r for index, r in enumerate(rows) if index % args.shard_count == args.shard_index]

    sparsity_rows: list[dict] = []
    for row in rows:
        setting, dataset = row["setting"], row["dataset"]
        horizon, seed = int(row["horizon"]), int(row["seed"])
        cell = row["cell"]
        mode_path = modes_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz"
        cache_path = features_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz"
        if not (mode_path.is_file() and cache_path.is_file()):
            print(f"  [skip] {setting} seed={seed} {cell}", flush=True)
            continue

        modes = np.load(mode_path)
        u, s, vt = modes["u"], modes["s"], modes["vt"]
        contribution = modes["contribution"]
        order = np.argsort(-contribution, kind="stable")

        cache = {k: v.astype(np.float64) for k, v in np.load(cache_path).items()}
        # The mode contribution is a scalar functional of the direction, so the
        # effect of sparsifying v_i is recomputed rather than approximated.
        ew, eb = cache["encoder_weight"], cache["encoder_bias"]
        dw, db = cache["decoder_weight"], cache["decoder_bias"]
        sigma, gate = cache["sigma"], cache["gate"]
        phase, target, z = cache["phase"], cache["target"], cache["z"]
        hidden = cache["hidden"]
        mapped_eb = dw @ eb
        x_last = cache["x_last_norm"] + float(mapped_eb.mean())
        last_abs = sigma * x_last + cache["mu"]
        c_const = mapped_eb + db
        branch_0 = last_abs + sigma * c_const[None, :, None]
        fused_0 = (1.0 - gate) * phase + gate * branch_0
        e0 = target - fused_0
        mse_0 = float(np.mean(e0 ** 2))
        gs2 = (gate * sigma)[:, 0, :]

        operators = lag_sparsifiers(z.shape[1])
        input_groups = build_groups(input_templates(z.shape[1], 96 if dataset == "ETTm2" else 24))
        output_groups = build_groups(output_templates(horizon, 96 if dataset == "ETTm2" else 24))
        input_dictionary = np.concatenate([g.basis for g in input_groups.values()], axis=1)
        input_dictionary, _ = np.linalg.qr(input_dictionary)
        output_dictionary = np.concatenate([g.basis for g in output_groups.values()], axis=1)
        output_dictionary, _ = np.linalg.qr(output_dictionary)

        for rank_position, index in enumerate(order[: args.top_modes], start=1):
            direction = vt[index]
            output_direction = u[:, index]
            variants: dict[str, np.ndarray] = {"dense": direction}
            for name, operator in operators.items():
                variants[name] = operator(direction)
            coefficients = sparse_regression(input_dictionary, direction)
            variants["semantic_lasso"] = input_dictionary @ coefficients

            for name, candidate in variants.items():
                norm = float(np.linalg.norm(candidate))
                r2 = 1.0 - float(np.linalg.norm(direction - candidate)) ** 2 / float(
                    np.linalg.norm(direction) ** 2
                )
                # Downstream cost: this mode alone, with its sparse input kernel,
                # against the fused forecast.
                if norm <= 1e-12:
                    delta_mse = contribution[index]
                else:
                    activation = np.einsum("l,nlc->nc", candidate, z)
                    scaled = s[index] * activation
                    contribution_sparse = (
                        2.0 * np.mean((gs2 * scaled) * np.einsum(
                            "nhc,h->nc", e0, output_direction))
                        - np.mean((gs2 * scaled) ** 2) / horizon
                    )
                    delta_mse = contribution[index] - contribution_sparse
                sparsity_rows.append({
                    "setting": setting, "dataset": dataset, "horizon": horizon,
                    "seed": seed, "cell": cell, "rank": int(modes["rank"]),
                    "mode_index": int(index), "contribution_rank": rank_position,
                    "variant": name,
                    "reconstruction_r2": r2,
                    "nonzero_lags": int(np.count_nonzero(candidate)),
                    "rare_atoms": int(np.count_nonzero(coefficients)) if name == "semantic_lasso" else "",
                    "fused_mse_increase": delta_mse,
                })
        print(f"  {setting} seed={seed} {cell}: {args.top_modes} modes x "
              f"{len(operators) + 2} variants", flush=True)
        del cache, modes

    path = output_dir / "mode_sparsity.csv"
    fieldnames = list(sparsity_rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sparsity_rows)
    print(f"wrote {len(sparsity_rows)} rows to {path}", flush=True)

    # Cross-seed mode matching is a whole-inventory question, so it is built
    # from every in-scope cell rather than from this shard's slice; each shard
    # would otherwise write a different subset of the same file.
    if args.shard_index == 0:
        seed_stability(modes_dir, all_rows, output_dir)


if __name__ == "__main__":
    main()
