#!/usr/bin/env python3
"""Stage 4 (plus the Stage 0 audit and feature cache) of the low-rank checkpoint
information analysis.

For every formal checkpoint this script

1. builds a fresh model, loads the frozen checkpoint and audits it:
   * **effective-map equivalence** ``decoder(encoder(pool(z))) == M z + c`` on a
     fixed validation batch must hold to ``1e-6``;
   * the intervention path must be numerically identical to the untouched head
     when no intervention is requested;
   * ``pool_factor``, ``smooth_ratio``, rank and gate are read out of the frozen
     parameters instead of being assumed;
2. caches the per-sample validation quantities the offline Stages 1-2 need; and
3. evaluates every intervention arm of plan section 5.

All interventions touch only the residual branch's private centered input.  The
Phase backbone always reads the original full input, the synthetic bias ``c`` is
kept except in the dedicated ``Bias-off`` arm, nothing is trained, no checkpoint
is modified and the test split is never read.

Because the low-rank branch is affine, every arm is evaluated exactly in closed
form: with ``r(h) = W h + b`` and a subspace basis ``Q`` (``P = Q Q^T``),

    r(P h) = W Q Q^T h + b
    r(h - P h) = r(h) - (W Q Q^T h)

so ``h`` is cached once per checkpoint and all arms, including the 100 random
controls, are replayed without any further forward pass.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lowrank_checkpoint_core import (  # noqa: E402
    adaptive_avg_pool_operator,
    build_groups,
    effective_map,
    input_templates,
    orthonormalize,
)
from scripts.lowrank_checkpoint_inventory import inventory_rows  # noqa: E402
from scripts.lowrank_checkpoint_model import (  # noqa: E402
    DATASET_PERIOD_STEPS,
    build_loaders,
    build_model,
    instrument_model,
    set_seed,
)

RANDOM_SEED = 20261116
RANDOM_QUANTILES = (2.5, 97.5)


def random_orthogonal_basis(
    dimension: int, rank: int, rng: np.random.Generator
) -> np.ndarray:
    gaussian = rng.standard_normal((dimension, rank))
    basis, _ = np.linalg.qr(gaussian)
    return np.ascontiguousarray(basis[:, :rank])


def semantic_basis(dataset: str, limit: int) -> np.ndarray:
    """Union of the input semantic groups, optionally truncated.

    The Semi/8 arm uses the leading eight directions of this union; the
    Semantic-only and Semantic-drop arms use the full union (dimension 22-28 per
    setting).  Both are data-free templates, so they cannot leak validation
    information.
    """
    groups = build_groups(input_templates(720, DATASET_PERIOD_STEPS[dataset]))
    basis = orthonormalize(
        np.concatenate([group.basis for group in groups.values()], axis=1).T
    )
    return np.ascontiguousarray(basis[:, :limit])


def pca_basis(moments_path: Path, rank: int) -> np.ndarray:
    payload = np.load(moments_path)
    covariance = payload["covariance"].astype(np.float64)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1]
    return np.ascontiguousarray(vectors[:, order[:rank]])


def arm_metrics(
    hidden: np.ndarray,
    decoder_weight: np.ndarray,
    decoder_bias: np.ndarray,
    gate: np.ndarray,
    phase_abs: np.ndarray,
    target: np.ndarray,
    correction_reference: np.ndarray,
    last_abs: np.ndarray,
    basis: np.ndarray | None,
    mode: str,
) -> dict:
    """Branch, fused and reconstruction metrics of one arm."""
    if mode == "bias":
        delta_hidden = np.einsum(
            "ncr,hr->nhc", hidden, decoder_weight
        )
    elif mode == "identity":
        delta_hidden = np.einsum("ncr,hr->nhc", hidden, decoder_weight)
    elif basis is None:
        delta_hidden = np.zeros((hidden.shape[0], decoder_weight.shape[0], hidden.shape[1]))
    else:
        projected = np.einsum("ncr,rk->nck", hidden, basis)
        if mode == "only":
            delta_hidden = np.einsum("nck,hr,rk->nhc", projected, decoder_weight, basis)
        elif mode == "drop":
            full = np.einsum("ncr,hr->nhc", hidden, decoder_weight)
            delta_hidden = full - np.einsum(
                "nck,hr,rk->nhc", projected, decoder_weight, basis
            )
        else:
            raise ValueError(f"unknown arm mode {mode!r}")
    if mode == "bias":
        # The synthetic bias ``c`` is the affine part of the synthetic map; with
        # it removed the branch writes only the input-driven correction.
        branch_abs = last_abs + delta_hidden
        correction_abs = branch_abs - last_abs
    else:
        branch_abs = last_abs + delta_hidden + decoder_bias
        correction_abs = branch_abs - last_abs
    branch_delta = branch_abs - target
    fused = (1.0 - gate) * phase_abs + gate * branch_abs
    fused_delta = fused - target
    recon = correction_abs - correction_reference
    count = branch_delta.size
    reference_energy = float(np.sum((correction_reference - correction_reference.mean()) ** 2))
    return {
        "correction_reconstruction_r2": (
            float(1.0 - np.sum(recon ** 2) / reference_energy)
            if reference_energy > 0
            else 0.0
        ),
        "correction_rmse": float(np.sqrt(np.mean(recon ** 2))),
        "branch_mse": float(np.sum(branch_delta ** 2) / count),
        "branch_mae": float(np.sum(np.abs(branch_delta)) / count),
        "fused_mse": float(np.sum(fused_delta ** 2) / count),
        "fused_mae": float(np.sum(np.abs(fused_delta)) / count),
        "correction_energy": float(np.sum(correction_abs ** 2) / count),
        "pair_count": int(count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir", default="research_runs/lowrank_checkpoint_information_v1"
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--cells", default="")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--semantic-rank", type=int, default=8)
    parser.add_argument(
        "--arms-limit",
        default="",
        help="comma separated arm names to evaluate (default: all)",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--debug-checks", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(exist_ok=True)

    gpu = int(args.gpus.split(",")[0]) if args.gpus else 0
    if not torch.cuda.is_available():
        raise RuntimeError("this evaluator requires CUDA on the analysis server")
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    print(f"device: {device}", flush=True)

    rows, _, _ = inventory_rows(repo_root, hash_checkpoints=False)
    formal = [row for row in rows if not row["is_diagnostic_only"]]
    wanted_settings = {item for item in args.settings.split(",") if item}
    wanted_seeds = {int(item) for item in args.seeds.split(",") if item}
    wanted_cells = {item for item in args.cells.split(",") if item}
    if wanted_settings:
        formal = [row for row in formal if row["setting"] in wanted_settings]
    if wanted_seeds:
        formal = [row for row in formal if int(row["seed"]) in wanted_seeds]
    if wanted_cells:
        formal = [row for row in formal if row["cell"] in wanted_cells]
    formal = formal[args.start_index :]
    if args.limit:
        formal = formal[: args.limit]
    print(f"checkpoints to process: {len(formal)}", flush=True)
    if not formal:
        return

    models: dict[tuple[str, int, int], dict] = {}
    audit_rows: list[dict] = []
    intervention_rows: list[dict] = []
    started = time.time()

    for row in formal:
        setting = row["setting"]
        dataset = row["dataset"]
        horizon = int(row["horizon"])
        seed = int(row["seed"])
        cell = row["cell"]
        run_dir = repo_root / row["selected_run_dir"]
        config = json.loads((run_dir / "config.json").read_text())
        hyperparams = dict(config["hyperparams"])
        # ``batch_size`` lives in the run spec, not in ``hyperparams``; keep it
        # explicit so the validation batches match the training protocol.
        batch_size = int(
            config.get("batch_size") or hyperparams.get("batch_size") or 256
        )
        group_key = (dataset, horizon, seed)
        if group_key not in models:
            exp_args, handles = build_loaders(
                dataset, 720, horizon, hyperparams, batch_size, repo_root,
                splits=("val",),
            )
            models[group_key] = {
                "loader": handles["val"][1],
                "model": build_model(exp_args, 720, horizon, hyperparams),
            }
        model = models[group_key]["model"]
        val_loader = models[group_key]["loader"]
        checkpoint_path = repo_root / row["checkpoint_path"]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        model.load_state_dict(state, strict=True)
        if getattr(model.args, "use_adaptive_weak_period_gate", False) or getattr(
            model.args, "use_adaptive_residual_gate", False
        ):
            raise RuntimeError(
                "the audited cells must use the fixed sigmoid gate; an adaptive "
                "gate would make the fused metric input dependent"
            )
        model.to(device).eval()
        head = model.weak_period_residual
        encoder_weight = (
            state["weak_period_residual.encoder.weight"].detach().double().cpu().numpy()
        )
        encoder_bias = (
            state["weak_period_residual.encoder.bias"].detach().double().cpu().numpy()
        )
        decoder_weight = (
            state["weak_period_residual.decoder.weight"].detach().double().cpu().numpy()
        )
        decoder_bias = (
            state["weak_period_residual.decoder.bias"].detach().double().cpu().numpy()
        )
        matrix, _ = effective_map(encoder_weight, decoder_weight, head.pooled_len)

        set_seed(20260916)
        chunks: dict[str, list] = {
            "z": [], "hidden": [], "phase": [], "phase_norm": [], "residual": [],
            "residual_norm": [], "mu": [], "sigma": [], "gate": [], "target": [],
            "fused": [],
        }
        equivalence_max = 0.0
        identity_max = 0.0
        batches = 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                if args.max_batches and batch_index >= args.max_batches:
                    break
                batch = [
                    item.to(device) if torch.is_tensor(item) else item for item in batch
                ]
                x, y, x_mark, y_mark = batch
                dec = model._build_decoder_input(y.float())
                clean_out, _, _ = model(x.float(), x_mark.float(), dec, y_mark.float())
                with instrument_model(model, None) as instrumented:
                    patched_out, _, _ = instrumented(
                        x.float(), x_mark.float(), dec, y_mark.float()
                    )
                    records = instrumented.last_lowrank_records
                    hidden = records["hidden"]
                    centered = records["z"]
                    mu, sigma = records["stats"]
                    gate = records["gate"]
                    weight = torch.as_tensor(
                        decoder_weight, dtype=hidden.dtype, device=hidden.device
                    )
                    bias = torch.as_tensor(
                        decoder_bias, dtype=hidden.dtype, device=hidden.device
                    )
                    if records["residual_norm"] is None:
                        raise RuntimeError(
                            "the model never denormalized a residual forecast; "
                            "the audited cells must use the plain R4 fusion path"
                        )
                    # The plan's equivalence check: the branch's *effective map*
                    # must reproduce the normalized residual the model actually
                    # wrote.  Evaluated in float64 because the in-model float32
                    # matmul is TF32 on this platform.
                    fp64_residual = torch.nn.functional.linear(
                        centered.double(),
                        torch.as_tensor(matrix, dtype=torch.float64, device=centered.device),
                    ).permute(0, 2, 1) + torch.as_tensor(
                        decoder_bias, dtype=torch.float64, device=centered.device
                    )
                    fp64_reference = records["residual_norm"].double()
                    equivalence_max = max(
                        equivalence_max, float((fp64_residual - fp64_reference).abs().max())
                    )
                    identity_max = max(
                        identity_max, float((patched_out - clean_out).abs().max())
                    )
                    gate_full = gate.reshape(1, 1, -1).expand(x.shape[0], 1, gate.shape[-1])
                    chunks["z"].append(centered.permute(0, 2, 1).double().cpu().numpy())
                    chunks["hidden"].append(hidden.double().cpu().numpy())
                    chunks["phase"].append(
                        instrumented.last_phase_forecast.double().cpu().numpy()
                    )
                    chunks["residual"].append(
                        instrumented.last_residual_forecast.double().cpu().numpy()
                    )
                    chunks["phase_norm"].append(
                        records["phase_norm"].double().cpu().numpy()
                    )
                    chunks["residual_norm"].append(
                        records["residual_norm"].double().cpu().numpy()
                    )
                    chunks["mu"].append(mu.double().cpu().numpy())
                    chunks["sigma"].append(sigma.double().cpu().numpy())
                    chunks["gate"].append(gate_full.double().cpu().numpy())
                    chunks["target"].append(y.float().double().cpu().numpy())
                    chunks["fused"].append(patched_out.double().cpu().numpy())
                    if args.debug_checks:
                        branch = (
                            torch.nn.functional.linear(hidden, weight, bias).permute(0, 2, 1)
                            + centered[:, :, -1:].permute(0, 2, 1) * sigma
                            + mu
                        )
                        print(
                            "  [debug-batch] residual fp32 vs branch absmax "
                            f"{float(instrumented.last_residual_forecast.abs().max()):.6f}/"
                            f"{float(branch.abs().max()):.6f} diff "
                            f"{float((instrumented.last_residual_forecast - branch).abs().max()):.6f} "
                            f"| fp64 map diff {equivalence_max:.3e} | "
                            f"mu {float(mu.mean()):.6f} sigma {float(sigma.mean()):.6f}",
                            flush=True,
                        )
                batches += 1
        features = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
        del chunks, clean_out, patched_out, records, hidden, centered

        rank_from_checkpoint = int(head.rank)
        # ``last_abs`` is the branch's own persistence anchor in the original
        # value space: the last step of its private normalized history, scaled
        # back with the exact RevIN statistics of this forward pass.  The
        # identity ``residual_norm == decoder(hidden) + bias + z_last`` is the
        # algebraic form of the head's own forward, so it is audited too.
        hidden = features["hidden"].astype(np.float64)
        sigma = features["sigma"].astype(np.float64)
        mu = features["mu"].astype(np.float64)
        residual_abs = features["residual"].astype(np.float64)
        residual_norm = features["residual_norm"].astype(np.float64)
        z_last_norm = features["z"].astype(np.float64)[:, -1, :][:, None, :]
        last_abs = z_last_norm * sigma + mu
        head_identity_error = float(
            np.abs(
                residual_norm
                - (
                    np.einsum("ncr,hr->nhc", hidden, decoder_weight)
                    + decoder_bias[None, :, None]
                    + z_last_norm
                )
            ).max()
        )
        denormalization_error = float(
            np.abs(residual_abs - (residual_norm * sigma + mu)).max()
        )
        audit = {
            "setting": setting,
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "cell": cell,
            "rank": row["rank"],
            "rank_from_checkpoint": rank_from_checkpoint,
            "rank_matches_inventory": bool(rank_from_checkpoint == int(row["rank"])),
            "pool_factor": int(head.pool_factor),
            "pooled_len": int(head.pooled_len),
            "pool_factor_is_one": bool(int(head.pool_factor) == 1),
            "smooth_ratio": float(head.smooth_ratio),
            "smooth_ratio_is_zero": bool(float(head.smooth_ratio) == 0.0),
            "effective_map_equivalence_max_abs": equivalence_max,
            "equivalence_pass": bool(equivalence_max < 1e-6),
            "head_decomposition_max_abs": head_identity_error,
            "head_decomposition_pass": bool(head_identity_error < 1e-9),
            "rev_in_denormalization_max_abs": denormalization_error,
            "intervention_identity_max_abs": identity_max,
            "intervention_identity_pass": bool(identity_max < 1e-9),
            "validation_batches": batches,
            "validation_samples": int(features["z"].shape[0]),
            "gate_mean": float(features["gate"].mean()),
            "gate_min": float(features["gate"].min()),
            "gate_max": float(features["gate"].max()),
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "run_val_mse": row["selected_val_mse"],
            "run_val_mae": row["selected_val_mae"],
        }
        audit_rows.append(audit)
        print(
            f"[audit] {setting} seed={seed} {cell} rank={rank_from_checkpoint} "
            f"equiv={equivalence_max:.3e} identity={identity_max:.3e} "
            f"anchor={anchor_error:.3e} n={audit['validation_samples']}",
            flush=True,
        )
        if args.debug_checks:
            recomputed = np.einsum(
                "ncr,hr->nhc", hidden, decoder_weight
            ) + decoder_bias[None, :, None] + last_abs
            stored = residual_abs
            print(
                "  [debug] residual absmax stored/recomputed "
                f"{np.abs(stored).max():.6f}/{np.abs(recomputed).max():.6f} "
                f"| diff {np.abs(stored - recomputed).max():.6f} "
                f"| last_abs absmax {np.abs(last_abs).max():.6f} "
                f"| sigma mean {sigma.mean():.6f}",
                flush=True,
            )
        if args.audit_only:
            del features, hidden, residual_abs, residual_norm
            models.pop(group_key, None)
            continue

        gate = features["gate"].astype(np.float64)
        phase_abs = features["phase"].astype(np.float64)
        target = features["target"].astype(np.float64)
        correction_reference = residual_abs - last_abs

        baseline = arm_metrics(
            hidden, decoder_weight, decoder_bias, gate, phase_abs, target,
            correction_reference, last_abs, None, "identity",
        )
        del residual_abs, residual_norm, phase_abs, target
        rank_dim = int(hidden.shape[-1])
        rng = np.random.default_rng(RANDOM_SEED)
        semantic_full = semantic_basis(dataset, rank_dim)
        semantic_small = semantic_basis(dataset, min(args.semantic_rank, rank_dim))
        moments_path = repo_root / args.output_dir / "train_moments" / f"{setting}.npz"
        pca = pca_basis(moments_path, rank_dim) if moments_path.is_file() else None
        conditional_path = (
            repo_root / args.output_dir / "subspaces" / f"{setting}_seed{seed}.npz"
        )
        conditional = None
        independent = None
        if conditional_path.is_file():
            payload = np.load(conditional_path)
            if "conditional_basis" in payload.files:
                conditional = payload["conditional_basis"].astype(np.float64)
            if "independent_basis" in payload.files:
                independent = payload["independent_basis"].astype(np.float64)

        requested = {item for item in args.arms_limit.split(",") if item}
        arms: list[tuple[str, np.ndarray | None, str]] = [
            ("Original", None, "identity"),
            ("Semantic-only", semantic_full, "only"),
            ("Semantic-drop", semantic_full, "drop"),
            ("Semantic8-only", semantic_small, "only"),
            ("Semantic8-drop", semantic_small, "drop"),
            ("Bias-off", None, "bias"),
        ]
        if pca is not None:
            arms.append(("PCA-only", pca, "only"))
            arms.append(("PCA-drop", pca, "drop"))
        if conditional is not None:
            arms.append(("Conditional-RRR-only", conditional, "only"))
        if independent is not None:
            arms.append(("Independent-RRR-only", independent, "only"))
        if requested:
            arms = [arm for arm in arms if arm[0] in requested]

        random_metrics = []
        random_count = 0
        for repeat in range(args.random_repeats):
            basis = random_orthogonal_basis(rank_dim, min(args.semantic_rank, rank_dim), rng)
            random_metrics.append(
                arm_metrics(
                    hidden, decoder_weight, decoder_bias, gate, phase_abs, target,
                    correction_reference, last_abs, basis, "drop",
                )
            )
            random_count += 1

        random_mse = np.asarray([item["fused_mse"] for item in random_metrics])
        random_mae = np.asarray([item["fused_mae"] for item in random_metrics])
        low_mse, high_mse = np.percentile(random_mse, RANDOM_QUANTILES)
        low_mae, high_mae = np.percentile(random_mae, RANDOM_QUANTILES)

        def summarise_against_random(arm_name: str, metrics: dict) -> dict:
            """Rank one arm against the 100 same-dimension random subspaces.

            The random controls are matched on dimension only, exactly as the
            plan prescribes, so they are a calibration band for "how much does
            removing an arbitrary eight-dimensional subspace hurt", not a
            variance-matched control.
            """
            is_sensitivity_arm = arm_name.endswith("-drop")
            return {
                "random_mean_fused_mse": float(random_mse.mean()),
                "random_mean_fused_mae": float(random_mae.mean()),
                "random_low_fused_mse": float(low_mse),
                "random_high_fused_mse": float(high_mse),
                "random_low_fused_mae": float(low_mae),
                "random_high_fused_mae": float(high_mae),
                "random_fused_mse_percentile_of_arm": float(
                    100.0 * np.mean(random_mse <= metrics["fused_mse"])
                ),
                "random_fused_mae_percentile_of_arm": float(
                    100.0 * np.mean(random_mae <= metrics["fused_mae"])
                ),
                "worse_than_random_95pct_fused_mse": bool(
                    is_sensitivity_arm and metrics["fused_mse"] > high_mse
                ),
                "worse_than_random_95pct_fused_mae": bool(
                    is_sensitivity_arm and metrics["fused_mae"] > high_mae
                ),
                "reference_matches_random_band": bool(
                    is_sensitivity_arm
                    and low_mse <= metrics["fused_mse"] <= high_mse
                    and low_mae <= metrics["fused_mae"] <= high_mae
                ),
            }

        for arm_name, basis, mode in arms:
            metrics = arm_metrics(
                hidden, decoder_weight, decoder_bias, gate, phase_abs, target,
                correction_reference, last_abs, basis, mode,
            )
            record = {
                "setting": setting,
                "dataset": dataset,
                "horizon": horizon,
                "seed": seed,
                "cell": cell,
                "rank": row["rank"],
                "arm": arm_name,
                "subspace_dimension": int(basis.shape[1]) if basis is not None else 0,
                "random_repeats": int(random_count),
                "baseline_branch_mse": baseline["branch_mse"],
                "baseline_branch_mae": baseline["branch_mae"],
                "baseline_fused_mse": baseline["fused_mse"],
                "baseline_fused_mae": baseline["fused_mae"],
                "delta_fused_mse_vs_checkpoint": metrics["fused_mse"] - baseline["fused_mse"],
                "delta_fused_mae_vs_checkpoint": metrics["fused_mae"] - baseline["fused_mae"],
            }
            record.update(metrics)
            record.update(summarise_against_random(arm_name, metrics))
            intervention_rows.append(record)
            print(
                f"  arm {arm_name:<22} branch={metrics['branch_mse']:.6f} "
                f"fused={metrics['fused_mse']:.6f} R2={metrics['correction_reconstruction_r2']:.4f}",
                flush=True,
            )

        np.savez_compressed(
            features_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz",
            encoder_weight=encoder_weight,
            decoder_weight=decoder_weight,
            decoder_bias=decoder_bias,
            hidden=hidden.astype(np.float32),
            z=features["z"].astype(np.float32),
            sigma=sigma.astype(np.float32),
            gate=gate.astype(np.float32),
            **{
                key: features[key].astype(np.float32)
                for key in ("phase", "target", "residual", "fused")
            },
        )
        del (features, hidden, sigma, mu, gate, phase_abs, target,
             correction_reference, last_abs, x_last_norm, semantic_full,
             semantic_small, pca, conditional, independent, random_metrics, arms)
        models.pop(group_key, None)
        del model, val_loader

    write_csv(audit_rows, audit_dir / "stage0_audit.csv")
    write_csv(intervention_rows, output_dir / "intervention_results.csv")
    if audit_rows:
        failures = [row for row in audit_rows if not row["equivalence_pass"]]
        identity_failures = [
            row for row in audit_rows if not row["intervention_identity_pass"]
        ]
        print(
            f"audited {len(audit_rows)} checkpoints; equivalence failures "
            f"{len(failures)}; identity failures {len(identity_failures)}",
            flush=True,
        )
        for row in failures + identity_failures:
            print(
                "  FAIL", row["setting"], row["seed"], row["cell"],
                row["effective_map_equivalence_max_abs"],
                row["intervention_identity_max_abs"],
                flush=True,
            )
    print(f"elapsed {time.time() - started:.1f}s", flush=True)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
