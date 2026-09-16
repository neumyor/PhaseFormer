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


def semantic_basis(dataset: str, limit: int | None = None) -> np.ndarray:
    """Union of the input semantic groups, optionally truncated.

    The dictionary groups overlap heavily (several contain the plain recent-level
    templates), so the union is a superset of any single group's semantics; the
    truncation keeps the leading directions of the union.
    """
    groups = build_groups(input_templates(720, DATASET_PERIOD_STEPS[dataset]))
    basis = orthonormalize(
        np.concatenate([group.basis for group in groups.values()], axis=1).T
    )
    if limit is not None:
        basis = np.ascontiguousarray(basis[:, :limit])
    return basis


def latent_image(z_basis: np.ndarray, encoder_weight: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal basis of the latent image of a z-space subspace.

    A linear head applies ``decoder @ encoder`` to ``Q Q^T z``, so keeping the
    z-space subspace ``Q`` writes corrections in the latent span of
    ``encoder @ Q``.  That span -- not ``Q`` itself -- is what an intervention on
    the hidden state can name.
    """
    image = encoder_weight @ z_basis
    basis = orthonormalize(image.T)
    return np.ascontiguousarray(basis[:, : min(rank, basis.shape[1])])


def latent_input_pca_basis(hidden: np.ndarray, rank: int) -> np.ndarray:
    """Top ``rank`` principal directions of the branch's hidden state.

    The plan's same-dimension PCA control, evaluated in the coordinate system the
    interventions actually touch.
    """
    centered = hidden.reshape(-1, hidden.shape[-1])
    centered = centered - centered.mean(axis=0)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1]
    return np.ascontiguousarray(vectors[:, order[:rank]])





def read_cached_features(path: Path) -> dict:
    """Load a feature cache written by an earlier pass."""
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def affine_bias(encoder_bias: np.ndarray, decoder_weight: np.ndarray, decoder_bias: np.ndarray) -> np.ndarray:
    """Constant term of the branch's effective affine map.

    The head computes ``decoder(encoder(z)) + decoder_bias``, and
    ``encoder(z) = encoder.weight @ z + encoder.bias``, so the input-independent
    part of the correction is ``decoder.weight @ encoder.bias + decoder.bias``.
    Dropping the mapped encoder bias silently shifts every arm.
    """
    return decoder_weight @ encoder_bias + decoder_bias


def arm_metrics(
    hidden: np.ndarray,
    decoder_weight: np.ndarray,
    decoder_bias: np.ndarray,
    encoder_bias: np.ndarray,
    gate: np.ndarray,
    phase_abs: np.ndarray,
    target: np.ndarray,
    correction_reference: np.ndarray,
    last_abs: np.ndarray,
    sigma: np.ndarray,
    basis: np.ndarray | None,
    mode: str,
) -> dict:
    """Branch, fused and reconstruction metrics of one arm.

    The branch writes, in the original value space,

        r = last_abs + sigma * (decoder(h) + decoder.weight @ encoder.bias
                                + decoder.bias)

    with ``last_abs = sigma * x_last_norm + mu`` the persistence anchor.  The
    arm's **correction** is everything except the anchor; keeping the bias inside
    it makes ``only`` and ``drop`` two independent projections whose corrections
    sum back to the untouched one, and ``Bias-off`` is the arm that removes the
    synthetic bias term.
    """
    bias_term = affine_bias(encoder_bias, decoder_weight, decoder_bias)[None, :, None]
    transformed = np.einsum("ncr,hr->nhc", hidden, decoder_weight)
    if mode == "bias":
        correction = transformed * sigma
    elif mode == "identity":
        correction = (transformed + bias_term) * sigma
    elif basis is None:
        correction = np.zeros_like(transformed)
    elif mode == "only":
        projected = np.einsum("ncr,rk->nck", hidden, basis)
        kept = np.einsum("nck,hr,rk->nhc", projected, decoder_weight, basis)
        correction = (kept + bias_term) * sigma
    elif mode == "drop":
        projected = np.einsum("ncr,rk->nck", hidden, basis)
        removed = np.einsum("nck,hr,rk->nhc", projected, decoder_weight, basis)
        correction = (transformed - removed + bias_term) * sigma
    else:
        raise ValueError(f"unknown arm mode {mode!r}")
    branch_abs = last_abs + correction
    branch_delta = branch_abs - target
    fused = (1.0 - gate) * phase_abs + gate * branch_abs
    fused_delta = fused - target
    recon = correction - correction_reference
    reference_energy = float(
        np.sum((correction_reference - correction_reference.mean()) ** 2)
    )
    return {
        "correction_reconstruction_r2": (
            float(1.0 - np.sum(recon ** 2) / reference_energy)
            if reference_energy > 0
            else 0.0
        ),
        "correction_rmse": float(np.sqrt(np.mean(recon ** 2))),
        "branch_mse": float(np.mean(branch_delta ** 2)),
        "branch_mae": float(np.mean(np.abs(branch_delta))),
        "fused_mse": float(np.mean(fused_delta ** 2)),
        "fused_mae": float(np.mean(np.abs(fused_delta))),
        "correction_energy": float(np.mean(correction ** 2)),
        "pair_count": int(branch_delta.size),
    }


def random_drop_band(
    hidden: np.ndarray,
    decoder_weight: np.ndarray,
    decoder_bias: np.ndarray,
    encoder_bias: np.ndarray,
    sigma: np.ndarray,
    last_abs: np.ndarray,
    gate: np.ndarray,
    phase_abs: np.ndarray,
    target: np.ndarray,
    rank: int,
    count: int,
    rng: np.random.Generator,
    chunk: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Fused MSE/MAE of ``count`` random same-dimension drop arms.

    The block size bounds the transient ``(n, h, c, arms)`` tensor: with the
    largest validation split of this analysis (thousands of windows, hundreds of
    channels) an unbounded block reached several gigabytes per process, which
    exhausted the shared host when several shards ran at once.

    The arms are all linear projections of the same hidden state, so the whole
    band is evaluated with one blocked contraction over a stacked basis tensor.
    A per-arm Python loop over the largest validation split of this analysis
    (thousands of windows, hundreds of channels) dominates the runtime, which is
    why the evaluation is vectorised.
    """
    count = max(int(count), 1)
    dimension = max(1, min(rank, 6))
    # (arms, r, k)
    bases = np.stack(
        [random_orthogonal_basis(rank, dimension, rng) for _ in range(count)]
    ).astype(np.float64)
    samples = hidden.shape[0]
    mse = np.zeros(count)
    mae = np.zeros(count)
    bias_term = affine_bias(encoder_bias, decoder_weight, decoder_bias)
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        piece = np.ascontiguousarray(hidden[start:stop], dtype=np.float64)
        n = piece.shape[0]
        # ``full`` is the untouched correction of this block: (n, h, c)
        full = np.einsum("ncr,hr->nhc", piece, decoder_weight) + bias_term[
            None, :, None
        ]
        # ``removed`` is the correction of the state projected onto each arm:
        # (n, r, arms) then (n, h, arms) then (n, h, c, arms).
        coefficients = np.einsum("ncr,mrk->nckm", piece, bases)
        back = np.einsum("nckm,mrk->ncrm", coefficients, bases)
        decoded = np.einsum("ncrm,hr->nhcm", back, decoder_weight)
        corrections = (full[:, :, :, None] - decoded) * sigma[start:stop][
            :, None, :, None
        ]
        branch = last_abs[start:stop][:, None, :, None] + corrections
        fused = (1.0 - gate[start:stop])[:, None, :, None] * phase_abs[
            start:stop
        ][:, :, :, None] + gate[start:stop][:, None, :, None] * branch
        delta = fused - target[start:stop][:, :, :, None]
        weight = n / samples
        mse += weight * np.mean(delta ** 2, axis=(0, 1, 2))
        mae += weight * np.mean(np.abs(delta), axis=(0, 1, 2))
        del piece, full, coefficients, back, decoded, corrections, branch, fused, delta
    return mse, mae


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
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="re-evaluate the arms of already-cached cells without any GPU work",
    )
    parser.add_argument("--reuse-cache", action="store_true",
                        help="skip the sweep for cells that already have a cache")
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
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
    else:
        # Every arm is evaluated in closed form from the cached features, so a
        # cache-only run needs no GPU at all.
        device = torch.device("cpu")
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
        cache_path = (
            features_dir
            / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz"
        )
        if (args.cache_only or args.reuse_cache) and cache_path.is_file():
            # Arms are closed-form functions of the cache, so a cached cell can
            # be recomputed without loading a model or touching the GPU -- this
            # is how the Conditional-RRR and Independent-RRR arms are filled in
            # after Stage 3 has written the subspaces.
            payload = read_cached_features(cache_path)
            cached_only = {
                key: payload[key].astype(np.float64)
                for key in ("hidden", "z", "sigma", "mu", "gate", "phase", "target")
            }
            encoder_weight = payload["encoder_weight"].astype(np.float64)
            encoder_bias = payload["encoder_bias"].astype(np.float64)
            decoder_weight = payload["decoder_weight"].astype(np.float64)
            decoder_bias = payload["decoder_bias"].astype(np.float64)
            hidden_only = cached_only["hidden"]
            sigma_only = cached_only["sigma"]
            mu_only = cached_only["mu"]
            head_map_only = np.einsum("ncr,hr->nhc", hidden_only, decoder_weight)
            bias_only = affine_bias(
                encoder_bias, decoder_weight, decoder_bias
            )[None, :, None]
            # ``x_last_norm`` is the head's own persistence anchor, cached
            # directly so this path reproduces the original one exactly.
            x_last_only = payload["x_last_norm"].astype(np.float64)
            residual_norm_only = head_map_only + bias_only + x_last_only
            residual_abs_only = residual_norm_only * sigma_only + mu_only
            last_abs_only = x_last_only * sigma_only + mu_only
            invariant = evaluate_arms(
                cached_only, hidden_only, sigma_only, mu_only, residual_abs_only,
                residual_norm_only, last_abs_only, encoder_weight,
                decoder_weight, decoder_bias, encoder_bias, setting, dataset,
                horizon, seed, cell, int(row["rank"]), row["selected_val_mse"],
                row["selected_val_mae"], repo_root, output_dir, args,
                intervention_rows,
            )
            print(
                f"[cache] {setting} seed={seed} {cell}: arms recomputed "
                f"(invariant {invariant['untouched_arm_reproduces_run_metric']})",
                flush=True,
            )
            del payload, cached_only, hidden_only, sigma_only, mu_only
            del residual_abs_only, residual_norm_only, last_abs_only, x_last_only
            del head_map_only, bias_only
            continue
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
        # The composed map used for the equivalence audit is rebuilt in float64
        # from the checkpoint tensors.  ``decoder @ encoder`` evaluated in
        # float32 and then widened to float64 keeps a TF32-level rounding error
        # of ~1e-3, which would masquerade as an equivalence failure.
        matrix64 = torch.as_tensor(
            encoder_weight, dtype=torch.float64, device=device
        )
        matrix64 = (
            torch.as_tensor(decoder_weight, dtype=torch.float64, device=device)
            @ matrix64
        )

        set_seed(20260916)
        sweep_started = time.time()
        chunks: dict[str, list] = {
            "z": [], "hidden": [], "phase": [], "phase_norm": [], "residual": [],
            "residual_norm": [], "mu": [], "sigma": [], "gate": [], "target": [],
            "fused": [],
        }
        equivalence_max = 0.0
        decomposition_max = 0.0
        tf32_max = 0.0
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
                with instrument_model(
                    model,
                    None,
                    {"matrix": matrix64},
                ) as instrumented:
                    patched_out, _, _ = instrumented(
                        x.float(), x_mark.float(), dec, y_mark.float()
                    )
                    records = instrumented.last_lowrank_records
                    hidden = records["hidden"]
                    centered = records["z"]
                    mu, sigma = records["stats"]
                    gate = records["gate"]
                    if records["residual_norm"] is None:
                        raise RuntimeError(
                            "the model never denormalized a residual forecast; "
                            "the audited cells must use the plain R4 fusion path"
                        )
                    weight = torch.as_tensor(
                        decoder_weight, dtype=hidden.dtype, device=hidden.device
                    )
                    bias = torch.as_tensor(
                        decoder_bias, dtype=hidden.dtype, device=hidden.device
                    )
                    audit_math_values = head.last_audit
                    decoder_weight64 = torch.as_tensor(
                        decoder_weight, dtype=torch.float64, device=device
                    )
                    # 1) The plan's effective-map equivalence, in float64:
                    #    decoder(encoder(pool(z))) must equal ``M z + c`` on the
                    #    head's own private input.
                    equivalence_max = max(
                        equivalence_max,
                        float(
                            (
                                audit_math_values["hidden64"] @ decoder_weight64.T
                                - audit_math_values["map64"]
                            )
                            .abs()
                            .max()
                        ),
                    )
                    # 2) The head's full decomposition: the normalized residual
                    #    the model actually wrote equals the mapped hidden state
                    #    plus the mapped encoder bias, the decoder bias and the
                    #    persistence anchor.
                    decomposition_max = max(
                        decomposition_max,
                        float(
                            (
                                audit_math_values["head64"]
                                - records["residual_norm"].double()
                            )
                            .abs()
                            .max()
                        ),
                    )
                    tf32_max = max(
                        tf32_max,
                        float(
                            (
                                audit_math_values["fp32_head"].double()
                                - audit_math_values["head64"]
                            )
                            .abs()
                            .max()
                        ),
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
                    # ``y`` carries ``label_len + pred_len`` steps: the first
                    # ``label_len`` are the decoder context, not targets.  The
                    # model's own validation metric scores only the trailing
                    # ``pred_len`` steps, so the cache keeps exactly those.
                    chunks["target"].append(
                        y.float()[:, -horizon:, :].double().cpu().numpy()
                    )
                    chunks["fused"].append(patched_out.double().cpu().numpy())
                    if args.debug_checks:
                        print(
                            "  [check] fp64 equiv "
                            f"{float((audit_math_values['hidden64'] @ torch.as_tensor(decoder_weight, dtype=torch.float64, device=device).T - audit_math_values['map64']).abs().max()):.3e} "
                            f"decomposition {decomposition_max:.3e} "
                            f"tf32 deviation {tf32_max:.3e} "
                            f"| residual_norm absmax "
                            f"{float(records['residual_norm'].abs().max()):.4f} "
                            f"| head-vs-residual_norm "
                            f"{records['residual_consistency_max_abs']}",
                            flush=True,
                        )
                batches += 1
        features = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
        del chunks, clean_out, patched_out, records, hidden, centered

        sweep_seconds = time.time() - sweep_started
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
        # The persistence anchor in the original value space.  It is recovered
        # from the model's own normalized residual as
        #     x_last_norm = residual_norm - (decoder(h) + mapped encoder bias
        #                                    + decoder bias)
        # rather than from a slice of ``z``: ``z`` is cached as ``(N, L, C)``, so
        # its last index is the *channel* axis and cannot be used here.
        head_map = np.einsum("ncr,hr->nhc", hidden, decoder_weight)
        x_last_norm = (
            residual_norm
            - head_map
            - affine_bias(encoder_bias, decoder_weight, decoder_bias)[None, :, None]
        )
        last_abs = x_last_norm * sigma + mu
        # The recovered anchor must be a single per-sample vector broadcast over
        # the horizon, which is a direct consequence of the head's algebra.
        anchor_spread = float(
            np.abs(x_last_norm - x_last_norm.mean(axis=1, keepdims=True)).max()
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
            "head_decomposition_max_abs": decomposition_max,
            "head_decomposition_pass": bool(decomposition_max < 1e-6),
            "fp32_tf32_deviation_max_abs": tf32_max,
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
            f"decomp={decomposition_max:.3e} n={audit['validation_samples']}",
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
        cached = {
            key: features[key].astype(np.float64)
            for key in ("hidden", "z", "sigma", "mu", "gate", "phase", "target")
        }
        # ``x_last_norm`` is the branch's persistence anchor before it is scaled
        # by the RevIN scale; caching it lets the arms be recomputed later with
        # no model and no approximation.
        cached["x_last_norm"] = x_last_norm
        # The cache is what lets the intervention arms be recomputed without a
        # second GPU sweep over the frozen checkpoints.
        np.savez_compressed(
            cache_path,
            encoder_weight=encoder_weight,
            encoder_bias=encoder_bias,
            decoder_weight=decoder_weight,
            decoder_bias=decoder_bias,
            **{key: value.astype(np.float32) for key, value in cached.items()},
        )
        if args.audit_only:
            del features, hidden, residual_abs, residual_norm, cached
            models.pop(group_key, None)
            continue

        arms_started = time.time()
        invariant = evaluate_arms(
            cached, hidden, sigma, mu, residual_abs, residual_norm, last_abs,
            encoder_weight, decoder_weight, decoder_bias, encoder_bias, setting,
            dataset, horizon, seed, cell, int(row["rank"]),
            row["selected_val_mse"], row["selected_val_mae"], repo_root,
            output_dir, args, intervention_rows,
        )
        print(
            f"  [timing] {setting} seed={seed} {cell}: sweep {sweep_seconds:.1f}s "
            f"arms {time.time() - arms_started:.1f}s",
            flush=True,
        )
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


def to_float_or(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
