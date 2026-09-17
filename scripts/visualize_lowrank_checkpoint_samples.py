#!/usr/bin/env python3
"""Stage 5 of the low-rank checkpoint information analysis: real sample figures.

For every included setting this script selects three representative validation
windows **programmatically** (plan section 5 forbids picking panels by eye):

* ``reproduced``   -- the window where ``Semantic-only`` reproduces the original
  low-rank correction best (evidence that the identified semantics are enough);
* ``necessary``    -- the window where ``Semantic-drop`` hurts the fused forecast
  most (evidence that the semantics carry usable predictive information);
* ``counterexample`` -- the window where deleting the semantics is no worse than
  a same-dimension random deletion, i.e. where the semantic reading fails.

Every figure carries, as plan section 5 requires, (1) the history window and the
true future, (2) the ``Phase-only`` / original / ``Semantic-only`` /
``Semantic-drop`` curves, (3) the correction decomposed into the canonical modes
plus the synthetic bias, (4) the main input statistics, and (5) the sample-level
MSE/MAE of every curve, so the selection is auditable rather than visual.

Everything is read from the Stage 4 feature cache, so this stage needs no GPU and
no model forward pass: the low-rank branch is affine, hence each intervention is
a closed-form function of the cached hidden state.  Figures are written as
modest-resolution PNGs (``--dpi``, default 110) to keep the footprint small.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
from scripts.lowrank_checkpoint_model import DATASET_PERIOD_STEPS  # noqa: E402

RANDOM_SEED = 20261117
FEATURE_BLOCK = 512
SEMANTIC_RANK = 8


# --------------------------------------------------------------------------- #
# cached-feature arithmetic
# --------------------------------------------------------------------------- #
def affine_bias(encoder_bias, decoder_weight, decoder_bias) -> np.ndarray:
    """The synthetic bias ``W b_enc + b_dec`` written by the low-rank branch."""
    return decoder_weight @ encoder_bias + decoder_bias


def semantic_basis(dataset: str, limit: int | None = None) -> np.ndarray:
    """Union of the plan's input semantic groups, optionally truncated."""
    groups = build_groups(input_templates(720, DATASET_PERIOD_STEPS[dataset]))
    basis = orthonormalize(
        np.concatenate([group.basis for group in groups.values()], axis=1).T
    )
    if limit is not None:
        basis = np.ascontiguousarray(basis[:, :limit])
    return basis


def latent_image(z_basis: np.ndarray, encoder_weight: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal basis of the latent image of a z-space subspace."""
    image = encoder_weight @ z_basis
    basis = orthonormalize(image.T)
    return np.ascontiguousarray(basis[:, : min(rank, basis.shape[1])])


def hidden_from_z(
    z: np.ndarray, encoder_weight: np.ndarray, encoder_bias: np.ndarray, pool_operator: np.ndarray
) -> np.ndarray:
    """Recompute the head's hidden state from the cached private input.

    ``z`` is cached as ``(N, lookback, C)``; the head consumes
    ``(N, C, lookback)``, pools the lookback axis and encodes, so this is the
    exact inverse of the caching convention.  The channel axis is the *last*
    axis of the cache, so the transpose is explicit rather than left to a
    shape coincidence.
    """
    centered = np.transpose(z, (0, 2, 1))  # (N, C, lookback)
    pooled = centered @ pool_operator.T  # (N, C, pooled_len)
    # ``encoder_weight`` is ``(rank, pooled_len)``, so the contraction is over
    # the pooled axis and the channel axis is carried through untouched.
    # ``encoder_weight`` is ``(rank, pooled_len)``, so it is transposed to put
    # the pooled axis first and contract against ``pooled``'s last axis; the
    # channel axis is carried through untouched.
    return (
        np.einsum("ncl,lr->ncr", pooled, encoder_weight.T)
        + encoder_bias[None, None, :]
    )


def correction_from_hidden(hidden: np.ndarray, decoder_weight: np.ndarray, bias_term: np.ndarray, basis, mode: str):
    """Correction the branch writes for one intervention ``mode``.

    Mirrors ``arm_metrics``: ``only`` keeps the branch's latent state inside the
    subspace span, ``drop`` removes it, and both keep the synthetic bias so that
    they partition the untouched correction.
    """
    transformed = np.einsum("ncr,hr->nhc", hidden, decoder_weight)
    if mode == "identity":
        return transformed + bias_term[None, :, None]
    if basis is None:
        return np.zeros_like(transformed)
    projected = np.einsum("ncr,rk->nck", hidden, basis)
    kept = np.einsum("nck,hr,rk->nhc", projected, decoder_weight, basis)
    if mode == "only":
        return kept + bias_term[None, :, None]
    if mode == "drop":
        return transformed - kept + bias_term[None, :, None]
    raise ValueError(f"unknown mode {mode!r}")


def fused_from_correction(correction, last_abs, sigma, gate, phase_abs):
    branch_abs = last_abs + correction * sigma
    return (1.0 - gate) * phase_abs + gate * branch_abs, branch_abs


def per_sample_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window MSE/MAE over the horizon and channels."""
    axes = tuple(range(1, prediction.ndim))
    mse = np.mean((prediction - target) ** 2, axis=axes)
    mae = np.mean(np.abs(prediction - target), axis=axes)
    return mse, mae


# --------------------------------------------------------------------------- #
# arm evaluation over one cached checkpoint
# --------------------------------------------------------------------------- #
def evaluate_sample_arms(
    setting: str,
    dataset: str,
    horizon: int,
    seed: int,
    cell: str,
    feature_path: Path,
    random_repeats: int,
    rng: np.random.Generator,
) -> dict:
    """Per-sample metrics for every Stage 5 arm of one checkpoint."""
    payload = np.load(feature_path)
    encoder_weight = payload["encoder_weight"]
    encoder_bias = payload["encoder_bias"]
    decoder_weight = payload["decoder_weight"]
    decoder_bias = payload["decoder_bias"]
    rank = int(encoder_weight.shape[0])
    pooled_len = int(encoder_weight.shape[1])

    hidden = payload["hidden"].astype(np.float64)          # (N, C, r)
    z = payload["z"].astype(np.float64)                    # (N, 720, C)
    sigma = payload["sigma"].astype(np.float64)            # (N, 1, C)
    mu = payload["mu"].astype(np.float64)
    x_last_norm = payload["x_last_norm"].astype(np.float64)
    target = payload["target"].astype(np.float64)          # (N, H, C)
    phase_abs = payload["phase"].astype(np.float64)
    samples = target.shape[0]
    channels = target.shape[2]
    gate = np.broadcast_to(payload["gate"].astype(np.float64), target.shape)
    last_abs = x_last_norm * sigma + mu
    bias_term = affine_bias(encoder_bias, decoder_weight, decoder_bias)

    pool_operator = adaptive_avg_pool_operator(pooled_len, z.shape[1])
    # The pooled-to-hidden identity is verified rather than assumed: it is the
    # one place where the z-space reconstruction could silently disagree with
    # the cached hidden state.
    hidden_check = hidden_from_z(
        z[:2], encoder_weight, encoder_bias, pool_operator
    )
    reconstruction_gap = float(np.abs(hidden_check - hidden[:2]).max())

    plan = semantic_basis(dataset)
    semantic_dim = plan.shape[1]
    semantic = latent_image(plan, encoder_weight, semantic_dim)
    semantic8 = latent_image(plan, encoder_weight, min(SEMANTIC_RANK, rank))
    pca = latent_image(plan, encoder_weight, semantic8.shape[1])
    # The variance control of plan section 5 is the top PCA directions of the
    # hidden state at the same dimension as the semantic subspace.
    flat_hidden = hidden.reshape(-1, rank)
    _, _, vt = np.linalg.svd(flat_hidden - flat_hidden.mean(axis=0), full_matrices=False)
    pca = np.ascontiguousarray(vt[: semantic8.shape[1]].T)

    random_bases = []
    for _ in range(random_repeats):
        gaussian = rng.standard_normal((rank, semantic8.shape[1]))
        q, _ = np.linalg.qr(gaussian)
        random_bases.append(q[:, : semantic8.shape[1]])

    collected: dict[str, list[np.ndarray]] = {"original": []}
    random_chunks: list[np.ndarray] = []
    basis_of = {
        "semantic-only": (semantic, "only"),
        "semantic-drop": (semantic, "drop"),
        "semantic8-only": (semantic8, "only"),
        "semantic8-drop": (semantic8, "drop"),
        "pca-only": (pca, "only"),
        "pca-drop": (pca, "drop"),
    }
    for start in range(0, samples, FEATURE_BLOCK):
        stop = min(start + FEATURE_BLOCK, samples)
        h = hidden[start:stop]
        sg = sigma[start:stop]
        la = last_abs[start:stop]
        gt = gate[start:stop]
        ph = phase_abs[start:stop]
        tg = target[start:stop]
        for name, (basis, mode) in basis_of.items():
            correction = correction_from_hidden(h, decoder_weight, bias_term, basis, mode)
            fused, _ = fused_from_correction(correction, la, sg, gt, ph)
            mse, mae = per_sample_metrics(fused, tg)
            collected.setdefault(name, []).append(np.stack([mse, mae], axis=0))
        # Same-dimension random deletions, kept as a band.
        random_mse = np.empty((stop - start, len(random_bases)))
        for index, basis in enumerate(random_bases):
            correction = correction_from_hidden(h, decoder_weight, bias_term, basis, "drop")
            fused, _ = fused_from_correction(correction, la, sg, gt, ph)
            random_mse[:, index] = per_sample_metrics(fused, tg)[0]
        random_chunks.append(random_mse)

    random_all = np.concatenate(random_chunks, axis=0)

    # The untouched arm is the model's own fused output, so it is read from the
    # cache rather than rebuilt: the cached gate is broadcast over the horizon, so
    # a closed-form rebuild is only accurate up to that broadcast.  That is fine
    # for the arm *differences* this analysis reports, but it is not exact enough
    # to draw as "the original curve".  The rebuild is still measured so the
    # residual error is disclosed rather than assumed.
    cached_fused = payload["fused"].astype(np.float64)
    reconstructed = fused_from_correction(
        correction_from_hidden(hidden, decoder_weight, bias_term, None, "identity"),
        last_abs, sigma, gate, phase_abs,
    )[0]
    original_gap = float(np.abs(reconstructed - cached_fused).max())
    del reconstructed
    for start in range(0, samples, FEATURE_BLOCK):
        stop = min(start + FEATURE_BLOCK, samples)
        collected["original"].append(
            np.stack(
                per_sample_metrics(cached_fused[start:stop], target[start:stop]),
                axis=0,
            )
        )

    metrics = {
        name: np.concatenate(parts, axis=1) for name, parts in collected.items()
    }

    return {
        "setting": setting,
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "cell": cell,
        "feature_path": feature_path,
        "rank": rank,
        "pooled_len": pooled_len,
        "samples": samples,
        "channels": channels,
        "hidden_reconstruction_gap": reconstruction_gap,
        "original_arm_gap_vs_cached_fused": original_gap,
        "metrics": metrics,
        "random_mse": random_all,
        "semantic_dim": semantic_dim,
        "semantic8_dim": semantic8.shape[1],
        "pca_dim": pca.shape[1],
        "bases": {"semantic": semantic, "semantic8": semantic8, "pca": pca},
        "decoder_weight": decoder_weight,
        "bias_term": bias_term,
    }


def select_samples(state: dict) -> list[dict]:
    """Pick the three representative windows of plan section 5."""
    mse = state["metrics"]
    baseline = mse["original"][0]
    only, drop = mse["semantic-only"][0], mse["semantic-drop"][0]
    only_rel = (only - baseline) / np.maximum(np.abs(baseline), 1e-30)
    drop_gain = drop - baseline
    random_median = np.median(state["random_mse"], axis=1)
    # "Reproduced": smallest relative degradation of the fused forecast.
    reproduced = int(np.argmin(only_rel))
    # "Necessary": largest absolute damage from deleting the semantics.
    necessary = int(np.argmax(drop_gain))
    # "Counterexample": deleting the semantics is no better than a same-dimension
    # random deletion, i.e. the semantic reading is not special here.
    extra_vs_random = drop - random_median
    counterexample = int(np.argmin(extra_vs_random))
    return [
        {"kind": "reproduced", "index": reproduced,
         "note": "Semantic-only 复现原低秩修正最好"},
        {"kind": "necessary", "index": necessary,
         "note": "Semantic-drop 融合退化最大"},
        {"kind": "counterexample", "index": counterexample,
         "note": "语义删除不优于随机同维删除"},
    ]


# --------------------------------------------------------------------------- #
# figure rendering
# --------------------------------------------------------------------------- #
def render_figure(state: dict, selection: dict, output_path: Path, dpi: int) -> dict:
    payload = np.load(state["feature_path"])
    index = selection["index"]
    horizon = state["horizon"]
    decoder_weight = state["decoder_weight"]
    bias_term = state["bias_term"]
    plan = semantic_basis(state["dataset"])
    semantic = state["bases"]["semantic"]

    z = payload["z"][index].astype(np.float64)             # (720, C)
    hidden = payload["hidden"][index: index + 1].astype(np.float64)
    sigma = payload["sigma"][index: index + 1].astype(np.float64)
    mu = payload["mu"][index: index + 1].astype(np.float64)
    x_last_norm = payload["x_last_norm"][index: index + 1].astype(np.float64)
    target = payload["target"][index: index + 1].astype(np.float64)
    phase_abs = payload["phase"][index: index + 1].astype(np.float64)
    gate = payload["gate"][index: index + 1].astype(np.float64)
    gate = np.broadcast_to(gate, target.shape).copy()
    last_abs = x_last_norm * sigma + mu
    # The cached ``z`` is RevIN-normalized *and* mean-centered per channel, so the
    # level cannot be recovered by subtracting it from the last step.  Its final
    # step is identically zero by construction while ``x_last_norm`` holds that
    # step's normalized value, hence the normalized history is ``z + x_last_norm``
    # and the model's value-space history follows from the cached RevIN stats.
    normalized_history = z + np.transpose(x_last_norm[0], (1, 0)).reshape(
        1, -1
    )  # (720, C)
    history = normalized_history * sigma[0, 0][None, :] + mu[0, 0][None, :]

    # The panel channel is the one carrying the largest correction energy.
    correction = correction_from_hidden(hidden, decoder_weight, bias_term, None, "identity")[0]
    channel = int(np.argmax(np.linalg.norm(correction * sigma[0], axis=0)))

    only_corr = correction_from_hidden(hidden, decoder_weight, bias_term, semantic, "only")[0]
    drop_corr = correction_from_hidden(hidden, decoder_weight, bias_term, semantic, "drop")[0]
    # The original curve is the model's own cached output (see evaluate_sample_arms).
    original_fused = payload["fused"][index: index + 1].astype(np.float64)
    only_fused, _ = fused_from_correction(only_corr, last_abs, sigma, gate, phase_abs)
    drop_fused, _ = fused_from_correction(drop_corr, last_abs, sigma, gate, phase_abs)

    curves = {
        "Phase-only": phase_abs[0, :, channel],
        "Original": original_fused[0, :, channel],
        "Semantic-only": only_fused[0, :, channel],
        "Semantic-drop": drop_fused[0, :, channel],
    }
    truth = target[0, :, channel]
    hist = history[:, channel]

    # Canonical mode decomposition of the original correction on this channel.
    # The branch's bottleneck is the latent state, so the canonical modes live in
    # the *rank*-dimensional latent space: ``v_i`` is the i-th latent input
    # direction of the decoder, and the branch writes ``sigma_i * v_i * <v_i, h>``
    # along it.  Decomposing in the 720-wide input space would not name any
    # direction the checkpoint can express.
    # All rank modes are computed so their sum reproduces the total correction
    # exactly; only the leading few are drawn to keep the panel readable.
    max_modes = state["rank"]
    u_dec, singular, vt_dec = np.linalg.svd(decoder_weight, full_matrices=False)
    modes = []
    for mode_index in range(max_modes):
        direction = vt_dec[mode_index]                # (rank,)
        score = np.einsum("r,cr->c", direction, hidden[0])          # (channels,)
        # The decoder's left singular vector carries the mode into the horizon.
        profile = u_dec[:, mode_index] * singular[mode_index]       # (horizon,)
        decoded = profile[:, None] * score[None, :]                 # (horizon, C)
        modes.append(decoded[:, channel] * sigma[0, 0, channel])
    modes_shown = 6

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), dpi=dpi)
    x_axis = np.arange(hist.size + horizon)
    top = axes[0]
    top.plot(
        np.arange(hist.size), hist, color="0.55", lw=1.0,
        label=f"history ({hist.size} steps)",
    )
    top.plot(np.arange(hist.size, hist.size + horizon), truth, color="black", lw=1.6, label="true future")
    colors = {"Phase-only": "#1f77b4", "Original": "#2ca02c",
              "Semantic-only": "#ff7f0e", "Semantic-drop": "#d62728"}
    for name, curve in curves.items():
        top.plot(np.arange(hist.size, hist.size + horizon), curve,
                 color=colors[name], lw=1.1, ls="--", label=name)
    top.axvline(hist.size - 0.5, color="0.8", lw=0.8)
    top.set_xlim(0, hist.size + horizon)
    top.set_ylabel("value")
    top.legend(fontsize=7, ncol=3, loc="best")
    top.set_title(
        f"{state['setting']} seed={state['seed']} {state['cell']} | channel {channel} | "
        f"{selection['kind']} (window {index}) — {selection['note']}",
        fontsize=9,
    )

    bottom = axes[1]
    for mode_index, curve in enumerate(modes[:modes_shown]):
        bottom.plot(np.arange(hist.size, hist.size + horizon), curve,
                    lw=0.9, label=f"mode {mode_index}")
    if max_modes > modes_shown:
        rest = np.sum(modes[modes_shown:], axis=0)
        bottom.plot(np.arange(hist.size, hist.size + horizon), rest,
                    lw=0.9, ls="--", color="0.5",
                    label=f"modes {modes_shown}–{max_modes - 1}")
    bottom.plot(np.arange(hist.size, hist.size + horizon),
                bias_term[channel] * sigma[0, 0, channel] * np.ones(horizon),
                lw=0.9, ls=":", color="black", label="bias")
    bottom.plot(np.arange(hist.size, hist.size + horizon),
                correction[:, channel] * sigma[0, 0, channel], lw=1.3, color="0.2",
                label="total correction")
    bottom.axvline(hist.size - 0.5, color="0.8", lw=0.8)
    bottom.set_xlim(hist.size - 1, hist.size + horizon)
    bottom.set_xlabel("step")
    bottom.set_ylabel("correction")
    bottom.legend(fontsize=7, ncol=4, loc="best")

    sample_mse = {name: float(state["metrics"][name][0][index]) for name in
                  ("original", "semantic-only", "semantic-drop", "semantic8-only", "pca-only")}
    sample_mae = {name: float(state["metrics"][name][1][index]) for name in
                  ("original", "semantic-only", "semantic-drop")}
    random_here = state["random_mse"][index]
    stats = (
        f"|history| mean {hist.mean():.3f} std {hist.std():.3f} "
        f"last {hist[-1]:.3f} min {hist.min():.3f} max {hist.max():.3f} "
        f"| sigma {sigma[0, 0, channel]:.3f} mu {mu[0, 0, channel]:.3f} "
        f"gate {gate[0, 0, channel]:.3f}\n"
        f"fused MSE: original {sample_mse['original']:.6f}  "
        f"semantic-only {sample_mse['semantic-only']:.6f}  "
        f"semantic-drop {sample_mse['semantic-drop']:.6f}  "
        f"semantic8-only {sample_mse['semantic8-only']:.6f}  pca-only {sample_mse['pca-only']:.6f}\n"
        f"fused MAE: original {sample_mae['original']:.6f}  "
        f"semantic-only {sample_mae['semantic-only']:.6f}  semantic-drop {sample_mae['semantic-drop']:.6f}  "
        f"| random same-dim deletion MSE median {float(np.median(random_here)):.6f} "
        f"95pct {float(np.percentile(random_here, 95)):.6f}"
    )
    figure.text(0.012, 0.012, stats, fontsize=6.4, family="monospace", va="bottom")
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png", dpi=dpi)
    plt.close(figure)

    return {
        "setting": state["setting"],
        "seed": state["seed"],
        "cell": state["cell"],
        "kind": selection["kind"],
        "window_index": index,
        "channel": channel,
        "note": selection["note"],
        "fused_mse_original": sample_mse["original"],
        "fused_mse_semantic_only": sample_mse["semantic-only"],
        "fused_mse_semantic_drop": sample_mse["semantic-drop"],
        "fused_mae_original": sample_mae["original"],
        "fused_mae_semantic_only": sample_mae["semantic-only"],
        "fused_mae_semantic_drop": sample_mae["semantic-drop"],
        "random_same_dim_mse_median": float(np.median(random_here)),
        "random_same_dim_mse_95pct": float(np.percentile(random_here, 95)),
        "sigma": float(sigma[0, 0, channel]),
        "mu": float(mu[0, 0, channel]),
        "gate": float(gate[0, 0, channel]),
        "history_mean": float(hist.mean()),
        "history_std": float(hist.std()),
        "hidden_reconstruction_gap": state["hidden_reconstruction_gap"],
        "original_arm_gap_vs_cached_fused": state["original_arm_gap_vs_cached_fused"],
        "figure": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir", default="research_runs/lowrank_checkpoint_information_v1"
    )
    parser.add_argument("--settings", default="", help="comma separated subset")
    parser.add_argument("--seeds", default="", help="comma separated subset")
    parser.add_argument(
        "--cell", default="q=1/8",
        help="which compression level the figures are drawn for",
    )
    parser.add_argument("--random-repeats", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = repo_root / args.output_dir
    features_dir = output_dir / "features"
    figures_dir = output_dir / "stage5_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows, _, _ = inventory_rows(repo_root, hash_checkpoints=False)
    formal = [row for row in rows if not row["is_diagnostic_only"]]
    wanted_settings = {item for item in args.settings.split(",") if item}
    wanted_seeds = {int(item) for item in args.seeds.split(",") if item}
    if wanted_settings:
        formal = [row for row in formal if row["setting"] in wanted_settings]
    if wanted_seeds:
        formal = [row for row in formal if int(row["seed"]) in wanted_seeds]
    formal = [row for row in formal if row["cell"] == args.cell]

    records: list[dict] = []
    for row in formal:
        setting = row["setting"]
        seed = int(row["seed"])
        feature_path = features_dir / f"{setting}_seed{seed}_{row['cell'].replace('/', '-')}.npz"
        if not feature_path.is_file():
            print(f"[skip] {setting} seed={seed} {row['cell']}: no cached features", flush=True)
            continue
        rng = np.random.default_rng(RANDOM_SEED)
        state = evaluate_sample_arms(
            setting, row["dataset"], int(row["horizon"]), seed, row["cell"],
            feature_path, args.random_repeats, rng,
        )
        print(
            f"[load] {setting} seed={seed} {row['cell']}: samples={state['samples']} "
            f"rank={state['rank']} hidden-gap={state['hidden_reconstruction_gap']:.2e}",
            flush=True,
        )
        for selection in select_samples(state):
            figure_path = figures_dir / f"{setting}_seed{seed}_{selection['kind']}.png"
            records.append(render_figure(state, selection, figure_path, args.dpi))
        del state

    table_path = figures_dir / "stage5_samples.csv"
    if records:
        fields: list[str] = []
        for record in records:
            for name in record:
                if name not in fields:
                    fields.append(name)
        with table_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    total_bytes = sum(
        path.stat().st_size for path in figures_dir.glob("*.png")
    )
    print(f"figures: {len(records)}  disk: {total_bytes / 1024 / 1024:.2f} MiB")
    print(f"table: {table_path}")


if __name__ == "__main__":
    main()
