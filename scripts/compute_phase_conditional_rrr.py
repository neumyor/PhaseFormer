#!/usr/bin/env python3
"""Stage 3 of the low-rank checkpoint information analysis.

Fits the **Phase-conditional target** of the plan's section 5 on the train split
and compares three input subspaces per checkpoint:

1. the subspace the checkpoint actually learned -- the row space of its
   effective map ``W = decoder @ encoder``, which is exactly the set of input
   directions the frozen branch can read;
2. the top-``r`` input directions of the *independent* reduced-rank regression on
   ``y - x_last`` (the target used by the earlier frozen-direction experiments);
3. the top-``r`` input directions of the *conditional* reduced-rank regression,
   fitted on the target the branch must actually supply once the Phase backbone
   and the learned gate are fixed:

       y_t = (y - mu)/sigma - (1-g) * p_norm - g * x_last_norm
       w   = g^2

   with ``mu``/``sigma`` the exact RevIN statistics of the forward pass, so the
   RevIN affine map is inside the weighted least squares instead of being
   divided out.

The comparison is a projection overlap in ``[0, 1]`` normalized by rank, together
with the weighted residual energy each subspace can explain.  A stable higher
alignment with the conditional target supports H1.

Reads the train split only; no model is trained and no checkpoint is modified.
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
    independent_rrr,
    projection_overlap,
    weighted_rrr_subspace,
)
from scripts.lowrank_checkpoint_inventory import inventory_rows  # noqa: E402
from scripts.lowrank_checkpoint_model import (  # noqa: E402
    build_loaders,
    build_model,
    instrument_model,
    set_seed,
)

RIDGE = 1e-6
# Row block for the Gram-matrix accumulation.  The full design on
# Electricity-336 is ~5.6e6 rows x 720 float64 (32 GiB); accumulating in blocks
# keeps the transients small without changing the (associative) sums.
ACCUMULATION_BLOCK = 131_072


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir", default="research_runs/lowrank_checkpoint_information_v1"
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0, help="override the RRR rank")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    subspace_dir = output_dir / "subspaces"
    subspace_dir.mkdir(exist_ok=True)
    moments_dir = output_dir / "train_moments"
    moments_dir.mkdir(exist_ok=True)

    gpu = int(args.gpus.split(",")[0]) if args.gpus else 0
    if not torch.cuda.is_available():
        raise RuntimeError("this stage requires CUDA on the analysis server")
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    print(f"device: {device}", flush=True)

    rows, _, _ = inventory_rows(repo_root, hash_checkpoints=False)
    formal = [row for row in rows if not row["is_diagnostic_only"]]
    wanted_settings = {item for item in args.settings.split(",") if item}
    wanted_seeds = {int(item) for item in args.seeds.split(",") if item}
    if wanted_settings:
        formal = [row for row in formal if row["setting"] in wanted_settings]
    if wanted_seeds:
        formal = [row for row in formal if int(row["seed"]) in wanted_seeds]

    groups: dict[tuple[str, int, int], list[dict]] = {}
    for row in formal:
        groups.setdefault(
            (row["setting"], int(row["seed"]), int(row["rank"])), []
        ).append(row)
    print(f"train-fit groups: {len(groups)}", flush=True)

    alignment_rows: list[dict] = []
    started = time.time()

    for (setting, seed, rank) in sorted(groups):
        entries = groups[(setting, seed, rank)]
        dataset = entries[0]["dataset"]
        horizon = int(entries[0]["horizon"])
        run_dir = repo_root / entries[0]["selected_run_dir"]
        config = json.loads((run_dir / "config.json").read_text())
        hyperparams = dict(config["hyperparams"])
        batch_size = int(config.get("batch_size") or hyperparams.get("batch_size") or 256)
        exp_args, handles = build_loaders(
            dataset, 720, horizon, hyperparams, batch_size, repo_root,
            splits=("train",),
        )
        train_loader = handles["train"][1]
        model = build_model(exp_args, 720, horizon, hyperparams)

        accumulators = {
            "zz": np.zeros((720, 720)),
            "zy": np.zeros((720, horizon)),
            "yy": np.zeros((horizon, horizon)),
            "wzz": np.zeros((720, 720)),
            "wzy": np.zeros((720, horizon)),
            "wyy": np.zeros((horizon, horizon)),
            "count": 0,
        }
        per_checkpoint: dict[str, dict] = {}

        for entry in entries:
            model.load_state_dict(
                torch.load(
                    repo_root / entry["checkpoint_path"],
                    map_location="cpu",
                    weights_only=False,
                )["state_dict"],
                strict=True,
            )
            model.to(device).eval()
            set_seed(25601)
            collection = {
                "z": [], "delta": [], "target": [], "weight": [],
            }
            with torch.inference_mode():
                for batch_index, batch in enumerate(train_loader):
                    if args.max_batches and batch_index >= args.max_batches:
                        break
                    batch = [
                        item.to(device) if torch.is_tensor(item) else item
                        for item in batch
                    ]
                    x, y, x_mark, y_mark = batch
                    dec = model._build_decoder_input(y.float())
                    with instrument_model(model) as instrumented:
                        instrumented(
                            x.float(), x_mark.float(), dec, y_mark.float()
                        )
                        records = instrumented.last_lowrank_records
                    mu, sigma = records["stats"]
                    centered = records["z"]
                    hidden = records["hidden"]
                    # The conditional target is what the branch still has to
                    # supply after the Phase backbone and the gate are fixed.
                    target_norm = (y.float() - mu) / sigma
                    phase_norm = records["phase_norm"]
                    anchor = centered[:, :, -1:].permute(0, 2, 1)
                    gate = records["gate"].reshape(1, 1, -1)
                    conditional_target = (
                        target_norm - (1.0 - gate) * phase_norm - gate * anchor
                    )
                    independent_target = target_norm - anchor
                    collection["z"].append(
                        centered.permute(0, 2, 1).double().cpu().numpy()
                    )
                    collection["delta"].append(
                        hidden.permute(0, 2, 1).double().cpu().numpy()
                    )
                    collection["target"].append(
                        conditional_target.double().cpu().numpy()
                    )
                    collection["weight"].append(
                        (gate ** 2).expand(
                            x.shape[0], conditional_target.shape[1], gate.shape[-1]
                        ).double().cpu().numpy()
                    )
                    del independent_target
            stacked = {
                key: np.concatenate(value, axis=0).reshape(-1, value[0].shape[-1])
                for key, value in collection.items()
            }
            del collection
            # ``z`` is ``(N, C, lookback)`` once concatenated, but the RRR design
            # is per (sample, channel) row with the *lookback* dimension as the
            # regressor, so its last axis has to become the 720-wide lookback
            # axis rather than the channel axis that ``shape[-1]`` produced.
            z_flat = stacked["z"].reshape(-1, 720)
            delta_flat = stacked["delta"]
            target_flat = stacked["target"].reshape(-1, horizon)
            weight_flat = stacked["weight"].reshape(-1, horizon)[:, 0]
            del stacked
            # The Gram accumulators are sums over training rows, so they are
            # evaluated in row blocks instead of as one product of the full
            # ``(rows, 720)`` design.  On Electricity-336 the full design is
            # 5.6e6 x 720 float64 (32 GiB), and materialising ``weighted_z``
            # beside it exhausted memory; blockwise accumulation is exact
            # (each block contributes its own partial sum) and bounds the
            # transient to one block.
            rows_total = z_flat.shape[0]
            residual_square_sum = 0.0
            for start in range(0, rows_total, ACCUMULATION_BLOCK):
                stop = min(start + ACCUMULATION_BLOCK, rows_total)
                z_block = np.ascontiguousarray(z_flat[start:stop])
                target_block = np.ascontiguousarray(target_flat[start:stop])
                weight_block = np.ascontiguousarray(weight_flat[start:stop])
                accumulators["zz"] += z_block.T @ z_block
                accumulators["zy"] += z_block.T @ target_block
                accumulators["yy"] += target_block.T @ target_block
                accumulators["wzz"] += (z_block * weight_block[:, None]).T @ z_block
                accumulators["wzy"] += (z_block * weight_block[:, None]).T @ target_block
                accumulators["wyy"] += (
                    target_block * weight_block[:, None]
                ).T @ target_block
                accumulators["count"] += z_block.shape[0]
                if delta_flat is not None:
                    delta_block = delta_flat[start:stop]
                    residual_square_sum += float(
                        np.einsum("ij,ij->", delta_block, delta_block)
                    )
                    del delta_block
                del z_block, target_block, weight_block
            # ``residual_trace`` is ``np.mean(delta**2)``, i.e. normalised by the
            # *element* count (rows x latent rank), not by the row count.
            residual_trace = residual_square_sum / max(delta_flat.size, 1)

            encoder_weight = model.weak_period_residual.encoder.weight.detach()
            decoder_weight = model.weak_period_residual.decoder.weight.detach()
            effective = (
                decoder_weight.double().cpu().numpy()
                @ encoder_weight.double().cpu().numpy()
            )
            effective_rank = int(
                entries[0]["rank"] if args.rank == 0 else args.rank
            )
            _, singular, vt = np.linalg.svd(effective, full_matrices=False)
            learned_basis = np.ascontiguousarray(vt[:effective_rank].T)
            per_checkpoint[entry["cell"]] = {
                "learned_basis": learned_basis,
                "effective_singular_values": singular,
                "decoder_weight": decoder_weight.double().cpu().numpy(),
                "encoder_weight": encoder_weight.double().cpu().numpy(),
                "residual_trace": float(residual_trace),
                "sample_count": int(z_flat.shape[0]),
            }
            print(
                f"[train] {setting} seed={seed} rank={rank} cell={entry['cell']} "
                f"pairs={z_flat.shape[0]}",
                flush=True,
            )
        del model, train_loader, exp_args, handles

        count = max(accumulators["count"], 1)
        szz = accumulators["zz"] / count
        szy = accumulators["zy"] / count
        syy = accumulators["yy"] / count
        m_zz = accumulators["wzz"] / count
        m_zy = accumulators["wzy"] / count
        m_yy = accumulators["wyy"] / count
        np.savez_compressed(
            moments_dir / f"{setting}.npz", covariance=szz, count=count
        )

        independent_basis = independent_rrr(szz, szy, rank, ridge=RIDGE)[0]
        conditional_basis, eigenvalues = weighted_rrr_subspace(m_zz, m_zy, rank)
        conditional_energy = float(np.trace(m_yy))

        for cell, payload in sorted(per_checkpoint.items()):
            learned = payload["learned_basis"]
            # Both RRR bases live in the 720-dimensional input space and have
            # the same rank as the learned row space, so the overlap is
            # dimension matched without any truncation.
            overlap_independent = projection_overlap(learned, independent_basis)
            overlap_conditional = projection_overlap(learned, conditional_basis)
            alignment_rows.append(
                {
                    "setting": setting,
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "cell": cell,
                    "rank": learned.shape[1],
                    "overlap_with_independent_rrr": overlap_independent,
                    "overlap_with_conditional_rrr": overlap_conditional,
                    "overlap_difference": overlap_conditional - overlap_independent,
                    "supports_h1": bool(overlap_conditional > overlap_independent),
                    "conditional_eigenvalue_leading_share": float(
                        eigenvalues[0] / max(eigenvalues.sum(), 1e-30)
                    ),
                    "conditional_weighted_target_energy": conditional_energy,
                    "independent_ridge": RIDGE,
                    "train_pairs": int(count),
                    "gate_mean": float(
                        np.mean(
                            np.sqrt(
                                np.clip(
                                    np.diag(m_zz) / np.maximum(np.diag(szz), 1e-30),
                                    0.0,
                                    None,
                                )
                            )
                        )
                    ),
                    "checkpoint_path": next(
                        entry["checkpoint_path"]
                        for entry in entries
                        if entry["cell"] == cell
                    ),
                }
            )
            np.savez_compressed(
                subspace_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz",
                conditional_basis=conditional_basis,
                independent_basis=independent_basis,
                learned_basis=learned,
                decoder_weight=payload["decoder_weight"],
                encoder_weight=payload["encoder_weight"],
            )
        print(
            f"[fit] {setting} seed={seed} rank={rank}: independent overlap "
            f"{np.mean([row['overlap_with_independent_rrr'] for row in alignment_rows[-len(per_checkpoint):]]):.4f} "
            f"conditional overlap "
            f"{np.mean([row['overlap_with_conditional_rrr'] for row in alignment_rows[-len(per_checkpoint):]]):.4f}",
            flush=True,
        )

    write_csv(alignment_rows, output_dir / "conditional_rrr_alignment.csv")
    print(f"alignment rows: {len(alignment_rows)}")
    print(f"elapsed {time.time() - started:.1f}s")


def write_csv(rows: list[dict], path: Path) -> None:
    """Merge ``rows`` into ``path``, replacing rows with the same checkpoint key.

    Stage 3 is run one setting at a time (each setting is a separate GPU job), so
    a plain overwrite would leave only the last setting in the table.  The rows
    are keyed by ``(setting, seed, cell)``; a re-run of one setting refreshes only
    its own rows and the other settings are preserved.
    """
    if not rows:
        return
    key = lambda row: (  # noqa: E731
        row.get("setting", ""),
        str(row.get("seed", "")),
        row.get("cell", ""),
    )
    merged: dict[tuple, dict] = {}
    if path.is_file():
        with path.open(newline="") as handle:
            for existing in csv.DictReader(handle):
                merged[key(existing)] = existing
    for row in rows:
        merged[key(row)] = row
    ordered = [merged[k] for k in sorted(merged, key=str)]
    fields: list[str] = []
    for row in ordered:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)


if __name__ == "__main__":
    main()
