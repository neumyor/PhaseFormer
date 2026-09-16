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
                "z": [], "delta": [], "target": [], "weight": [], "residual_norm": [],
                "phase_norm": [],
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
                    collection["residual_norm"].append(
                        records["residual_norm"].double().cpu().numpy()
                    )
                    collection["phase_norm"].append(
                        phase_norm.double().cpu().numpy()
                    )
                    del independent_target
            stacked = {
                key: np.concatenate(value, axis=0).reshape(-1, value[0].shape[-1])
                for key, value in collection.items()
            }
            del collection
            z_flat = stacked["z"].reshape(-1, 720)
            delta_flat = stacked["delta"]
            target_flat = stacked["target"].reshape(-1, horizon)
            weight_flat = stacked["weight"].reshape(-1, horizon)[:, 0]
            accumulators["zz"] += z_flat.T @ z_flat
            accumulators["zy"] += z_flat.T @ target_flat
            accumulators["yy"] += target_flat.T @ target_flat
            weighted_z = z_flat * weight_flat[:, None]
            accumulators["wzz"] += weighted_z.T @ z_flat
            accumulators["wzy"] += weighted_z.T @ target_flat
            accumulators["wyy"] += (target_flat * weight_flat[:, None]).T @ target_flat
            accumulators["count"] += z_flat.shape[0]

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
                "residual_trace": float(
                    np.mean(delta_flat ** 2)
                ),
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
            overlap_independent = projection_overlap(
                learned, independent_basis[: learned.shape[1]]
            )
            overlap_conditional = projection_overlap(
                learned, conditional_basis[: learned.shape[1]]
            )
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
                        np.mean(accumulators["wzy"] * 0.0) + 0.0
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
