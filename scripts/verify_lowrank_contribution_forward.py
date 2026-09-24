#!/usr/bin/env python3
"""Section 8.3 of the plan: verify the analytic mode contributions against real
forward passes of the frozen model.

The whole functional-rank analysis rests on one identity: removing a subset of
canonical modes changes the fused MSE by exactly the sum of their ``I_i``.  That
follows from the modes' output directions being orthonormal, but an identity
that is only ever checked against itself proves nothing, so this script patches
the residual head so that an arbitrary subset of modes is deleted *inside the
model's own forward pass* and compares the measured MSE with the predicted one.

The deletion is exact rather than approximate: a mode is removed by subtracting
``(dw^+ u_i) * s_i * (v_i^T z)`` from the hidden state, so the decoder writes
the same correction minus that mode and nothing else changes.

Reads the validation split only.  No training, no checkpoint modification, no
test read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lowrank_checkpoint_model import (  # noqa: E402
    build_loaders,
    build_model,
    instrument_model,
    load_checkpoint_into,
)
from analyze_lowrank_functional_rank import analyse_cell, canonical_modes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="research_runs/lowrank_checkpoint_information_v1/checkpoint_inventory.csv")
    parser.add_argument("--features-dir", default="research_runs/lowrank_checkpoint_information_v1/features")
    parser.add_argument("--output-dir", default="research_runs/lowrank_functional_rank_v1")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="2021")
    parser.add_argument("--cells", default="q=1/4,q=1/8")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    inventory = list(csv_dict_rows(repo_root / args.inventory))
    wanted_settings = {v for v in args.settings.split(",") if v}
    wanted_seeds = {int(v) for v in args.seeds.split(",") if v}
    wanted_cells = {v for v in args.cells.split(",") if v}
    rows = [
        row for row in inventory
        if row["is_diagnostic_only"] == "False"
        and row["dataset"] != "Electricity"
        and (not wanted_settings or row["setting"] in wanted_settings)
        and (not wanted_seeds or int(row["seed"]) in wanted_seeds)
        and (not wanted_cells or row["cell"] in wanted_cells)
    ]
    rows.sort(key=lambda row: (row["setting"], int(row["seed"]), row["cell"]))
    rows = [r for index, r in enumerate(rows) if index % args.shard_count == args.shard_index]

    results: list[dict] = []
    for row in rows:
        setting, dataset = row["setting"], row["dataset"]
        horizon, seed = int(row["horizon"]), int(row["seed"])
        cell = row["cell"]
        cache_path = repo_root / args.features_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz"
        if not cache_path.is_file():
            continue
        payload = {k: v.astype(np.float64) for k, v in np.load(cache_path).items()}
        _, _, _, _, modes = analyse_cell(
            payload, setting, dataset, horizon, seed, cell, int(row["rank"]),
            random_repeats=1, rng=np.random.default_rng(0), top_modes=1,
        )
        contribution = modes["contribution"]
        # Analytic baseline in the same algebra as analyze_lowrank_functional_rank.
        mapped_eb = payload["decoder_weight"] @ payload["encoder_bias"]
        last_abs = payload["sigma"] * (
            payload["x_last_norm"] + float(mapped_eb.mean())
        ) + payload["mu"]
        branch_0 = last_abs + payload["sigma"] * (
            mapped_eb + payload["decoder_bias"]
        )[None, :, None]
        fused_0 = (1.0 - payload["gate"]) * payload["phase"] + payload["gate"] * branch_0
        mse_0 = float(np.mean((payload["target"] - fused_0) ** 2))
        del branch_0, fused_0

        run_dir = repo_root / row["selected_run_dir"]
        config = json.loads((run_dir / "config.json").read_text())
        hyperparams = dict(config["hyperparams"])
        batch_size = int(config.get("batch_size") or hyperparams.get("batch_size") or 256)
        exp_args, handles = build_loaders(
            dataset, 720, horizon, hyperparams, batch_size, repo_root, splits=("val",),
        )
        model = build_model(exp_args, 720, horizon, hyperparams)
        load_checkpoint_into(model, repo_root / row["checkpoint_path"])
        model.eval()
        model.to(device)
        _, val_loader = handles["val"]

        head = model.weak_period_residual
        dw = head.decoder.weight.detach().double().cpu().numpy()
        ew = head.encoder.weight.detach().double().cpu().numpy()
        u, s, vt = canonical_modes(dw @ ew)
        rank = int(min(payload["hidden"].shape[-1], s.size))
        u, s, vt = u[:, :rank], s[:rank], vt[:rank]
        pinv = np.linalg.pinv(dw)                       # (r, H)
        images = (pinv @ u)                             # (r, r), one column per mode

        order = np.argsort(-contribution, kind="stable")
        subsets = {
            "zero": np.arange(rank),
            "drop_top1": order[:1],
            "drop_top3": order[:3],
            "keep_top1": np.setdiff1d(np.arange(rank), order[:1]),
            "drop_negative": np.nonzero(contribution < 0)[0],
        }

        def relative_error(predicted: float, measured: float) -> float:
            scale = max(abs(predicted), abs(measured), 1e-12)
            return abs(predicted - measured) / scale

        for name, dropped in subsets.items():
            dropped = np.asarray(dropped, dtype=int)
            measured = measure_mse(
                model, val_loader, device, dropped, s, vt, images, horizon,
            )
            if name == "zero":
                predicted = mse_0
            else:
                predicted = mse_0 - float(contribution.sum()) + float(contribution[dropped].sum())
            results.append({
                "setting": setting, "dataset": dataset, "horizon": horizon,
                "seed": seed, "cell": cell, "rank": rank, "subset": name,
                "n_dropped": int(dropped.size),
                "predicted_fused_mse": predicted,
                "measured_fused_mse": measured,
                "relative_error": relative_error(predicted, measured),
            })
            print(
                f"  {setting} seed={seed} {cell} {name:14s} "
                f"pred={predicted:.8f} meas={measured:.8f} "
                f"rel={relative_error(predicted, measured):.2e}", flush=True,
            )
        del model, val_loader

    path = repo_root / args.output_dir / "contribution_forward_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        import csv as csv_module
        with path.open("w", newline="") as handle:
            writer = csv_module.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)


def csv_dict_rows(path: Path):
    import csv as csv_module
    with path.open(newline="") as handle:
        yield from csv_module.DictReader(handle)


def measure_mse(model, loader, device, dropped, s, vt, images, horizon) -> float:
    """Fused MSE with the modes in ``dropped`` deleted inside the forward pass."""
    dropped = np.asarray(dropped, dtype=int)
    scaling = torch.as_tensor(
        (s[dropped][:, None] * vt[dropped]), dtype=torch.float64, device=device,
    )                                                # (k, L)
    image = torch.as_tensor(images[:, dropped], dtype=torch.float64, device=device)

    def intervention(centered, hidden):
        # centered: (B, C, L); hidden: (B, C, r)
        activations = torch.einsum("bcl,kl->bck", centered.double(), scaling)  # (B,C,k)
        removal = torch.einsum("bck,rk->bcr", activations, image)              # (B,C,r)
        return (hidden.double() - removal).to(hidden.dtype)

    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = [t.to(device) if torch.is_tensor(t) else t for t in batch]
            x, y, x_mark, y_mark = batch
            dec = model._build_decoder_input(y.float())
            with instrument_model(model, intervention) as instrumented:
                out, _, _ = instrumented(x.float(), x_mark.float(), dec, y_mark.float())
            error = (out.double() - y.float()[:, -horizon:, :].double())
            total += float((error ** 2).sum())
            count += error.numel()
    return total / count


if __name__ == "__main__":
    main()
