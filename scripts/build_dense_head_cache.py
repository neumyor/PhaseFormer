#!/usr/bin/env python3
"""Build a validation feature cache for the *dense* (`l_main`) residual head.

The plan's section 10 compares the low-rank head's retained subspace against the
dense head it compresses.  The dense feature cache that comparison needs was
never written: E16 dissected the dense head but kept only summary tables, so the
canonical mode vectors and per-mode contributions do not exist on disk.

This script writes the dense cells in exactly the format
``evaluate_lowrank_semantic_interventions.py`` uses for the low-rank cells, so
``analyze_lowrank_functional_rank.py`` consumes them unchanged:

* ``encoder_weight = I_L`` and ``encoder_bias = 0``: the dense head's hidden
  state *is* its private centered input, so expressing it as an identity encoder
  makes the shared algebra exact rather than approximate;
* ``decoder_weight = linear.weight`` and ``decoder_bias = linear.bias``;
* ``x_last_norm`` is the head's own anchor (the low-rank caches store a deflated
  version, but with a zero encoder bias the recovery step is the identity).

Reads the validation split only.  No training, no checkpoint modification, and
the test split is never touched.
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
    load_checkpoint_into,
)


def resolve_cells(summary_path: Path, settings: set[str], seeds: set[int]) -> list[dict]:
    summary = json.loads(summary_path.read_text())
    cells = []
    for cell in summary["cells"]:
        if cell.get("arm") != "l_main":
            continue
        if settings and cell["dataset"] not in settings:
            continue
        if seeds and int(cell["seed"]) not in seeds:
            continue
        cells.append(cell)
    cells.sort(key=lambda c: (c["dataset"], int(c["horizon"]), int(c["seed"])))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="research_runs/phaseformer_L_e16_dissection_v1/e16_summary.json")
    parser.add_argument("--output-dir", default="research_runs/lowrank_functional_rank_v1/dense_features")
    parser.add_argument("--datasets", default="ETTh2,ETTm2,Weather")
    parser.add_argument("--seeds", default="2021,2022,2023")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    if not (repo_root / "src").is_dir():
        repo_root = Path.cwd().resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    settings = {value for value in args.datasets.split(",") if value}
    seeds = {int(value) for value in args.seeds.split(",") if value}
    cells = resolve_cells(repo_root / args.summary, settings, seeds)
    cells = [
        cell for index, cell in enumerate(cells)
        if index % args.shard_count == args.shard_index
    ]
    print(f"dense cells to cache: {len(cells)}  device={device}", flush=True)

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so the module's import cost is paid only when actually
    # instrumenting; ``e16_dissection`` also imports the training stack.
    sys.path.insert(0, str(repo_root / "scripts" / "phaseformer_L"))
    from e16_dissection import instrument_branch  # noqa: E402

    for position, cell in enumerate(cells, start=1):
        dataset, horizon, seed = cell["dataset"], int(cell["horizon"]), int(cell["seed"])
        run_dir = repo_root / cell["run_dir"]
        checkpoint = repo_root / cell["checkpoint"]
        output_path = output_dir / f"{dataset}-{horizon}_seed{seed}_dense.npz"
        if output_path.is_file():
            print(f"  [{position}/{len(cells)}] cached, skip {output_path.name}", flush=True)
            continue

        config = json.loads((run_dir / "config.json").read_text())
        hyperparams = dict(config["hyperparams"])
        batch_size = int(config.get("batch_size") or hyperparams.get("batch_size") or 256)
        started = time.time()

        exp_args, handles = build_loaders(
            dataset, 720, horizon, hyperparams, batch_size, repo_root, splits=("val",),
        )
        model = build_model(exp_args, 720, horizon, hyperparams)
        info = load_checkpoint_into(model, checkpoint)
        head = model.weak_period_residual
        if not hasattr(head, "linear"):
            raise RuntimeError(
                f"expected the dense shared head for {dataset}-{horizon} seed={seed}, "
                f"got {type(head).__name__}"
            )
        model.eval()
        model.to(device)
        _, val_loader = handles["val"]

        linear_weight = head.linear.weight.detach().double().cpu().numpy()
        linear_bias = head.linear.bias.detach().double().cpu().numpy()
        assert linear_weight.shape == (horizon, 720), linear_weight.shape

        chunks: dict[str, list] = {key: [] for key in (
            "z", "hidden", "phase", "mu", "sigma", "gate", "target", "fused", "x_last_norm",
        )}
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                if args.max_batches and batch_index >= args.max_batches:
                    break
                batch = [item.to(device) if torch.is_tensor(item) else item for item in batch]
                x, y, x_mark, y_mark = batch
                dec = model._build_decoder_input(y.float())
                with instrument_branch(model, "shared") as instrumented:
                    out, _, _ = instrumented(
                        x.float(), x_mark.float(), dec, y_mark.float()
                    )
                    records = instrumented.last_lowrank_records
                    mu, sigma = records["stats"]
                    gate = records["gate"]
                    gate_full = gate.reshape(1, 1, -1).expand(
                        x.shape[0], 1, gate.shape[-1]
                    )
                    chunks["z"].append(records["z"].permute(0, 2, 1).double().cpu().numpy())
                    chunks["hidden"].append(records["hidden"].double().cpu().numpy())
                    chunks["phase"].append(
                        instrumented.last_phase_forecast.double().cpu().numpy()
                    )
                    chunks["mu"].append(mu.double().cpu().numpy())
                    chunks["sigma"].append(sigma.double().cpu().numpy())
                    chunks["gate"].append(gate_full.double().cpu().numpy())
                    chunks["target"].append(
                        y.float()[:, -horizon:, :].double().cpu().numpy()
                    )
                    chunks["fused"].append(out.double().cpu().numpy())
                    chunks["x_last_norm"].append(
                        records["anchor64"].double().cpu().numpy()
                    )

        features = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
        samples = features["z"].shape[0]
        identity = np.eye(720, dtype=np.float64)
        np.savez(
            output_path,
            encoder_weight=identity,
            encoder_bias=np.zeros(720, dtype=np.float64),
            decoder_weight=linear_weight,
            decoder_bias=linear_bias,
            **features,
        )
        print(
            f"  [{position}/{len(cells)}] {dataset}-{horizon} seed={seed} "
            f"n={samples} missing={len(info['missing_keys'])} "
            f"({time.time() - started:.1f}s)", flush=True,
        )
        del model, val_loader


if __name__ == "__main__":
    main()
