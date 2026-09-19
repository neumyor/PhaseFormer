#!/usr/bin/env python3
"""E14 parameter accounting: the §4.2 parameter columns.

minipaper §4.2 requires the parameter count "按仓库 ``metrics.csv`` 的
``parameter_count`` 口径报告（主干 + 修正器 + 门），并单列修正器参数".

The total is already recorded per run by the runner, but the *corrector-only*
share is not, and it is the number the paper needs: the whole point is to show
how small the level corrector is relative to the phase backbone.  This script
recovers it exactly by reading each cell's checkpoint parameter shapes (no
weights are materialised: ``torch.load(..., mmap=True)`` on CPU), splitting them
into

* ``residual_params``  -- everything owned by the residual branch
  (``weak_period_residual.*``), which is the corrector plus its gate;
* ``backbone_params``  -- the remaining (phase trunk) parameters;
* ``total_params``     -- their sum, cross-checked against ``metrics.csv``.

FLOPs are deliberately not reported: minipaper §4.2 states the original Table 4
FLOP convention was not reproduced in this repository, so comparing FLOPs would
be a fabricated artefact.

Cells whose run directory cannot be resolved yet (E14 still training) are listed
under ``unresolved`` rather than silently skipped.

Usage::

    python scripts/phaseformer_L/e14_params.py \\
        --manifest research_runs/phaseformer_L_e14_main_v1/stage_a_manifest.json \\
        --output-root research_runs/phaseformer_L_e14_main_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phaseformer_L.e14_main_matrix import (  # noqa: E402
    DEFAULT_ARMS as ARM_ORDER,
    _arm_match,
)

CORRECTOR_PREFIXES = ("weak_period_residual.",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--e14-root", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cell_key(cell) -> str:
    return f"{cell['arm']}__{cell['dataset']}-{cell['horizon']}-s{cell['seed']}"


def find_run_dir(cell, e14_root: Path):
    """Resolve one cell's run directory (reused cells live outside e14_root)."""
    source = cell.get("source") or {}
    if cell.get("status") == "reused" and source.get("run_dir"):
        candidate = ROOT / str(source["run_dir"])
        if (candidate / "config.json").is_file():
            return candidate
    runs = e14_root / "runs"
    if runs.is_dir():
        for config_path in sorted(runs.glob("*/config.json")):
            try:
                config = json.loads(config_path.read_text())
            except json.JSONDecodeError:
                continue
            if (
                str(config.get("dataset")) == str(cell["dataset"])
                and int(config.get("horizon", -1)) == int(cell["horizon"])
                and int(config.get("seed", -1)) == int(cell["seed"])
                and _arm_match(config, str(cell["arm"]))
            ):
                return config_path.parent
    return None


def read_metrics(run_dir: Path):
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle), None)


def checkpoint_of(run_dir: Path, record) -> Path | None:
    recorded = str((record or {}).get("checkpoint", "")).strip()
    if recorded:
        for candidate in (ROOT / recorded, run_dir / recorded):
            if candidate.is_file():
                return candidate
    for path in sorted(run_dir.glob("attempts/*/checkpoints/*.ckpt")):
        return path
    return None


def count_params(checkpoint: Path):
    """Split a checkpoint's parameter count into corrector vs backbone."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state = payload.get("state_dict", payload)
    residual = 0
    total = 0
    detail: dict[str, int] = {}
    for name, tensor in state.items():
        if not hasattr(tensor, "numel"):
            continue
        n = int(tensor.numel())
        total += n
        if name.startswith(CORRECTOR_PREFIXES):
            residual += n
            group = name.split(".")[1] if "." in name else name
            detail[group] = detail.get(group, 0) + n
    return residual, total, detail


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    e14_root = ROOT / (args.e14_root or Path(args.manifest).parent.name)
    if not (e14_root / "stage_a_manifest.json").is_file():
        e14_root = Path(args.manifest).parent
        if not e14_root.is_absolute():
            e14_root = ROOT / e14_root

    rows, unresolved = [], []
    for cell in manifest.get("cells", []):
        if str(cell.get("arm")) not in ARM_ORDER:
            continue
        run_dir = find_run_dir(cell, e14_root)
        if run_dir is None:
            unresolved.append(cell_key(cell))
            continue
        record = read_metrics(run_dir)
        if record is None:
            unresolved.append(cell_key(cell) + " (no metrics.csv)")
            continue
        checkpoint = checkpoint_of(run_dir, record)
        if checkpoint is None:
            unresolved.append(cell_key(cell) + " (no checkpoint)")
            continue
        residual, total, detail = count_params(checkpoint)
        recorded_total = str(record.get("parameter_count", "")).strip()
        rows.append({
            "arm": cell["arm"],
            "dataset": cell["dataset"],
            "horizon": int(cell["horizon"]),
            "seed": int(cell["seed"]),
            "setting": f"{cell['dataset']}-{cell['horizon']}",
            "status": cell.get("status"),
            "total_params": total,
            "residual_params": residual,
            "backbone_params": total - residual,
            "residual_share": round(residual / total, 6) if total else None,
            "residual_detail": json.dumps(detail, sort_keys=True),
            "metrics_parameter_count": int(recorded_total) if recorded_total else None,
            "total_matches_metrics": (
                int(recorded_total) == total if recorded_total else None
            ),
            "parameter_count_source": "checkpoint state_dict, mmap read (keys/shapes only)",
            "flops_reported": False,
            "flops_note": "minipaper 4.2: the original Table 4 FLOP convention is not "
                          "reproduced in this repository, so FLOPs are not compared",
        })

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else [
        "arm", "dataset", "horizon", "seed", "setting", "status", "total_params",
        "residual_params", "backbone_params", "residual_share", "residual_detail",
        "metrics_parameter_count", "total_matches_metrics",
        "parameter_count_source", "flops_reported", "flops_note"]
    if not args.dry_run:
        with (out_root / "parameter_table.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    mismatches = [r["setting"] + "/" + r["arm"] for r in rows
                  if r["total_matches_metrics"] is False]
    print(json.dumps({
        "event": "finished",
        "cells_with_parameters": len(rows),
        "unresolved": len(unresolved),
        "total_mismatches": mismatches,
        "wrote": None if args.dry_run else str(out_root / "parameter_table.csv"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
