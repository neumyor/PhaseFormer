#!/usr/bin/env python3
"""Audit reuse of the existing ``direct_nlinear`` control runs (plan section 6).

The plan allows reusing the already trained ``direct_nlinear`` arm only when
setting, seed, split, training configuration, code semantics and metric
implementation all match.  This script finds those runs across the existing
experiment roots, checks each against the frozen protocol of this plan, and
writes an auditable reuse manifest.  It trains nothing.

Usage::

    python scripts/audit_top2_direction_retention_reuse.py \
        --root research_runs/top2_direction_retention_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Search roots that may contain a reusable direct_nlinear control.
SEARCH_ROOTS = (
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_v4",
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_repair_v1",
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_summary",
    "research_runs/rank_sweep_2_stage1",
    "research_runs/joint_lowrank_rank_sweep_v1",
    "research_runs/structured_lowrank_round1_scratch",
    "research_runs/structured_lowrank_round1_scratch_INVALID_defaults",
)

SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
)
SEEDS = (2021, 2022, 2023)

FROZEN = {
    ("ETTh2", 96): {"gate": 0.5, "lr": 0.001},
    ("ETTh2", 720): {"gate": 0.5, "lr": 0.001},
    ("ETTm2", 96): {"gate": 0.5, "lr": 0.0003},
    ("ETTm2", 192): {"gate": 0.2, "lr": 0.001},
    ("Weather", 96): {"gate": 0.2, "lr": 0.0003},
    ("Weather", 192): {"gate": 0.5, "lr": 0.001},
}

# Any run under these roots was produced by a protocol known to be invalid.
INVALID_MARKERS = ("INVALID",)


def protocol_checks(config, record, dataset, horizon):
    """Return (passed, failures) for one candidate run."""
    hyper = config.get("hyperparams", {})
    frozen = FROZEN[(dataset, horizon)]
    failures = []
    if config.get("mechanism") != "weak_residual":
        failures.append(f"mechanism={config.get('mechanism')!r} != 'weak_residual'")
    if hyper.get("weak_period_residual_head_type") != "shared":
        failures.append(
            "weak_period_residual_head_type="
            f"{hyper.get('weak_period_residual_head_type')!r} != 'shared'"
        )
    if int(config.get("lookback", -1)) != 720:
        failures.append(f"lookback={config.get('lookback')} != 720")
    if int(config.get("period", -1)) != 24:
        failures.append(f"period={config.get('period')} != 24")
    if int(config.get("max_epochs", -1)) != 30:
        failures.append(f"max_epochs={config.get('max_epochs')} != 30")
    if config.get("loss") != "huber":
        failures.append(f"loss={config.get('loss')!r} != 'huber'")
    if int(config.get("percent", -1)) != 100:
        failures.append(f"percent={config.get('percent')} != 100")
    if abs(float(hyper.get("learning_rate", -1)) - frozen["lr"]) > 1e-12:
        failures.append(
            f"learning_rate={hyper.get('learning_rate')} != {frozen['lr']}"
        )
    if (
        abs(
            float(hyper.get("weak_period_residual_gate_init", -1))
            - frozen["gate"]
        )
        > 1e-12
    ):
        failures.append(
            "weak_period_residual_gate_init="
            f"{hyper.get('weak_period_residual_gate_init')} != {frozen['gate']}"
        )
    if float(hyper.get("weak_period_residual_smooth_ratio", 0.0)) != 0.0:
        failures.append("smooth_ratio != 0")
    if not record.get("checkpoint"):
        failures.append("no best-validation checkpoint recorded")
    if not record.get("test_mse") or not record.get("test_mae"):
        failures.append("missing test metrics")
    if record.get("dataset") != dataset:
        failures.append(f"dataset={record.get('dataset')} != {dataset}")
    if int(record.get("horizon", -1)) != horizon:
        failures.append(f"horizon={record.get('horizon')} != {horizon}")
    return (not failures), failures


def candidate_runs(repo_root: Path):
    for root in SEARCH_ROOTS:
        base = repo_root / root
        if not base.exists():
            continue
        invalid = any(marker in root for marker in INVALID_MARKERS)
        for metrics in sorted(base.rglob("metrics.csv")):
            config_path = metrics.with_name("config.json")
            if not config_path.exists():
                continue
            yield root, metrics, config_path, invalid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    output_root = Path(args.root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    accepted: dict[tuple, dict] = {}
    rejected: list[dict] = []
    for root, metrics, config_path, invalid in candidate_runs(ROOT):
        with metrics.open(newline="") as handle:
            record = next(csv.DictReader(handle), None)
        if record is None:
            continue
        try:
            dataset = record["dataset"]
            horizon = int(record["horizon"])
            seed = int(record["seed"])
        except (KeyError, ValueError):
            continue
        if (dataset, horizon) not in FROZEN or seed not in SEEDS:
            continue
        config = json.loads(config_path.read_text())
        hyper = config.get("hyperparams", {})
        if hyper.get("weak_period_residual_head_type") != "shared":
            continue
        if config.get("mechanism") != "weak_residual":
            continue
        passed, failures = protocol_checks(config, record, dataset, horizon)
        entry = {
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "run_dir": str(metrics.parent.relative_to(ROOT)),
            "root": root,
            "from_invalidated_batch": invalid,
            "test_mse": record.get("test_mse"),
            "test_mae": record.get("test_mae"),
            "val_mse": record.get("val_mse"),
            "val_mae": record.get("val_mae"),
            "elapsed_sec": record.get("elapsed_sec"),
            "parameter_count": record.get("parameter_count"),
            "checkpoint": record.get("checkpoint"),
            "config_hash": record.get("config_hash"),
            "failures": failures,
        }
        if invalid:
            entry["failures"] = failures + ["root is an invalidated batch"]
            passed = False
        if passed:
            key = (dataset, horizon, seed)
            # Prefer the run from the audited v4 root when duplicates exist.
            if key not in accepted or "v4" in root:
                accepted[key] = entry
        else:
            rejected.append(entry)

    rows = []
    missing = []
    for dataset, horizon in SETTINGS:
        for seed in SEEDS:
            entry = accepted.get((dataset, horizon, seed))
            if entry is None:
                missing.append(f"{dataset}-{horizon}-s{seed}")
                continue
            rows.append(entry)

    manifest = {
        "protocol": "top2-direction-retention-reuse-audit-v1",
        "criteria": [
            "mechanism=weak_residual, weak_period_residual_head_type=shared",
            "lookback=720, period=24, huber loss, 30 epochs, percent=100",
            "setting-frozen (gate_init, learning_rate) exactly matched",
            "smooth_ratio=0, best-validation checkpoint recorded",
            "test metrics present in the original run",
            "run not from an invalidated batch",
        ],
        "accepted": len(rows),
        "missing": missing,
        "rejected_candidates": len(rejected),
        "runs": rows,
    }
    (output_root / "reuse_audit.json").write_text(
        json.dumps({**manifest, "rejected": rejected}, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# direct_nlinear 复用审计",
        "",
        f"- 通过并复用的 run：**{len(rows)}**",
        f"- 缺失的格子：**{len(missing)}**" + (f"（{', '.join(missing)}）" if missing else ""),
        f"- 被拒绝的候选：**{len(rejected)}**",
        "",
        "## 复用清单",
        "",
        "| Setting | seed | test MSE | test MAE | 来源 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']}-{row['horizon']} | {row['seed']} | "
            f"{float(row['test_mse']):.4f} | {float(row['test_mae']):.4f} | "
            f"`{row['run_dir']}` |"
        )
    lines.append("")
    if rejected:
        lines.append("## 被拒绝的候选（前 20 条）")
        lines.append("")
        lines.append("| Setting | seed | 原因 |")
        lines.append("|---|---:|---|")
        for row in rejected[:20]:
            lines.append(
                f"| {row['dataset']}-{row['horizon']} | {row['seed']} | "
                f"{'; '.join(row['failures'])} |"
            )
        lines.append("")
    (output_root / "reuse_audit.md").write_text("\n".join(lines))
    print(json.dumps({"accepted": len(rows), "missing": missing,
                      "rejected": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
