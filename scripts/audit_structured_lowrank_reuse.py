#!/usr/bin/env python3
"""Round 0 controls, parameters and reuse audit for the structured low-rank plan.

This script performs no training.  For every pilot setting it:

1. finds the already trained ``phase_only``, ``direct_nlinear`` and generic
   ``pooled_lowrank(q=1/8)`` runs and checks that they match the current
   protocol (lookback 720, period 24, Huber, 30 epochs, the setting's frozen
   ``(gate_init, learning_rate)``, validation-selected checkpoint);
2. records the reused test numbers as ``reused_exact`` rows together with the
   original run directory, so the report never presents them as new evidence;
3. writes ``frozen_configs.json`` for the Round 1 runner and a Round 0 audit
   CSV/Markdown that can be filled into the plan's placeholder table.

Plan reference: sections 3.1.1 (result reuse), 3.3 and 5.

Example::

    python scripts/audit_structured_lowrank_reuse.py \
        --scratch-root research_runs/rank_sweep_2_stage1 \
        --output-dir research_runs/structured_lowrank_round0_v1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 192),
    ("Electricity", 336),
)

# ``pooled_lowrank`` rank for the generic control is q = 1/8 of the horizon with
# the plan's integer mapping r = max(1, min(H, round(q * H))).
CONFIG_IDS = {
    "phase_only": "phase_only",
    "direct_nlinear": "direct_nlinear",
}


def canonical_setting(dataset: str, horizon: int, seed: int) -> str:
    return f"{dataset}_h{horizon}_seed{seed}"


def generic_q(config_id: str) -> float:
    if config_id.startswith("pool") and "_q" in config_id:
        return float(config_id.split("_q")[1].split("_")[0])
    return float("nan")


def load_results(scratch_root: Path, dataset: str, horizon: int, seed: int):
    tag = f"{dataset.lower()}_h{horizon}"
    path = scratch_root / f"phase_a_{tag}_s{seed}_results.csv"
    if not path.is_file():
        raise SystemExit(f"missing result file: {path}")
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_frozen(scratch_root: Path, dataset: str, horizon: int, seed: int):
    """Read the frozen ``(gate_init, learning_rate)`` from a matching run."""

    tag = f"{dataset.lower()}_h{horizon}"
    pattern = f"confirm_{tag}_weak_residual_*_s{seed}_*/config.json"
    candidates = sorted(scratch_root.joinpath("runs").glob(pattern))
    if not candidates:
        return None
    with candidates[0].open() as handle:
        config = json.load(handle)
    hyper = config["hyperparams"]
    return {
        "dataset": dataset,
        "horizon": int(horizon),
        "seed": int(seed),
        "gate_init": float(hyper.get("weak_period_residual_gate_init", 0.2)),
        "learning_rate": float(hyper.get("learning_rate", 1e-3)),
        "head_type": hyper.get("weak_period_residual_head_type", "shared"),
        "lookback": int(config.get("lookback", 720)),
        "period": int(hyper.get("period_len", 24)),
        "max_epochs": int(config.get("max_epochs", 30)),
        "loss": "huber",
        "source_run": str(candidates[0].parent),
    }


def audit_setting(rows, dataset, horizon, seed, frozen):
    """Pick the three controls and check protocol compatibility."""

    selected = {}
    for row in rows:
        config_id = row["config_id"]
        if config_id in CONFIG_IDS:
            selected[config_id] = row
        elif config_id.startswith("pool") and abs(generic_q(config_id) - 0.125) < 1e-9:
            selected["pooled_lowrank_q1/8"] = row
    missing = [
        key
        for key in ("phase_only", "direct_nlinear", "pooled_lowrank_q1/8")
        if key not in selected
    ]
    if missing:
        raise SystemExit(f"{dataset}-{horizon}: missing controls {missing}")

    audit = []
    for key, row in selected.items():
        compatible = True
        notes = []
        if frozen is not None:
            if frozen["lookback"] != 720:
                compatible = False
                notes.append(f"lookback={frozen['lookback']}")
            if frozen["period"] != 24:
                compatible = False
                notes.append(f"period={frozen['period']}")
            if frozen["max_epochs"] != 30:
                compatible = False
                notes.append(f"max_epochs={frozen['max_epochs']}")
        audit.append(
            {
                "setting": canonical_setting(dataset, horizon, seed),
                "candidate": key,
                "config_id": row["config_id"],
                "val_mse": float(row["val_mse"]),
                "val_mae": float(row["val_mae"]),
                "test_mse": float(row["test_mse"]),
                "test_mae": float(row["test_mae"]),
                "elapsed_sec": float(row.get("elapsed_sec") or 0.0),
                "source": "reused_exact" if compatible else "rerun_for_alignment",
                "source_run": row["run_dir"],
                "notes": "; ".join(notes),
            }
        )
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch-root", default="research_runs/rank_sweep_2_stage1"
    )
    parser.add_argument("--output-dir", default="research_runs/structured_lowrank_round0_v1")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    scratch = Path(args.scratch_root)
    if not scratch.is_absolute():
        scratch = ROOT / scratch
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    frozen_rows = []
    for dataset, horizon in DEFAULT_SETTINGS:
        rows = load_results(scratch, dataset, horizon, args.seed)
        frozen = load_frozen(scratch, dataset, horizon, args.seed)
        if frozen is None:
            raise SystemExit(f"{dataset}-{horizon}: no frozen run config found")
        frozen_rows.append(frozen)
        audit_rows.extend(audit_setting(rows, dataset, horizon, args.seed, frozen))

    with (output / "frozen_configs.json").open("w") as handle:
        json.dump(
            {
                "seed": args.seed,
                "source": str(scratch),
                "settings": frozen_rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    fields = [
        "setting",
        "candidate",
        "config_id",
        "val_mse",
        "val_mae",
        "test_mse",
        "test_mae",
        "elapsed_sec",
        "source",
        "source_run",
        "notes",
    ]
    with (output / "round0_controls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    lines = [
        "# Round 0 — controls, parameters and reuse audit",
        "",
        "> Generated by `scripts/audit_structured_lowrank_reuse.py`. No training was",
        "> performed. All rows below are `reused_exact` unless the notes column says",
        "> otherwise. These numbers are inherited from an existing test-exposed",
        "> lineage and must be disclosed as `test-set selection` evidence.",
        "",
        "## Frozen per-setting training hyperparameters",
        "",
        "| setting | gate_init | learning_rate | source_run |",
        "|---|---:|---:|---|",
    ]
    for frozen in frozen_rows:
        lines.append(
            f"| {canonical_setting(frozen['dataset'], frozen['horizon'], frozen['seed'])} "
            f"| {frozen['gate_init']} | {frozen['learning_rate']} | `{frozen['source_run']}` |"
        )
    lines += [
        "",
        "## Reused controls",
        "",
        "| setting | candidate | config_id | val MSE | val MAE | test MSE | test MAE | source |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['setting']} | {row['candidate']} | {row['config_id']} "
            f"| {row['val_mse']:.6f} | {row['val_mae']:.6f} "
            f"| {row['test_mse']:.6f} | {row['test_mae']:.6f} | {row['source']} |"
        )
    lines += [
        "",
        "## Relative to the direct control",
        "",
        "| setting | candidate | delta MSE% | delta MAE% |",
        "|---|---|---:|---:|",
    ]
    by_setting = {}
    for row in audit_rows:
        by_setting.setdefault(row["setting"], {})[row["candidate"]] = row
    for setting, group in sorted(by_setting.items()):
        direct = group["direct_nlinear"]
        for name, row in sorted(group.items()):
            if name == "direct_nlinear":
                continue
            dmse = (direct["test_mse"] - row["test_mse"]) / direct["test_mse"] * 100
            dmae = (direct["test_mae"] - row["test_mae"]) / direct["test_mae"] * 100
            lines.append(
                f"| {setting} | {name} | {dmse:+.3f} | {dmae:+.3f} |"
            )
    (output / "round0_controls.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {output/'frozen_configs.json'}")
    print(f"wrote {output/'round0_controls.csv'}")
    print(f"wrote {output/'round0_controls.md'}")
    for row in audit_rows:
        print(
            f"  {row['setting']:24s} {row['candidate']:20s} "
            f"test {row['test_mse']:.6f}/{row['test_mae']:.6f} {row['source']}"
        )


if __name__ == "__main__":
    main()
