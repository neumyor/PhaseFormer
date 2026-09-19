#!/usr/bin/env python3
"""E16 write-back: build the minipaper §4.4 dissection and intervention tables.

Consumes E16's own products and reshapes them into the two §4.4 tables, which
are per (arm, setting) with the three seeds aggregated.

Intervention table columns required by §4.4:

``Semantic-only Δfused``, ``Semantic-drop Δfused``, ``随机 95% 区间``,
``PCA-drop``, ``随机 RRR 子空间 drop`` (the new control), ``支路自身 Δ``,
``融合 Δ`` -- every arm reporting BOTH its own branch error and the fused error.

Dissection table columns required by §4.4:

leading mode input group / explanation, output group / explanation, correction
energy share, cross-seed ``leading4`` overlap, stable-semantics verdict.

Aggregation rules (frozen here, so the table is reproducible):

* numeric columns are the mean over the available seeds, with the seed count and
  the spread reported alongside so a 1-seed cell cannot masquerade as a 3-seed one;
* the stable-semantics verdict is a **majority over seeds** of E16's per-seed
  verdict, and the count of seeds agreeing is reported in a separate column;
* the random band's low/high are the mean of the per-seed bounds, and an arm's
  percentile inside the band is reported as-is per seed then averaged, because the
  percentile is what "worse than the 95% null" is defined on.

Usage::

    python scripts/phaseformer_L/e16_writeback.py \\
        --intervention research_runs/phaseformer_L_e16_dissection_v1/intervention_table.csv \\
        --dissection   research_runs/phaseformer_L_e16_dissection_v1/dissection_table.csv \\
        --output-root  research_runs/phaseformer_L_e16_dissection_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The intervention arms the §4.4 table names, mapped to their output columns.
INTERVENTION_ARMS = ("Semantic-only", "Semantic-drop", "PCA-drop", "RandomRRR-drop")
ARM_DISPLAY = {
    "l_main": "PhaseFormer-L",
    "l_q1_4": "L-q1/4",
    "l_q1_8": "L-q1/8",
}
MIN_SEED_MAJORITY = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--dissection", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def as_float(value):
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path) -> list:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_of(rows, column):
    values = [as_float(row.get(column)) for row in rows]
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    if len(values) == 1:
        return float(values[0]), 0.0, 1
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)), int(arr.size)


def missing_columns(rows, needed: list) -> list:
    if not rows:
        return needed
    present = set(rows[0].keys())
    return [name for name in needed if name not in present]


def build_intervention(rows: list) -> tuple[list, list]:
    needed = ["setting", "arm", "seed", "intervention_arm",
              "delta_fused_mse_vs_checkpoint", "delta_branch_mse_vs_checkpoint",
              "delta_fused_mae_vs_checkpoint", "delta_branch_mae_vs_checkpoint",
              "random_low_fused_mse", "random_high_fused_mse",
              "random_rrr_low_fused_mse", "random_rrr_high_fused_mse",
              "random_rrr_fused_mse_percentile_of_arm",
              "worse_than_random_rrr_95pct_fused_mse",
              "subspace_dimension", "same_dimension_controls_available"]
    absent = missing_columns(rows, needed)
    grouped: dict = {}
    for row in rows:
        grouped.setdefault((row["arm"], row["setting"],
                            row["intervention_arm"]), []).append(row)
    cells = sorted({(row["arm"], row["setting"]) for row in rows},
                   key=lambda k: (k[0], k[1]))
    out = []
    for arm, setting in cells:
        entry = {"arm": arm, "model": ARM_DISPLAY.get(arm, arm),
                 "setting": setting,
                 "dataset": setting.rsplit("-", 1)[0],
                 "horizon": setting.rsplit("-", 1)[1]}
        for name in INTERVENTION_ARMS:
            block = grouped.get((arm, setting, name), [])
            for metric in ("fused", "branch"):
                mean, sd, n = mean_of(block, f"delta_{metric}_mse_vs_checkpoint")
                entry[f"{name}__delta_{metric}_mse"] = (
                    None if mean is None else round(mean, 6))
                entry[f"{name}__delta_{metric}_mse_std"] = (
                    None if sd is None else round(sd, 6))
            entry[f"{name}__seeds"] = max(
                (len(grouped.get((arm, setting, name), []))), 0)
            dim = mean_of(block, "subspace_dimension")[0]
            entry[f"{name}__dimension"] = None if dim is None else int(dim)
        # Random same-dimension band (the existing control) and the new RRR band.
        band = grouped.get((arm, setting, "Semantic-drop"), []) or rows
        entry["random_band_low_fused_mse"] = round(
            mean_of(band, "random_low_fused_mse")[0], 6) \
            if mean_of(band, "random_low_fused_mse")[0] is not None else None
        entry["random_band_high_fused_mse"] = round(
            mean_of(band, "random_high_fused_mse")[0], 6) \
            if mean_of(band, "random_high_fused_mse")[0] is not None else None
        entry["random_rrr_band_low_fused_mse"] = round(
            mean_of(band, "random_rrr_low_fused_mse")[0], 6) \
            if mean_of(band, "random_rrr_low_fused_mse")[0] is not None else None
        entry["random_rrr_band_high_fused_mse"] = round(
            mean_of(band, "random_rrr_high_fused_mse")[0], 6) \
            if mean_of(band, "random_rrr_high_fused_mse")[0] is not None else None
        entry["random_rrr_percentile_of_arm_mean"] = round(
            mean_of(band, "random_rrr_fused_mse_percentile_of_arm")[0], 3) \
            if mean_of(band, "random_rrr_fused_mse_percentile_of_arm")[0] is not None else None
        flags = [str(r.get("worse_than_random_rrr_95pct_fused_mse", "")).strip()
                 for r in band]
        entry["semantic_drop_worse_than_random_rrr_95pct_seeds"] = sum(
            1 for f in flags if f.lower() == "true")
        entry["semantic_drop_worse_than_random_rrr_95pct_seed_total"] = len(flags)
        entry["same_dimension_controls_available"] = (
            str(band[0].get("same_dimension_controls_available", "")).strip()
            if band else None)
        out.append(entry)
    return out, absent


def build_dissection(rows: list) -> tuple[list, list]:
    # NOTE: E16's dissection_table.csv has no majority_input_group_label column
    # (verified against the real 48-column header).  It has the per-seed leading
    # mode's group and label (`leading_input_group` / `leading_input_group_label`)
    # plus a `majority_input_group` key with its own vote string.  Voting on the
    # per-seed label is therefore the correct route; demanding a
    # majority_input_group_label column would have blanked the whole column.
    needed = ["setting", "arm", "seed", "majority_input_group",
              "leading_input_group", "leading_input_group_label",
              "mean_input_group_explanation",
              "leading_output_group_label", "mean_output_group_explanation",
              "leading_correction_energy_share",
              "cross_seed_leading4_input_overlap",
              "cross_seed_leading4_output_overlap",
              "stable_semantics_verdict", "criterion_1_group_stable",
              "criterion_2_input_explanation_ge_0p5",
              "criterion_3_output_explanation_ge_0p8",
              "criterion_4_drop_beyond_random_95pct",
              "criterion_5_only_within_0p5pct",
              "criterion_6_drop_beyond_random_rrr_95pct", "head_kind"]
    absent = missing_columns(rows, needed)
    grouped: dict = {}
    for row in rows:
        grouped.setdefault((row["arm"], row["setting"]), []).append(row)
    out = []
    for (arm, setting) in sorted(grouped, key=lambda k: (k[0], k[1])):
        block = grouped[(arm, setting)]
        entry = {"arm": arm, "model": ARM_DISPLAY.get(arm, arm),
                 "setting": setting,
                 "dataset": setting.rsplit("-", 1)[0],
                 "horizon": setting.rsplit("-", 1)[1],
                 "head_kind": block[0].get("head_kind"),
                 "seeds": len(block)}
        # majority input group across seeds
        # Vote on the per-seed leading mode's group label, which is the column
        # E16 actually writes.  `majority_input_group` is recorded alongside as a
        # cross-check (it is E16's own majority over ranks within a seed).
        group_votes: dict = {}
        label_by_group: dict = {}
        for row in block:
            group = str(row.get("leading_input_group", "") or "").strip()
            label = str(row.get("leading_input_group_label", "") or "").strip()
            group_votes[group] = group_votes.get(group, 0) + 1
            if label:
                label_by_group.setdefault(group, {})
                label_by_group[group][label] = label_by_group[group].get(label, 0) + 1
        if group_votes:
            group, count = max(group_votes.items(), key=lambda kv: kv[1])
            entry["input_group"] = group
            labels = label_by_group.get(group) or {}
            entry["input_group_label"] = (
                max(labels.items(), key=lambda kv: kv[1])[0] if labels else group
            )
            entry["input_group_votes"] = f"{count}/{len(block)}"
        majority = {}
        for row in block:
            key = str(row.get("majority_input_group", "") or "").strip()
            majority[key] = majority.get(key, 0) + 1
        if majority:
            key, count = max(majority.items(), key=lambda kv: kv[1])
            entry["e16_majority_input_group"] = key
            entry["e16_majority_input_group_votes"] = f"{count}/{len(block)}"
        for column, target in (("mean_input_group_explanation",
                                "input_group_explanation"),
                               ("mean_output_group_explanation",
                                "output_group_explanation"),
                               ("leading_correction_energy_share",
                                "correction_energy_share"),
                               ("cross_seed_leading4_input_overlap",
                                "leading4_input_overlap"),
                               ("cross_seed_leading4_output_overlap",
                                "leading4_output_overlap")):
            mean, sd, n = mean_of(block, column)
            entry[target] = None if mean is None else round(mean, 6)
            entry[target + "_std"] = None if sd is None else round(sd, 6)
        out_votes: dict = {}
        for row in block:
            key = str(row.get("leading_output_group_label", ""))
            out_votes[key] = out_votes.get(key, 0) + 1
        if out_votes:
            label, count = max(out_votes.items(), key=lambda kv: kv[1])
            entry["output_group_label"] = label
            entry["output_group_votes"] = f"{count}/{len(block)}"
        # stable verdict: majority over seeds, with the per-seed criterion counts
        verdicts = [str(r.get("stable_semantics_verdict", "")).strip().lower()
                    for r in block]
        true_count = sum(1 for v in verdicts if v == "true")
        entry["stable_semantics_seed_true"] = true_count
        entry["stable_semantics_seed_total"] = len(verdicts)
        entry["stable_semantics_verdict"] = (
            true_count >= MIN_SEED_MAJORITY if verdicts else None)
        for index in range(1, 7):
            column = {
                1: "criterion_1_group_stable",
                2: "criterion_2_input_explanation_ge_0p5",
                3: "criterion_3_output_explanation_ge_0p8",
                4: "criterion_4_drop_beyond_random_95pct",
                5: "criterion_5_only_within_0p5pct",
                6: "criterion_6_drop_beyond_random_rrr_95pct",
            }[index]
            passed = sum(1 for r in block
                         if str(r.get(column, "")).strip().lower() == "true")
            entry[f"criterion_{index}_seeds_true"] = f"{passed}/{len(block)}"
        out.append(entry)
    return out, absent


def main() -> None:
    args = parse_args()
    intervention_rows = read_csv(Path(args.intervention)
                                 if Path(args.intervention).is_absolute()
                                 else ROOT / args.intervention)
    dissection_rows = read_csv(Path(args.dissection)
                              if Path(args.dissection).is_absolute()
                              else ROOT / args.dissection)

    intervention, missing_i = build_intervention(intervention_rows)
    dissection, missing_d = build_dissection(dissection_rows)

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("intervention_table_44.csv", intervention),
                       ("dissection_table_44.csv", dissection)):
        fields = list(rows[0].keys()) if rows else ["setting"]
        with (out_root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # markdown for the minipaper
    def qr_label(arm, horizon):
        """§4.4's q/r column, derived from the arm and horizon (not guessed)."""
        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            return "—"
        if arm == "l_main":
            return f"dense（r={horizon}）"
        if arm == "l_q1_4":
            return f"q=1/4（r={horizon // 4}）"
        if arm == "l_q1_8":
            return f"q=1/8（r={horizon // 8}）"
        return arm

    lines = []
    for row in intervention:
        def delta(name, metric="fused"):
            value = row.get(f"{name}__delta_{metric}_mse")
            return "—" if value is None else f"{value:+.4f}"
        band = ("—" if row["random_band_low_fused_mse"] is None else
                f"[{row['random_band_low_fused_mse']:.4f}, "
                f"{row['random_band_high_fused_mse']:.4f}]")
        rrr_band = ("—" if row["random_rrr_band_low_fused_mse"] is None else
                    f"[{row['random_rrr_band_low_fused_mse']:.4f}, "
                    f"{row['random_rrr_band_high_fused_mse']:.4f}]")
        # Eleven cells: model, dataset, H, q/r, Semantic-only, Semantic-drop,
        # random band, PCA-drop, random-RRR band, branch delta, fused delta.
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row["model"], row["dataset"], row["horizon"],
                qr_label(row["arm"], row["horizon"]),
                delta("Semantic-only"), delta("Semantic-drop"), band,
                delta("PCA-drop"), rrr_band,
                delta("Semantic-drop", "branch"), delta("Semantic-drop")))
    (out_root / "intervention_table_44.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")

    summary = {
        "experiment": "E16 write-back (minipaper section 4.4)",
        "intervention_rows": len(intervention),
        "dissection_rows": len(dissection),
        "missing_columns_intervention": missing_i,
        "missing_columns_dissection": missing_d,
        "aggregation_rules": {
            "numeric": "mean over available seeds; std and seed count reported "
                       "alongside so a 1-seed cell cannot look like a 3-seed one",
            "stable_verdict": f"majority over seeds (>= {MIN_SEED_MAJORITY}); the "
                              "per-seed true count and each of the six criteria's "
                              "seed counts are kept in separate columns",
            "random_band": "low/high are the mean of the per-seed bounds; the "
                           "percentile is averaged because 'worse than the 95% "
                           "null' is defined on the percentile",
        },
        "disclosures": [
            "the seven settings are test-set-selection-derived, not a blind sample",
            "the reference dissection artifacts cover only 6 settings; "
            "Electricity-336 has no registered counterpart, so its anatomy is "
            "newly computed rather than reproduced",
            "the random-RRR null band is drawn from the cell's RRR-achievable "
            "subspace; on rank-bottlenecked probe cells that subspace spans the "
            "whole latent space, so the RRR family coincides with the ambient "
            "family there (random_rrr_pool_covers_ambient flags it per cell)",
            "the same-dimension controls are flagged per cell rather than assumed",
        ],
    }
    (out_root / "e16_writeback_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "finished",
        "intervention_rows": len(intervention),
        "dissection_rows": len(dissection),
        "missing_columns_intervention": missing_i,
        "missing_columns_dissection": missing_d,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
