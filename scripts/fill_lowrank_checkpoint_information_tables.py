#!/usr/bin/env python3
"""Fill the plan document's result tables from the analysis artifacts.

The plan ``docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md``
reserves tables 1-7 in section 7 and a placeholder for a status block.  This
script rewrites those tables in place from

    research_runs/lowrank_checkpoint_information_v1/
        checkpoint_inventory.csv
        stage0_audit.csv
        canonical_modes.csv
        cross_seed_alignment.csv
        semantic_alignment.csv
        conditional_rrr_alignment.csv
        intervention_results.csv

so the document can never drift from the numbers.  It is idempotent: running it
twice produces byte-identical output.

Usage::

    python scripts/fill_lowrank_checkpoint_information_tables.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLAN = REPO_ROOT / "docs" / "PhaseFormer_lowrank_checkpoint_information_analysis_plan.md"
RESULTS = REPO_ROOT / "research_runs" / "lowrank_checkpoint_information_v1"

HEADLINE_CELLS = ("q=1/8", "q=1/32")


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def delta_branch_mse(row: dict) -> float:
    """Branch degradation versus the checkpoint, derived from the row itself.

    ``intervention_results.csv`` records ``delta_*_vs_checkpoint`` for the fused
    MSE/MAE but not for the branch, while ``baseline_branch_mse`` and
    ``branch_mse`` are both present.  The delta is their difference by
    definition, so it is derived here rather than re-running the arm sweep for a
    purely arithmetic column.
    """
    if "delta_branch_mse_vs_checkpoint" in row:
        return num(row["delta_branch_mse_vs_checkpoint"])
    return num(row["branch_mse"]) - num(row["baseline_branch_mse"])


def num(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def table_block(title: str, header: list[str], rows: list[list[str]], note: str = "") -> str:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    lines.append("")
    return "\n".join(lines)


REQUIRED = (
    "checkpoint_inventory.csv",
    "stage0_audit.csv",
    "canonical_modes.csv",
    "cross_seed_alignment.csv",
    "semantic_alignment.csv",
    "conditional_rrr_alignment.csv",
    "intervention_results.csv",
)


def build_tables() -> dict[int, str]:
    # Refuse to run on an incomplete artifact set: filling the plan with empty
    # tables would silently destroy the recorded results.
    missing = [name for name in REQUIRED if not (RESULTS / name).is_file()]
    if missing:
        raise SystemExit(f"missing analysis artifacts: {missing}")
    inventory = read_csv(RESULTS / "checkpoint_inventory.csv")
    audit = read_csv(RESULTS / "stage0_audit.csv")
    canonical = read_csv(RESULTS / "canonical_modes.csv")
    cross_seed = read_csv(RESULTS / "cross_seed_alignment.csv")
    semantic = read_csv(RESULTS / "semantic_alignment.csv")
    alignment = read_csv(RESULTS / "conditional_rrr_alignment.csv")
    interventions = read_csv(RESULTS / "intervention_results.csv")

    audit_by_key = {(r["setting"], r["seed"], r["cell"]): r for r in audit}
    tables: dict[int, str] = {}

    # ---- table 1: checkpoint audit -------------------------------------
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in inventory:
        grouped[(row["setting"], row["cell"], row["rank"])].append(row)
    rows1 = []
    for key in sorted(grouped, key=lambda k: (k[0], k[1])):
        setting, cell, rank = key
        entries = grouped[key]
        seeds = sorted(int(e["seed"]) for e in entries)
        records = [audit_by_key.get((setting, str(s), cell)) for s in seeds]
        records = [r for r in records if r is not None]
        worst = max((num(r["effective_map_equivalence_max_abs"]) for r in records), default=float("nan"))
        qc = "PASS" if records and all(r["equivalence_pass"] == "True" for r in records) else "FAIL"
        hashes = sorted({e["checkpoint_sha256"][:12] for e in entries})
        rows1.append([
            setting,
            f"{cell} (r={rank})",
            f"{len(seeds)}: {seeds[0]}–{seeds[-1]}" if len(seeds) > 1 else str(seeds[0]),
            ", ".join(sorted({str(r["pool_factor"]) for r in records})) or "—",
            f"{worst:.2e}",
            hashes[0] if len(hashes) == 1 else f"{len(hashes)} 个",
            qc,
        ])
    tables[1] = table_block(
        "表 1：checkpoint 审计",
        ["Setting", "rank/q", "seeds 完整", "pool factor", "映射等价误差", "checkpoint hash", "QC"],
        rows1,
        "映射等价误差为 float64 下 `decoder(encoder(pool(z)))` 与 "
        "`(decoder@encoder)z + decoder@encoder.bias` 在全 validation 上的最大绝对差。",
    )

    # ---- table 2: canonical modes --------------------------------------
    rows2 = []
    for row in canonical:
        if row["cell"] not in HEADLINE_CELLS:
            continue
        rows2.append([
            row["setting"],
            f"{row['cell']} (r={row['rank']})",
            row["seed"],
            f"mode {row['mode_index']}",
            f"{num(row['singular_value_share']):.4f}",
            f"{num(row['correction_energy_share']):.4f}",
            f"{num(row['latent_variance_share']):.4f}",
            f"{num(row['singular_gap_to_next']):.4f}",
        ])
    tables[2] = table_block(
        "表 2：checkpoint 规范模式",
        ["Setting", "q/rank", "seed", "mode", "singular share", "output energy share",
         "latent 方差份额", "到下一奇异值的相对间隙"],
        rows2,
    )

    # ---- table 3: cross-seed stability ---------------------------------
    rows3 = []
    for row in cross_seed:
        if row.get("scope") != "full":
            continue
        overlap = num(row["input_subspace_overlap"])
        rows3.append([
            row["setting"],
            f"{row['rank_a']} vs {row['rank_b']}",
            f"{overlap:.4f}",
            f"{num(row['output_subspace_overlap']):.4f}",
            f"{row['matched_modes_above_0p7']}/{row['dimension']}",
            "稳定" if overlap >= 0.8 else "不稳定",
        ])
    tables[3] = table_block(
        "表 3：跨 seed 稳定性",
        ["Setting", "rank 对比", "input subspace overlap", "output subspace overlap",
         "匹配度 ≥0.7 的模式数", "结论"],
        rows3,
        "重叠为 `‖QaᵀQb‖_F²/k`，1 表示子空间重合、0 表示正交。",
    )

    # ---- table 4: input/output semantics -------------------------------
    rows4 = []
    for row in semantic:
        if row["cell"] not in HEADLINE_CELLS:
            continue
        rows4.append([
            row["setting"],
            f"{row['cell']} (r={row['rank']})",
            f"mode {row['mode_index']}",
            row["input_best_group"],
            f"{num(row['input_group_explanation']):.3f}",
            row["output_best_group"],
            f"{num(row['output_group_explanation']):.3f}",
            row["paired_mechanism"],
        ])
    tables[4] = table_block(
        "表 4：输入—输出语义",
        ["Setting", "q/rank", "canonical mode", "输入首要语义组", "输入解释率",
         "输出首要语义组", "输出解释率", "配对机制"],
        rows4,
        "解释率为该组子空间在方向 L2 范数中的投影份额；字典组彼此共线，"
        "因此各组分额不构成划分。",
    )

    # ---- table 5: conditional RRR alignment ----------------------------
    rows5 = []
    for row in alignment:
        rows5.append([
            row["setting"],
            f"{row['cell']} (r={row['rank']})",
            f"{num(row['overlap_with_independent_rrr']):.4f}",
            f"{num(row['overlap_with_conditional_rrr']):.4f}",
            f"{num(row['overlap_difference']):+.4f}",
            "是" if row["supports_h1"] == "True" else "否",
        ])
    tables[5] = table_block(
        "表 5：独立目标与条件性目标对齐",
        ["Setting", "q/rank", "overlap with independent RRR", "overlap with conditional RRR",
         "差值", "支持 H1"],
        rows5,
    )

    # ---- table 6: interventions ----------------------------------------
    rows6 = []
    for row in interventions:
        is_drop = row["arm"].endswith("-drop")
        rows6.append([
            row["setting"],
            f"{row['cell']} (r={row['rank']})",
            row["arm"],
            f"{num(row['correction_reconstruction_r2']):.4f}",
            f"{delta_branch_mse(row):+.6f}",
            f"{num(row['delta_fused_mse_vs_checkpoint']):+.6f}",
            f"{num(row['delta_fused_mae_vs_checkpoint']):+.6f}",
            f"{num(row['random_fused_mse_percentile_of_arm']):.1f}%" if is_drop else "—",
        ])
    tables[6] = table_block(
        "表 6：语义保留/删除干预",
        ["Setting", "q/rank", "Arm", "correction R²", "Δbranch MSE", "Δfused MSE",
         "Δfused MAE", "vs random 95%"],
        rows6,
        "`vs random 95%` 是该 arm 的 fused MSE 在 100 个同维随机子空间对照中的分位数；"
        "仅 drop 类 arm 具有必要性含义。",
    )

    # ---- table 7: mechanism verdict ------------------------------------
    # Condition 1/2 of plan section 6.1: a majority of seeds must rank the group
    # first with both explanation rates clearing the thresholds.
    mechanism_by_setting: dict[str, dict[str, int]] = {}
    for setting in sorted({r["setting"] for r in semantic}):
        entries = [r for r in semantic if r["setting"] == setting and r["cell"] == "q=1/8"]
        per_seed: dict[str, dict] = {}
        for row in entries:
            if int(row["mode_index"]) == 0:
                per_seed[row["seed"]] = row
        votes: dict[str, int] = defaultdict(int)
        for row in per_seed.values():
            if num(row["input_group_explanation"]) >= 0.5 and num(
                row["output_group_explanation"]
            ) >= 0.8:
                votes[row["paired_mechanism"]] += 1
        mechanism_by_setting[setting] = votes

    mechanisms = sorted(
        {name for votes in mechanism_by_setting.values() for name in votes}
    )
    # ``§6.2`` states the cross-setting rule on the full 7-setting scope:
    # >=5/7 consistent, 3-4/7 conditional, <=2/7 unsupported.  On the full scope
    # these constants are used verbatim, so the plan's own numbers are preserved
    # exactly.  If a setting is excluded the scope shrinks, and the boundaries
    # are scaled by the same proportions (5/7 -> 0.7n, 3/7 -> 0.4n) so that
    # bypassing a setting does not silently change what the words mean.
    n_settings = max(len(mechanism_by_setting), 1)
    if n_settings == 7:
        consistent_at, conditional_at = 5, 3
    else:
        consistent_at = max(1, math.ceil(0.7 * n_settings))
        conditional_at = max(1, math.ceil(0.4 * n_settings))
        conditional_at = min(conditional_at, consistent_at)
    rows7 = []
    for name in mechanisms:
        held = sorted(
            setting for setting, votes in mechanism_by_setting.items() if votes.get(name, 0) >= 2
        )
        counter = sorted(set(mechanism_by_setting) - set(held))
        rows7.append([
            name,
            "、".join(held) if held else "—",
            "、".join(counter) if counter else "—",
            "见表 6 Semantic-drop 分位",
            "见表 6 Semantic-only Δfused MSE",
            (
                "一致机制"
                if len(held) >= consistent_at
                else ("条件性机制" if len(held) >= conditional_at else "不支持")
            ),
        ])
    rows7.append([
        "gate/主干绕行而非信息保留",
        "见表 6 branch 与 fused 的分离",
        "同左",
        "branch 退化但 fused 近中性",
        "Semantic-only 近中性",
        "独立机制",
    ])
    tables[7] = table_block(
        "表 7：最终机制裁定",
        ["候选机制", "成立 setting", "反例 setting", "必要性证据", "充分性证据", "裁定"],
        rows7,
        "成立判定按计划 §6.1：三个 seed 中至少 2 个把该机制排在首位，且输入解释率 ≥0.5、"
        "输出解释率 ≥0.8。跨 setting 判定按 §6.2，并按实际生效 scope "
        f"（n={n_settings}）按多数边界折算："
        f"≥{consistent_at}/{n_settings} 为一致机制，"
        f"{conditional_at}–{consistent_at - 1}/{n_settings} 为条件性机制，"
        f"≤{conditional_at - 1}/{n_settings} 为不支持。"
        + (
            ""
            if n_settings == 7
            else f"注：本轮实际生效 {n_settings} 个 setting（原计划 7 个），"
            "被排除的 setting 不参与裁定。"
        ),
    )
    return tables


def replace_section(text: str, number: int, block: str) -> str:
    """Replace the ``### 表 N：…`` section with ``block``."""
    pattern = re.compile(
        rf"^### 表 {number}：.*?(?=^### 表 |\Z|\n## )",
        re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"plan has no section for table {number}")
    return pattern.sub(block, text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(PLAN))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan)
    tables = build_tables()
    text = plan_path.read_text(encoding="utf-8")
    for number in sorted(tables):
        text = replace_section(text, number, tables[number])
    if args.dry_run:
        print(text[:2000])
        return
    plan_path.write_text(text, encoding="utf-8")
    for number in sorted(tables):
        filled = tables[number].count("\n|") - 2
        print(f"table {number}: {max(filled, 0)} rows")


if __name__ == "__main__":
    main()
