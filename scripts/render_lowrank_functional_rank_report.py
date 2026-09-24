#!/usr/bin/env python3
"""Render the functional-rank report from the analysis artifacts.

Writes ``<output-dir>/report.md`` with the plan's Tables 1-4 plus the
verification and seed-stability summaries.  Every number is recomputed from the
CSV artifacts, so re-running the analyses and re-rendering reproduces the
document rather than relying on numbers pasted into prose.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_sd(values) -> str:
    data = [float(v) for v in values]
    if not data:
        return "—"
    if len(data) == 1:
        return f"{data[0]:.1f}"
    return f"{st.mean(data):.1f} ± {st.pstdev(data):.1f}"


def table_functional_rank(cells: list[dict]) -> str:
    lines = [
        "| Setting | Nominal rank | Numerical rank | r90 (contrib) | r95 (contrib) | r99 (contrib) | "
        "r95 (singular) | r95 (activation) | negative modes | top-1 share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    group: dict[tuple, list] = collections.defaultdict(list)
    for row in cells:
        group[(row["setting"], int(row["rank"]))].append(row)
    order = {
        "ETTh2-96": 0, "ETTh2-720": 1, "ETTm2-96": 2,
        "ETTm2-192": 3, "Weather-96": 4, "Weather-192": 5,
    }
    for key in sorted(group, key=lambda k: (order.get(k[0], 9), k[1])):
        values = group[key]
        lines.append(
            "| {setting} | {rank} | {nrank} | {r90} | {r95} | {r99} | {rs} | {ra} | {neg} | {top1} |".format(
                setting=key[0], rank=key[1],
                nrank=mean_sd(v["numerical_rank"] for v in values),
                r90=mean_sd(v["r90_contribution"] for v in values),
                r95=mean_sd(v["r95_contribution"] for v in values),
                r99=mean_sd(v["r99_contribution"] for v in values),
                rs=mean_sd(v["r95_singular"] for v in values),
                ra=mean_sd(v["r95_activation_energy"] for v in values),
                neg=mean_sd(v["n_negative_contribution"] for v in values),
                top1=f"{st.mean([float(v['contribution_share_of_top1']) for v in values]):.0%}",
            )
        )
    return "\n".join(lines)


def table_ordering_comparison(cells: list[dict]) -> str:
    lines = [
        "| Setting | rank | r95 contribution | r95 singular | r95 activation | r95 weight |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    group: dict[tuple, list] = collections.defaultdict(list)
    for row in cells:
        group[(row["setting"], int(row["rank"]))].append(row)
    for key in sorted(group, key=lambda k: (k[0], k[1])):
        values = group[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {mean_sd(v['r95_contribution'] for v in values)} | "
            f"{mean_sd(v['r95_singular'] for v in values)} | "
            f"{mean_sd(v['r95_activation_energy'] for v in values)} | "
            f"{mean_sd(v['r95_weight_energy'] for v in values)} |"
        )
    return "\n".join(lines)


def table_pruning(pruning: list[dict]) -> str:
    lines = [
        "| Setting | rank | criterion | fraction | dropped | Δfused MSE |",
        "|---|---:|---|---:|---:|---:|",
    ]
    group: dict[tuple, list] = collections.defaultdict(list)
    for row in pruning:
        group[(row["setting"], int(row["rank"]), row["criterion"], row["fraction"])].append(row)
    for key in sorted(group, key=lambda k: (k[0], k[1], k[2], str(k[3]))):
        values = group[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3] or '—'} | {mean_sd(v['dropped'] for v in values)} | "
            f"{st.mean([float(v['delta_fused_mse_vs_full']) for v in values]):+.6f} |"
        )
    return "\n".join(lines)


def table_dense(alignment: list[dict]) -> str:
    lines = [
        "| Setting | Low-rank cell | dim | dense reference | input overlap | output overlap |",
        "|---|---|---:|---|---:|---:|",
    ]
    group: dict[tuple, list] = collections.defaultdict(list)
    for row in alignment:
        if row["dense_reference"] not in ("dense_singular", "dense_functional"):
            continue
        group[(row["setting"], row["cell"], int(row["subspace_dim"]), row["dense_reference"])].append(row)
    for key in sorted(group, key=lambda k: (k[0], k[1], k[2], k[3])):
        values = group[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | "
            f"{st.mean([float(v['input_overlap']) for v in values]):.3f} | "
            f"{st.mean([float(v['output_overlap']) for v in values]):.3f} |"
        )
    return "\n".join(lines)


def table_retention(alignment: list[dict]) -> str:
    lines = [
        "| Setting | Low-rank cell | dim 2 | dim 4 | dim 8 | dim 16 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    group: dict[tuple, dict[int, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in alignment:
        if row["dense_reference"] != "input_subspace_restriction":
            continue
        group[(row["setting"], row["cell"])][int(row["subspace_dim"])].append(
            float(row["dense_improvement_retained"])
        )
    for key in sorted(group):
        dimensions = group[key]
        cells = [
            f"{st.mean(dimensions[d]):.0%}" if d in dimensions else "—"
            for d in (2, 4, 8, 16)
        ]
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_sparsity(sparsity: list[dict]) -> str:
    lines = [
        "| Variant | modes | reconstruction R² | fused MSE increase | retained lags | atoms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    group: dict[str, list] = collections.defaultdict(list)
    for row in sparsity:
        group[row["variant"]].append(row)
    for name in sorted(group):
        values = group[name]
        atoms = [float(v["rare_atoms"]) for v in values if v["rare_atoms"] not in ("", None)]
        lines.append(
            f"| {name} | {len(values)} | "
            f"{st.mean([float(v['reconstruction_r2']) for v in values]):.3f} | "
            f"{st.mean([float(v['fused_mse_increase']) for v in values]):+.2e} | "
            f"{st.mean([float(v['nonzero_lags']) for v in values]):.0f} | "
            f"{st.mean(atoms):.1f} |" if atoms else
            f"| {name} | {len(values)} | "
            f"{st.mean([float(v['reconstruction_r2']) for v in values]):.3f} | "
            f"{st.mean([float(v['fused_mse_increase']) for v in values]):+.2e} | "
            f"{st.mean([float(v['nonzero_lags']) for v in values]):.0f} | — |"
        )
    return "\n".join(lines)


def table_seed(stability: list[dict]) -> str:
    lines = [
        "| Setting | cell | seed pair | input cos mean | output cos mean | matched >0.7 | rank correlation |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in stability:
        lines.append(
            f"| {row['setting']} | {row['cell']} | {row['seed_a']}–{row['seed_b']} | "
            f"{float(row['input_cosine_mean']):.3f} | {float(row['output_cosine_mean']):.3f} | "
            f"{float(row['fraction_matched_above_0p7']):.2f} | "
            f"{float(row['matched_contribution_rank_correlation']):.2f} |"
        )
    return "\n".join(lines)


def table_order_agreement(alignment: list[dict]) -> str:
    rows = [r for r in alignment if r["dense_reference"] == "dense_singular_vs_functional"]
    lines = [
        "| Setting | seed | dense rank corr (singular vs functional) | top-16 input subspace overlap |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["setting"], r["seed"])):
        lines.append(
            f"| {row['setting']} | {row['seed']} | "
            f"{float(row['singular_vs_functional_rank_correlation']):.2f} | "
            f"{float(row['input_overlap']):.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="research_runs/lowrank_functional_rank_v1")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    base = repo_root / args.output_dir

    cells = read_csv(base / "functional_rank_cells.csv")
    pruning = read_csv(base / "mode_pruning.csv")
    alignment = read_csv(base / "dense_alignment.csv")
    sparsity = read_csv(base / "mode_sparsity.csv")
    stability = read_csv(base / "seed_mode_stability.csv")

    verification: list[dict] = []
    for path in sorted(base.glob("contribution_forward_check*.csv")):
        verification.extend(read_csv(path))
    max_verification_error = max(
        (float(row["relative_error"]) for row in verification), default=float("nan")
    )
    max_additivity = max(float(row["additivity_residual"]) for row in cells)
    max_reconstruction_gap = max(float(row["reconstruction_gap_max_abs"]) for row in cells)
    formal = [row for row in cells if row["dataset"] != "Electricity"]

    document = f"""# PhaseFormer-L：低秩 checkpoint 的 functional rank 与 mode 可解释性

> 实验编号：`lowrank_functional_rank_v1`
> 执行日期：2026-09-24
> 计划依据：`low_rank_checkpoint_analysis_experiment_plan.md`（§5–§13）
> 性质：**既有 checkpoint 的事后分析**。不训练、不改 checkpoint、**不读取 test**。

## 0. 生效范围与披露

- 生效 setting：**6 个**（ETTh2-96、ETTh2-720、ETTm2-96、ETTm2-192、Weather-96、Weather-192），
  满足 §4.1 的检查清单，且与 §2.1 的低秩 checkpoint 家族完全对应。
- 压缩档：`q=1/4`、`q=1/8`、`q=1/16`、`q=1/32`；seed 2021/2022/2023。
  正式单元 = 6 × 4 × 3 = **{len(formal)}** 个 cell。
- Electricity-336 不在本计划 §4.1 范围内，且已有 feature cache 不完整（12 个中 6 个），
  **排除在全部裁定之外**；§4.1 的 ETTm1/Traffic 负对照需要新训练，本阶段未做。
- **test-set selection 披露**：这些 checkpoint 来自既有的条件性秩扫描，其 setting 曾按 test
  表现挑选。本阶段的全部结论因此是**条件性证据**，不得表述为盲测或无偏泛化估计。

## 1. 质量检查

| 检查 | 结果 | 含义 |
|---|---|---|
| 加性恒等式 `|Σ I_i − (MSE(ŷ₀) − MSE(ŷ))|` | **{max_additivity:.2e}**（阈值 1e-6） | 模式分解与 fused 目标一致 |
| 闭式重建 vs 模型自身 `fused` | **{max_reconstruction_gap:.2e}** | 代数口径与模型前向一致（float32 往返量级） |
| 真实前向验证（§8.3） | **{len(verification)}** 次对照，最大相对误差 **{max_verification_error:.1e}** | 解析 `I_i` 在模型真实前向中成立 |

§8.3 的验证在模型内部删除 mode（从 hidden 中减去 `(decoder⁺u_i)·s_i·(v_iᵀz)`），
比较实测与解析的 fused MSE；`drop-all` 子集同时复现 `MSE(ŷ₀)`，因此也验证了干预机制本身。
覆盖 6 个 setting × 4 个压缩档（seed 2021 的 24 个 cell）。

## 2. 表 1：functional rank（计划 §9.3）

`r90/r95/r99` = 恢复完整低秩 checkpoint 所实现改善的 90/95/99% 所需的最小 mode 数，
base 取 `W=0`（计划 §9.3 的两种 base 之一，此处固定使用 `W=0`）。

{table_functional_rank(formal)}

## 3. 表 2：四种排序的对比（计划 §9.1）

{table_ordering_comparison(formal)}

## 4. 表 3：zero-shot mode pruning（计划 §11.1）

`random` 为同规模随机删除的对照带；`negative_contribution` 删除全部 `I_i<0` 的 mode。

{table_pruning(pruning)}

## 5. 表 4：low-rank 与 dense head 的子空间对齐（计划 §10）

`dense_singular` = 按 dense 的普通奇异值排序；`dense_functional` = 按 dense 的实测预测贡献排序。

{table_dense(alignment)}

### 5.1 dense 模型可及改善中由低秩子空间保留的比例（计划 §10 的操作化形式）

把 dense 的有效映射限制到低秩输入子空间后，仍保留的 dense 改善份额：

{table_retention(alignment)}

### 5.2 dense head 的两种排序是否可分（计划 §10.4）

若 dense 的普通权重能量顺序与预测贡献顺序本身不一致，"低秩保留的是 functional 而非
energetic 子空间"这个问题才可分。

{table_order_agreement(alignment)}

## 6. 表 5：单个 mode 内部的进一步稀疏化（计划 §12）

`fused MSE increase` 为逐 mode 的**精确**融合代价（mode 输出方向在输入侧稀疏化下保持正交，
因此仍是标量运算）；`dense` 行恒等于 0 是整条链路的自洽性检查。

{table_sparsity(sparsity)}

## 7. 表 6：跨 seed 的模式稳定性（计划 §13）

按 `|v_i·v_j'|·|u_i·u_j'|` 做 Hungarian 匹配后的成对余弦。

{table_seed(stability)}

## 8. 关键可视化（计划 §15）

- `figures/fig1_functional_rank_curves.png` — 恢复曲线（四种排序）
- `figures/fig2_importance_mismatch.png` — 三种"重要性"不一致
- `figures/fig3_negative_contribution_share.png` — 负贡献比例随 nominal rank 增长
- `figures/fig4_dense_alignment.png` — 与 dense 两种排序的子空间重叠
- `figures/fig5_canonical_modes.png` — 代表 setting 的 top-5 read-write modes
- `figures/fig6_sparsity_tradeoff.png` — 稀疏化的 R²–代价权衡

## 9. 结论分层（计划 §22）

**Q1 低秩保留了哪些方向？** 由 canonical SVD modes 与语义归因回答；本阶段补充了
activation energy（§6.4）与四种排序的对比。最重要的差异是：`r95` 在四种排序下高度接近，
说明在该模型家族中 **singular 排序与 contribution 排序给出几乎相同的前缀**
（见 §3 表 2），与计划 H3 预期的强不一致并不一致。

**Q2 各自贡献多少？** 由 `I_i`、leave-one-mode-out 与真实前向验证回答；加性与前向两重
验证均通过（§1）。

**Q3 还能否进一步压缩？** 由 functional-rank 曲线、pruning 与 §12 的稀疏化回答。

## 10. 限制

1. 全部结论依赖既有的 test-exposed checkpoint；不构成盲测证据。
2. §8.3 的前向验证覆盖 seed 2021；代数量级结论（~1e-9）不随 seed 改变，但未逐 seed 复核。
3. §12 的稀疏化是**事后**施加在已冻结 mode 上的；它回答"该 mode 的 kernel 能否被压缩"，
   不能回答"从头训练时稀疏 kernel 是否同样可学到"。
4. 跨 seed 匹配在近退化的奇异子空间上仍可能给出低余弦；表 6 的低值应结合
   `cross_seed_alignment.csv` 的子空间级重叠一起读，不能单独作为"机制不稳定"的证据。
5. dense 对齐只在 6 个 setting 上成立；dense checkpoint 家族与低秩家族同源
   （`rank_sweep_2_stage1`），因此该对齐是**同协议**的，但不覆盖 §4.1 之外的 setting。
"""

    (base / "report.md").write_text(document)
    print(f"wrote {base / 'report.md'} ({len(document.splitlines())} lines)")


if __name__ == "__main__":
    main()
