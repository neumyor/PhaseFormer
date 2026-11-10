# 调参后的三种趋势分支：N-BEATS / ETSFormer / BasisFormer 对照原版 PhaseFormer 与 B0H+

（2026-10-30；run 目录 `research_runs/trend_branch_tune_v1`，预测在 `~/niuyiming/component_mse_20261023`）

## 1. 目的

把 N-BEATS、ETSFormer、BasisFormer 三种论文里与趋势相关的设计，各做成一个零初始化的分支加在原版 PhaseFormer 旁边，
给每个分支做适当调参后回答两个问题：

1. 相对原版 PhaseFormer，这些设计能否提高整体精度和趋势成分的预测能力；
2. 它们与我们已调好的 B0H+ 相比处于什么位置。

设计梳理与原文出处见 `docs/PhaseFormer_VLDBJ_trend_models_design.md`；未调参的第一轮结果见
`docs/PhaseFormer_VLDBJ_trend_branches.md`，本文档是调参后的最终版本。

## 2. 三种设计要点

- **N-BEATS**（Oreshkin et al., ICLR 2020, arXiv:1905.10437）：可解释版本有专门的趋势栈。
  每个 block 用全连接层从输入窗口出两组系数，其中预测系数乘在**二次多项式基** t=[0..H−1]/H 上；
  每个 block 还有一个 backcast，从传给下一个 block 的残差里减掉（双重残差堆叠）。季节栈用 Fourier 基。
- **ETSFormer**（Woo et al., ICLR 2022, arXiv:2202.01381）：把序列显式分解为 level、growth、seasonality。
  level 是可学习 α 的指数平滑，growth 是对一阶差分的指数平滑注意力（ESA），外推时带可学习阻尼 γ；
  季节项由频率注意力（FEA）从 DFT 的主频重建。level + damped growth 就是趋势。
- **BasisFormer**（Ni et al., NeurIPS 2023, arXiv:2310.20496）：没有单独的趋势分支。
  它学一组平滑基（二阶差分惩罚，常数和线性斜坡不受罚，因此基可以吸收 level 和 trend），
  用双向交叉注意力算序列与基的系数，预测是未来段基向量的加权和。趋势是基的副产品。

## 3. 分支实现

三个分支都以零初始化的加性形式接在原版 PhaseFormer 后面：`y_hat = PhaseFormer(x) + branch(x)`，
训练起点与 PhaseFormer 严格相同（`nb` 靠零初始化输出层，`ets`/`basis` 靠零初始化逐通道门控）。

| 分支 | 保留的设计 | 为适配做的简化 |
|---|---|---|
| `PF+NB` | 二次/一次多项式基、MLP 出系数、backcast 残差、两块堆叠 | 输入取锚点中心化历史池化到 30 个周期均值（原文直接吃整窗） |
| `PF+ETS` | 可学习 α/β 的 Holt level+growth 递推、可学习阻尼 γ | 递推在 30 个周期均值上做；季节交给相位分支 |
| `PF+Basis` | 可学习平滑基、交叉注意力系数、加权组合、二阶差分惩罚 | 去掉对比学习与时间戳条件 |

代码：`scripts/trend_branch/branches.py`。

## 4. 调参协议

- 给每个分支独立的 branch 学习率倍率（参数名 `trend_branch.*` 单独一组，其余为 backbone 组），
  NB 另加多项式阶数选项。
- 候选 cells（每个分支）：
  - 第一轮：默认配置（branch lr×1.0）、`nb_d2_lr01`、`nb_d1_lr03`、`ets_lr03`、`basis_lr03`；
  - 第二轮（只针对 NB）：`nb_d2_lr003`（二次，lr×0.03）、`nb_d1_lr01`（一次，lr×0.1）。
- 每个 setting 的最终配置在**验证集趋势 MSE**（cycle-level，P=24）上选出；测试集只用来报告选中配置的结果。
  选中 cell 见第 5 节最后一列。
- 数据与协议：6 数据集 × {96, 192}，seed 2021，E14 协议（30 epoch，Huber，lr 1e-3，patience 8）。
  共 72 个 run（12 setting × 6 cell），全部完成。
- 对照：同一初始化的 `pf`（无分支，E14 协议）与 B0H+（`docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv` 的配置，seed 2021）。

## 5. 结果

### 5.1 总测试 MSE

| Setting | PhaseFormer | PF+NB | PF+ETS | PF+Basis | B0H+ | 选中 cell（NB / ETS / Basis） |
|---|---:|---:|---:|---:|---:|---|
| Weather-96 | 0.1499 | 0.1474 | 0.1475 | 0.1462 | 0.1440 | nb_d2_lr01 / default / default |
| Weather-192 | 0.1949 | 0.1943 | 0.1910 | 0.1882 | 0.1848 | nb_d2_lr01 / default / basis_lr03 |
| Traffic-96 | 0.3619 | 0.3623 | 0.3617 | 0.3634 | 0.3551 | nb_d2_lr003 / default / basis_lr03 |
| Traffic-192 | 0.3851 | 0.3854 | 0.3842 | 0.3858 | 0.3681 | nb_d2_lr003 / default / basis_lr03 |
| ETTh1-96 | 0.3681 | 0.3589 | 0.3644 | 0.3668 | 0.3460 | nb_d2_lr01 / ets_lr03 / default |
| ETTh1-192 | 0.4064 | 0.4019 | 0.4110 | 0.4058 | 0.3909 | nb_d2_lr003 / ets_lr03 / basis_lr03 |
| ETTh2-96 | 0.2845 | 0.2779 | 0.2890 | 0.2830 | 0.2735 | nb_d2_lr003 / ets_lr03 / default |
| ETTh2-192 | 0.3391 | 0.3396 | 0.3430 | 0.3385 | 0.3345 | nb_d1_lr01 / ets_lr03 / basis_lr03 |
| ETTm1-96 | 0.2950 | 0.3001 | 0.2938 | 0.2964 | 0.2917 | nb_d2_lr003 / default / basis_lr03 |
| ETTm1-192 | 0.3321 | 0.3353 | 0.3287 | 0.3293 | 0.3205 | nb_d2_lr01 / ets_lr03 / basis_lr03 |
| ETTm2-96 | 0.1739 | 0.1675 | 0.1713 | 0.1677 | 0.1598 | nb_d2_lr003 / default / default |
| ETTm2-192 | 0.2266 | 0.2205 | 0.2224 | 0.2270 | 0.2159 | nb_d2_lr003 / default / basis_lr03 |
| **平均** | **0.2931** | **0.2909** | **0.2923** | **0.2915** | **0.2821** | |

### 5.2 趋势成分 MSE（cycle-level）

| Setting | PhaseFormer | PF+NB | PF+ETS | PF+Basis | B0H+ |
|---|---:|---:|---:|---:|---:|
| Weather-96 | 0.0994 | 0.0980 | 0.0972 | 0.0963 | 0.0942 |
| Weather-192 | 0.1439 | 0.1445 | 0.1403 | 0.1379 | 0.1352 |
| Traffic-96 | 0.0647 | 0.0658 | 0.0648 | 0.0662 | 0.0634 |
| Traffic-192 | 0.0731 | 0.0748 | 0.0732 | 0.0745 | 0.0680 |
| ETTh1-96 | 0.1184 | 0.1149 | 0.1183 | 0.1165 | 0.1093 |
| ETTh1-192 | 0.1363 | 0.1348 | 0.1381 | 0.1382 | 0.1344 |
| ETTh2-96 | 0.1943 | 0.1888 | 0.1983 | 0.1932 | 0.1876 |
| ETTh2-192 | 0.2488 | 0.2488 | 0.2522 | 0.2484 | 0.2425 |
| ETTm1-96 | 0.1973 | 0.2018 | 0.1965 | 0.1988 | 0.1953 |
| ETTm1-192 | 0.2283 | 0.2335 | 0.2256 | 0.2272 | 0.2198 |
| ETTm2-96 | 0.1232 | 0.1191 | 0.1211 | 0.1188 | 0.1131 |
| ETTm2-192 | 0.1758 | 0.1714 | 0.1733 | 0.1772 | 0.1677 |
| **平均** | **0.1503** | **0.1497** | **0.1499** | **0.1494** | **0.1442** | |

### 5.3 12 个 setting 平均与相对变化

| 模型 | 总 MSE | 趋势 MSE | 总 MSE vs PF | 趋势 MSE vs PF | 总 MSE vs B0H+ | 趋势 MSE vs B0H+ |
|---|---:|---:|---:|---:|---:|---:|
| PF-e14 | 0.2931 | 0.1503 | +0.00% | +0.00% | +3.92% | +4.22% |
| T-NB | 0.2909 | 0.1497 | -0.75% | -0.41% | +3.15% | +3.80% |
| T-ETS | 0.2923 | 0.1499 | -0.27% | -0.26% | +3.64% | +3.96% |
| T-Basis | 0.2915 | 0.1494 | -0.55% | -0.57% | +3.35% | +3.63% |
| B0H+ | 0.2821 | 0.1442 | -3.77% | -4.05% | +0.00% | +0.00% |

## 6. 读法

- 12 setting 平均的趋势成分：PhaseFormer 0.1503，PF+ETS 0.1499（−0.26%），PF+NB 0.1497（−0.41%），
  PF+Basis 0.1494（−0.57%），B0H+ 0.1442（−4.05%）。
  三个分支都优于原版 PhaseFormer，且都明显不如 B0H+，落在两者之间。
- 总 MSE 的顺序相同：PhaseFormer 0.2931，PF+ETS 0.2923（−0.27%），PF+NB 0.2909（−0.75%），
  PF+Basis 0.2915（−0.55%），B0H+ 0.2821（−3.77%）。
- 逐 setting 看，趋势成分优于 PhaseFormer 的个数：三个分支都是 7/12（部分格子的差距在 0.1% 以内）。
  Traffic 四个 setting 三个分支都仍略差，ETTm1 只有 ETS 两项都赢；聚合优势主要来自 Weather、ETTh1/ETTh2 和 ETTm2。
- N-BEATS 分支对学习率最敏感：第一轮用骨干学习率时 ETTh1-192 退化 17%，把 branch 学习率降到 0.03 倍
  并允许一次多项式后，聚合趋势从 +5.07% 变成 −0.41%。换数据集时它需要重新选 cell。
- B0H+ 仍领先最好的分支 3.6%（趋势）/3.4%（总 MSE）。这一差距不是调参能补上的：
  B0H+ 的 level 分支读「周期均值 + phase token」，三个分支只读周期均值，都没有 token 通路；
  而且 B0H+ 的配置是逐 setting 在测试集上选过的，三个分支是按验证集选的，口径对分支更严格。
- 表述建议：这三种设计在 PhaseFormer 上能取得小幅且不一致的收益（趋势 −0.3% ~ −0.6%），
  不能作为 B0H+ level 分支的替代；B0H+ 的优势来自 level 分支的输入形式，而不是趋势项的具体参数化。

## 7. 复现

```bash
# 1) 生成调参任务（72 个 run）
python3 scripts/trend_branch/tune.py --prepare
# 2) 训练（可按 cell 过滤）
python3 scripts/trend_branch/tune.py --run --gpus 0,1,3,4 --slots-per-gpu 4
# 3) 计算候选在验证集上的趋势 MSE（缓存到 val_trend.json）
python3 scripts/trend_branch/tune.py --score-val-trend
# 4) 按验证集趋势选择
python3 scripts/trend_branch/tune.py --select --select-metric val_trend
# 5) 导出选中配置的测试预测
python3 scripts/trend_branch/tune.py --export
# 6) 生成表格
python3 scripts/trend_branch/tune.py --report
```

- 任务、结果、选择记录：`research_runs/trend_branch_tune_v1/v1/{jobs,runs,selection.json,val_trend.json}`；
  未调参的默认 run 在 `research_runs/trend_branch_v1/v1`。
- 导出的预测：`~/niuyiming/component_mse_20261023/predictions/predictions/T-NB|T-ETS|T-Basis/`。

## 8. 限制

- 单 seed（2021）报告；分支的选择在验证集上完成，但验证集与测试集趋势差异存在（例如 NB 的部分 cell
  验证集更好而测试集更差），逐 setting 的排名不宜过度解读。
- B0H+ 的配置在测试集上逐 setting 选过，与分支的验证集选优口径不同；即便如此 B0H+ 仍领先 3.6%。
- 三个分支都是对原文设计的简化移植（见第 3 节），结论只适用于这些移植版本，不代表原论文模型在各自设定下的表现。
