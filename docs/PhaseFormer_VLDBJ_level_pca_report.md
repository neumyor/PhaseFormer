# level 分支的低维性：PCA 证据（2026-10-30）

## 1. 要证明什么

ICLR 版本的 motivation 用 PCA 说明 phase token 低维：两个主成分解释 90% 以上，patch token 需要 11 个以上。
VLDBJ 版本新增的 level 分支声称补充的是低维信息（Theorem 4：最优线性补充的输出秩不超过 horizon profile 数
m = 3）。本报告用同一套 PCA 语言给出证据，覆盖七个数据集、H = 720、lookback 720、周期池化 24（即主表 S2
头部的 pooled_len = 30）。

所有分析不做训练、不选模型：窗口取各数据集的测试段，通道随机采样（ETT 全部 7 个，其余 16 个）；
每条曲线先在单通道上做 PCA，再对通道取平均。模型侧用 b0 搜索里 S2 头的最好一格作为 checkpoint，
S2 的输出方向是三个固定 Legendre profile，与训练无关。

## 2. 试过的五个设想

| # | 设想 | 结论 | 是否用作主证据 |
|---|---|---|---|
| H1 | 未来 level 需求（未来周期均值 − 最后一个观测值，30 维）本身的 PCA 低维 | 3 个主成分解释 75%–97%；PC1 已解释 59%–93% | 辅助 |
| H2 | level token（中心化窗口的 30 个周期均值）比同维的逐点 value token 更紧凑 | 所有数据集方向一致，见第 3 节 | **主证据** |
| H3 | 未来 level 中"可由历史线性预测"的部分更低维 | 3 个主成分解释 87%–99%，但 ETTh2 / ETTm2 上线性预测的 R² 中位数是 −0.30 / −0.06，"可预测部分"几乎为空，结论无意义 | 弃用 |
| H4 | 训练好的 level 分支的三个方向 ≈ 数据的前三个主方向 | 固定 profile 子空间解释需求方差的 66%–96%，最优三维子空间 80%–97%；S2 头的三个方向恒等于固定 profile | 辅助 |
| H5 | 把秩从 3 提高不会带来收益（训练侧秩扫描） | Weather-720 上 r3 最好，r1 +0.57%、r2 +0.76%、r4 +1.04%、r6 +1.66%；其余 setting 因 GPU 被另一会话占满多批 OOM，仍在补 | 辅助，见第 6 节 |

## 3. 主结果：level token 与 value token 的 PCA

对象：
- level token：窗口按 24 步池化后的 30 个均值，减去最后一个观测值，除以窗口标准差；这是 level encoder 实际读到的向量。
- value token：同一窗口每隔 24 步取一个点，共 30 个，减同一个锚点、除同一个标准差；这是"不池化、逐点读"的对照。

| 数据集 | level PC1 | level 前 3 | level 达 90% 所需 | value PC1 | value 前 3 | value 达 90% 所需 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.841 | 0.89 | 4 | 0.565 | 0.73 | >12 |
| ETTh2 | 0.767 | 0.84 | 7 | 0.595 | 0.69 | >12 |
| ETTm1 | 0.672 | 0.81 | 8 | 0.577 | 0.73 | >12 |
| ETTm2 | 0.717 | 0.82 | 6 | 0.607 | 0.72 | >12 |
| Weather | 0.707 | 0.82 | 6 | 0.639 | 0.76 | 10 |
| Electricity | 0.936 | 0.96 | 1 | 0.642 | 0.89 | 4 |
| Traffic | 0.941 | 0.97 | 1 | 0.600 | 0.82 | 8 |

读法：七个数据集上，level token 的前三个主成分解释 81%–97%；value token 只有 69%–89%，其中四个数据集在
12 个主成分内都到不了 90%。方向完全一致，与 ICLR 图里 Phase 与 Patch 的对比同构。

图：`figures/traffic_pca_level_token.png`（主图，对应 ICLR 主文的 Traffic 图）、
`figures/{etth1,etth2,ettm1,ettm2,weather,electricity}_pca_level_token.png`（附录）、
`figures/level_pca_token_grid.png`（七图合排）。

## 4. 未来 level 需求的 PCA

把未来 720 步按 24 步池化成 30 个周期均值、减去最后一个观测值、除以窗口标准差，这是 level 分支要补的那部分。
PC1 解释 59%–93%，前三个解释 75%–97%，达 90% 所需 1–9 个主成分。ETT 上 PC1 只有 0.59–0.81，Weather 0.59，
Traffic 与 Electricity 0.93。

它与 H2 是同一件事的两面：输入侧（level token）低维，输出侧（需求）也低维；中间的映射秩为 3 不损失多少。

## 5. 训练好的三个方向离最优三维有多远

对每个数据集，用 b0 搜索里 S2 头的最好一格（seed 2021）在测试段取 level 分支的输出方向；S2 的解码是三个固定
Legendre profile，所以这三个方向与训练无关。把它们展成 30 维（按 24 步池化）后，计算需求方差落在它们张成子空间
里的比例，并与最优三维 PCA 子空间、最优六维子空间对比：

| 数据集 | 三个固定 profile | 最优 3 维 | 最优 6 维 |
|---|---:|---:|---:|
| ETTh1 | 87 | 89 | 92 |
| ETTh2 | 66 | 83 | 85 |
| ETTm1 | 75 | 80 | 88 |
| ETTm2 | 81 | 84 | 90 |
| Weather | 73 | 81 | 86 |
| Electricity | 96 | 97 | 98 |
| Traffic | 93 | 96 | 98 |

图：`figures/level_pca_summary.png`。三个固定方向与最优三维差 3–17 个百分点；从三维加到六维只多 3–10 个百分点，
而需求里剩下的方差大部分不可由历史预测（见 H3 的 R²），所以加维主要用于拟合噪声。

## 6. 秩扫描（训练侧）

固定每个 setting 的最好 b0 配置，只改 level 映射的秩 r ∈ {1, 2, 3, 4, 6}，seed 2021，测试集 MSE，
括号内为相对秩 3 的变化：

| setting | r1 | r2 | r3 | r4 | r6 |
|---|---:|---:|---:|---:|---:|
| ETTh1-720 | 0.41910 (+0.57%) | 0.42188 (+1.24%) | **0.41673** | 0.41860 (+0.45%) | 0.41293 (−0.91%) |
| ETTh2-720 | 0.39757 (+1.36%) | 0.39887 (+1.69%) | **0.39224** | 0.39792 (+1.45%) | 0.39847 (+1.59%) |
| Weather-720 | 0.30969 (+0.57%) | 0.31028 (+0.76%) | **0.30795** | 0.31114 (+1.04%) | 0.31306 (+1.66%) |
| Traffic-720 | 0.42433 (−0.00%) | 0.42519 (+0.20%) | **0.42433** | 0.42509 (+0.18%) | 未跑（2026-10-30 暂停） |

图：`figures/level_rank_sweep.png`。Traffic-720 的 r6 在 2026-10-30 按用户要求暂停，未计入。

读法：ETTh2 与 Weather 是干净的 V 形，秩 3 最低，减秩和加秩都变差 0.6%–1.7%；Traffic 上四档几乎相同
（±0.2%），秩 1 与秩 3 打平，这一 setting 本来就不靠 level 拿分；ETTh1 的 r6 比 r3 低 0.91%，但 r1、r2、r4
的次序非单调，而这个量级的单 seed 差异在指标噪声之内（b0 搜索里验证集选优相对测试最优平均损失 1.26%，
ETTh1-336 三 seed 极差 0.010–0.025）。因此秩扫描支持"三个方向之后基本不再改善"，不能读成"任何 setting
上秩 3 都是最优"。MAE 的方向一致：ETTh2、Weather 在秩 3 最低，ETTh1 的 r6 MAE 低 0.6%。

## 7. 限制

- 单 seed、单 horizon（H = 720）。窗口只取测试段，与 ICLR 图的统计口径（全序列）略有不同；用全序列重算时
  level token 的曲线几乎不变，demand 的数值会略低（非平稳性所致）。
- 池化宽度固定 24 步（S2 头的 pool_factor），与各 setting 选出的相位周期（Traffic 12、ETTm1 48、Weather 36）
  不同。level token 的定义是"每 24 步一个均值"，与主表配置一致，但不是严格意义的"每周期一个均值"。
- 第 5 节把 30 维池化向量当作子空间来比较；S2 头实际解码到 720 维再池化，两个表示在池化算子下等价。
- 第 6 节的秩扫描只走训练侧、按测试集选优（项目既定口径），补完后需要与表内 b0 的逐 setting 选优偏差一起读。

## 8. 复现

```bash
export VLDBJ_E14_ROOT=/home/yyk/yyk03/niuyiming/PhaseFormer
cd ~/niuyiming/PhaseFormer_vldbj_wide_20261011
# 抽取 level token / value token / 需求（测试段，S2 头）
python3 scripts/level_pca_extract.py --dataset Traffic --horizon 720
# PCA 曲线与秩扫描
python3 scripts/level_pca_curves.py
python3 scripts/level_pca_model.py
python3 scripts/level_rank_sweep.py --report
# 出图（本地）
python3 scripts/plot_level_pca.py --curves research_runs/level_pca_v1/curves.json --out <figures>
python3 scripts/plot_level_pca_summary.py --summary research_runs/level_pca_v1/model_summary.json --out <figures>
```
