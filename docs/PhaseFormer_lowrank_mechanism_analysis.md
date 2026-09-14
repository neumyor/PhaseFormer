# 低秩压缩机制分析 — 为什么压缩近似中性、平滑有害

> 状态：**4 个实验已在服务器全部跑完，结果已回填（见 §4）。实验 4 的频段
> 扰动探针因设计局限未能直接验证核心假设，详见 §4.4 与结论。**

---

## 1. 实验目标

`PhaseFormer_residual_smooth_ratio_sweep_experiment.md`（boxcar 平滑）与
`PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md`（causal EMA 平滑）
两轮实验一致表明：对全秩残差头做平滑（无论对称有限窗口还是单侧指数衰减）总体
有害或中性偏害；而更早的 `PhaseFormer_rank_sweep_conditioned_experiment.md`
表明低秩压缩（限制 `weak_period_residual` 线性映射的秩）在很宽的压缩比范围内
近似性能中性。

用户要求：明确低秩压缩具体为什么有效而不影响性能——它到底保留了时间序列的
什么成分。工作假设：压缩（秩）限制的是线性映射能输出的**独立模式数量**，但每个
模式本身仍可以是高频/相位敏感的，即压缩不预先过滤输入；平滑则是直接对**输入**
做低通滤波，无论下游映射的"容量"多大，高频信息已经被物理删除。这是"容量轴"
与"时间分辨率轴"的区分。

本轮设计 4 个互补实验检验该假设。除实验 4 外均不新增任何模型训练（纯粹对已有
checkpoint 的事后分析），按 `CLAUDE.md` 的界定不触发 `/experiment-and-error-analysis`
skill（假设讨论/已有聚合结果分析/不含新训练执行的请求不触发该 skill）；实验 4
同样不引入新的训练网格，只是对已有 checkpoint 做前向扰动评测。

---

## 2. 实验设置

### 2.1 关键前提披露（不可省略）

1. **7 个 setting 及其 (gate_init, learning_rate) 组合仍是此前 test-exposed
   事后挑选的结果**（见 `PhaseFormer_rank_sweep_conditioned_experiment.md` §2），
   本轮复用同一批 checkpoint，不产生新的盲测数据，但同样不得表述为无偏泛化估计。
2. **单 seed 2021**，与此前所有轮一致。
3. **本轮不是预注册假设检验**：4 个实验均为探索性/诊断性分析（三个训练无关的
   事后分析 + 一个基于已有 checkpoint 的前向扰动探针），不设跨 setting 计数式
   判定规则，也不触发任何止损条款；结论只是"这 7 个 setting 上观察到的机制性
   证据"，不构成"低秩压缩机制"的普适性证明。
4. **实验 4 的"低秩变体"选用每个 setting 实际测试过的最小秩**（如 H=96 时
   rank=3），"平滑变体"选用 causal-EMA `smooth_ratio=1.0`（全平滑），均为已有
   checkpoint，不新增训练配置。
5. 本实验不修改任何 preset 默认值。

---

### 2.2 方法（4 个实验设计）

#### 实验 1：全秩权重矩阵 SVD 谱分析

`scripts/analyze_weak_residual_svd_spectrum.py`。对 7 个 setting 各自的全秩
（`head_type="shared"`，即 `direct_nlinear`/`weak_residual` 配置）checkpoint，
取出 `weak_period_residual.linear.weight`（`pred_len × seq_len`），计算 SVD。
指标：奇异值衰减曲线、90/95/99%-累计能量有效秩、participation ratio
`PR=(Σs)²/Σs²`、条件数。检验"低秩扫描性能曲线的拐点"是否对应"奇异值谱的拐点"。
纯 CPU，无需数据集，不需要重建完整模型。

#### 实验 2：SVD 截断（不重新训练）+ 真实 test 评测

`scripts/analyze_weak_residual_svd_truncation_eval.py`。对每个 setting、每个
该 setting 实际测试过的秩值 `r`：取全秩 checkpoint 的权重做 SVD 截断
`W_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]`，原地替换进一个用全秩 checkpoint
其余权重初始化的完整 `PhaseFormer`，在真实 test 集上评测（无梯度更新）。与
(a) 该 setting 该 rank **实际训练**的低秩模型结果、(b) 全秩基线自身结果 并排
比较。三者接近 ⟹ 学到的映射本身已接近低秩，压缩没有丢掉多少东西；SVD截断显著
差于实际训练的低秩模型 ⟹ 低秩训练学到了比截断更优的解。

#### 实验 3：训练出的低秩基向量 FFT 分析

`scripts/analyze_weak_residual_lowrank_basis_fft.py`。对每个 setting 的最小
测试秩 checkpoint，取 `encoder.weight`（`rank×pooled_len`，"读取输入的时间
模式"）与 `decoder.weight`（`pred_len×rank`，"写到输出的时间模式"）的每个模式
向量，做 FFT，报告频谱质心与高频能量占比。验证"压缩后剩下的少数模式，是否仍
覆盖高频/局部化模式"（预期：是），区别于平滑"直接删除高频成分"。

#### 实验 4：频段敏感性探针

`scripts/analyze_weak_residual_freq_band_sensitivity.py`。每个 setting 取 3
个模型变体：全秩基线、最小测试秩的低秩模型、causal-EMA 全平滑模型
（`smooth_ratio=1.0`，仅服务器 `causal_ema_smooth_sweep_v1` 有 checkpoint）。
对测试集输入 `x` 做 `rfft`，切成 6 个对数间隔频段（低→高频），逐段清零后
`irfft` 还原为扰动输入 `x'`（`y` 不变），前向评测，记录相对未扰动基线的
`ΔMSE%`/`ΔMAE%`（为控制成本，test 集采样上限 2000 条）。预期：全秩与低秩模型
对高频段扰动都有明显敏感度且形状相近；全平滑变体对高频段扰动近乎不敏感（该
信息已在输入侧被物理移除）——这是"容量轴 vs. 时间分辨率轴"假设最直接的实证。

---

## 3. 实验结果

> 状态：**7 个 setting 全部跑完，以下为服务器实测结果回填。**
> 原始数据：`research_runs/lowrank_mechanism_analysis_v1/*.csv`，图见同目录 `figures/`。

### 3.1 实验 1：SVD 谱

`svd_spectrum_summary.csv`。90%/95%/99% 累计能量所需秩（`rank_90/95/99`）与
实际测试过的秩网格（`tested_ranks`）对比：

| setting | rank_90 | rank_95 | rank_99 | participation_ratio | 已测秩网格 |
|---|---|---|---|---|---|
| ETTh2-96 | 13 | 18 | 33 | 20.7 | 3,6,12,24,96 |
| ETTh2-720 | 35 | 64 | 212 | 101.1 | 22,45,90,180,720 |
| ETTm2-96 | 14 | 18 | 27 | 20.0 | 3,6,12,24,96 |
| ETTm2-192 | 13 | 18 | 36 | 24.9 | 6,12,24,48,192 |
| Weather-96 | 4 | 11 | 57 | 16.1 | 3,6,12,24,96 |
| Weather-192 | 8 | 34 | 122 | 33.4 | 6,12,24,48,192 |
| Electricity-336 | 106 | 126 | 166 | 160.2 | 10,21,42,84,336 |

除 Electricity 外，6 个 setting 的 90%-能量秩都落在已测网格的中低段
（约 4–35），participation ratio 也普遍较低（16–33），说明全秩权重矩阵本身
的谱衰减很快。Electricity-336 明显是例外：90%-能量秩高达 106，
participation ratio 160，谱衰减慢得多——全秩权重矩阵本身并不"天然低秩"。
但结合实验 2/4.2 可见，Electricity 在 rank=10（远低于 90%-能量秩 106）时
训练出的低秩模型性能仍几乎不掉（见下），说明"性能中性所需的秩"系统性地
低于"保留原始权重矩阵 90% 能量所需的秩"——**任务相关的有效秩比原始权重矩阵
的谱有效秩更低**，这是低秩压缩能中性生效的第一层证据。

### 3.2 实验 2：SVD 截断 vs. 实际训练低秩

`svd_truncation_eval.csv`。多数 setting（ETTh2-96/720、ETTm2-96/192、
Weather-96/192）上，"SVD 截断不重训"与"实际训练的低秩模型"的 test
MSE/MAE 在各秩上都相差很小（多在 1–3% 以内），且两者都接近全秩基线——
说明这些 setting 上全秩映射本身已经足够接近低秩，直接截断也能保留大部分
性能，训练与否差别不大。

**Electricity-336 是明显反例**：rank=10 时，SVD 截断 MSE=0.2104，而实际
训练的低秩模型 MSE=0.1629（几乎等于全秩基线 0.1617，仅高 0.7%），截断比
训练结果差 **约 29%**；rank=21 时差距仍有 16%（0.1878 vs 0.1628）；直到
rank=84 才基本追平。这与实验 1 的发现一致：Electricity 的全秩权重本身谱
衰减慢（非低秩），因此单纯截断会丢失大量信息；但秩约束下的**训练**能找到
一个和截断解完全不同、明显更优的低秩解——说明低秩压缩的"性能中性"不是
（或不仅是）因为"权重矩阵本来就是低秩的"，训练过程本身能在秩约束下重新
分配权重、找到任务相关的低维子空间，而不是被动地丢弃截断掉的方向。

### 3.3 实验 3：低秩基向量 FFT

`lowrank_basis_fft_summary.csv`。所有 7 个 setting 的 encoder 模式（"从输入
读取的时间模式"）高频能量占比普遍在 0.18–0.46 之间，谱质心多在
`seq_len` 的中高段（如 ETTh2-720 的 22 个模式质心集中在 137–161，接近
`rfft` 谱总长 361 的中点偏高），并非集中在低频。decoder 模式（"写到输出的
时间模式"）高频占比整体更低（约 0.03–0.19）、谱质心更低，符合直觉——预测
horizon 更短、输出端天然更平滑一些，但仍非零、仍保留局部/高频结构，并未
被"抹平"成纯低通模式。整体上，**低秩压缩保留下来的少数模式本身仍是
高频/局部化敏感的**，与"平滑=直接对输入做低通滤波"在性质上不同——这支持
了"容量轴"假设的定性部分：压缩限制的是模式数量，不是模式的频率内容。

### 3.4 实验 4：频段敏感性（核心探针，结果与预期不符）

`freq_band_sensitivity.csv`。**关键发现：本实验未能观察到"全平滑变体对高频
段扰动明显更不敏感"的预期效应。** 以最高频段（band_index=5）的 ΔMSE% 为例：

| setting | full_rank | low_rank | smoothed |
|---|---|---|---|
| ETTh2-96 | -0.43% | -0.69% | -0.87% |
| ETTh2-720 | -0.82% | -0.82% | -0.98% |
| ETTm2-96 | -0.14% | -0.59% | -0.36% |
| ETTm2-192 | 0.09% | -0.53% | 0.47% |
| Weather-96 | 1.41% | **5.50%** | 3.38% |
| Weather-192 | 1.14% | 1.26% | 2.00% |
| Electricity-336 | 6.50% | 6.27% | 6.20% |

三个变体在每个 setting、每个频段上的 ΔMSE%/ΔMAE% 都高度接近（同号、量级
相近），部分 setting（如 Weather-96）甚至是 low_rank 或 smoothed 变体比
full_rank 更敏感，而非预期中 smoothed 明显更不敏感。此外所有变体在
band_index=0（最低频段，含直流/趋势分量）上都出现灾难性劣化（ETTm2-96
上高达 5000%+），且三变体几乎完全一致——这是抹掉均值/趋势导致的分布外
输入，与"高频信息是否被保留"这一核心问题无关。

**诊断**：这不代表"平滑没有滤掉高频信息"的假设本身错误（实验 3 已从基向量
谱直接证实低秩模式确实携带高频内容），而是**探针设计的局限**：本实验的
`x_transform` 扰动作用在送入整个 `PhaseFormer` 的输入 `x` 上，而非只作用在
`weak_period_residual` 分支内部——`smooth_ratio=1.0` 的因果 EMA 平滑只发生
在残差分支内部对（分支私有的）输入表示做平滑，PhaseFormer 的主干（phase
attention 等）三个变体都是同一套架构、都直接吃未平滑的原始 `x`。因此外部
频段扰动测的是"整个模型对输入频段的敏感度"，由主干（三变体共享、未变）主导，
残差分支内部平滑与否的差异被主干的响应淹没，探针没有分离出目标机制。
若要真正验证"平滑变体的残差分支对高频扰动不敏感"，需要把频段扰动做在残差
分支的内部输入（即分支自己看到的那份序列）上，而不是模型的公共输入 `x`——
这是本轮遗留的方法学局限，留待后续单独设计探针验证，不在本轮结论中计入。

## 4. 实验分析

### 4.1 综合结论

- 全秩权重矩阵谱衰减普遍较快（6/7 个 setting 的 90%-能量秩仅 4–35），
  但即使在谱衰减慢、参与比高的 Electricity-336 上，**训练出的低秩模型**
  性能仍几乎中性——说明性能中性主要靠"秩约束下的训练能找到任务相关的低维
  子空间"，而不只是"全秩权重恰好已经低秩"（实验 1+2 共同支持）。
- 压缩保留下来的少数模式（encoder/decoder 基向量）本身仍含有相当比例的
  高频能量、并非被压成纯低通模式（实验 3 支持"容量轴≠时间分辨率轴"的
  定性假设）。
- 但本轮设计的频段敏感性探针（实验 4）**未能给出该假设的直接因果验证**：
  由于扰动作用在模型公共输入而非残差分支私有输入，三个变体（全秩/低秩/
  全平滑）对每个频段的敏感度几乎一致，观测不到"平滑变体对高频扰动更不
  敏感"的预期信号——这既可能是探针设计缺陷（更可能），也不能排除是
  假设本身需要修正的可能性，本轮证据不足以区分这两者。
- **综合判断（受限于上述探针局限）**：低秩压缩之所以近似性能中性，现有
  证据更支持"训练能在秩约束下主动重组出任务相关的低维子空间，且该子空间
  本身仍保留高频/局部时间模式"，而不是"低秩子空间恰好只是全秩解的能量
  主成分截断"。平滑之所以有害，本轮未能通过频段扰动探针直接量化验证，
  需要修正探针（作用于残差分支私有输入）后才能给出更可靠的因果证据。

### 4.2 边界与披露（重申，见 §2.1）

- 7 个 setting 及其 (gate_init, learning_rate) 组合均为用户指定的 test-exposed
  条件性配置，非盲测、非无偏泛化估计。
- 单 seed 2021。
- 本结果为探索性机制分析，不构成"低秩压缩机制"的普适性/可对外声明证据。

---

## 5. 复现

### 5.1 运行命令

```bash
/home/yyk/yyk03/miniconda3/envs/time/bin/python scripts/analyze_weak_residual_svd_spectrum.py
/home/yyk/yyk03/miniconda3/envs/time/bin/python scripts/analyze_weak_residual_svd_truncation_eval.py
/home/yyk/yyk03/miniconda3/envs/time/bin/python scripts/analyze_weak_residual_lowrank_basis_fft.py
/home/yyk/yyk03/miniconda3/envs/time/bin/python scripts/analyze_weak_residual_freq_band_sensitivity.py
```

无命令行参数：全部 7 个 setting 一次性跑完；结果写入
`research_runs/lowrank_mechanism_analysis_v1/`。

### 5.2 结果与产物位置

- `research_runs/lowrank_mechanism_analysis_v1/{svd_spectrum_summary.csv,
  svd_truncation_eval.csv, lowrank_basis_fft_summary.csv,
  freq_band_sensitivity.csv, figures/}`。
- `research_runs/` 已 gitignore，产物仅本地/服务器保留，不入库。

### 5.3 复现环境

- 服务器：8× A800-80GB，conda env `time`
  （`/home/yyk/yyk03/miniconda3/envs/time/bin/python`）；实验 1、3 纯 CPU 也可
  跑，实验 2、4 需要构建模型+跑 dataloader，用单卡 GPU 即可。
- 同步规范：见 `REMOTE_SERVER.md`。

---

## 6. 关联文档

- **低秩压缩比扫描（本轮分析对象的来源）**：[`PhaseFormer_rank_sweep_conditioned_experiment.md`](PhaseFormer_rank_sweep_conditioned_experiment.md)
- **boxcar 平滑扫描**：[`PhaseFormer_residual_smooth_ratio_sweep_experiment.md`](PhaseFormer_residual_smooth_ratio_sweep_experiment.md)
- **causal-EMA 平滑扫描（本轮"平滑变体"checkpoint 来源）**：[`PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md`](PhaseFormer_residual_causal_ema_smooth_sweep_experiment.md)
- **金标准参照**：[`PhaseFormer_gold_standard.md`](PhaseFormer_gold_standard.md)
- **操作记录**：[`agent-log.md`](agent-log.md)
