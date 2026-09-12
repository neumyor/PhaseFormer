# 低秩压缩机制分析 — 为什么压缩近似中性、平滑有害

> 状态：**计划已冻结，脚本已实现，结果待服务器运行完成后回填。**

---

## 1. 背景与动机

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

## 2. 关键前提披露（不可省略）

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

## 3. 方法（4 个实验）

### 实验 1：全秩权重矩阵 SVD 谱分析

`scripts/analyze_weak_residual_svd_spectrum.py`。对 7 个 setting 各自的全秩
（`head_type="shared"`，即 `direct_nlinear`/`weak_residual` 配置）checkpoint，
取出 `weak_period_residual.linear.weight`（`pred_len × seq_len`），计算 SVD。
指标：奇异值衰减曲线、90/95/99%-累计能量有效秩、participation ratio
`PR=(Σs)²/Σs²`、条件数。检验"低秩扫描性能曲线的拐点"是否对应"奇异值谱的拐点"。
纯 CPU，无需数据集，不需要重建完整模型。

### 实验 2：SVD 截断（不重新训练）+ 真实 test 评测

`scripts/analyze_weak_residual_svd_truncation_eval.py`。对每个 setting、每个
该 setting 实际测试过的秩值 `r`：取全秩 checkpoint 的权重做 SVD 截断
`W_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]`，原地替换进一个用全秩 checkpoint
其余权重初始化的完整 `PhaseFormer`，在真实 test 集上评测（无梯度更新）。与
(a) 该 setting 该 rank **实际训练**的低秩模型结果、(b) 全秩基线自身结果 并排
比较。三者接近 ⟹ 学到的映射本身已接近低秩，压缩没有丢掉多少东西；SVD截断显著
差于实际训练的低秩模型 ⟹ 低秩训练学到了比截断更优的解。

### 实验 3：训练出的低秩基向量 FFT 分析

`scripts/analyze_weak_residual_lowrank_basis_fft.py`。对每个 setting 的最小
测试秩 checkpoint，取 `encoder.weight`（`rank×pooled_len`，"读取输入的时间
模式"）与 `decoder.weight`（`pred_len×rank`，"写到输出的时间模式"）的每个模式
向量，做 FFT，报告频谱质心与高频能量占比。验证"压缩后剩下的少数模式，是否仍
覆盖高频/局部化模式"（预期：是），区别于平滑"直接删除高频成分"。

### 实验 4：频段敏感性探针

`scripts/analyze_weak_residual_freq_band_sensitivity.py`。每个 setting 取 3
个模型变体：全秩基线、最小测试秩的低秩模型、causal-EMA 全平滑模型
（`smooth_ratio=1.0`，仅服务器 `causal_ema_smooth_sweep_v1` 有 checkpoint）。
对测试集输入 `x` 做 `rfft`，切成 6 个对数间隔频段（低→高频），逐段清零后
`irfft` 还原为扰动输入 `x'`（`y` 不变），前向评测，记录相对未扰动基线的
`ΔMSE%`/`ΔMAE%`（为控制成本，test 集采样上限 2000 条）。预期：全秩与低秩模型
对高频段扰动都有明显敏感度且形状相近；全平滑变体对高频段扰动近乎不敏感（该
信息已在输入侧被物理移除）——这是"容量轴 vs. 时间分辨率轴"假设最直接的实证。

---

## 4. 结果

> 状态：**待服务器运行完成后回填。**

### 4.1 实验 1：SVD 谱

(占位)

### 4.2 实验 2：SVD 截断 vs. 实际训练低秩

(占位)

### 4.3 实验 3：低秩基向量 FFT

(占位)

### 4.4 实验 4：频段敏感性

(占位)

### 结论

(占位)

### 边界与披露（重申，见 §2）

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
