# PhaseFormer 输入成分利用问题：现有实验全景汇总

> 更新：2026-09-03。本文汇总所有已经完成、与下述研究问题直接相关的实验；它不是新的效果
> 声明。除 H1/H3/H4 D0 外，所有候选发现实验都限定于 ETTm1、lookback=720、horizon=192、
> seed=2021、validation-only。除非单独标注，表中的“损失”均为 MAE 相对同模型完整输入匹配基线的
> 百分比变化，正数表示更差。

## 1. 研究问题与三种必须区分的结论

目标不是笼统地问“成分 A 是否有用”，而是寻找以下模式：**A 对预测有用，原版 PhaseFormer 没有
充分使用 A，而含 NLinear 的增强模型在使用 A。**

模型定义固定如下。

|简称|机制|结构含义|
|---|---|---|
|M0|`original`|原版 phase-only PhaseFormer|
|M1|`weak_residual`|M0 + 共享 NLinear-style 残差分支 + 静态融合 gate|
|M2|`rcrf_nlinear_plain`|M0 + 同一 NLinear-style 分支 + RCRF 融合；不含额外 phase calibration|

这项问题至少有三种不同的可观测量，不能互相替代。

|证据|比较|它回答什么|不能回答什么|
|---|---|---|---|
|即时依赖（frozen remove）|full 训练 → remove 验证，相对 full → full|完整输入下、已经训练好的模型是否立刻依赖 A|模型在长期缺失 A 后能否学习替代策略|
|恢复能力（remove-trained）|remove 训练 → remove 验证，相对 full → full|没有 A 时模型能恢复到什么程度、会不会利用替代信息|完整输入模型当前是否实际使用 A|
|分支实际利用|固定 phase 输出与 gate，仅以 remove 输入重算/替换 NLinear 分支|NLinear 分支在完整模型中是否使用 A|没有 A 时整个模型是否可恢复|

因此，“M1/M2 在 remove-trained 后损失较小”严格表示**增强模型更能用 A 以外的信息补偿 A 的
缺失**。它不能单独推出“增强分支不依赖 A”；该判断还需要同一 A 的 frozen remove 和分支反事实。

后续应补齐的 2×2 矩阵为：

|训练输入|验证输入|用途|
|---|---|---|
|full|full|匹配基线|
|full|remove A|完整模型对 A 的即时依赖|
|remove A|remove A|本文 D1/D2/D3 已测的恢复能力|
|remove A|full|重新训练的分布适应代价（辅助诊断）|

只有当 M0 的即时依赖近零、M1/M2 的即时依赖/分支反事实显著为正、而 matched control 不足以解释
差异时，才能称 A 为目标成分。

## 2. 共同实验边界与基线

- 输入均在训练集 scaler 标准化后、模型内部 RevIN 前处理；目标 `y` 和时间标记不改写。
- D1/D2/D3 的 matched full-input anchor（30 epoch 上限）为：M0 MAE=0.458641、M1=0.461455、
  M2=0.462059。这是**实验内 matched rerun**，不是论文金标准；正式模型提升仍应以
  `docs/PhaseFormer_gold_standard.md` 为准。
- D1/D2/D3 没有读取 test；均为单 seed，不报告跨 seed 显著性或泛化结论。
- 全部原始运行、checkpoint 和日志位于 `research_runs/` 的 gitignore 目录；本文只引用其汇总 CSV。

## 3. H1/H3/H4：跨数据集的早期因果消融（D0）

来源：[阶段性汇报](PhaseFormer_input_component_H1_H3_H4_stage_report_D0.md)。这是目前范围最广的
实验：7 数据集、horizon=192、seed=2021，且含 frozen、重训练、matched sham 与 test 阶段；结论为
`provisional (seed2021 only)`。它检验的成分为：

|假设|成分|
|---|---|
|H1|跨 24 步周期、同一 phase slot 的残差|
|H3|去除稳定 phase 模板后的近期水平漂移/局部趋势|
|H4|相位漂移/时间扭曲及其跨周期演化|

### 3.1 关键结果（7 数据集宏平均）

下表是 `minus_A` 的 retrain MAE 损失；括号为相对 M0 的 interaction。正 interaction 才支持“增强
模型更依赖”。

|成分|M0|M1|M2|主要结论|
|---|---:|---:|---:|---|
|H1 跨周期残差|+30.9%|+33.4% (+2.4pp)|+32.7% (+1.7pp)|三模型均高度依赖；增量 interaction 很小|
|H3 近期水平漂移|+1.3%|+1.0% (-0.3pp)|+1.4% (+0.1pp)|效应弱，未出现稳定增强依赖|
|H4 相位漂移演化|+3.3%|+2.3% (-1.0pp)|+2.3% (-1.0pp)|增强模型反而更可恢复|

但 H1/H3/H4 都存在关键限制：matched sham 与删除同样有害甚至更有害。例如 frozen H1 的
M0 `minus_A=+31.9%`、`sham=+63.1%`（MAE）；H3/H4 中 sham 也经常约等于或大于删除。故这些结果主要
表明输入扰动/分布变化的鲁棒性差异，不能给出成分专属的“原版未用、增强在用”证据。

### 3.2 D0 的可用结论

1. H1、H3、H4 都没有通过目标模式的必要条件。
2. H4 的负 interaction 和 H1 frozen sham 的明显负 interaction 提示增强模型在某些扰动下更鲁棒，
   不等于它们对相关成分更不依赖。
3. 由于单 seed、sham 混淆和部分 test exposure，D0 不能作为最终拒绝或最终确认；其价值主要是排除
   这些具体处理方式的强结论。

## 4. C1--C7：冻结的候选发现（ETTm1-H192）

来源：[候选发现计划 §3、§9](PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md) 与审计报告
`research_runs/input_candidate_discovery_ettm1_h192_v1/objective_error_analysis.md`。

### 4.1 设定

- 三个 full-input anchor 各训练最多30 epoch；S1a 在 validation 均匀抽取512个 origins，测试7个
  候选、两种低剂量 remove 和 matched sham；C2/C3/C7 晋级 S1b 全 validation（11,329 origins）。
- C1: 96步日周期增量；C2: 672步周内低频；C3: 非24整齐频带；C4: 周期边界连续性；C5: 周期间
  幅度包络；C6: 平滑相位速度；C7: 最后24步的 train-fitted 因果创新。
- 设计还包含 PF 残差 probe、移动块 CI 与 RCRF 的四类输出重组；若无候选通过门槛，就不重训、
  不读 test。

### 4.2 结果与解释

|候选/阶段|代表性结果|结论|
|---|---|---|
|C1、C4、C5、C6、S1a|未晋级 S1b|没有出现预注册的模型差异信号|
|C2、S1b|remove-vs-sham MAE：M0 -7.92%、M1 -8.64%、M2 -7.94%|sham 比真实删除更有害，control/intervention mismatch；不可解释为成分专属利用|
|C3、S1b、50%剂量|sham-adjusted MAE：M0 +0.69%、M1 +0.93%、M2 +0.24%；全部移动块CI跨0|weak 的名义优势不足 1% 效应门和 interaction 门|
|C7、S1b、预测步1--24|remove/sham MAE 变化（pp）：M0 +0.822/+0.496，M1 +0.999/+1.073，M2 +0.316/+1.261|M0 并不等效不敏感，增强模型也没有更大的 sham-adjusted 依赖|

结论：C1--C7 没有候选通过 S1 的必要门，因此按预注册早停，不进行候选重训或 test。这个阴性结果只
说明该候选库和该 matched-sham 构造未发现目标模式，并不表示不存在任何合适 A。

## 5. D1/D2：频率与近程信息

### 5.1 旧定义（历史保留，不作当前结论）

最初 D1 用训练集拟合的连续正弦/余弦谐波删除，D2 用 train-fitted 因果创新删除；之后用户明确否定
这两个 remove 定义。旧冻结/重训练结果保留在[候选发现计划 §10--§11](PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md)，仅作为审计历史：96步日周期对三模型都重要，近期创新也没有产生稳定的
正 interaction。它们**不应与下节的当前 D1/D2 结果混用**。

### 5.2 当前定义与 remove-trained 结果

当前 D1/D2 设定见[候选发现计划 §12](PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md)：

- D1：每个标准化的720步输入窗口做 rFFT，对目标频率 `f0=1/P` 施加 Gaussian notch，
  `sigma_f=1/720`，保留 DC；测试训练集 periodogram 固定得到的 96、48、32、24、677.647、205.714 步。
- D2：直接将每个标准化输入窗口最后 24/48/96/192 步的所有变量置零。
- 每个成分均在 train 与 validation 同时 remove，并从头训练 M0/M1/M2。它测的是恢复能力，不是
  完整模型的即时依赖。

|成分|M0|M1（相对M0）|M2（相对M0）|
|---|---:|---:|---:|
|D1-96 高斯陷波|+2.95%|+0.64% (-2.31pp)|+0.39% (-2.56pp)|
|D1-48 高斯陷波|+6.38%|+5.37% (-1.01pp)|+4.85% (-1.53pp)|
|D1-32 高斯陷波|+1.35%|+0.44% (-0.92pp)|+0.35% (-1.00pp)|
|D1-24 高斯陷波|+1.10%|+0.59% (-0.51pp)|+0.58% (-0.52pp)|
|D1-677.647 高斯陷波|+7.59%|+7.88% (+0.29pp)|+7.80% (+0.20pp)|
|D1-205.714 高斯陷波|+1.09%|+0.33% (-0.76pp)|+0.29% (-0.80pp)|
|D2-末尾24步置零|+6.53%|+6.10% (-0.43pp)|+6.34% (-0.19pp)|
|D2-末尾48步置零|+11.73%|+10.02% (-1.71pp)|+10.67% (-1.06pp)|
|D2-末尾96步置零|+21.15%|+19.38% (-1.77pp)|+18.54% (-2.61pp)|
|D2-末尾192步置零|+33.52%|+30.20% (-3.32pp)|+29.90% (-3.62pp)|

解释：近期原始观测的缺失对三模型都很重要，且置零长度越长损失越大；但 M1/M2 在每个 D2 长度的
恢复能力都高于 M0。D1 也几乎全部为负 interaction；只有约678步的 +0.20/+0.29pp 单点效应很小，
不能作为候选。这组结果支持“增强分支提高信息缺失下的鲁棒性/替代能力”，不支持“增强分支更依赖
被删除成分”。

## 6. D3：末值锚定的跨周期轨迹

来源：[候选发现计划 §13](PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md)。D3 专门针对
PhaseFormer 的24步 folding 与 NLinear 全时间轴线性映射的结构差异。每个成分都仅从当前720步历史
提取，且移除后最后一个输入值严格不变，避免把 NLinear 的 persistence anchor 丢失误判为结构效应。

|D3 成分|remove 定义|M0|M1（相对M0）|M2（相对M0）|
|---|---|---:|---:|---:|
|global-linear|全窗线性趋势，末点锚定|+1.50%|+1.09% (-0.41pp)|+1.26% (-0.24pp)|
|recent-linear|只用末96步估计趋势，末点锚定后删除全窗方向|+7.20%|+1.65% (-5.55pp)|+2.81% (-4.39pp)|
|cycle-levels|每个24步块均值相对最后块均值的轨迹|+5.11%|+1.95% (-3.16pp)|+1.89% (-3.21pp)|
|phase-drift|每个 phase 在30个周期间的线性漂移|+1.22%|+0.46% (-0.76pp)|+0.61% (-0.62pp)|
|cycle-amplitude|逐周期幅度相对最后周期的包络|+3.16%|+0.98% (-2.18pp)|+1.03% (-2.13pp)|

解释：五项 interaction 全为负。最强的例子是 recent-linear、cycle-levels 和 cycle-amplitude：M0
分别损失 7.20%、5.11%、3.16%，而 M1/M2 只损失约 1--3%。这说明含 NLinear 的模型在这些轨迹信息
缺失时有明显替代能力；它不是“增强模型更依赖这些轨迹”的证据。

## 7. 总结：哪些结论已经成立，哪些尚未成立

### 已有证据支持

1. 对 D1、D2、D3 的当前 remove-trained 定义，NLinear 和 RCRF+NLinear 都通常比原版更能在缺失
   输入成分后恢复；D2 尾部置零与 D3 跨周期轨迹的证据尤其一致。
2. 这更符合“增强路径提供冗余/替代信息利用能力”的解释，而不是“增强路径额外依赖被删除 A”。
3. H1/H3/H4 与 C1--C7 都没有可靠地找到目标模式；其中很多失败由 sham/扰动本身的伤害主导。

### 7.1 D4 补充：分支利用与恢复能力不是同一结论

D4 对 D3 中 `recent-linear`、`cycle-levels` 补充了 full-trained 的冻结输入视图和“仅替换 NLinear
branch”的反事实（详见 [D4 报告](PhaseFormer_input_component_D4_complementary_frozen_report.md)）：

- `recent-linear` 被去除时，M0/M1/M2 的完整输出都立即显著变差（MAE +140.1/+201.0/+172.4%），所以
  M0 绝非没有使用它；M1/M2 的 NLinear-only 反事实也显著变差。
- `cycle-levels` 被去除时，M0 的完整输出损失略大于 M1/M2（+51.8% vs +46.0/+44.8%），但 M1/M2 的
  NLinear-only 反事实仍显著变差（+33.3/+30.1%）。因此“增强模型更稳健”和“增强分支不使用 A”并不
  等价；该分支可以使用 A，同时利用其他信息提供补偿。

这进一步排除了把 D1--D3 的 remove-trained 负 interaction 叙述为“增强分支专门使用 A 以外的信息”的
写法。它们只能支撑增强的**替代/恢复能力**，而不是原版遗漏某个已命名成分的直接证明。

### 尚不能成立的结论

1. 不能说 M1/M2 的增强分支“不依赖”D1/D2/D3 成分：D4 已经直接显示其对两个 D3 成分存在实际利用；
   其余 D1/D2/D3 成分仍未完成对应的 frozen 分支反事实。
2. 不能说 M0“没使用”任何已测成分：D4 的 recent-linear frozen 结果也显示 M0 有很强即时依赖；D1/D2/D3
   的 remove-trained M0 损失均为正，至少说明这些
   成分缺失后 M0 不能完全补偿；这不同于完整模型的即时使用，但绝不是等效不敏感证据。
3. 不能把单 seed、单数据集 validation 结果推广到其他 horizon、数据集或论文金标准。

## 8. 最有信息量的下一步

优先不再扩大 remove-trained 候选库。D4 已完成 `D3-recent-linear`、`D3-cycle-levels` 的冻结和分支级
诊断，均不符合“M0不使用、增强在使用”的目标模式；若继续，应将冻结分支诊断限定到尚未检验的
`D2-192`，并优先寻找一个在 M0 上即时近零、却令 NLinear-only 反事实显著变差的新候选：

1. 对 full-trained checkpoint 做 full/remove validation，测即时依赖；
2. 固定 phase 输出和 RCRF gate，只替换 NLinear 分支在 remove 输入上的预测，测分支实际利用；
3. 运行 remove-trained/full-validation，量化训练后适应造成的输入分布错配；
4. 若某项显示 M0 即时近零、M1/M2 即时与分支反事实显著为正，才扩大到多 seed，并使用未暴露任务做确认。

在此之前，最准确的总体表述是：**增强版本表现出更强的成分缺失恢复能力；尚无实验能够证明存在某个
输入成分被原版忽略、却被增强分支实际使用。**

## 9. 可审计结果位置

|实验|主汇总/报告|
|---|---|
|H1/H3/H4 D0|`docs/PhaseFormer_input_component_H1_H3_H4_stage_report_D0.md`；`research_runs/result_summary_d0.csv`|
|C1--C7|`research_runs/input_candidate_discovery_ettm1_h192_v1/objective_error_analysis.md`；同目录 `results.csv`|
|D1/D2 当前定义|`research_runs/d1_d2_gaussian_tailzero_control/d1_d2_retrained_summary.csv`|
|D3|`research_runs/d3_trajectory_remove_control/d3_trajectory_summary.csv`|
|D4 互补冻结诊断|[D4 报告](PhaseFormer_input_component_D4_complementary_frozen_report.md)；`research_runs/d4_complementary_frozen_probe_control/frozen_complementary_results.csv`|
