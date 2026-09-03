# 从输入成分到结构缺陷：PhaseFormer 的研究叙事与实验证据

> 更新：2026-09-03。本文是当前研究主线的实验记录，而非最终论文效果声明。除单独说明外，所有近期
> 诊断均固定在 ETTm1、lookback=720、horizon=192、seed=2021、validation-only。它们用于机制发现，
> 不代表跨数据集、跨 horizon 或跨 seed 的泛化结论。

## 1. 要回答的问题

研究的起点是一个结构问题：PhaseFormer 将历史按固定24步周期重排为 phase series，再以 phase slot
为核心进行表征与预测。这种归纳偏置擅长稳定的周期形状，但可能弱化跨周期的连续时间状态。

为检验这一猜想，我们先引入一个最小、可归因的增强参照：共享的 NLinear-style 全时间轴残差分支。
它不是作为论文创新本身，而是一个**诊断探针**：如果它稳定地修正 PhaseFormer 的某类残差，就能反向
定位 phase-only 表示的不足。RCRF 进一步根据 phase reliability 融合相位与残差路径。

|简称|结构角色|
|---|---|
|M0|原始 PhaseFormer；固定24步 phase folding 的 phase-only 路径|
|M1|M0 + 共享 NLinear-style 全时间轴残差路径 + 静态 gate|
|M2|M0 + 相同 NLinear-style 路径 + RCRF reliability-coupled fusion；无额外 phase calibration|

最初的强假设是：存在输入成分 A，M0 没有充分利用而 NLinear 路径实际利用。实验随后表明这一定义过强；
最终收敛到更可证实、也更贴合模型结构的假设：

> **PhaseFormer 会使用跨周期水平信息，但 phase-only 表示对非平稳的周期水平状态（尤其最后周期相对
> 历史的偏移）建模不足；全时间轴残差路径可以用轻量修正补偿其预测残差。**

## 2. 证据路线与每一步学到的内容

### 2.1 早期 H1/H3/H4：没有找到成分专属的“原版未用”证据

H1（跨周期同相位残差）、H3（局部水平漂移）、H4（相位漂移演化）在7数据集、H192、seed2021上做了
remove/retrain/frozen 诊断。其宏平均 remove-trained MAE 损失如下：

|成分|M0|M1|M2|可支持的观察|
|---|---:|---:|---:|---|
|H1 跨周期残差|+30.9%|+33.4%|+32.7%|三者都依赖，增强并非明显更专属|
|H3 局部水平漂移|+1.3%|+1.0%|+1.4%|效应弱|
|H4 相位漂移演化|+3.3%|+2.3%|+2.3%|增强较可恢复|

但 matched sham 在多项上与真实删除同样甚至更有害（例如 H1 frozen 的 M0：remove `+31.9%`、sham
`+63.1%`），所以不能将这些扰动直接解释为成分利用。这个阶段的价值是建立了重要边界：**输入扰动下
的鲁棒性差异，不等于模型对被删成分的专属依赖差异。**

### 2.2 C1--C7：预注册候选发现得到阴性结果

在 ETTm1-H192 的 validation-only 冻结筛选中，候选包括日/周周期、非整齐频带、周期边界、幅度包络、
相位速度和近期创新。预注册要求同时通过残差可预测、sham 校正 interaction、NLinear-only 反事实等门。

- C2 的 sham 比真实删除更有害，属于 control/intervention mismatch；
- C3 的 sham 校正效应置信区间跨零；
- C7 中 M0 并不等效不敏感，增强模型也没有更大专属依赖。

因此无候选进入重训或 test。此阴性结果排除了“在该候选构造和对照下，存在显著的 M0 未用/M1-M2 在用
成分”的强结论。

### 2.3 D1--D3：增强模型更会恢复，不代表它不依赖被删信息

随后按当前有效定义进行了 remove-trained 对照：D1 为窗口 rFFT Gaussian notch，D2 为末尾原始输入
直接置零，D3 为末值锚定的跨周期轨迹删除。所有任务均在移除后的 train/validation 上重新训练。

|代表成分|M0 训练后损失|M1|M2|现象|
|---|---:|---:|---:|---|
|D1-96频率|+2.95%|+0.64%|+0.39%|增强更能从其余信息恢复|
|D2尾部192置零|+33.52%|+30.20%|+29.90%|近期原始观测对三者都重要，增强恢复略好|
|D3近期线性轨迹|+7.20%|+1.65%|+2.81%|增强对缺失轨迹有很强补偿|
|D3周期水平轨迹|+5.11%|+1.95%|+1.89%|增强对缺失水平状态有很强补偿|
|D3周期幅度轨迹|+3.16%|+0.98%|+1.03%|增强对缺失幅度演化有补偿|

这里的 claim 必须严格限定为：**M1/M2 在缺失输入后能更好地利用剩余上下文。** 不能据此声称 NLinear
不依赖被删除的成分，或 PhaseFormer 完全没有使用它。

### 2.4 D4/D5：NLinear 确实使用许多被删信息，原版也通常立即依赖

D4 以 full-trained checkpoint 比较 `X`、`X-A` 与末值锚定的 A-only 视图，并固定 full-input phase/gate、
只替换 NLinear branch。D5 再将这套冻结分支诊断扩展至当前 D1六项、D2四项、D3五项。

关键的 D4 结果：

|成分与输入|M0 full→remove|M1 full→remove|M2 full→remove|M1 NLinear-only|M2 NLinear-only|
|---|---:|---:|---:|---:|---:|
|D3 recent-linear，`X-A`|+140.1%|+201.0%|+172.4%|+222.3%|+31.2%|
|D3 cycle-levels，`X-A`|+51.8%|+46.0%|+44.8%|+33.3%|+30.1%|

所有 branch replay 误差小于 `4.8e-6`。这直接证明两点：

1. NLinear branch **实际使用** recent-linear 和 cycle-levels；
2. M0 对它们也有很强即时依赖，故它们不是“原版未用”的 A。

尤其 cycle-levels 同时呈现“分支使用 A”与“增强整体 remove 更稳健”。这是最重要的反例：**分支使用某
信息，与整模型在该信息缺失后是否更鲁棒可以同时成立，二者不能互相推导。**

D5 的15项广泛筛查没有出现 M0 近零、M1/M2 branch 显著为正的分离模式。D1-32/D1-24 的 M0 损失虽小
（+0.59%/+0.37%），增强支路也同样小；其余13项 M0 均有超过1%的即时损失。

### 2.5 D6：结构关系也是共同信息

为避免只删除数值成分，D6 直接扰动关系：周期顺序、跨 phase 同步、相邻 phase edge。三种扰动均保留
最后输入点；前两者还保留特定 phase/cycle 边际统计。

|结构扰动|M0|M1|M2|结论|
|---|---:|---:|---:|---|
|早期周期顺序反转|+70.33%|+68.46%|+68.59%|跨周期顺序为共同关键信息|
|phase 去同步|+61.71%|+48.18%|+44.29%|同步关系为共同信息；增强更可恢复|
|相邻 phase pair交换|+0.77%|+0.08%|+0.10%|简单相邻 pair 不是增强专属关系|

M1/M2 的 NLinear-only 反事实在前两项同样明显恶化。因此，结构关系扰动也没有发现“原版不利用、增强
利用”的强候选；它们强化了“增强的主要差异是恢复能力”的解释。

## 3. 关键转折：从输入盲区转向 phase path 的系统性残差

前述实验排除了一个过于简化的故事：不是 NLinear 找到了一种 PhaseFormer 从不看的输入。于是 D7 不再
改变输入，而在完整输入上直接诊断内部路径。

对每个窗口，定义 NLinear 融合收益：

```text
gain = MAE(phase path) - MAE(fused output)
```

并检验 NLinear correction 是否与 phase residual 同方向，以及哪些预先固定的历史描述能预测 `gain`。

|D7 可测量结果|M1|M2|
|---|---:|---:|
|correction 与 phase residual 的平均余弦对齐|0.703|0.791|
|gain 与周期水平波动的相关性|**+0.490**|**+0.534**|
|gain 与最后周期水平偏移的相关性|**+0.483**|**+0.488**|
|六个固定描述量的连续5折 OOF R²|0.205|0.299|
|平均 phase→fused MAE 绝对收益|0.231|0.355|

这组结果的含义是：NLinear correction 并非任意改动输出；它通常朝 phase residual 的正确方向修正，且其
收益集中在**跨周期水平状态不稳定**、尤其最后周期相对历史偏移的窗口。

## 4. 当前可主张的科研 claim

### Claim A：PhaseFormer 的固定 phase folding 对非平稳跨周期水平状态存在建模缺口

**证据：** D7 中周期水平波动和最后周期偏移是预测 NLinear 校正收益的两个最强描述量；NLinear
correction 与 phase residual 高度同向。D3 的 remove-trained 中，缺失周期水平/近期线性轨迹时，增强
模型能明显更好恢复。

**准确表述：** “PhaseFormer 会编码这些信息，但其 phase-only 预测在非平稳周期水平状态下残差更大，
需要额外的连续时间状态校正。”

### Claim B：全时间轴的轻量线性路径为该缺口提供互补校正

**证据：** D4/D5 的 NLinear-only 反事实在 cycle-levels、recent-linear、主要频率和近期观测等条件下
显著变差，证明路径实际使用输入；D7 显示其 correction 与 phase residual 方向一致。

**准确表述：** “NLinear-style 路径提供可与 phase path 融合的、基于完整时间轴的预测校正。”

### Claim C：RCRF 的角色是条件融合，而不是发现一个独占输入成分

**证据：** M2 的分支反事实说明它同样使用这些信息；D1--D3、D6 中，M2 常具有较好的缺失恢复，但并未
呈现一个稳定的增强专属输入候选。

**准确表述：** “RCRF 根据 phase reliability 在 phase 与连续时间校正之间进行条件融合；其价值应以
融合/消融和鲁棒性评估验证，而不是宣称它独占某输入成分。”

## 5. 被实验否定或尚不能主张的说法

- “存在一个已发现的输入 A，原始 PhaseFormer 完全没用、NLinear 专门在用。”——当前证据不支持；
  D1--D6 均未发现此模式。
- “增强模型在 remove-trained 下更稳健，因此它不依赖被删信息。”——被 D4 反事实直接否定。
- “M1/M2 在 ETTm1-H192 的完整输入上已优于原版。”——当前 matched anchor 中 M0 MAE=0.458641，
  M1=0.461455，M2=0.462059，不能作此声明。
- “D7 已证明新机制有效或可泛化。”——不支持；D7 是单数据集、单 seed 的内部关联诊断，仍需候选机制
  的严格对照验证。

## 6. 从诊断到新机制的下一步

最自然的后续结构不是泛化地再加一个 NLinear 分支，而是提出一个**末值锚定的跨周期水平状态校正头**：

1. 从每个24步周期抽取 level，并以最后周期/最后值作为锚；
2. 只预测低维的未来周期水平状态或其相对锚点的校正；
3. 将该校正以受限方式加到 phase forecast，而非训练第二个完整预测器；
4. 以同参数量的非结构化线性头为控制，检验收益是否确实来自“跨周期状态”归纳偏置；
5. 先在 ETTm1-H192 多 seed validation 进行小规模验证，再决定是否扩展数据集和 horizon。

这条路线将创新点放在“由 phase residual 诊断推导的结构化状态校正”，而不是“堆叠一个 residual branch”。

## 7. 审计索引

|阶段|主要文档/产物|
|---|---|
|H1/H3/H4|[D0阶段汇报](PhaseFormer_input_component_H1_H3_H4_stage_report_D0.md)|
|C1--C7|[候选发现计划](PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md)；`research_runs/input_candidate_discovery_ettm1_h192_v1/`|
|D1--D3|[证据汇总](PhaseFormer_input_component_evidence_summary.md)|
|D4|[互补冻结报告](PhaseFormer_input_component_D4_complementary_frozen_report.md)|
|D5|[广泛冻结报告](PhaseFormer_input_component_D5_broad_frozen_report.md)|
|D6|[结构关系报告](PhaseFormer_input_component_D6_structural_relation_report.md)|
|D7|[内部路径报告](PhaseFormer_input_component_D7_internal_path_report.md)|
