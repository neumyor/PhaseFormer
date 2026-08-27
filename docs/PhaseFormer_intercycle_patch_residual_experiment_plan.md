# PhaseFormer × ICPT 周期残差模型与位置编码实验计划

> 状态：**Stage 0 实现与静态验证全部通过；Stage A 架构筛选已运行但门槛失败，
> 按 §13 停止 ICPT 主路线（未进入 Stage B/C/D）。** 结果表 9.1/9.2 已填充；
> 9.3–9.8 因停止不适用。详细数据见
> `research_runs/phaseformer_icpt_pe_screen/screen_summary.csv` 与
> `freeze_record.json`。

## 0. 上一轮结论与本轮边界

上一轮 `periodic_residual_pe_v1` 已结束：在 NLinear 外增加 LFF 周期检索后，ETTh2、
ETTm2 获得小幅收益，Electricity 略有退化。该结果说明“显式周期归纳偏置”有条件地有效，
但也暴露了两个限制：

1. NLinear 本身是固定的 `history→future` 线性映射，位置编码不是其天然组成部分；
2. 上一轮的周期检索只是在 NLinear 旁增加一个加权复制通路，没有真正建模“一个完整周期
   如何演化成下一个周期”。

本轮不继续调整 NLinear，也不复用上一轮 LFF 头作为候选。目标是用一个原生具备周期块
建模能力的新残差模型替换 NLinear，然后在同一模型上公平比较位置编码。

## 1. 核心研究问题

### 1.1 主假设

将时间序列按周期 `P` 切成完整周期块，并显式建模周期块之间的形态、幅值和趋势演化，
应比 NLinear 的逐预测步线性映射更适合承担 PhaseFormer 的周期补偿支路。

### 1.2 为什么与 PhaseFormer 互补

令周期化输入为 `X[k,l]`：`k` 是第几个周期，`l` 是周期内相位位置。

```text
                   周期内相位 l →
              ┌────────────────────┐
周期序号 k ↓  │ X[0,0] ... X[0,P-1]│  ← 一个完整周期
              │ X[1,0] ... X[1,P-1]│
              │          ...       │
              └────────────────────┘

PhaseFormer：以每个 l 为中心，汇总同相位跨周期信息，再建模相位位置间关系。
ICPT：       以每个 k 的完整一行为 token，建模整个周期块从 k 到 k+1 的演化。
```

- PhaseFormer 回答：“当前处于周期中的什么相位，不同相位如何交互？”
- 新支路回答：“最近几个完整周期的形状、幅值和整体水平如何逐周期变化？”

二者观察同一周期矩阵的两个正交轴。新支路不会再次建立“同相位聚合器”，避免只是复制
PhaseFormer 的相位视角。

### 1.3 客观风险

- `P` 选错时，完整周期 patch 会把不相关片段绑在一起；
- 周期内存在快速相位漂移时，固定边界的周期块可能错位；
- Electricity 等多通道数据可能具有不同主周期，而共享 `P=24` 可能不足；
- Transformer 的提升可能来自容量增加，而非周期结构或位置编码；
- PhaseFormer 与 ICPT 仍使用同一输入，融合收益不等于信息独立。

因此计划必须包含参数量、非周期 patch、无位置编码、单支路和融合消融。

## 2. 调研依据

- [PatchTST（ICLR 2023）](https://openreview.net/pdf?id=Jbdc0vTOcol)表明，将局部连续片段
  作为 token 并使用 channel-independent Transformer，可以减少 token 数并保留局部语义。
- [TimesNet（ICLR 2023）](https://openreview.net/pdf/98c0a5bad8225b6d1baf5c74047c4d04bacfcfa1.pdf)
  将多周期时间变化展开为周期内与周期间两个二维方向，支持“周期间演化应被单独建模”的动机；
  本轮不复制其多周期 Inception 结构。
- [CycleNet](https://openreview.net/pdf?id=clBiQUgj4w)用可学习循环模板显式建模周期，再预测
  剩余成分。本轮把它作为“简单循环模板”文献对照，但 ICPT 的目标不同：ICPT 直接建模完整
  周期块之间的动态关系，而不是只学习一个重复模板。
- [Time2Vec](https://arxiv.org/abs/1907.05321)提供可学习线性项与周期项；
  [RoPE](https://arxiv.org/abs/2104.09864)通过 Q/K 旋转把绝对坐标转化为相对位置依赖；
  [相对位置表示](https://aclanthology.org/N18-2074/)直接让 attention 感知 token 距离；
  [ALiBi](https://openreview.net/pdf?id=R8sQPpGCv0)以线性距离偏置表达近远关系；
  [Learnable Fourier Features](https://proceedings.neurips.cc/paper_files/paper/2021/file/84c2d4860a0fc27bcf854c444fb8b400-Paper.pdf)
  提供可学习、依赖相对位置差的 Fourier 相似度。

这些工作只提供设计依据，不代表它们在本仓库协议上一定有效。

## 3. 新模型：ICPT（Inter-Cycle Patch Transformer）

中文名：**周期间块变换器残差头**。建议模式名：`rcrf_icpt_<pe>`。

### 3.1 输入与周期块

沿用 PhaseFormer 完成 RevIN 后的输入 `x∈R^(B×L×C)`，核心实验固定 `L=720, P=24`：

```text
K_in  = L / P = 30
K_out = ceil(H / P)
X     = reshape(x, B, C, K_in, P)
anchor = X[:, :, -1, :]                         # 最近完整周期
M_k    = X_k - anchor                            # 历史周期相对最近周期的形变
```

若将来允许不能整除的 `L/H/P`，输入左侧复制最早值补齐，输出尾部裁剪；本轮所选 horizon
均可被 24 整除，不引入该混杂因素。

### 3.2 周期 token 编码

每个完整周期块是一个 token，而不是把每个时间点当 token：

```text
z_k = LayerNorm(W_patch M_k + b_patch)           # W_patch: P → d_model
```

`B` 和 `C` 合并为 `B×C`，所有通道共享同一个 ICPT，保持 channel-independent，避免
Electricity/Traffic 参数量随通道数爆炸。第一轮固定：

| 参数 | 固定值 | 说明 |
|---|---:|---|
| `period_len` | 24 | 与当前 PhaseFormer/RCRF 可比；本轮不同时搜索周期 |
| `d_model` | 32 | 参数规模受控 |
| encoder / decoder depth | 1 / 1 | 先验证机制，不堆深度 |
| attention heads | 4 | 每头维度 8 |
| FFN hidden | 64 | `2×d_model` |
| dropout | 0.0 | 与当前小模型设置保持一致 |
| channel mode | independent/shared | 每个通道独立前向、共享权重 |

### 3.3 周期间编码与未来周期查询

历史周期 token 经过一层 self-attention；`K_out` 个不含未来真值的 query token 经过
future-query self-attention 和对历史 token 的 cross-attention：

```text
Z_hist = Encoder(z_0 ... z_(K_in-1), position)
Q_fut  = learned_query(0 ... K_out-1) + future_position
Z_fut  = Decoder(Q_fut, Z_hist, position)
Delta_j = W_out Z_fut[j]                         # d_model → P
Y_icpt[j] = anchor + Delta_j
```

`W_out` 零初始化，使模型初始输出为“重复最近完整周期”，而不是随机输出。future queries
之间没有真实未来值，不存在 label leakage；可以使用非因果 query self-attention并行预测全部周期。

### 3.4 与 PhaseFormer 融合

第一轮完全复用当前 RCRF，不修改可靠度定义、门控公式或相位增强模块：

```text
r     = phase reliability from raw phase series
alpha = sigmoid(gate_bias + sensitivity*(1-r))
y     = (1-alpha)*y_phase + alpha*y_icpt
```

这样 baseline 与 candidate 的唯一结构差异是 `NLinear → ICPT`。不得同时调整 `alpha_init`、
`sensitivity`、loss、学习率或 PhaseFormer 模块来制造收益。

## 4. 为什么位置编码在 ICPT 上是合理的

NLinear 中直接加入固定位置 `p` 会出现 `W(x+p)=Wx+Wp`，位置项容易退化为固定偏置。
ICPT 不同：self/cross-attention 本身对 token 排列不敏感；位置编码会改变任意周期 token
之间的 Q/K 匹配或 pairwise attention bias，因此“第几个历史周期、距未来多远”是模型
不可缺少且可审计的信息。

但仍需客观区分：

- cycle patch 已通过固定切块隐含了局部顺序；PE 主要解决**周期块之间**的顺序和距离；
- learned query 也含未来周期编号，因此 learned absolute PE 可能与 query 参数部分重复；
- PE 的效果必须相对 `ICPT-none` 证明，不能只相对 NLinear；
- calendar PE 使用额外时间戳信息，不能与纯 index PE 混称为完全等价的输入条件。

## 5. 位置编码候选

所有候选共享完全相同的 ICPT、初始化、训练协议和参数预算；只改变位置注入方式。

| ID / 模式 | 注入方式 | 直白含义 | 主要风险 | 主筛选资格 |
|---|---|---|---|---|
| P0 `icpt_none` | 不加 PE | 检验模型是否仅靠周期内容就够用 | attention 不知道顺序 | 架构基线 |
| P1 `icpt_sincos` | token 加固定正余弦 | 告诉模型绝对第几个周期 | 绝对坐标未必泛化 | 是 |
| P2 `icpt_learned_abs` | token 加可学习位置表 | 每个周期序号单独学习 | 易记住训练长度 | 是 |
| P3 `icpt_time2vec` | token 加线性项+可学习周期项 | 同时表达时间方向和重复频率 | 小数据可能学偏频率 | 是 |
| P4 `icpt_rope` | self/cross-attention 的 Q/K 旋转 | 通过相位差表达相对周期距离 | 不直接表达日历事件 | 是 |
| P5 `icpt_relative` | attention logit 加 bucketed `Δcycle` 偏置 | 直接学习“相隔几个周期” | bucket 数过多会过拟合 | 是 |
| P6 `icpt_alibi` | attention logit 加按距离线性衰减 | 默认更信任较近周期 | 远周期重复可能被压制 | 是 |
| P7 `icpt_lff` | Fourier 位置相似度加到 attention logit | 学习哪些周期距离会重复 | 可能重现上一轮的小信号 | 是 |
| P8 `icpt_sincos_relative` | P1 token PE + P5 pairwise bias | 同时有绝对序号和相对距离 | 两类 PE 可能冗余 | 是，预注册组合 |
| P9 `icpt_calendar` | 周期起点的 hour/weekday/day/month 循环编码 | 使用真实日历位置 | 多了外部时间信息 | 单独排名 |

### 5.1 公平性约束

- 纯 index 主排名：P1–P8；P9 单列为 calendar-aware 候选。
- P1–P9 都必须与 P0 `ICPT-none`、当前 `RCRF-NLinear` 同设置配对。
- PE 参数使用同一随机 seed；不因数据集单独修改维度、温度、bucket 或频率数。
- `position_dim=d_model=32`；relative bucket 固定 16；LFF 频率数固定 16；ALiBi slope
  使用确定性 head-wise 配置；不搜索这些超参数。
- calendar 候选只能读取模型原本已提供的 `x_mark`，不得增加未来目标或额外数据源。

## 6. 对照与消融矩阵

| ID | 模型 | 目的 | 是否进入正式确认 |
|---|---|---|---|
| A0 | 固定 Golden PhaseFormer | 正式提升参照；不重新训练替换 | 必须报告 |
| A1 | matched original PhaseFormer | 排查本机协议偏差 | 是 |
| A2 | 当前 RCRF + NLinear | 直接 baseline | 是 |
| A3 | RCRF + RepeatLastCycle | 判断仅重复最近周期能否解释收益 | Stage A |
| A4 | RCRF + CycleNet-style recurrent template | 简单显式周期文献对照 | Stage A |
| A5 | RCRF + ICPT-none | 新架构、不加 PE | 是 |
| A6 | RCRF + ICPT-best-index-PE | 主候选 | 是 |
| A7 | RCRF + ICPT-calendar | 含真实时间信息的独立候选 | 合格才进入 |
| B1 | ICPT-best 单支路 | 判断 ICPT 单独能力 | 消融 |
| B2 | PhaseFormer + fixed 0.5 ICPT | 判断收益是否依赖 RCRF | 消融 |
| B3 | ICPT-best，patch size=16 | 判断周期对齐 patch 是否必要 | 消融 |
| B4 | ICPT-best，无 last-cycle anchor | 判断最近周期锚点作用 | 消融 |
| B5 | Cycle-token MLP，移除 attention | 判断 attention/PE 是否必要 | 消融 |

容量归因必须同时报告参数量、训练时间、推理时间和峰值显存。若 A6 参数明显更多，则补跑
`d_model=16` compact 版本；不得只拿大模型和 NLinear 比准确率。

## 7. 数据集与实验阶段

### Stage 0：实现与静态验证（不读 test）

必须通过：

1. `P=24` 下 `L=720`，四个 horizon 输出 shape 正确；
2. channel-independent 在 7、21、321、862 通道上均可前向；
3. `W_out=0` 时严格等于 RepeatLastCycle；
4. P0–P9 均有 finite forward/backward，PE 参数能收到梯度；
5. flag-off 与当前 RCRF 共享参数、随机初始化和输出完全一致；
6. future query 不接收未来 `batch_y`；calendar 只用 timestamp mark；
7. batch=1、非连续 tensor、CPU/GPU、checkpoint round-trip 均通过；
8. smoke test：ETTm2-96，5% 数据，1 epoch，validation-only。

### Stage A：新架构可行性筛选

四个锚点 setting，seed 2021，30% train，最多 8 epoch，只读 validation：

| Setting | 选择理由 | Golden MSE/MAE | loss | lr | batch | period |
|---|---|---:|---|---:|---:|---:|
| ETTh2-720 | 长预测、当前 RCRF 已强 | 0.402 / 0.436 | Huber | 1e-3 | 256 | 24 |
| ETTm2-96 | 高频短预测、当前 RCRF 已强 | 0.163 / 0.256 | MAE | 3e-4 | 256 | 24 |
| Electricity-336 | 321 通道、上一轮 PE 退化 | 0.165 / 0.257 | MAE | 3e-4 | 16 | 24 |
| Weather-336 | 自然信号、周期较弱且有漂移 | 0.242 / 0.278 | MAE | 3e-4 | 256 | 24 |

运行 A2/A3/A4/A5。A5 满足以下任一条件才进入完整 PE 筛选：

- 相对 A2 的 8 个 validation 指标比值均值 `<1`，最差比值 `≤1.01`；或
- 至少 3/4 settings 双指标改善，剩余 setting 最大回退 `≤0.5%`。

否则停止 ICPT 主路线，仅保留失败分析，不用位置编码搜索掩盖架构失败。

### Stage B：位置编码 validation-only 广筛

> **未运行。** Stage A 架构门槛失败后按 §13 停止条件跳过 PE 广筛，避免用 PE 搜索掩盖架构失败。

- 设置：同 Stage A，复用 A5；新增 P1–P9，共 36 个 candidate runs。
- 预算：30% train，最多 8 epoch，seed 2021，best-validation checkpoint。
- 不构造 test loader；所有候选完成后才写 freeze record。
- 对 P1–P8 计算相对 P0 的 8 项比值均值和最差值；资格为均值 `<1` 且最差 `≤1.01`。
- 合格者按“均值比值、最差比值、参数量、运行时间”依次排序，冻结一个 index-PE。
- P9 相对 P0 独立使用相同门槛；若合格，作为 calendar-aware 候选进入 Stage C，但不替代
  index-PE 冠军。
- test 读取后不得更改模型、PE、period、维度、loss 或学习率；若更改，必须重新命名实验并
  披露 test-set selection。

### Stage C：六数据集、三 seed 正式确认

> **未运行。** 无冻结 PE；Stage A 门槛失败后不执行正式确认。

冻结后才运行。全量训练，seeds `2021/2022/2023`，validation early stopping，恢复
best-validation checkpoint 后一次性读取 test。

| Setting | 覆盖属性 | Golden MSE | Golden MAE | loss | lr | batch | period |
|---|---|---:|---:|---|---:|---:|---:|
| ETTh1-96 | 未专门优化的 ETT 迁移 | 0.359 | 0.382 | MAE | 3e-4 | 256 | 24 |
| ETTh2-720 | 小通道、长预测 | 0.402 | 0.436 | Huber | 1e-3 | 256 | 24 |
| ETTm2-96 | 高频、短预测 | 0.163 | 0.256 | MAE | 3e-4 | 256 | 24 |
| Weather-336 | 弱周期与自然噪声 | 0.242 | 0.278 | MAE | 3e-4 | 256 | 24 |
| Electricity-336 | 高维通道异质性 | 0.165 | 0.257 | MAE | 3e-4 | 16 | 24 |
| Traffic-96 | 862 通道、强日周期、资源压力 | 0.361 | 0.238 | MAE | 3e-4 | 8 | 24 |

每个 setting 运行 A1、A2、A5、A6；若 P9 合格再运行 A7。Golden A0 只从固定文档读取。

### Stage D：机制消融与样本分析

> **未运行。** 依赖冻结的 A6；Stage A 门槛失败后不执行。

- 消融 setting：ETTh2-720、Electricity-336；seed 2021；运行 B1–B5。
- 最终对比：A2 当前 RCRF-NLinear vs A6 ICPT-best-index-PE；按
  `sample×channel` 计算 MSE/MAE。
- 每 setting/seed 程序化选取 Top-10：baseline 高误差、candidate 显著改善、candidate
  显著退化；同通道重叠窗口按至少一个 horizon 间隔去重，不人工挑例。
- 必须导出 `r/alpha`、PE attention、各 head 的主要 cycle lag、attention entropy、
  last-cycle anchor 与预测增量范数。

### 预计运行规模

| 阶段 | 新运行数（上限） | 数据预算 | 扩展条件 |
|---|---:|---|---|
| Stage 0 | 1 smoke + 单元测试 | 5% / 1 epoch | 静态测试全过 |
| Stage A | 16 | 30% / ≤8 epoch | A5 通过架构门槛 |
| Stage B | 36 | 30% / ≤8 epoch | Stage A 通过 |
| Stage C | 72；calendar 合格时 90 | 100% / ≤30 epoch | PE 已冻结 |
| Stage D | 10 | 两 setting，按正式预算 | Stage C 完成 |
| 合计 | 135；含 calendar 时 153 | 不含可选 Stage E | 支持断点续跑 |

不得为了节省成本跳过 matched A2 或三 seed。Traffic 优先单独排队并记录 OOM；若显存不足，
baseline/candidate 必须成对使用相同 batch 调整，不能只降低 candidate batch。

### 可选 Stage E：全 Golden 网格扩展

只有 Stage C 通过最终门槛后才扩展到 ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity/
Traffic × horizons 96/192/336/720 × 3 seeds。Exchange 没有固定 Golden，只报告 matched
RCRF 配对结果。未完成 Stage E 前，不更新全局 `latest` preset。

## 8. 预注册判断标准

### 8.1 “新模型有效”

Stage C 中 A6 相对 A2：

1. 至少 4/6 settings 的三 seed 平均 MSE、MAE 同时改善；
2. 其余 setting 任一平均指标回退不超过 0.5%；
3. 六 setting 的 12 项指标宏平均相对改善为正；
4. 参数量不超过 A2 的 1.5 倍，或准确率收益足以覆盖资源代价并明确披露；
5. A6 优于 A5，证明位置编码在新模型内部有独立贡献。

### 8.2 “稳定超过 Golden”

逐 setting 同时满足：

- 三个 seed 的 MSE、MAE 全部低于固定 Golden；
- `mean + sample_std < Golden` 对 MSE、MAE 同时成立。

Stage C 至少 4/6 settings 达标，才允许称“在多个数据集稳定超过 Golden”。否则必须逐项
表述，不得用宏平均掩盖失败 setting。

### 8.3 “互补而不是更大模型”

至少同时观察到：

- A6 融合优于 A2 当前 RCRF；
- A6 融合优于 B1 ICPT-only；
- 周期对齐 patch A6 优于 B3 非周期 patch；
- A6 优于 A5 ICPT-none；
- 资源开销和参数量已配对报告。

若只满足其中一两项，只能称工程候选，不能声称验证了“相位×周期间演化互补”。

## 9. 实验结果表

### 9.1 Stage 0 实现与测试

| 检查项 | 命令/配置 | 预期 | 实际 | 状态 |
|---|---|---|---|---|
| flag-off 等价 | `pytest tests/ -q`（124 个既有用例全绿）；gold_combo 头仍为 `WeakPeriodResidualHead` | 参数/输出一致 | 既有 124 用例通过；构造分支互斥，gold_combo 路径未触碰 | ✅ |
| 10 种 PE forward/backward | `tests/test_intercycle_patch.py::test_all_pe_parameters_receive_gradients` | finite + 有梯度 | P0–P9 全部 forward/backward finite，PE 参数收到非零梯度 | ✅ |
| 四 horizon shape | `test_four_horizons_and_862_channels_forward` | shape 正确 | 96/192/336/720 输出 `(1,H,862)` 均正确 | ✅ |
| 862 通道 smoke | 同上 | 无 OOM/shape 错误 | Traffic 862 通道单前向正常 | ✅ |
| future leakage 检查 | 代码审计：decoder 仅用 learned_query + 对历史 token 的 cross-attn | 不读取 future target | 无 `batch_y` 进入 head；calendar 只读 x_mark | ✅ |
| ETTm2 5% smoke | `search_phaseformer.py --stage smoke --mechanism rcrf_icpt_none / rcrf_icpt_calendar --percent 5 --max-epochs 1` | 1 epoch 完成 | 两模式均 4.5s 完成，val 指标落盘，`test_*` 为空（未建 test loader） | ✅ |

### 9.2 Stage A 架构筛选（validation）

**结果：A5 架构门槛未通过，ICPT 主路线停止**（validation-only，30% 数据、≤8 epoch、seed 2021）。A5 vs A2 的 8 个比值均值 **1.137**，最差 **1.278**；仅 ETTh2-720 双指标改善（1/4 settings）。不满足 §8 任一条件 → 按 §13 停止条件结束 ICPT，不进入 Stage B PE 广筛，不把架构失败掩盖成 PE 差异。

| Setting | A2 RCRF-NLinear MSE/MAE | A3 RepeatCycle | A4 CycleNet-style | A5 ICPT-none | A5/A2 比值 | 结论 |
|---|---|---|---|---|---|---|
| ETTh2-720 | 0.63207 / 0.55913 | 0.63928 / 0.55241 | 0.62422 / 0.55686 | 0.60690 / 0.54418 | 0.960 / 0.973 | A5 改善双指标 |
| ETTm2-96 | 0.11817 / 0.23265 | 0.15772 / 0.28119 | 0.11816 / 0.23262 | 0.15099 / 0.27222 | 1.278 / 1.170 | A5 明显回退 |
| Electricity-336 | 0.13936 / 0.23060 | 0.16031 / 0.24731 | 0.13942 / 0.23071 | 0.16461 / 0.24755 | 1.181 / 1.073 | A5 回退 |
| Weather-336 | 0.54072 / 0.36653 | 0.86699 / 0.49921 | 0.53945 / 0.36626 | 0.68911 / 0.43357 | 1.274 / 1.183 | A5 明显回退 |
| 宏平均/最差 | — | — | — | 1.137 / 1.278 | 1/4 settings 双改善 | **停止** |

可测量观察：

- A3 RepeatLastCycle 只在 ETTh2-720 接近 A2（MSE +1.1% / MAE −1.2%），在其余三 setting 大幅劣化
  （ETTm2 +33%/+21%、Electricity +15%/+7%、Weather +60%/+36%），说明“只重复最近周期”不是
  A2 优势来源。
- A4 CycleNet-style 在四 setting 均与 A2 几乎持平（MSE 变化 −1.2% ~ +0.04%，MAE −0.4% ~
  +0.05%），作为显式周期模板文献对照，其性能与 NLinear 相当。
- A5 ICPT-none（26.8K–28.2K 参数）参数量远小于 A2 NLinear（ETTh2 519K、其余 72K–247K），
  仅在大容量 NLinear 的 ETTh2-720 上反超；在 ETTm2/Electricity/Weather 上显著回退。ICPT
  残差在 RCRF 融合下平均劣于 NLinear，架构假设未获支持。
- 参数/容量差异是已知混杂（plan §6），但方向与 A5 收益相反（A5 更小却更差），不构成掩盖。

冻结记录：`research_runs/phaseformer_icpt_pe_screen/freeze_record.json` → `stage_a_passed: false`，
未做任何 PE freeze；test 从未在 freeze 前读取。

### 9.3–9.8 后续阶段表格

**不适用。** Stage A 架构门槛失败（§9.2），按 §13 停止条件：不进入 Stage B PE 广筛
（9.3）、不执行 Stage C 正式确认（9.4/9.5）、不执行 Stage D 消融与样本分析（9.7/9.8）。
表 9.6 中的 A2/A5 参数量与训练耗时已在 §9.2 表格与 `screen_summary.csv` 记录
（A5 ICPT-none 参数量 26.8K–28.2K，各 setting 训练耗时见 screen 记录）。

## 10. 实现文件与开关计划

建议最小改动：

| 文件 | 计划改动 |
|---|---|
| `src/models/intercycle_patch.py` | 新增 ICPT、PE 模块和诊断值；避免继续膨胀 `phase_adapters.py` |
| `src/models/PhaseFormer.py` | 仅在新 flag 下构造/调用 ICPT；RCRF 公式不改 |
| `src/models/phaseformer_presets.py` | 注册 A3–A7、P0–P9、B1–B5 模式 |
| `scripts/run_intercycle_patch_experiment.py` | Stage A/B/freeze/C/D，支持断点续跑和 validation-only |
| `scripts/analyze_intercycle_patch.py` | 逐 cell 误差、内部量、报告、ZIP 和校验 |
| `tests/test_intercycle_patch.py` | 等价性、shape、梯度、无泄漏、PE、checkpoint 测试 |

建议配置字段：

```text
use_intercycle_patch_residual
intercycle_period_len
intercycle_d_model
intercycle_heads
intercycle_ffn_dim
intercycle_encoder_layers
intercycle_decoder_layers
intercycle_pe_type
intercycle_relative_buckets
intercycle_lff_frequencies
intercycle_use_last_cycle_anchor
```

flag-off 必须保持当前 RCRF 的模块构造、state keys、随机数消耗和输出不变。

## 11. 复现命令模板（待实现）

```bash
# Stage A：架构筛选
python scripts/run_intercycle_patch_experiment.py --stage architecture_screen \
  --settings ETTh2:720,ETTm2:96,Electricity:336,Weather:336 --seed 2021

# Stage B：PE validation-only 筛选并冻结
python scripts/run_intercycle_patch_experiment.py --stage pe_screen --seed 2021
python scripts/run_intercycle_patch_experiment.py --stage freeze

# Stage C：冻结后正式确认
python scripts/run_intercycle_patch_experiment.py --stage full \
  --seeds 2021,2022,2023

# Stage D：消融和审计
python scripts/run_intercycle_patch_experiment.py --stage ablation
python scripts/analyze_intercycle_patch.py --device cuda:0
```

所有命令最终需写入 `run.yaml`；在 runner 实现前，上述仅为接口约定，不可执行。

## 12. 证据目录与最终交付

实验编号：`phaseformer_icpt_pe_v1`。最终规范目录：

```text
research_runs/phaseformer_icpt_pe_v1/
├── run.yaml
├── results.csv
├── sample_errors.csv
├── selected_cases.npz
├── objective_error_analysis.md
├── objective_error_analysis.zip
└── figures/
```

报告必须明确分开：validation 选择结果、冻结后的 test 结果、相对 matched RCRF 的变化、
相对固定 Golden 的变化、可测量观察和待验证机制假设。最终 ZIP 只包含 Markdown 与其实际
引用图片；不得打包 checkpoint、日志、全量预测或未引用图片。

## 13. 停止条件与决策出口

- Stage A 架构门槛失败：结束 ICPT，不进入 PE 广筛；
- Stage B 无任何 index-PE 合格：保留 ICPT-none 进入一次正式诊断，结论为“PE 无独立收益”；
- Stage C 未达到 4/6 跨数据集门槛：不替换 NLinear，只保留局部适用结论；
- Stage C 通过但资源超标：运行 compact 后再决定；
- Stage C 与 D 同时支持互补性：再进入 28 个 Golden 任务的 Stage E；
- 无论成功或失败，都填写本文表格、生成客观错误分析并更新 `docs/agent-log.md`；不得通过
  继续查看 test 后修改 PE 来追逐结果。
