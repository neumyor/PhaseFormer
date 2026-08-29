# A2 锚定式 PCTF 融合修复与复测计划

> 实验编号：`pctf_anchor_fusion_v2`。状态：模型、preset、训练入口、三阶段 runner 和测试已完成；
> **尚未启动任何训练，也未读取新的 validation/test 结果**。

## 1. 为什么需要重做融合

PCTF v1 的 F1/F2 使用

`Y = μ_T + mix(L_T,L_C) + mix(S_P,S_C)`，

其中 `P/T/C` 分别是 PhaseFormer、NLinear、ICPT，`μ/L/S` 分别是 horizon 绝对均值、周期间
相对水平和周期内形状。该公式有四个已由结果支持的问题：

1. 它完全删除 NLinear 的周期内形状 `S_T`，而 A2 的优势并不只来自 NLinear 的均值；
2. 任意 gate 取值都不能还原 A2，因此复杂候选没有安全退路；
3. F2 的 shape 证据比较 ICPT 与 NLinear，却被用于控制 ICPT 与 PhaseFormer；
4. 证据只看两个一步伪起点，并广播到全部未来周期，不能反映 lead-specific 可靠度。

结果与诊断一致：F1/F2/F3 相对 A2 宏平均退化约 2.5%–3.3%，而保留完整 NLinear 与外层
RCRF 的 F0 基本持平。最佳 checkpoint 中，F1/F3 的系数仍接近初始化，F1a/F1b 几乎相同；
这说明首先应修正候选空间与训练信号，而不是继续增加 gate 复杂度。

## 2. 修复原则

### 2.1 完整 A2 是可训练锚点

先按原样计算 A2：

`A = HighFreqDamp(RCRF(P, T_LFF))`。

- PhaseFormer、LFF-NLinear、RCRF 和既有相位校准均保留；
- 它们与 ICPT 在同一个模型、同一个 optimizer、同一个 checkpoint 中端到端训练；
- 不是加载或路由多个冻结模型；
- ICPT 构造使用隔离 RNG，不改变同 seed 下 A2 的任何初始参数。

候选只在完整锚点后加入结构化创新：

`Y = A + β_L·D_L + β_S·D_S`。

其中：

- `D_L = L_C - L_T`：ICPT 相对轨迹分支提供的周期间水平创新；
- `D_S = S_C - S_P`：ICPT 相对相位分支提供的周期内形状创新；
- `D_L` 在实际逐周期缩放后重新投影为 horizon 均值零；
- `D_S` 每个周期均值为零，二者正交，不改变 A2 的绝对均值。

`β` 使用 `β_max·tanh(raw)`，默认 `|β|≤0.25`，并令 `raw=0`。因此所有候选在初始化时
逐点严格等于 A2，而不是“接近 A2”。有符号系数允许 ICPT 表示相对参考分量的修正方向；边界
限制其成为新的完整预测捷径。

### 2.2 ICPT 从第一步就获得匹配的训练信号

零初始化保证安全，但会让 ICPT 在第一批数据上收不到主损失梯度。训练时增加只作用于 ICPT
的组件辅助目标：

`L = L_forecast(Y,target) + 0.05 L_shape(S_C,S_y) + 0.05 L_level(L_C,L_y)`。

shape-only、level-only 和 phase-modulation 消融只启用实际使用的辅助项。validation/checkpoint
选择仍只看最终融合预测，不把辅助损失混入 validation 指标。

### 2.3 证据对象和预测 lead 对齐

F2 修复版在历史中构造多个严格因果、与当前 horizon 等长的滚动伪起点：

- shape 风险比较 ICPT 与由伪起点之前周期形成的 PhaseFormer-style phase template；
- level 风险比较 ICPT 与同一个 A2 LFF-NLinear trajectory head；
- 对每个未来周期分别计算风险，不再把一步证据广播到全部 lead；
- 风险使用有符号 `log(error_ICPT/error_reference)`，保留“更好多少”和“更差多少”，不再用
  ReLU 把所有正收益压成同一个值；
- 跨伪起点标准差作为不确定性惩罚；历史不足覆盖完整 horizon 时，只对未观测的尾部周期保守
  延用最后一个可测置信度。

### 2.4 相位周期与 ICPT 周期解耦

PhaseFormer/A2 的 phase period 固定为24，保持与 incumbent 完全可比；ICPT cycle period 单独
设置。若 720 不能整除 cycle period，ICPT 只使用最近的完整周期，不填充未来或读取窗口外
数据。例如 period 96 使用最近672步的7个完整周期，舍弃最早48步；A2 仍使用完整720步。

## 3. 修复候选

| ID | preset | 设计 | 相对 A2 初值 |
|---|---|---|---:|
| B1 | `pctf_anchor_component_scalar` | level/shape 各一个全 horizon 系数 | 完全相等 |
| B2 | `pctf_anchor_component_cycle` | 每个未来周期独立 level/shape 系数 | 完全相等 |
| B3 | `pctf_anchor_monotonic` | B2 × 匹配对象、逐 lead 的因果置信度 | 完全相等 |
| B4 | `pctf_anchor_mlp` | 7维历史证据经零输出初始化 MLP 产生逐周期系数 | 完全相等 |
| B5 | `pctf_anchor_phase_modulation` | ICPT 只提供相位偏移、幅度和小幅零均值形变 | 完全相等 |
| S | `pctf_anchor_shape_only` | 只加入 `D_S`，组件消融 | 完全相等 |
| L | `pctf_anchor_level_only` | 只加入 `D_L`，组件消融 | 完全相等 |

B4 的七个输入为：相位不可靠度、近期漂移、shape/level 历史置信度、`C-P` shape 分歧、
`C-T` level 分歧和 `C-A` 完整预测分歧。特征 detach，末层权重和 bias 为零。

B5 不再用 ICPT 替换 NLinear 周期均值。它把 ICPT 预测与 PhaseFormer shape 做可微 circular
alignment，温度从 v1 的0.10放宽到0.25，然后只把“对齐模板相对原 phase shape 的变化”和
剩余形变作为零初始化更新；形变上限0.10，主调制上限0.25。

## 4. 复测协议

### Stage P：cycle period 选择，48 runs

- 数据：六数据集，H96/H192，30% train，12 epoch，seed 2021，Huber；
- PhaseFormer period 永远为24；
- 每个 setting 运行一次 A2，共12次；
- 用 B2 探测每数据集三个 ICPT period，共36次；
- ETTh1/ETTh2/Weather/Electricity：12、24、48；
- ETTm1/ETTm2：24、48、96；
- 每数据集在 H96/H192 四个指标上，先按相对 A2 宏平均、再按最差比值冻结唯一 period；
- 只读 validation，任何 test 字段都会终止汇总。

| Dataset | P12 | P24 | P48 | P96 | 冻结 period |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 待填 | 待填 | 待填 | — | 待填 |
| ETTh2 | 待填 | 待填 | 待填 | — | 待填 |
| ETTm1 | — | 待填 | 待填 | 待填 | 待填 |
| ETTm2 | — | 待填 | 待填 | 待填 | 待填 |
| Weather | 待填 | 待填 | 待填 | — | 待填 |
| Electricity | 待填 | 待填 | 待填 | — | 待填 |

表中填写 H96/H192、MSE/MAE 四指标相对 A2 的宏平均比值。

### Stage S：融合策略筛选，132 runs

使用 Stage P 冻结的 period，在六数据集×H96/H192 上运行 A1、A2、I0、旧 F0、B1–B5、S、L，
共 `6×2×11=132` 次；仍为30% train、12 epoch、seed 2021、validation-only。

论文候选须同时满足：

1. 24个指标相对 A2 宏平均比值 `≤0.998`；
2. 至少8/12 setting 的 MSE、MAE 同时改善；
3. 最差单指标比值 `≤1.01`；
4. 相对 A1/A2/I0/旧 F0 逐指标包络的宏平均比值 `≤1.005`。

| Candidate | macro/A2 | 双指标改善/12 | worst/A2 | macro/参考包络 | 决策 |
|---|---:|---:|---:|---:|---|
| B1 scalar | 待填 | 待填 | 待填 | 待填 | 待填 |
| B2 cycle | 待填 | 待填 | 待填 | 待填 | 待填 |
| B3 monotonic | 待填 | 待填 | 待填 | 待填 | 待填 |
| B4 MLP | 待填 | 待填 | 待填 | 待填 | 待填 |
| B5 phase modulation | 待填 | 待填 | 待填 | 待填 | 待填 |
| S shape-only | 待填 | 待填 | 待填 | 待填 | 仅消融 |
| L level-only | 待填 | 待填 | 待填 | 待填 | 仅消融 |

汇总器同时固定输出 B2/B1、B3/B2、B4/B2、B5/B2、B2/S、B2/L 六组嵌套对照。

### Stage F：冻结后三 seed test，最多144 runs

只有 Stage S 有论文候选通过时，冻结唯一冠军及六个数据集的 period。在六数据集、H96/H192、
seeds 2021/2022/2023 上统一重跑 A1、A2、I0、冠军：`6×2×3×4=144` 次。全训练集、30 epoch，
只按最低 validation loss 选 checkpoint，然后读取一次 test。

| Test setting | Golden | A2三seed | 冠军三seed | 相对A2 | 稳定低于Golden |
|---|---:|---:|---:|---:|---:|
| ETTh1-96 | 0.359 / 0.382 | 待填 | 待填 | 待填 | 待填 |
| ETTh1-192 | 0.397 / 0.404 | 待填 | 待填 | 待填 | 待填 |
| ETTh2-96 | 0.275 / 0.338 | 待填 | 待填 | 待填 | 待填 |
| ETTh2-192 | 0.341 / 0.376 | 待填 | 待填 | 待填 | 待填 |
| ETTm1-96 | 0.293 / 0.344 | 待填 | 待填 | 待填 | 待填 |
| ETTm1-192 | 0.323 / 0.361 | 待填 | 待填 | 待填 | 待填 |
| ETTm2-96 | 0.163 / 0.256 | 待填 | 待填 | 待填 | 待填 |
| ETTm2-192 | 0.219 / 0.293 | 待填 | 待填 | 待填 | 待填 |
| Weather-96 | 0.148 / 0.195 | 待填 | 待填 | 待填 | 待填 |
| Weather-192 | 0.193 / 0.237 | 待填 | 待填 | 待填 | 待填 |
| Electricity-96 | 0.129 / 0.221 | 待填 | 待填 | 待填 | 待填 |
| Electricity-192 | 0.148 / 0.238 | 待填 | 待填 | 待填 | 待填 |

替换 A2 要求：24指标宏平均 `<0.998`、至少8/12 setting 双指标改善、最差平均单指标回退
`≤0.5%`。同时报告三 seed sample std、参数量、显存和耗时。Golden 始终来自
`docs/PhaseFormer_gold_standard.md`；matched A2 只用于协议内配对。

## 5. 设备与泄漏审计

v1 的环境文件显示55次 CUDA、11次 CPU；因此 F0 的亚千分位差异不能作为可靠结论。v2 做
双重阻断：

- 每条训练命令包含 `--require-cuda`，PyTorch 不识别 CUDA 时直接失败；
- 汇总器要求全部 run 的 GPU 名称、torch、CUDA runtime 和 Lightning 版本完全一致；
- period/screen 汇总拒绝任何非空 test 指标；
- 每个锚定候选训练前跑真实 batch，并要求 `candidate == internal A2 anchor` 的最大绝对差严格
  等于0，否则不训练。

## 6. 命令

只预览矩阵，不训练：

```bash
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage period-dry
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage screen-dry
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage confirm-dry \
  --champion pctf_anchor_component_cycle
```

经人工确认后才允许依次执行：

```bash
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage period --progress
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage period-summarize
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage screen --progress
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage screen-summarize
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage confirm --progress
.venv/bin/python scripts/run_pctf_anchor_fusion_retest.py --stage confirm-summarize
```

输出根目录为 `research_runs/pctf_anchor_fusion_v2/`。Stage P/S 没有合格结论时，runner 会阻断
后续 test。项目历史已经查看过部分 test，因此即使按冻结协议完成 Stage F，也必须披露项目级
test exposure，不能表述为完全盲测。

## 7. 实现验收条件

- 所有七种策略在初始 forward 中逐点严格等于 A2；
- 同 seed 下，候选包含的全部 A2 state tensor 与独立 A2 逐项相等；
- 非零 level/shape 更新保持 horizon 均值守恒、周期 shape 零均值和二者正交；
- period 96 能以720输入的最近672步完成严格因果 forward；
- 逐 lead 历史伪起点不读取目标之后的数据；
- 零校正时 ICPT 仍通过组件辅助损失获得非零梯度；
- dry-run 数量严格为48/132/144，前两阶段不含 `--evaluate-test`，正式阶段全部包含；
- 汇总器拒绝 test 泄漏、CPU fallback、异构 GPU/软件环境、非零锚点误差、矩阵缺失和重复结果。

这些检查只证明代码能实现实验目标，不构成模型效果结论。
