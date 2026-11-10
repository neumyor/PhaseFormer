# PhaseFormer-LA 的 level 分支

## 1. 它做什么

相位分支把历史按周期折成"相位槽 × 周期"的视图，输出未来每个相位槽上的值。它对预测窗口的**整体高度**没有独立的自由度：窗口整体抬升或下移，要靠每一个槽同时写出同样的偏移来实现。

level 分支补的就是这一块。它只读历史的水平信息，写出一条与预测长度对齐的曲线，再把这条曲线与相位预测逐通道混合。

`b0` 是这个模型的相位部分加 level 分支；更完整的版本在此之上另有改动，不在本文范围。

## 2. 输入与记号

| 张量 | 形状 | 说明 |
|---|---|---|
| `x_enc` | (B, L, C) | 原始输入窗口 |
| `x_in` | (B, L, C) | 归一化后的窗口，level 分支读它 |
| `y_phase` | (B, H, C) | 相位分支的预测 |

B 是 batch，L 是输入长度（本项目 720），H 是预测长度（96/192/336/720），C 是通道数。周期长度 P 默认 24，瓶颈秩 k 固定 3，池化因子 f 默认 24。

## 3. 计算过程

**归一化。** 按样本、按通道沿时间算均值 μ 和标准差 σ，把输入标准化。level 与相位两条分支都在这个空间里工作，最后用同一组 μ、σ 还原。

**取锚点并中心化。** 用窗口最后一个观测值 a 作锚点，历史减去它：

```
x~ = x_in - a
```

**压到每周期一个值。** 沿时间做自适应平均池化，把长度 L 压到 n = ⌈L/f⌉。当 f = P = 24、L = 720 时 n = 30，每个池化分箱正好覆盖一个周期，池化结果就是每个周期的平均高度（相对锚点的偏移）。这就是 "level" 的来源。

**编码。** 一个线性层把 n 维压到 3 维：`z = W_e · p + b_e`。

**解码。** 把 z 写成一条长度为 H 的曲线，再各点加回锚点 a：

- S2 head 用三个固定的正交多项式基（常数、倾斜、弯曲），先把 z 映射成三个系数，再加权求和；
- S1/S3 head 用一次线性投影直接映射到 H 维。

**混合。** 两条预测各按周期拆开，用两个门控分别控制"水平"与"形状"的混合比例，门控逐通道独立。输出是相位预测与 level 曲线的加权组合，权重可以按通道不同。

拼接回完整预测后，用同一组 μ、σ 还原到原始尺度。

## 4. 参数

| 名称 | 形状 | 初始化 |
|---|---|---|
| `encoder.weight` / `.bias` | (k, n) / (k) | 默认 |
| `decode` 的线性层（S1/S3） | (H, k) / (H) | 零初始化 |
| `mix` 的线性层（S2） | (3, k) / (3) | 零初始化 |
| `profiles`（S2） | (H, 3) | 固定的正交多项式，不参与训练 |

k = 3、f = 24（S3 为 8）、P = 24。L = 720 时 encoder 是 3×30 + 3 = 93 个参数，S2 的 mix 是 3×3 + 3 = 12 个，合计 105，与运行记录里的 `level_branch = 105` 一致。S1/S3 的解码层参数量随 H 增长。

## 5. 代码对应

| 环节 | 文件 | 位置 |
|---|---|---|
| 归一化 | `src/models/phase_adapters.py` | `RevIN.normalize`，第 27 行 |
| 分支输入 | `src/models/PhaseFormer.py` | 第 1970 行 |
| 分支构造 | `src/models/PhaseFormer.py` | 第 590 行 |
| 分支调用 | `src/models/PhaseFormer.py` | 第 2145 行 |
| 混合调用 | `src/models/PhaseFormer.py` | 第 2240 行 |
| 编码 / 解码（S1、S3） | `src/models/phase_adapters.py` | `PooledLowRankWeakPeriodResidualHead`，第 165 与 185 行 |
| 编码 / 解码（S2） | `scripts/vldbj_m1/model.py` | `ProfileLevelHead`，第 91 与 96 行 |
| 多项式基 | `scripts/vldbj_m1/model.py` | `legendre_profiles`，第 31 行 |
| 混合器 | `src/models/lowrank_fusion.py` | 第 57 行 |

## 6. 最小示例

取 B = 1、L = 16、C = 2、H = 4、f = 4、k = 2，于是 n = 4。

| 步骤 | 形状 |
|---|---|
| 输入 `x` | (1, 16, 2) |
| 锚点 `last = x[:, -1:]` | (1, 1, 2) |
| 中心化 `x - last` | (1, 16, 2) |
| 转置到通道维 | (1, 2, 16) |
| 自适应平均池化到 4 | (1, 2, 4) |
| 编码 | (1, 2, 2) |
| S2 系数 | (1, 2, 3) |
| 乘多项式基 | (1, 2, 4) |
| 转回 | (1, 4, 2) |
| 加锚点 | (1, 4, 2) |

复现（在仓库根目录执行，CPU 即可）：

```bash
CUDA_VISIBLE_DEVICES="" python3 - <<'PY'
import torch
from scripts.vldbj_m1.model import ProfileLevelHead
from src.models.phase_adapters import PooledLowRankWeakPeriodResidualHead

x = torch.randn(1, 16, 2)
for cls in (ProfileLevelHead, PooledLowRankWeakPeriodResidualHead):
    head = cls(16, 4, pool_factor=4, rank=2)
    latent, last = head.encode_latent(x)
    print(cls.__name__, latent.shape, head.decode_latent(latent, last).shape)
PY
```

输出为 `(1, 2, 2) (1, 4, 2)`，两个 head 一致。

## 7. 常见误解

**把 level 与趋势当成一回事。** level 指每个周期平均高度随周期变化的那条曲线，趋势还包括窗口内的线性漂移和长期走向。分支输出的是一条能在预测窗口上弯曲的曲线（S2 下最多二次），它同时承载整体高度和缓慢变化，不能当成一个标量偏置。

**以为零初始化意味着分支没用。** 解码层零初始化，让训练起点精确等于"把最后一个观测值复制 H 次"，也就是一条持续性基线。隐状态 z 在此之前已经算好，梯度从混合门控传回来会立即更新解码层。这个初始化排除了"分支一开始就扰动相位预测"，让两条通路从同一个基线出发。

**以为分支读的是折叠后的相位视图。** 它读的是未折叠的归一化历史 (B, L, C)。折叠只发生在相位分支。两条分支看同一段历史的不同表示，这是它们能互补的前提。
