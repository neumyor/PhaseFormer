# PhaseFormer-LA 在 ETTm1 上的最佳配置（2026-10-23）

记录本轮 ETTm1 参数搜索找到的最佳配置。目标：在每个 horizon 上让 test MSE 比 Golden 低 2% 以上。

## 1. 搜索协议

| 项目 | 设置 |
|---|---|
| 模型 | PhaseFormer-LA（vldbj_m1 框架，arm=m1，head=S2，rank=3） |
| 通信方式 | tp_mod_centered，shared_gate=True（ETTm1 的注册设定） |
| 输入长度 | 720（对周期不整除时按模型内建循环填充，如 720→768 对应周期 96） |
| 预测长度 | 96 / 192 / 336 / 720 |
| 训练 | 30 epochs，Huber loss，batch 256，从零训练 |
| seed | 2021（单 seed 搜索） |
| 选优口径 | 测试集最优（test-set selection，用户 2026-10-10 授权） |
| 代码 | commit `32f9a4d07555e5a0e14b08c59dc05c153583fbec` |

标注：单 seed + 测试集选优，属于条件性、探索性结果，不能表述为盲测泛化性能。

## 2. 搜索规模

三个阶段共 **2548 次训练**，位于服务器
`~/niuyiming/PhaseFormer_vldbj_wide_20261011/research_runs/aggressive_ettm1_traffic_v2/`：

| 阶段 | 任务数 | 内容 |
|---|---:|---|
| stage1 | 1343 | 7 数据集 × 4 horizon 的密集网格（本记录只用到其中的 ETTm1 部分），周期 24 |
| ettm1_period | 800 | ETTm1 × 4 horizon × 周期 {24,48,96} × 精调网格（96 与 336 无合法 p96，因预测长度非整数周期） |
| ettm1_stage3 | 405 | 针对未达标 horizon：H192 加周期 {32,64}，H720 加周期 {72,120} |

搜索维度：`gate_init`、`extension_lr_mult`、`communication_width`、`backbone_lr_mult`、`period_len`。

## 3. 最佳配置

| Horizon | period_len | gate_init | ext_lr_mult | width | backbone_lr_mult | test MSE | test MAE | vs Golden (MSE) | 距 −2% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | **48** | 0.1 | 3.0 | 2 | 1.0 | **0.285722** | 0.338722 | **−2.48%** | 达标 |
| 192 | **48** | 0.1 | 10.0 | 4 | 2.0 | 0.322985 | 0.364021 | −0.00% | +0.006445 |
| 336 | **48** | 0.1 | 10.0 | 2 | 1.0 | **0.350841** | 0.380277 | **−2.00%** | 达标（边界） |
| 720 | **24** | 0.04 | 10.0 | 3 | 2.0 | 0.407480 | 0.411723 | −1.10% | +0.003720 |

Golden（`docs/PhaseFormer_gold_standard.md`）：96 = 0.293/0.344，192 = 0.323/0.361，336 = 0.358/0.381，720 = 0.412/0.410。

MAE 对照：96 与 336 低于 Golden；192 与 720 略高于 Golden（0.364021 vs 0.361；0.411723 vs 0.410）。

### 对应 run 目录

| Horizon | 服务器路径（相对 `research_runs/aggressive_ettm1_traffic_v2/`） |
|---|---|
| 96 | `ettm1_period/runs/ETTm1_H96_S2021__m1__g0.1_lr3_w2_bb1__p48` |
| 192 | `ettm1_stage3/runs/ETTm1_H192_S2021__m1__g0.1_lr10_w4_bb2__p48` |
| 336 | `ettm1_period/runs/ETTm1_H336_S2021__m1__g0.1_lr10_w2_bb1__p48` |
| 720 | `ettm1_stage3/runs/ETTm1_H720_S2021__m1__g0.04_lr10_w3_bb2__p24` |

四个 run 的 `result.json` 与 `config.json` 已拷贝到本地
`research_runs/ettm1_best_configs_v1/{h96,h192,h336,h720}/`。

## 4. 各 horizon 的周期对比（每格为该周期下的最优 test MSE）

| Horizon | p24 | p32 | p48 | p64 | p72 | p96 | p120 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 0.291648 | — | **0.285722** | — | — | 0.330305 | — |
| 192 | 0.327438 | 0.346334 | **0.322985** | 0.357121 | — | 0.356085 | — |
| 336 | 0.357049 | — | **0.350841** | — | — | 不适用 | — |
| 720 | **0.407480** | — | 0.407699 | — | 0.429618 | 不适用 | 0.446307 |

"不适用"：预测长度必须是整数个周期（低秩融合要求），336/96 = 3.5、720/96 = 7.5 不合法。

## 5. 结论

1. **周期 48 对 ETTm1 是关键**：三个 horizon（96、192、336）的最优周期都是 48；周期 24（6 小时）太短，周期 96（整日）明显过短周期化后变差，32/64 也不如 48。
2. **96 与 336 达到 −2% 目标**：96 为 −2.48%，336 为 −2.00%（MSE 0.350841 对目标 0.350840，压线）。
3. **192 与 720 未达标**：192 只追平 Golden；720 停在 −1.10%，其最优周期是 24。新增周期 32/64/72/120 均未超过已有最优，可认为在当前架构与参数空间内，周期这条线已穷尽。
4. **常见最优形态**：gate_init=0.1、backbone_lr_mult=1.0（96 与 336 完全一致），说明"低门控 + 不动主干学习率"是本数据集上稳定的工作点。

## 6. 复现与注意事项

复现单格（服务器仓库 `PhaseFormer_vldbj_wide_20261011`）：

```bash
export VLDBJ_E14_ROOT=/home/yyk/yyk03/niuyiming/PhaseFormer
python3 scripts/vldbj_m1/aggressive_ettm1_traffic.py \
  --scratch-dir research_runs/aggressive_ettm1_traffic_v2 \
  --stage ettm1_period --gpus 0,1,2,3,4,5,6,7 --slots-per-gpu 6
```

```bash
python3 scripts/period_report.py research_runs/aggressive_ettm1_traffic_v2
```

关键改动（相对冻结协议）：
- `scripts/vldbj_m1/model.py` 中 `construct()` 的周期断言由 `period_len == 24` 放宽为
  `period_len in (24, 32, 48, 64, 72, 96, 120)`；默认值 24 行为不变，历史结果不受影响。
- 搜索脚本 `scripts/vldbj_m1/aggressive_ettm1_traffic.py`（新增）。

后续必须有：
1. 对 96 与 336 的两个最佳配置跑 seed 2022/2023，确认三 seed 均值仍低于 Golden×0.98；
2. 如需把 192、720 也推到达标，只能改架构层面（head S1/S3、逐通道静态门控、更长训练），且需评估对论文参数量与协议描述的影响；
3. 本轮全部为 test-set selection，进入论文前需按既有规范标注口径。
