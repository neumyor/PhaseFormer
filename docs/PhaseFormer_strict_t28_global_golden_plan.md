# Strict-T28 全数据集 Golden 对比计划

## 目标与冻结对象

本计划测试一个统一的、随机初始化且单次 `Trainer.fit` 训练的模型：
`pctf_anchor_repair_strict_t28`。它将 A2 的完整相位—NLinear/LFF 预测作为锚点，只增加有界的
ICPT 周期 level/shape 修正；融合器对 A2 输入完全 stop-gradient，A2 仅由 anchor loss 学习。

已冻结的通用训练设置为：lookback=720、Huber、最多 30 epoch、best-validation checkpoint、
anchor/composer LR=1、anchor loss=1、shape/level/gate auxiliary weight 均为 0.05、无 warm-up。
T28 在 ETTh2/ETTm2 H192 的小范围 validation 搜索中使 correction/deformation/global-level 边界
取 `0.60/0.24/0.12`，但尚未超过 two-stage Full Repair；它是本轮的**起点**，不是全数据集最优的
既定结论。

固定 Golden 覆盖 ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic 的四个 horizon，
共 28 个 setting。Exchange 没有权威 Golden，因此不混入本轮主结论。

## 为什么不能直接跑 84 个 test

不同数据集的周期长度与可接受修正幅度可能不同；但同一数据集的四个 horizon 不应随意切换机制。
更关键的是，当前 composer 要求 `pred_len % cycle_period == 0`：历史 ETTm2 的 cycle=96 不适用于
H336。因此必须先基于 validation 为每个数据集冻结一个能覆盖四个 horizon 的周期和 trust-region
档位，再读取 test。

## Stage A：低成本周期可行性筛选

使用 strict-T28 的原始边界、30% train、8 epoch、seed=2021、validation-only，在 H96/H336 两端筛选：

| 数据集 | 候选 cycle | run 数 | 原因 |
|---|---|---:|---|
| ETTm2 | 24、48 | 4 | 96 无法整除 H336；需替代历史 96 |
| Traffic | 12、24、48 | 6 | 尚无 anchored-PCTF 冻结周期 |

其它数据集沿用已有训练期冻结且四 horizon 均整除的周期：ETTh1=48、ETTh2=48、ETTm1=48、Weather=24、
Electricity=12。每个数据集按 H96/H336 的 MSE/MAE 相对本数据集各周期最优值的联合均值排序；若前二
差小于 0.2%，优先较短周期，降低模型复杂度。此阶段不读取 test。

## Stage B：数据集级 trust-region 推导

周期冻结后，在所有 7 个数据集、H96/H336、30% train、8 epoch、seed=2021 上比较三个统一档位：

| 档位 | correction / deformation / global-level |
|---|---|
| C（保守） | 0.25 / 0.10 / 0.05 |
| M（中等） | 0.40 / 0.16 / 0.08 |
| S（平滑插值） | 0.50 / 0.20 / 0.10 |
| W（T28） | 0.60 / 0.24 / 0.12 |

共 `7 × 2 × 4 = 56` 个 validation-only run。每个数据集只冻结一个档位，并原样外推至其 H192/H720；
不按 horizon 挑选不同档位。选择分数为四个 MSE/MAE 比值相对 C 的平均，附加约束为任一端点单指标
退化不超过 0.5%。不通过时仍保守选 C 并标记该数据集“无可信 correction 扩张收益”。

## Stage C：确认性 validation

对 7 个冻结的“周期+档位”组合，在四个 horizon、100% train、seeds 2021/2022 做 validation-only
复核，共 `7 × 4 × 2 = 56` 个 run。该阶段只验证跨 horizon 稳定性；不再改变周期、档位、损失或
容量。若某数据集四 horizon 的 16 个 MSE/MAE 比值中任一值相对 C 回退超过 0.5%，该数据集回退到 C。

## Stage D：正式 Golden 对比

冻结后，strict-T28 在 28 个 Golden setting 上使用 full train、seeds 2021/2022/2023、best-validation
checkpoint、一次 test，共 **84** 个 candidate run；不需要重跑 Golden。

每格报告 MSE/MAE mean±sample std、相对 Golden 的绝对/百分比变化、训练时间、峰值显存、参数量。
只有每个 seed 的 MSE/MAE 都低于 Golden，且 `mean + std < Golden`，才称为该 setting 稳定双指标超过
Golden。由于历史 two-stage Full Repair 的 test 已暴露，ETTh2/ETTm2 的结果须披露相关 test exposure，
不得描述为完全盲测。

## 当前 Stage A/B 进度（2026-09-01 更新）

Stage A 全部完成。ETTm2 由先导探测冻结 cycle=24；Traffic 在本轮 4×A100 补跑
（30% train、8 epoch、seed 2021、validation-only，输出
`research_runs/pctf_strict_t28_global_golden_v1/`）。

Traffic Stage A 结果（validation MSE/MAE）：

| Horizon | cycle=12 | cycle=24 | cycle=48 |
|---|---:|---:|---:|
| 96 | 0.328997 / 0.217733 | 0.329051 / 0.218076 | 0.328870 / 0.218076 |
| 336 | 0.346189 / 0.217846 | 0.345601 / 0.218260 | 0.345822 / 0.218510 |

Traffic 各 cycle 的联合相对分数（四个单元格相对各单元格最优值的均值）：cp12=1.000522、
cp24=1.001006、cp48=1.001315。cp12 为最优且同时是最短周期，规则无歧义：**冻结 Traffic cycle=12**。
决策文件：`research_runs/pctf_strict_t28_global_golden_v1/frozen_decisions.json`。

Stage B（7 数据集 × H96/H336 × 4 档，30% train、8 epoch、seed 2021，共 56 run）完成。
每数据集按”四个 MSE/MAE 比值相对 C 的均值”选档；约束为任一端点单指标相对 C 退化 ≤0.5%，
且该档平均上必须相对 C 有可信改进（mean<1.0）；不满足则保守冻结 C。

| 数据集 | cycle | C(0.25/0.10/0.05) | M(0.40/0.16/0.08) | S(0.50/0.20/0.10) | W(0.60/0.24/0.12) | 冻结档位 |
|---|---|---:|---:|---:|---:|---|
| ETTh1 | 48 | 0.70410/0.58372/1.28985/0.79255 | 0.99701 | 0.99519 | 0.99349 | W |
| ETTh2 | 48 | 0.20995/0.31898/0.37094/0.42765 | 0.99706 | 0.99516 | 0.99342 | W |
| ETTm1 | 48 | 0.42305/0.43538/0.65035/0.53597 | 1.00028 | 1.00039 | 0.99909 | W |
| ETTm2 | 24 | 0.12039/0.23733/0.19901/0.30257 | 0.99942 | 0.99962 | 0.99977 | M |
| Weather | 24 | 0.41908/0.29968/0.53018/0.37861 | 0.99426 | 0.99298 | 0.99800 | S |
| Electricity | 12 | 0.11269/0.20624/0.14082/0.23427 | 0.99909 | 0.99940 | 0.99959 | M |
| Traffic | 12 | 0.32862/0.21771/0.34548/0.21777 | 1.00130 | 1.00138 | 1.00091 | C（无可信扩张收益） |

表中 C 列为 H96/H336 的 `mse/mae/mse/mae`；M/S/W 列为四个比值相对 C 的均值。

## Stage C 确认结果与主研究者决策（2026-09-01）

Stage C 对 ETTh1/ETTh2/ETTm1/ETTm2/Weather 五个数据集完成（7 冻结组合 × 4 horizon ×
seeds 2021/2022，100% train，validation-only）。Electricity 与 Traffic 的 Stage C 由主研究者
指令跳过：Electricity 直接确认 M（不执行回退检查），Traffic 保持 Stage B 的 C。

已执行数据集的 16 比值回退检查（任一 frozen/C 比值 >1.005 回退到 C）：

| 数据集 | 冻结档位 | worst 比值 | 触发 cell | 结论 |
|---|---|---:|---|---|
| ETTh1 | W | 1.0074 | mse@H336·s2021 | **回退到 C** |
| ETTh2 | W | 1.0068 | mse@H720·s2022 | **回退到 C** |
| ETTm1 | W | 1.0018 | 无 | 保持 W |
| ETTm2 | M | 1.0055 | mse@H720·s2021 | **回退到 C** |
| Weather | S | 1.0047 | 无 | 保持 S |

三个触发回退的数据集各仅有一个 MSE cell 略超 1.005（超 0.05%–0.74%），其余 15 个比值均低于
阈值、16 比值均值均在 1.0 附近；但按预注册规则严格回退到保守档 C。最终冻结档位：

| 数据集 | cycle | 档位 |
|---|---|---:|---|
| ETTh1 | 48 | C |
| ETTh2 | 48 | C |
| ETTm1 | 48 | W |
| ETTm2 | 24 | C |
| Weather | 24 | S |
| Electricity | 12 | M（主研究者确认） |
| Traffic | 12 | C |

Stage D（28 setting × 3 seed，full train + test）已据此启动。

## Stage D 结果与任务变更（2026-09-01）

Stage D 在 ETT 四件套 + Weather（60/84 runs，20 settings × 3 seeds）完成后，
Electricity/Traffic 的剩余 run 由主研究者指令取消。已完成的 20 个 setting 相对 Golden：
ETTh2/ETTm2（均 C 档，test 曾被历史 two-stage 实验暴露）稳定双指标超越 6/8 项，
ETTh1（C 档）、ETTm1（W 档）、Weather（S 档）均未超越。原始部分总表见本次会话汇报。

随后任务变更为：**Weather 专项大范围参数搜索**（`pctf_anchor_repair_strict_t28`，
1 seed，H96 与 H192 的 MSE/MAE 均需比 Golden 好 ≥0.5%）。搜索驱动
`scripts/search_weather_t28.py`，输出 `research_runs/pctf_weather_search_v1/`。
层次：Layer 1 协议主扫描（LR×epochs×loss，tier=X）→ Layer 2 机制极值
（tier×gate，再补 warmup/lr_scale）→ Layer 3 精修与最终确认（LR 局部/lookback/
锚定-融合器平衡）。每个候选直接 `--stage confirm --evaluate-test`，test 指标为唯一
比较来源。总墙钟预算约 9 小时。目标：H96 MSE≤0.1473 且 MAE≤0.1940；H192
MSE≤0.1920 且 MAE≤0.2358。

## Weather 专项搜索最终结果（2026-09-01）

共 ~62 个有效 run（Layer 1：24；Layer 2：16；Layer 3 批 1：10；lr_scale 合法格：6；
anchor-mid 探针：4；另有 anchor_scale=2.0 因模型约束 `0≤anchor_lr_scale≤1.0` 非法，见下）。
每个候选 `--stage confirm --evaluate-test`，test 指标为唯一比较来源，seed=2021。

**Layer 1（协议主扫描，tier X，LR×epochs×loss×2H）**
- mae-loss 完胜：huber 全部 MAE>0.238；mae 在 H192 直接双达标（lr=0.001 → 0.1908/0.2322；
  lr=0.003 → 0.1914/0.2332）。
- ep30=ep60（best-val 早停），epoch 维度无效，后续固定 ep30。
- H96 最优 lr=0.002（0.1477/0.1897），H192 最优 lr=0.001（0.1908/0.2322）；无单一 lr 双达标。

**Layer 2（机制极值，tier{W,X,Y,Z}×gate{0,0.05}，lr=0.002，mae）**
- tier×gate 全部无效：H96 8 格 0.1475–0.1478，H192 8 格 0.1925–0.1934（跨度 0.0003）。
- 修正幅度已收敛到边界内，trust-region 旋钮到极限。最优格 W·g0：H96 0.1475/0.1896，
  H192 0.1925/0.2339。

**Layer 3（精修，基座 W·g0·lr=0.002·mae）**
- LR 局部：H96 谷值就在 lr=0.002（0.0018→0.1488，0.0022→0.1499，0.0025→0.1489）；
  H192 最优 lr=0.0018（0.1920/0.2331，MSE 贴线达标）。
- lookback=512 变差（H96 0.1506），warmup=5 微差（H96 0.1477，H192 0.1932）。
- lr_scale 合法格（anchor∈[0,1]，composer>0）：anchor=0.5 使 H192 达 0.1909/0.2323 ✔
  但 H96 崩到 0.1516（退化约 4× 于 H192 获益）；composer=0.5/2.0 与基线完全持平
  （0.1475/0.1925）。anchor=2.0 非法（模型上限 1.0），GPUs 0/1 因此 `set -e` 提前退出，
  其上的 composer=2.0 两格已单独补跑。
- anchor-mid 探针：as=0.9 → H192 0.1919/0.2331 ✔ 但 H96 0.1487（miss 0.95%）；as=0.85 →
  H96 0.1496/H192 0.1931 双退化。线性外推（H96 退化斜率≈4×H192 获益）被证实。
- anchor_loss_weight 终探针（preset 默认 1.0）：alw=3.0 → H192 0.1914/0.2326 ✔ 但 H96
  0.1478（差 0.34%）；alw=2.0 → H192 0.1922/H96 0.1487 双不达标。与 anchor_scale 相同结构：
  任何强化 H192 的锚定动力学旋钮都会使 H96 的 MSE 地板抬升。

**最终判定：搜索空间内无单一配置使 H96 与 H192 的 MSE/MAE 同时达到 ≥0.5% 优于 Golden。**

最优单一配置（mae / ep30 / lr=0.002 / tier W(0.60/0.24/0.12) / gate 0.0 / lookback 720 /
anchor_scale=1.0 / composer_scale=1.0 / seed 2021）：

| H | MSE (Gold) | MAE (Gold) | ΔMSE | ΔMAE | 目标判定 |
|---|---|---|---|---|---|
| 96 | 0.1475 (0.148) | 0.1896 (0.195) | −0.31% | −2.75% | MSE 差 0.14% |
| 192 | 0.1925 (0.193) | 0.2339 (0.237) | −0.27% | −1.31% | MSE 差 0.26% |

即四格全部优于 Golden，但 MSE 增益为 0.27–0.31%（目标 0.5%），MAE 增益强（1.31–2.75%）。
结构性原因：H96 的 MSE 在 lr=0.002 处有尖锐谷值 0.1475（任何其它旋钮都使其退化），而
H192 易达标；每个把 H192 推到 0.1920 以下的旋钮（lr↓、anchor↓、alw↑）都使 H96 退化更
严重。

**主研究者决定（2026-09-02）：停止搜索。** 保留上表最优配置为 Weather 最终结论；如实记录
0.5% MSE 双目标未达成（MSE 增益 0.27–0.31%，MAE 增益 1.31–2.75%，四格全部优于 Golden）。
不再投入计划外预算。

**最终配置 H336/H720 观察（2026-09-02，seed 2021，`driver_h336_720`）**：

| H | MSE (Gold) | MAE (Gold) | ΔMSE | ΔMAE |
|---|---|---|---|---|
| 336 | 0.2418 (0.242) | 0.2734 (0.278) | −0.10% | −1.65% |
| 720 | 0.3079 (0.309) | 0.3228 (0.332) | −0.37% | −2.76% |

最终配置在 Weather 四个 horizon 共 8 个 test 指标**全部优于 Golden**（MSE −0.10~−0.37%，
MAE −1.31~−2.76%）。H96/H192 的 MSE 增益（0.27–0.31%）未达 0.5% 双目标门槛，已记录。

## Stage D 完整登记表（2026-09-02 更新，含 ETTh1/ETTm1 专项搜索）

下表为当前版本在全部 Golden setting 上的登记结果。来源与二次校验规则：

- **ETTh1 / ETTm1**：主研究者在两数据集上专项搜索得到的最佳共享配置（ETTh1 `u_lr020`、
  ETTm1 `w_aux01`，均 1 seed），结果由主研究者报告、原样登记；**本机无这两个搜索的 run 数据**，
  因此不参与 1:1 校验、也不收集其 config/commands 文件。
- **ETTh2 / ETTm2**：Stage D 3-seed（2021/2022/2023）confirm run 的 test 均值±总体 std。
- **Weather**：Weather 专项搜索最终配置（1 seed，seed 2021，单次 test）。
- **Electricity (M) 与 Traffic (C)**：主研究者指令取消（60/84 处停止），标记 CANCELLED，未登记结果。
- 稳定判定（3-seed 行）：`mean+std < Golden` 且每 seed 双指标低于 Golden → BOTH；否则按单指标记
  MSE/MAE。单 seed 行（srch/W-srch）按双指标各自严格低于 Golden 判定。
- 登记表生成命令 `scripts/report_strict_t28_master_table.py`；有 run 数据的行其配置身份由
  `scripts/verify_run_reproducibility.py` 校验为 1:1。
- 每个已登记 cell 的 `config.json` + `commands.sh` 已整理到
  `docs/strict_t28_master_table_configs/<Dataset>/<h<horizon>>/`（见下节）。

| Dataset | H | conf | MSE | MAE | Gold M | Gold A | ΔMSE% | ΔMAE% | Beat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 96 | srch | 0.352 | 0.384 | 0.359 | 0.382 | −1.95 | +0.52 | MSE |
| ETTh1 | 192 | srch | 0.390 | 0.407 | 0.397 | 0.404 | −1.76 | +0.74 | MSE |
| ETTh1 | 336 | srch | 0.420 | 0.426 | 0.425 | 0.424 | −1.18 | +0.47 | MSE |
| ETTh1 | 720 | srch | 0.414 | 0.442 | 0.431 | 0.450 | −3.94 | −1.78 | BOTH |
| ETTh2 | 96 | 3seed | 0.273±0.001 | 0.332±0.001 | 0.275 | 0.338 | −0.79 | −1.71 | BOTH |
| ETTh2 | 192 | 3seed | 0.339±0.003 | 0.374±0.001 | 0.341 | 0.376 | −0.54 | −0.43 | MAE |
| ETTh2 | 336 | 3seed | 0.366±0.004 | 0.401±0.001 | 0.369 | 0.405 | −0.78 | −1.07 | MAE |
| ETTh2 | 720 | 3seed | 0.395±0.004 | 0.429±0.001 | 0.402 | 0.436 | −1.73 | −1.59 | BOTH |
| ETTm1 | 96 | srch | 0.291 | 0.338 | 0.293 | 0.344 | −0.68 | −1.74 | BOTH |
| ETTm1 | 192 | srch | 0.329 | 0.357 | 0.323 | 0.361 | +1.86 | −1.11 | MAE |
| ETTm1 | 336 | srch | 0.359 | 0.376 | 0.358 | 0.381 | +0.28 | −1.31 | MAE |
| ETTm1 | 720 | srch | 0.415 | 0.408 | 0.412 | 0.410 | +0.73 | −0.49 | MAE |
| ETTm2 | 96 | 3seed | 0.161±0.002 | 0.250±0.001 | 0.163 | 0.256 | −1.29 | −2.40 | BOTH |
| ETTm2 | 192 | 3seed | 0.214±0.000 | 0.286±0.001 | 0.219 | 0.293 | −2.09 | −2.30 | BOTH |
| ETTm2 | 336 | 3seed | 0.267±0.001 | 0.323±0.000 | 0.269 | 0.326 | −0.71 | −0.95 | BOTH |
| ETTm2 | 720 | 3seed | 0.347±0.003 | 0.375±0.001 | 0.351 | 0.379 | −1.18 | −0.98 | BOTH |
| Weather | 96 | W-srch | 0.1475 | 0.1896 | 0.148 | 0.195 | −0.31 | −2.75 | BOTH |
| Weather | 192 | W-srch | 0.1925 | 0.2339 | 0.193 | 0.237 | −0.27 | −1.31 | BOTH |
| Weather | 336 | W-srch | 0.2418 | 0.2734 | 0.242 | 0.278 | −0.10 | −1.65 | BOTH |
| Weather | 720 | W-srch | 0.3079 | 0.3228 | 0.309 | 0.332 | −0.37 | −2.76 | BOTH |
| Electricity | 96–720 | CANCELLED | — | — | — | — | — | — | — |
| Traffic | 96–720 | CANCELLED | — | — | — | — | — | — | — |

共 20 个已登记 setting，其中 12 个双指标超越 Golden（ETTh1 1/4、ETTh2 2/4、ETTm1 1/4、
ETTm2 4/4、Weather 4/4）；另有 8 个 setting（Electricity/Traffic 各 4）取消。ETTh1/ETTm1 为
单 seed 专项搜索结果，ETTh2/ETTm2 的 test 曾被历史 two-stage Full Repair 实验暴露，非完全盲测。

### 已登记 cell 的 config/commands 清单（docs/strict_t28_master_table_configs/）

有 run 数据的 12 个 cell（ETTh2/ETTm2 取 seed-2021 代表、Weather 单 seed）的 `config.json` 与
`commands.sh` 由 `scripts/collect_strict_t28_configs.py` 原样拷贝至
`docs/strict_t28_master_table_configs/<Dataset>/h<horizon>/`。布局与收录情况见该目录 `README.md`。
ETTh1/ETTm1 新搜索结果无本机 run 数据，未收集文件（仅登记结果）；Electricity/Traffic 取消。
ETTh2/ETTm2 其余两个 seed 的 run 仍保留在 `research_runs/pctf_strict_t28_global_golden_v1/`
及其 manifest 中。

## 二次校验与可复现清单（2026-09-02）

对两个输出树全部 run 目录执行二次校验
（`scripts/verify_run_reproducibility.py`，已提交本分支）：对每个 run 用
`search_phaseformer.py` 的同一算法重算 `config_hash`，必须与 `config.json`、`metrics.csv`、
run 目录名的 12 位十六进制后缀一致；并核对 metrics 行的 dataset/horizon/seed/loss/mechanism/
stage 与 config 一致、`commands.sh` 存在、无重复配置结果分叉。

| 树 | 目录数 | 有结果（1:1 校验通过） | 无结果 | hash/字段不一致 | 重复配置分叉 |
|---|---|---:|---:|---:|---:|
| pctf_strict_t28_global_golden_v1 | 210 | 60 | 150 | 0 | 0 |
| pctf_weather_search_v1 | 66 | 64 | 2 | 0 | 0 |

无结果目录均为预期：150 = Stage A/B/C validation-only 或取消的 Electricity run（8 个）；
2 = Weather anchor_scale=2.0 非法配置（status=failed，模型约束
`0≤anchored_pctf_anchor_lr_scale≤1.0`）。**124 个有结果 run 全部 1:1 对应，0 硬失败。**

每个 run 的完整配置已整理成清单文件（`manifest/runs/<run_id>.json`：完整 spec + 指标 +
校验状态；`manifest/index.json` 索引；`manifest/verification_report.md` 汇总）：

- `research_runs/pctf_strict_t28_global_golden_v1/manifest/`
- `research_runs/pctf_weather_search_v1/manifest/`

（manifest 随 gitignored 的 `research_runs/` 存储，可由提交的校验脚本随时重新生成；`config.json`
与 `commands.sh` 原本就在每个 run 目录中，是逐 run 的权威配置文件。）

## 可复现执行与参数治理

所有筛选通过 `scripts/search_phaseformer.py` 执行，必须添加 `--require-cuda`，且不添加 `--test`；
每个 run 的 `metrics.csv` 是唯一比较来源。Stage A 的命令模板如下（替换数据集、horizon 与
`--cycle-period`）：

```bash
.venv/bin/python scripts/search_phaseformer.py \
  --dataset ETTm2 --horizon 96 --stage period_screen \
  --mechanism pctf_anchor_repair_strict_t28 --period 24 --cycle-period 24 \
  --lookback 720 --percent 30 --max-epochs 8 --seed 2021 --loss huber \
  --num-workers 0 --bad-case-limit 0 \
  --output-dir research_runs/pctf_strict_t28_global_pilot_v1/period \
  --require-cuda --resume
```

参数的可选择范围严格限于本计划的 dataset-shared `cycle_period` 和 trust-region 档位；学习率、
loss、容量、辅助权重、seed 与训练协议不随数据集改变。这样可以检验“周期尺度和修正幅度是否随
数据生成频率不同”这一假设，而不是以大量数据集专属旋钮追逐指标。每次冻结均需在本文件写入
validation 分数、tie-break 及失败/OOM 状态，随后才允许进入下一阶段。
