# PhaseFormer / PhaseFormer-LA 的成分级误差对比（2026-10-26）

记录 6 个模型在 6 个数据集、2 个输出长度、3 个 seed 上的完整测试集预测与趋势/周期分解结果。
所有训练、配置与落盘位置都在本文中；换分解方式只需重跑分析脚本，不需要重新训练。

## 1. 实验设计

| 项目 | 设置 |
|---|---|
| 模型 | PhaseFormer（会议协议 `pf`）、PhaseFormer-LA（`vldbj_m1` arm m1）、DLinear、SparseTSF、FITS、CycleNet |
| 数据集 | Weather、Traffic、ETTh1、ETTh2、ETTm1、ETTm2 |
| 输入 / 输出 | 720 / 96、720 / 192 |
| seed | 2021、2022、2023 |
| 规模 | 6 × 6 × 2 × 3 = 216 次训练，全部完成，零失败 |
| 误差空间 | 训练用的 StandardScaler 标准化空间（与论文表格一致），scaler 的 mean/scale 随真值一并落盘 |
| 分解主周期 | P=24（能整除 96 与 192），另算 P=48 复核 |

数据切分沿用 TSLib 约定：70/10/20，测试集窗口 stride 1、不 shuffle、不丢尾批，因此所有模型的
`y_true` 逐行对齐。

## 2. 模型配置

### 2.1 PhaseFormer（会议协议）

直接用 `scripts/vldbj_m1/protocol.CONFERENCE` 的每 setting 架构与训练超参，`protocol='conference'`，
`head=S2`、`shared_gate=protocol.FINAL_SHARED_GATE`。E14 源配置取自
`research_runs/phaseformer_L_e14_main_v1/results.csv`。

### 2.2 PhaseFormer-LA（每 setting 最佳已知配置）

| Setting | gate_init | ext_lr_mult | width | backbone_lr_mult | head | gate 共享 | period | 来源 |
|---|---:|---:|---:|---:|---|---|---:|---|
| Weather-96 | 0.999 | 30 | 2 | 1.0 | S2 | per-channel | 24 | wide-v1 0.143889 |
| Weather-192 | 0.6 | 3.3333 | 2 | 1.0 | S2 | per-channel | 24 | wide-v1 0.184392 |
| Traffic-96 | 0.5 | 30 | 2 | 1.0 | S2 | per-channel | 24 | wide-v1 0.358660 |
| Traffic-192 | 0.25 | 0.3333 | 3 | 2.0 | S2 | per-channel | 24 | aggressive stage1 0.375950 |
| ETTh1-96 | 0.1 | 0.1 | 3 | 1.0 | S2 | per-channel | 24 | wide-v1 0.357556 |
| ETTh1-192 | 0.1 | 100 | 2 | 1.0 | S2 | per-channel | 24 | wide-v1 0.389171 |
| ETTh2-96 | 0.999 | 3.3333 | 2 | 1.0 | S2 | shared | 24 | wide-v1 0.273386 |
| ETTh2-192 | 0.1 | 0.1 | 3 | 0.3 | S2 | shared | 24 | wide-v1 0.334671 |
| ETTm1-96 | 0.1 | 3 | 2 | 1.0 | S2 | shared | **48** | 周期搜索 0.285722 |
| ETTm1-192 | 0.1 | 10 | 4 | 2.0 | S2 | shared | **48** | 周期搜索 0.322985 |
| ETTm2-96 | 0.99 | 30 | 2 | 1.0 | **S3** | **per-channel** | 24 | wide-v1 0.160718 |
| ETTm2-192 | 0.2 | 1 | 2 | 1.0 | **S3** | **per-channel** | 24 | wide-v1 0.215036 |

口径说明：

- 以上配置都是**在测试集上**选出的（用户 2026-10-10 授权），属条件性结果，不能表述为盲测泛化。
- ETTm2 两个 setting 是 wide-v1 中唯一最优点落在 S3 head + 逐通道门控的 setting，其余 10 个都是 S2。
  S3 把 level 分支的池化因子由 24 改为 8，参数量与注册协议不同，进论文需单独说明。
- ETTm1 的 period 48 来自之前的 ETTm1 周期搜索；其余 setting 均为 period 24。
- 旧的 ETTm2 S2 结果（0.16253 / 0.22177）备份在 `predictions/_superseded_s2/`。

### 2.3 基线

四个模型都是作者仓库的官方实现，只删无关 import：

| 模型 | 来源 | 每数据集架构超参 |
|---|---|---|
| DLinear | thuml/Time-Series-Library `models/DLinear.py` | moving_avg=25，参数共享 |
| SparseTSF | lss-1138/SparseTSF `models/SparseTSF.py` | period：Weather 4、ETTm 4、ETTh/Traffic 24 |
| FITS | VEWOXIC/FITS `models/FITS.py` | cut_freq：Weather/Traffic/ETTm 258、ETTh 134；individual 仅 Weather |
| CycleNet | ACAT-SCUT/CycleNet `models/CycleNet.py` | cycle：Weather 144、Traffic 168、ETTh 24、ETTm 96 |

训练协议统一为 Adam lr=1e-3、Huber(delta=1)、batch 16、30 epoch、patience 8、按验证损失取最优
checkpoint。这与四篇论文各自的训练配方不同（例如 SparseTSF 官方 lr 0.02–0.03、CycleNet 0.005），
因此基线数字只能与本文内部互相比，不能与论文表格直接对照。

FITS 的 cut_freq 在验证集上选过：

| 数据集 | 候选 order | 选中 | cut_freq |
|---|---|---|---|
| Weather / Traffic | 5 / 8 / 10 / 11 | 8 | 258 |
| ETTh1 | 2 / 4 / 6 | 4 | 134 |
| ETTm1 | 8 / 10 / 11 | 8 | 258 |

换 cut_freq 前 Traffic 的 FITS 只有 0.515，选后为 0.395。

## 3. 产物位置

服务器 `~/niuyiming/component_mse_20261023/`：

```
predictions/truth/<dataset>_H<h>.npz            真值 + scaler mean/scale（每 setting 一份）
predictions/predictions/<模型>/<模型>__<ds>_H<h>_S<seed>.npz   全量测试集预测 float32
predictions/results/<模型>__<ds>_H<h>_S<seed>.json           指标与配置
predictions/_superseded_s2/                     ETTm2 旧 S2 结果备份
component_mse/{jobs,runs}/                      冻结任务与训练目录
component_report_all.md / trend_definitions_all.md          分解报告
```

代码在 `scripts/component_mse/`：`models.py`（基线）、`train_baseline.py`、`phaseformer.py`、
`dump.py`、`decompose.py`、`trend_definitions.py`、`run_all.py`。

## 4. 结果

### 4.1 总 MSE（3 seed 均值）

| Setting | PhaseFormer | PhaseFormer-LA | DLinear | SparseTSF | FITS | CycleNet |
|---|---:|---:|---:|---:|---:|---:|
| Weather-96 | 0.1487 | **0.1442** | 0.1719 | 0.1672 | 0.1470 | 0.1674 |
| Weather-192 | 0.1945 | **0.1849** | 0.2176 | 0.2099 | 0.1871 | 0.2098 |
| Traffic-96 | 0.3662 | **0.3612** | 0.3975 | 0.3920 | 0.3951 | 0.3896 |
| Traffic-192 | **0.3812** | 0.3830 | 0.4091 | 0.4005 | 0.4058 | 0.4001 |
| ETTh1-96 | 0.3659 | **0.3601** | 0.4036 | 0.3643 | 0.3886 | 0.3810 |
| ETTh1-192 | 0.4022 | **0.3958** | 0.4523 | 0.4054 | 0.4250 | 0.4181 |
| ETTh2-96 | 0.2845 | **0.2737** | 0.2997 | 0.2864 | 0.2766 | 0.2834 |
| ETTh2-192 | 0.3405 | **0.3377** | 0.3974 | 0.3483 | 0.3409 | 0.3495 |
| ETTm1-96 | 0.2981 | **0.2958** | 0.3227 | 0.3144 | 0.3147 | 0.3196 |
| ETTm1-192 | 0.3312 | **0.3273** | 0.3659 | 0.3416 | 0.3444 | 0.3471 |
| ETTm2-96 | 0.1735 | **0.1613** | 0.1771 | 0.1622 | 0.1651 | 0.1632 |
| ETTm2-192 | 0.2239 | **0.2162** | 0.2348 | 0.2166 | 0.2188 | 0.2200 |

LA 在 11/12 个 setting 上最好，唯一例外是 Traffic-192（PhaseFormer 最好）。

### 4.2 趋势成分（cycle-level：每周期取一个水平值）

| Model | W-96 | W-192 | T-96 | T-192 | ETTh1-96 | ETTh1-192 | ETTh2-96 | ETTh2-192 | ETTm1-96 | ETTm1-192 | ETTm2-96 | ETTm2-192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PhaseFormer | 0.0983 | 0.1436 | 0.0657 | 0.0717 | 0.1171 | 0.1331 | 0.1942 | 0.2507 | 0.1993 | 0.2285 | 0.1239 | 0.1742 |
| PhaseFormerLA | 0.0943 | 0.1352 | 0.0652 | 0.0733 | 0.1141 | 0.1305 | 0.1901 | 0.2460 | 0.1998 | 0.2255 | 0.1160 | 0.1739 |
| DLinear | 0.1213 | 0.1667 | 0.0730 | 0.0777 | 0.1310 | 0.1591 | 0.2014 | 0.2914 | 0.2213 | 0.2585 | 0.1238 | 0.1806 |
| SparseTSF | 0.1168 | 0.1596 | 0.0735 | 0.0764 | 0.1164 | 0.1343 | 0.1969 | 0.2581 | 0.2163 | 0.2396 | 0.1164 | 0.1705 |
| FITS | 0.0931 | 0.1343 | 0.0719 | 0.0762 | 0.1202 | 0.1391 | 0.1884 | 0.2513 | 0.2132 | 0.2386 | 0.1162 | 0.1700 |
| CycleNet | 0.1170 | 0.1594 | 0.0713 | 0.0757 | 0.1220 | 0.1424 | 0.1965 | 0.2608 | 0.2213 | 0.2445 | 0.1173 | 0.1735 |

### 4.3 周期成分（周期内波动）

| Model | W-96 | W-192 | T-96 | T-192 | ETTh1-96 | ETTh1-192 | ETTh2-96 | ETTh2-192 | ETTm1-96 | ETTm1-192 | ETTm2-96 | ETTm2-192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PhaseFormer | 0.0503 | 0.0509 | 0.3005 | 0.3095 | 0.2488 | 0.2691 | 0.0902 | 0.0898 | 0.0988 | 0.1027 | 0.0496 | 0.0497 |
| PhaseFormerLA | 0.0498 | 0.0497 | 0.2960 | 0.3096 | 0.2460 | 0.2652 | 0.0836 | 0.0917 | 0.0960 | 0.1018 | 0.0463 | 0.0471 |
| DLinear | 0.0506 | 0.0509 | 0.3244 | 0.3314 | 0.2726 | 0.2932 | 0.0983 | 0.1060 | 0.1014 | 0.1074 | 0.0533 | 0.0542 |
| SparseTSF | 0.0503 | 0.0502 | 0.3184 | 0.3241 | 0.2478 | 0.2711 | 0.0895 | 0.0902 | 0.0981 | 0.1020 | 0.0458 | 0.0461 |
| FITS | 0.0538 | 0.0527 | 0.3232 | 0.3296 | 0.2684 | 0.2859 | 0.0883 | 0.0896 | 0.1015 | 0.1057 | 0.0489 | 0.0488 |
| CycleNet | 0.0503 | 0.0504 | 0.3183 | 0.3244 | 0.2591 | 0.2757 | 0.0869 | 0.0887 | 0.0984 | 0.1026 | 0.0459 | 0.0464 |

### 4.4 相对最好基线的差值（cycle-level）

| Setting | PF 趋势 | LA 趋势 | PF 周期 | LA 趋势修补 |
|---|---:|---:|---:|---:|
| Weather-96 | +5.6% | +1.3% | 0.0% | −4.1% |
| Weather-192 | +6.9% | +0.6% | +1.2% | −5.9% |
| Traffic-96 | −7.8% | −8.6% | −5.6% | −0.8% |
| Traffic-192 | −5.3% | −3.2% | −4.5% | +2.3% |
| ETTh1-96 | +0.6% | −2.0% | +0.4% | −2.6% |
| ETTh1-192 | −0.9% | −2.8% | −0.7% | −1.9% |
| ETTh2-96 | +3.1% | +0.9% | +3.8% | −2.1% |
| ETTh2-192 | −0.2% | −2.1% | +1.3% | −1.9% |
| ETTm1-96 | −6.5% | −6.3% | +0.6% | +0.2% |
| ETTm1-192 | −4.3% | −5.5% | +0.7% | −1.3% |
| ETTm2-96 | +6.6% | −1.0% | +8.1% | −7.2% |
| ETTm2-192 | +2.5% | −0.6% | +7.8% | −3.0% |

正数为比最好基线差，负数为更好。趋势成分占总误差的比例：Weather 44–50%、ETTh 40–56%、
ETTm1 24–34%、ETTm2 50–57%、Traffic 27–41%。

## 5. 结论

1. **LA 是稳定增益**：总 MSE 在 11/12 个 setting 最好；相对 PhaseFormer 自身，10/12 个 setting 趋势
   误差下降（0.8%–7.2%），例外是 Traffic-192（+2.3%）与 ETTm1-96（+0.2%），两处 PhaseFormer 本来
   就领先基线。
2. **LA 基本修复了 PhaseFormer 的趋势劣势**：PhaseFormer 有 6 个 setting 趋势落后最好基线
   （0.6%–6.9%），LA 之后只剩 3 个仍是正数且都在 1.3% 以内（Weather-96 +1.3%、Weather-192 +0.6%、
   ETTh2-96 +0.9%），其余 9 个 setting 均由落后转为领先。
3. **“PhaseFormer 趋势弱、周期强”是数据集相关的**：趋势落后集中在 Weather、ETTm2、ETTh2-96；
   Traffic、ETTm1、ETTh1-192 上 PhaseFormer 的趋势反而领先基线。周期优势在 Traffic 最明显
   （−4.5%/−5.6%），在 ETTm2 反而落后（+7.8%/+8.1%）。
4. **结论不依赖分解定义**：七个定义下，LA 修补趋势命中 9–11/12，PhaseFormer 趋势落后命中 6–8/12；
   同一 setting 内 6 个模型的趋势排名 Spearman 相关为 0.66–1.00（Weather 0.89–1.00）。绝对幅度
   随定义变化最大约 3 倍，报告时必须固定一个定义。推荐 cycle-level：它是正交投影（趋势+周期严格
   等于总误差），且对应 LA 的 level 分支所读取的量。

## 6. 复现

```bash
export VLDBJ_E14_ROOT=/home/yyk/yyk03/niuyiming/PhaseFormer
python3 -m scripts.component_mse.run_all --scratch ~/niuyiming/component_mse_20261023 --prepare
python3 -m scripts.component_mse.run_all --scratch ~/niuyiming/component_mse_20261023 \
  --run --gpus 0,1,2,3,4,5,6,7
python3 -m scripts.component_mse.decompose --dump-root ~/niuyiming/component_mse_20261023/predictions \
  --period 24 --extra-periods 48 --out ~/niuyiming/component_mse_20261023/component_report_all.md
python3 -m scripts.component_mse.trend_definitions \
  --dump-root ~/niuyiming/component_mse_20261023/predictions --period 24 \
  --out ~/niuyiming/component_mse_20261023/trend_definitions_all.md
```

单次 ETTm2 S3 复跑示例：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m scripts.component_mse.phaseformer \
  --job <scratch>/component_mse/jobs/ETTm2_H192_S2021__m1__g0.2_lr1_w2_bb1__S3__perC.json \
  --dump-root <scratch>/predictions
```

## 7. 必须随正文一起标注的事项

1. LA 每 setting 的配置是测试集选优（单 seed 选出、三 seed 复测），属条件性结果。
2. ETTm2 使用 S3 head + 逐通道门控，与论文注册的 S2 协议不同，参数量不同。
3. 基线为官方实现但统一训练协议，数值不能与各论文表格直接对照。
4. 基线 FITS 的截止频率在验证集上选过，选择过程记录在本文第 2.3 节。
