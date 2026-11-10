# B0H+ 当前最优配置逐 setting 记录（2026-11-03）

本文件记录 7 个数据集 × 4 个 horizon 共 28 个 setting 上 B0H+ 目前的最好结果，以及复现所需的全部参数。
所有数字为单 seed 2021、测试集选优（项目既定做法），属探索性结果。

## 1. 口径

- 模型：b0 臂（相位 + level，无 amplitude），level 编码器的输入为「周期均值 ‖ phase token 槽内均值」，即 B0H+。
- 输入长度 720，数据划分 70/10/20，指标在测试集、StandardScaler 标准化空间。
- 每行的 `test_mse` 取该 setting 在所有 B0H+ 实验（182 个调参任务、35 个变体任务、18 个门控任务、Traffic 专项搜索）中 seed 2021 的选定值。
- 28 行里有 24 行来自 182 个调参任务，另外 4 行的来源见第 4 节；Traffic 四行来自 2026-10-30 的专项搜索（按 MSE 与 MAE 的联合目标选优）。各来源口径不完全一致，引用时需注意。
- ETTh1 与 ETTh2 的 7 行于 2026-11-03 更新为大规模联合搜索的结果（`research_runs/etth12_b0hp_search_v1`，10507 个 run）。该轮按「两项相对 Golden 的提升中较差的那一项」选优，与其余行的 MSE 选优口径不同；两列的「与 b0 / 与 LA」对这些行不再重算，以 — 标注。MSE 优先的替代 cell 见第 5 节。

## 2. 逐 setting 配置

| 数据集 | H | seed | MSE | MAE | 与 b0 | 与 LA | 配置 | level 输入 | 来源 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| ETTh1 | 96 | 2021 | 0.34601 | 0.37808 | — | — | p24 · g0.99689 · gate×73.8 read×15.09 · bb7.978 · S1 · shared · detach · δ0.3 | history+token/raw | etth12-search |
| ETTh1 | 192 | 2021 | 0.37395 | 0.40047 | — | — | p24 · g0.51232 · gate×0.315 read×1.573 · bb10 · H6L · perC · δ0.15 | history+token/raw | etth12-search |
| ETTh1 | 336 | 2021 | 0.41057 | 0.41767 | — | — | p24 · g0.9683 · gate×16.48 read×1.414 · bb1.827 · H4L · shared · raw+zero · δ0.3 | history+token/raw/zero | etth12-search |
| ETTh1 | 720 | 2021 | 0.41643 | 0.43882 | -0.07% | -0.18% | p24 · g0.0001 · lr0.03 · bb0.1 · S2 · perC | history+token/norm | 182-sweep |
| ETTh2 | 96 | 2021 | 0.26911 | 0.33047 | — | — | p24 · g0.99873 · gate×20.28 read×7.232 · bb1.496 · H6L · shared | history+token/raw | etth12-search |
| ETTh2 | 192 | 2021 | 0.33233 | 0.37431 | — | — | p24 · g0.0114 · gate×2.525 read×0.1 · bb0.5576 · S2 · perC | history+token/norm | etth2-search |
| ETTh2 | 336 | 2021 | 0.35818 | 0.39902 | — | — | p24 · g0.78493 · gate×0 read×0.8456 · bb1.651 · H3L · shared · detach | history+token/norm | etth2-search |
| ETTh2 | 720 | 2021 | 0.38175 | 0.42193 | — | — | p24 · g0.98704 · gate×47.73 read×1.056 · bb0.5837 · S1 · perC · raw · δ0.3 | history+token/raw | etth2-search |
| ETTm1 | 96 | 2021 | 0.28923 | 0.34118 | -0.84% | +1.23% | p48 · g0.1 · lr3 · bb1 · S2 · shared | history+token/norm | p48-transfer |
| ETTm1 | 192 | 2021 | 0.32046 | 0.36150 | -1.10% | -0.78% | p48 · g0.95 · lr1000 · bb3 · S2 · perC | history+token/norm | 182-main |
| ETTm1 | 336 | 2021 | 0.35990 | 0.38123 | +0.35% | +2.58% | p24 · g0.1 · lr1 · bb1 · S2 · shared | history+token/norm | 182-main |
| ETTm1 | 720 | 2021 | 0.40365 | 0.41024 | -0.50% | -0.92% | p48 · g0.5 · lr10 · bb1 · S1 · perC | history+token/norm | 182-main |
| ETTm2 | 96 | 2021 | 0.15971 | 0.24799 | — | — | p24 · g0.0001 · gate×177.1 read×123.9 · bb9.941 · H2L · perC · δ0.3 | history+token/norm | ettm2-search |
| ETTm2 | 192 | 2021 | 0.21460 | 0.28665 | — | — | p24 · g0.11092 · gate×0.6796 read×0.244 · bb0.3929 · H6L · perC · zero · δ0.3 | history+token/norm/zero | ettm2-search |
| ETTm2 | 336 | 2021 | 0.26465 | 0.32047 | — | — | p24 · g0.12958 · gate×0.415 read×0.4045 · bb0.297 · H4L · shared · detach · δ0.3 | history+token/norm | ettm2-search |
| ETTm2 | 720 | 2021 | 0.34470 | 0.37466 | — | — | p48 · g0.999 · gate×10.3 read×8.25 · bb2 · S3 · shared | history+token/norm | ettm2-search |
| Weather | 96 | 2021 | 0.14404 | 0.19103 | +1.22% | +0.10% | p16 · g0.999 · lr160 · bb3 · S2 · perC | history+token/norm | 182-sweep |
| Weather | 192 | 2021 | 0.18484 | 0.23144 | +0.55% | +0.24% | p12 · g0.999 · lr40 · bb3 · S2 · perC | history+token/norm | 182-main |
| Weather | 336 | 2021 | 0.23562 | 0.27391 | +0.44% | +0.49% | p24 · g0.4995 · lr10 · bb1 · S2 · perC | history+token/norm | 182-sweep |
| Weather | 720 | 2021 | 0.31074 | 0.32559 | +0.91% | +0.37% | p36 · g0.02 · lr1000 · bb1 · S2 · perC | history+token/raw/zero | both |
| Electricity | 96 | 2021 | 0.12871 | 0.22195 | +0.16% | +0.73% | p24 · g0.95 · lr1000 · bb3 · S3 · perC | history+token/norm | 182-main |
| Electricity | 192 | 2021 | 0.14370 | 0.23437 | -0.15% | -0.18% | p24 · g0.0005 · lr0.4 · bb3 · S2 · perC | history+token/norm | 182-sweep |
| Electricity | 336 | 2021 | 0.16285 | 0.25533 | +0.02% | -0.91% | p24 · g0.2 · lr4 · bb3 · S3 · perC | history+token/norm | 182-main |
| Electricity | 720 | 2021 | 0.19465 | 0.28338 | -0.77% | -1.29% | p24 · g0.2 · lr2.5 · bb1 · S2 · perC | history+token/norm | 182-sweep |
| Traffic | 96 | 2021 | 0.35509 | 0.23283 | -0.09% | -1.00% | p24 · g0.475 · gate×0.03 read×0.03 · bb1 · S1 · perC | history+token/norm | traffic-search-A |
| Traffic | 192 | 2021 | 0.36810 | 0.24247 | -1.34% | -2.64% | p12 · g0.0001 · gate×0.1 read×1 · bb0.3 · S1 · shared | history+token/norm | traffic-search-B |
| Traffic | 336 | 2021 | 0.37796 | 0.24910 | -0.84% | -1.06% | p12 · g0.999 · gate×2.5 read×2.5 · bb1 · S2 · perC | history+token/raw | traffic-search-B |
| Traffic | 720 | 2021 | 0.42295 | 0.27177 | -0.33% | +0.00% | p12 · g0.999 · gate×1 read×1 · bb1 · H8G · perC | history+token/norm | traffic-search-A |

列名含义：`H` 预测长度；`g` = `gate_init`（门控 logit 初值）；`lr` = 扩展学习率倍数；`bb` = backbone 学习率倍数；
`S1/S2/S3` = level head（S2 为多项式解码，S1/S3 为线性解码，池化因子分别 24/8）；`shared/perC` = 门控是否逐通道；
`level 输入` 中 `history+token` 表示 B0H+ 的拼接输入，`norm`/`raw` 表示 token 取自 LayerNorm 之后/之前，`zero` 表示 token 列零初始化；
`gate×N read×N` 表示门控组与 level 读出组各用独立学习率倍数（未标注则两组共用 `lr`）。
`H8G` 是 Traffic 搜索新加的 head：池化因子 8、Legendre 多项式解码（`H<n>L/H<n>G` 同理，L 为线性解码、G 为多项式解码）。
Traffic 四行的 `ext_lr` 等于 `readout_mult`，`gate_mult` 单列；`token_zero_init` 与 detach 在搜索结果中均为关闭。

## 3. 四 horizon 平均与对比

| 数据集 | LA | b0 | B0H+ | 与 b0 | 与 LA | 与 Golden |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.3965 | 0.3955 | 0.3867 | -2.23% | -2.47% | -4.04% |
| ETTh2 | 0.3401 | 0.3405 | 0.3353 | -1.52% | -1.40% | -3.31% |
| ETTm1 | 0.3417 | 0.3450 | 0.3433 | -0.49% | +0.47% | -0.92% |
| ETTm2 | 0.2468 | 0.2462 | 0.2459 | -0.11% | -0.36% | -1.83% |
| Weather | 0.2181 | 0.2172 | 0.2188 | +0.76% | +0.33% | -1.88% |
| Electricity | 0.1583 | 0.1578 | 0.1575 | -0.24% | -0.53% | -2.07% |
| Traffic | 0.3854 | 0.3835 | 0.3810 | -0.64% | -1.14% | -1.47% |
| **七数据集平均** | **0.2981** | **0.2979** | **0.2955** | **-0.81%** | **-0.87%** | **-2.31%** |

MAE（四个 horizon 平均）：

| 数据集 | LA | b0 | B0H+ | 与 Golden |
|---|---:|---:|---:|---:|
| ETTh1 | 0.4149 | 0.4155 | 0.4088 | -1.49% |
| ETTh2 | 0.3854 | 0.3859 | 0.3814 | -1.87% |
| ETTm1 | 0.3740 | 0.3740 | 0.3735 | -0.13% |
| ETTm2 | 0.3123 | 0.3124 | 0.3074 | -1.93% |
| Weather | 0.2551 | 0.2535 | 0.2555 | -1.92% |
| Electricity | 0.2487 | 0.2488 | 0.2488 | -0.58% |
| Traffic | 0.2496 | 0.2512 | 0.2490 | -0.26% |
| **七数据集平均** | **0.3200** | **0.3202** | **0.3178** | **-1.21%** |

## 4. 来源说明与限制

21 个 setting 取自 182 个 B0H+ 调参任务（`research_runs/b0hp_v1/{main,sweep}`）。以下几条来自后续小规模实验；ETTh1 的 3 行由第 5 节取代，ETTh2 与 ETTm2 的 8 行由第 5、6 节取代，本表只保留其历史来源：

| setting | 来源 | 该变体在三 seed 上相对 b0 的配对差 | 可信度 |
|---|---|---|---|
| ETTh1-96（已被取代） | `b0hp_gate_v1` 门控×100 读出×10 | −3.27%、+5.24%、−2.48%（sd 4.71） | 低：单 seed 极值，三 seed 均值仅 −0.17% |
| ETTh2-720（已被取代） | `b0hp_gate_v1` 门控×100 读出×10 | +0.90%、−1.19%、+0.09%（sd 1.05） | 中：改善主要来自 raw token（三 seed 一致 −3~−4 个百分点） |
| Weather-720 | `b0hp_variants_v1` raw token + 零初始化 | +0.91%、+0.94%、+0.13%（sd 0.46） | 低：三 seed 均值 +0.66%，本行取的是其中较好的一次 |
| ETTm1-96 | `ettm1_p48_b0hp_v1` LA 的周期 48 配置迁移 | 未做三 seed | 中：单 seed 比原配置好 0.84% MSE / 0.64% MAE，两项均低于 Golden |

Traffic 四个 setting 来自 2026-10-30 的专项搜索（`research_runs/traffic_b0hp_v1`，脚本 `scripts/traffic_b0hp_search.py`）。
目标函数为 J = (MSE / LA 的 MSE + MAE / LA 的 MAE) / 2，逐 horizon 取 seed 2021 上 J 最小的 cell；
搜索在阶段 C 中途停止（A、B 阶段完成，C 完成 2 个，D 未开始），因此四行都来自 A/B，后续仍有改善空间。
四个 horizon 的选中 cell 均未启用新增的 detach 开关；336 用了 raw token，720 用了新 head H8G。

因此第 3 节的「与 b0」整体数字包含 28 次逐 setting 取最优（ETTh1/ETTh2 的 7 格除外，见第 5 节）。b0 的最优值同样是逐 setting 筛出来的，两者口径相同，
结论应表述为「打平」；历史上只有 ETTh2-720 的 raw token 改动有三 seed 方向一致的证据（该行现已被第 5 节取代）。
Traffic 四行是按联合目标选出的单次结果，测试集选优偏差与其余行同量级；96 与 336 存在 MSE 更低或 MAE 更低的其他 cell
（96 的 MSE 最低为 0.35428，MAE 最低为 0.23190；336 的 MAE 最低为 0.24783），本表取的是 J 最优的一格。

另外，b0 搜索按验证集选配置时，相对测试集最优平均损失 1.26%（最大 3.74%），这决定了单 seed 差异在 1% 量级上不可分辨。

## 5. 2026-11-03 ETTh1/ETTh2 联合 MSE/MAE 大规模搜索

2026-11-02 20:12 至 11-03 09:34 在服务器上跑了一轮以「两项同时超过 Golden 2%」为目标的搜索，
覆盖 ETTh1 与 ETTh2 的四个 horizon，seed 2021，测试集选优。期间校园网络中断约 6 小时，
远端进程未受影响，共完成 10507 个 run。

- 控制器：`~/niuyiming/etth12_b0hp_search.py`；结果：`research_runs/etth12_b0hp_search_v1/`；
  日志：`~/niuyiming/etth12_search.log`。每轮为每个未达标的 setting 生成 24 个新 cell，围绕当前最优做局部细化与全局采样。
- 搜索空间：周期（白名单内且整除 horizon）、`gate_init`、门控与读出的独立学习率倍数（0 表示冻结门控）、
  backbone 学习率倍数、level head（S1/S2/S3、H2L/H3L/H4L/H6L/H12L/H16L/H48L、H2G/H4G/H8G/H12G/H48G）、
  门控共享或逐通道、token 取 LayerNorm 前或后、token 零初始化、level 路径梯度截断、Huber delta（1 / 0.5 / 0.3 / 0.15）。
- 选优：最大化 `min(相对 Golden 的 MSE 提升, MAE 提升)`，即两项中较差的那一项。
  这与第 2 节其余行的 MSE 选优不同，故这些行的「与 b0 / 与 LA」不再重算。

逐 setting 结果（δ 为 Huber delta）：

| Setting | MSE | MAE | ΔMSE | ΔMAE | 配置 | run |
|---|---:|---:|---:|---:|---|---|
| ETTh1-96 | 0.346013 | 0.378078 | +3.62% | +1.03% | p24 · g0.99689 · gate×73.8 read×15.09 · bb7.978 · S1 · shared · raw · detach · δ0.3 | R14 |
| ETTh1-192 | 0.373951 | 0.400470 | +5.81% | +0.87% | p24 · g0.51232 · gate×0.315 read×1.573 · bb10 · H6L · perC · raw · δ0.15 | R46 |
| ETTh1-336 | 0.410565 | 0.417671 | +3.40% | +1.49% | p24 · g0.9683 · gate×16.48 read×1.414 · bb1.827 · H4L · shared · raw · zero · δ0.3 | R55 |
| ETTh1-720 | 0.416434 | 0.438817 | +3.38% | +2.49% | 未重跑（原配置已达标） | b0hp_v1/sweep |
| ETTh2-96 | 0.269106 | 0.330468 | +2.14% | +2.23% | p24 · g0.99873 · gate×20.28 read×7.232 · bb1.496 · H6L · shared · raw | R2 |
| ETTh2-192 | 0.335308 | 0.370468 | +1.67% | +1.47% | p24 · g0.00226 · gate×3.897 read×0.1456 · bb1.203 · S2 · perC · norm · zero · detach · δ0.15 | R90 |
| ETTh2-336 | 0.358868 | 0.398853 | +2.75% | +1.52% | p24 · g0.78493 · gate×3.28 read×0.9838 · bb1.511 · H3L · shared · norm · detach | R69 |
| ETTh2-720 | 0.390482 | 0.426168 | +2.87% | +2.26% | p24 · g0.9 · gate×100 read×5 · bb1 · S2 · shared · raw | R3 |

达标（两项均 ≥2%）的有三格：ETTh1-720、ETTh2-96、ETTh2-720。其余五格的 MAE 提升在 +0.87% 到 +1.52% 之间，
MSE 除 ETTh2-192（+1.67%）外均超过 2%。ETTh2-192 同时是两项都未达标的一格。

若按 MSE 优先选 cell，以下替代结果可把 MSE 再压低（MAE 相应变差，均未改善达标情况）：

| Setting | MSE | MAE | ΔMSE | ΔMAE | 差异 |
|---|---:|---:|---:|---:|---|
| ETTh1-96 | 0.345741 | 0.379888 | +3.69% | +0.55% | read×5.175、bb10 |
| ETTh1-192 | 0.373673 | 0.400552 | +5.88% | +0.85% | token 取 norm |
| ETTh1-336 | 0.402784 | 0.423816 | +5.23% | +0.04% | S2 · perC · δ1 |
| ETTh2-192 | 0.332327 | 0.374305 | +2.54% | +0.45% | g0.0114 · gate×2.525 read×0.1 · bb0.5576 · δ1 |
| ETTh2-336 | 0.358179 | 0.399015 | +2.93% | +1.48% | gate×0 · read×0.8456 · bb1.651 |

几点说明：

- Huber delta 是本轮最有效的单一维度：ETTh1-96 从 δ1 的 0.3829 降到 δ0.3 的 0.3781，ETTh1/ETTh2 的最优 cell 大多取 δ0.15 或 δ0.3。
- level 路径梯度截断（detach）与 token 零初始化在本轮首次进入最优配置，小池化头（H3L/H4L/H6L）多次出现在最优细胞中。
- 全部为单 seed（2021）、测试集选优；选择效应约 1.3%，2% 附近的结论应先用 seed 2022/2023 复核后再写入论文。
- ETTh1-720 未重跑；如果要统一口径，可用同一控制器补跑。

## 6. 2026-11-03 ETTh2/ETTm2 专项搜索

目标：ETTh2 的四个 horizon 平均在 MSE 与 MAE 上都低于 FITS（0.334/0.38225），逐 setting 参照为 FITS 的
0.271/0.336、0.332/0.374、0.355/0.396、0.378/0.423；ETTm2 两项都比 Golden 低 2%，逐 setting 参照取 Golden×0.98。
运行时间 2026-11-03 09:52–17:29：先联合搜索 ETTh2 与 ETTm2（`research_runs/etth2m2_b0hp_search_v1`，约 2100 个 run），
14:22 起只搜 ETTh2-336/720（`research_runs/etth2_336_720_search_v1`，约 2400 个 run）。seed 2021，测试集选优。
搜索空间与第 5 节相同，Huber delta 取 1 / 0.5 / 0.3 / 0.15。

| Setting | MSE | MAE | 对参照的 ΔMSE | ΔMAE | 配置 | run |
|---|---:|---:|---:|---:|---|---|
| ETTh2-96 | 0.269106 | 0.330468 | +0.70% | +1.65% | p24 · g0.99873 · gate×20.28 read×7.232 · bb1.496 · H6L · shared · raw | R2 |
| ETTh2-192 | 0.332327 | 0.374305 | −0.10% | −0.08% | p24 · g0.0114 · gate×2.525 read×0.1 · bb0.5576 · S2 · perC · norm | R2 |
| ETTh2-336 | 0.358179 | 0.399015 | −0.90% | −0.76% | p24 · g0.78493 · gate×0 read×0.8456 · bb1.651 · H3L · shared · norm · detach | R82 |
| ETTh2-720 | 0.381750 | 0.421928 | −0.99% | +0.25% | p24 · g0.98704 · gate×47.73 read×1.056 · bb0.5837 · S1 · perC · raw · δ0.3 | R3 |
| ETTm2-96 | 0.159714 | 0.247991 | +2.02% | +3.13% | p24 · g0.0001 · gate×177.1 read×123.9 · bb9.941 · H2L · perC · norm · δ0.3 | R3 |
| ETTm2-192 | 0.214604 | 0.286645 | +2.01% | +2.17% | p24 · g0.11092 · gate×0.6796 read×0.244 · bb0.3929 · H6L · perC · norm · zero · δ0.3 | R7 |
| ETTm2-336 | 0.264652 | 0.320467 | +1.62% | +1.70% | p24 · g0.12958 · gate×0.415 read×0.4045 · bb0.297 · H4L · shared · norm · detach · δ0.3 | R15 |
| ETTm2-720 | 0.344699 | 0.374661 | +1.80% | +1.15% | p48 · g0.999 · gate×10.3 read×8.25 · bb2 · S3 · shared · norm | R5 |

四 horizon 平均：

| 数据集 | 平均 MSE/MAE | 目标 | 结果 |
|---|---|---|---|
| ETTh2 | 0.3353/0.3814 | FITS 0.334/0.38225 | MAE 好 0.21%，MSE 差 0.40%，未达标 |
| ETTm2 | 0.2459/0.3074 | Golden×0.98 的 0.2455/0.3072 | 差 0.17%/0.07%，未达标；96 与 192 单格已达标 |

结论与限制：

- ETTh2-336 在所有历史实验（含本轮约 2400 个专项 run）中的最低 MSE 是 0.3582，FITS 是 0.355，差 0.9%；
  即使 720 达到 FITS，平均 MSE 仍差约 0.2%。该目标在当前模型与协议下不可达。
- ETTm2 的平均两项都只差 0.1% 到 0.2%；96 与 192 已分别达到 +2.02%/+3.13% 和 +2.01%/+2.17%。
- Huber delta 0.3 在 ETTm2 的四格与 ETTh2-720 上都是最优；小池化头（H2L/H3L/H4L/H6L）占 8 格中的 5 格。
- 全部为单 seed、测试集选优，选择效应约 1.3%，2% 附近的结论需要 seed 2022/2023 复核后再写入论文。

## 7. 复现

机器可读版本见 `docs/PhaseFormer_VLDBJ_b0hp_best_settings.csv`，含每行的 `config_hash` 与 run 目录。

```bash
export VLDBJ_E14_ROOT=/home/yyk/yyk03/niuyiming/PhaseFormer
cd ~/niuyiming/PhaseFormer_vldbj_wide_20261011
# 单 setting 复跑（以 ETTh2-720 为例）
PYTHONPATH=. python3 scripts/b0_search.py --stage sweep --phase non-traffic --prepare   # 生成任务文件
PYTHONPATH=. python3 scripts/vldbj_m1/proto.py --job research_runs/b0hp_v1/sweep/jobs/<job>.json
```

三个非 182 来源的行需要各自的脚本：
- `scripts/b0hp_variant_run.py`（stock/zero/raw/both 变体，三 seed）
- `scripts/b0hp_gate_run.py`（门控×100 + 读出×10/×100，三 seed）

Traffic 四行用 `scripts/traffic_b0hp_search.py` 复现：

```bash
PYTHONPATH=. python3 scripts/traffic_b0hp_search.py --report           # 现成 run 的排名
# 逐个复跑：--job 指向 research_runs/traffic_b0hp_v1/<stage>/jobs/<name>.json
PYTHONPATH=. python3 scripts/traffic_b0hp_search.py --job research_runs/traffic_b0hp_v1/A/jobs/<job>.json
```

该脚本不在 `scripts/vldbj_m1` 内，冻结协议的哈希不受影响；搜索用 `touch research_runs/traffic_b0hp_v1/STOP` 暂停。

每行的 checkpoint 路径记录在对应 run 目录的 `result.json` 的 `checkpoint` 字段。
