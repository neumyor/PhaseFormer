# PhaseFormer-LA 实验 1/2 计划：会议训练协议复现 + 从 PhaseFormer 热启动

状态：**计划，未启动**（2026-10-11）。用户批准实验 1 和实验 2，并要求“实验开始前先确认”；
本文件中的任何 run 均须在用户确认后才启动。

## 0. 目的与约束

- 首要目标（用户 2026-10-11）：证明 PhaseFormer-LA 强于会议版 PhaseFormer（Golden，
  `docs/PhaseFormer_gold_standard.md`）。PhaseFormer-L 只服务于 LA-vs-L 消融，可以弱。
  最重要的是主表与消融表。
- 报告口径：论文写 “averaged over three seeds”，因此所有臂一律报告 seeds 2021–2023 的
  均值 ± 标准差，三个 seed 全部计入；不做 best-of-N seed 报告（见 memory
  `test-set-selection-disclosure` 的 Limit 条款）。
- 允许按 test 选择协议与配置（用户决定 2026-10-10），但本文件和结果文档必须标注为
  **test-set selection**，保留完整搜索轨迹；论文只写 “small grid search”，不得写 blind/unbiased。
- Golden 不被任何重跑替换；重跑的 PhaseFormer 一律标为 matched rerun。

## 1. Review 结论（为什么做这两个实验）

### 1.1 现有差距

E14 协议重训的 PhaseFormer（论文消融表中的 “retrained PhaseFormer”）在全部 7 个数据集上
都比 Golden 差（数据集平均 MSE：ETTh1 .408/.403，ETTh2 .354/.346，ETTm1 .352/.346，
ETTm2 .258/.250，Weather .227/.223，Electricity .161/.160，Traffic .392/.386）。LA 在同协议
下相对重训 PhaseFormer 有提升，但相对 Golden 只在 Weather 上胜出（.220 vs .223）。
即 LA 的劣势主要来自骨干训练协议，而不是扩展组件。

### 1.2 会议代码与 E14 协议的实际差异（本次 review 新发现）

会议代码是仓库早期的逐数据集脚本 `run_{dataset}.py`（commit `d9cb0b3`，2025-10-05；
ETTh1-H720 配置来自 `ff14c16`，2026-03-25）。逐项比对 `get_best_config_for_horizon`
与 E14 源配置（服务器 `research_runs/phaseformer_L_e14_main_v1/`，seed 2021）：

| 项目 | 会议代码 | E14 协议（现 LA/L/重训 PF 全部使用） |
|---|---|---|
| 结构（layers/latent/hidden/routers/heads） | 28 setting 全部相同 | 相同 |
| 数据管线、划分、RevIN、test loader（drop_last=False） | 相同 | 相同 |
| Huber 损失 | 自写 Huber = `nn.HuberLoss/δ`（δ=1 时完全相同） | `nn.HuberLoss` |
| 优化器/学习率调度 | Adam，`type3` = 常数学习率 | 相同 |
| max epochs / patience | 30 / 8（ETTh1-720：70 / 14） | 30 / 8（ETTh1-720：30 / 14） |
| **batch size** | ETTh1/ETTh2 256；**ETTm1/ETTm2/Weather/Electricity/Traffic 16** | ETTh 256；ETTm1/ETTm2 **256**；Weather/Electricity **64**；Traffic **8** |
| **学习率** | Electricity-96 0.002；Weather-96、ETTm2-96 0.001 | Electricity-96 0.001；Weather-96、ETTm2-96 0.0003 |
| **测试所用权重** | 早停结束时的**最后一个 epoch** 权重（`trainer.test(model)`，无 ckpt） | best-validation checkpoint |
| 每 epoch 测试 | `EpochTestCallback` 每个 epoch 记录 test MSE/MAE | 无 |
| seed | 脚本固定 2021（单 seed） | 2021–2023 |

结论：ETTm1/ETTm2 的 batch 相差 16 倍、Weather/Electricity 相差 4 倍，是解释“重训 PF 比
Golden 差”的首要候选，也是 LA 至今没有在会议条件下训练过的原因。另外两点需要如实记录：
(a) 会议代码单 seed，而论文正文写 PhaseFormer “average results over three random seeds”；
(b) 会议代码每个 epoch 都读 test，Golden 是否是逐 epoch 中的最好值无法从代码判定。
Stage 0 用诊断方式回答 (b)，但诊断值**不进入任何报告数值**。

## 2. 实验 1：会议训练协议下的 PF / L / LA

### Stage 0：协议复现诊断（PF only，seed 2021，28 run）

- 臂：`pf`，会议协议（上表左列：batch、学习率、ETTh1-720 epochs 全部取自会议脚本）。
- 同一次训练记录三个 test 数：best-val checkpoint、last-epoch 权重（会议口径）、
  逐 epoch test 轨迹（**仅诊断**：判断 Golden 是否落在轨迹范围内，不用于选择或报告）。
- 对照：E14 协议 PF seed 2021（已有，不重跑）。
- 采纳规则（预注册；以 test 对比 Golden，属于 **test-set selection**）：
  1. 若会议协议 PF 的 28 setting 平均相对 MSE 差距（相对 Golden）≤ 1.0%，且在 ≥ 5/7 个
     数据集上数据集平均 MSE 优于 E14 PF seed 2021 → 采纳会议协议，进入 Stage 1。
  2. checkpoint 规则（best-val 或 last-epoch）在 Stage 0 一次性冻结，选 28 setting 平均更
     接近 Golden 者；两者差 < 0.2% 时取 best-val。冻结后对 PF/L/LA 三臂完全相同。
     两种规则都符合论文“early stopping on the validation loss (patience 8)”的表述。
  3. 若不满足 1 → 会议协议在当前环境不可复现：停止实验 1，结果如实写入结果文档；
     实验 2 改为在 E14 协议的 PF checkpoint 上进行（见 §3）。
- 每个数据集按 batch 调整后的单 run 耗时也在本阶段实测，用于修正 Stage 1 排期。

### Stage 1：三臂、三 seed（224 个新 run）

- setting：28；seeds：2021/2022/2023。
- 臂（均在冻结的会议协议下训练，同一 runner、同一份冻结代码）：
  - `pf`：PhaseFormer（seed 2021 复用 Stage 0，新跑 56）；
  - `b0` = PhaseFormer-L（84）；
  - `m1` = PhaseFormer-LA（84）。
- L/LA 的扩展超参：沿用当前论文的最终配置（主配置 + 2026-10-11 confirm round 的 8 个
  per-setting 切换），不重新调参；`b0` 与 `m1` 共享同 seed 的骨干随机状态（现有
  `paired_models` 机制）。
- 骨干学习率取会议协议值；扩展学习率乘子沿用最终配置。

### 实验 1 预注册目标（目标，不是结论；无论成败全部报告）

- PF matched rerun（3-seed 均值）与 Golden 的差距写入结果文档。
- LA（3-seed 均值）在 ≥ 4/7 个数据集上数据集平均 MSE 优于 Golden；
- LA 相对同协议 PF 在 ≥ 17/28 个 setting 上 MSE 改善（当前 E14 协议下为 17/28）。

## 3. 实验 2：从训练好的 PhaseFormer 热启动 LA（及 L）

依据：LA 精确嵌套 PhaseFormer —— `V2=0` 且 `g_ℓ=g_s=0` 时 LA 的输出等于 PhaseFormer。
因此可从同 seed 的 PF checkpoint 出发，令扩展初始近似关闭，再在验证集早停下训练，
使 LA 的起点就是 PF 的解。

- 来源 checkpoint：实验 1 采纳会议协议时，用 Stage 0/1 的 `pf` checkpoint（同 seed、
  同协议、同 checkpoint 规则）；否则用 E14 协议 PF checkpoint（启动前核实服务器上
  checkpoint 完整，sha256 入记录）。
- 初始化：骨干参数全部从 PF checkpoint 严格加载（缺失/多余的非扩展键即报错）；扩展参数用
  同 seed 随机状态初始化；`zero_init_exit=True`（`V2=0`，即 q=0）；level/shape 门
  `gate_init=1e-3`；`shape_bias_delta=0`。启动断言：初始 val loss 与 PF best val loss 的
  相对差 ≤ 0.5%。
- 训练：同协议（Huber、≤30 epoch、patience 8、同 batch），扩展学习率乘子沿用最终配置；
  骨干学习率乘子 `backbone_lr_mult ∈ {0.1, 1.0}`。
- Stage 2a（LA，seed 2021，28 × 2 = 56 run）：按 seed-2021 test MSE 为每个 setting 选
  `backbone_lr_mult`（**test-set selection**，标注）。
- Stage 2b（seeds 2022/2023，LA-warm 一臂，56 个新 run；seed 2021 复用 Stage 2a 的选中
  run）。用户决定（2026-10-11）不跑 L-warm。后果：热启动 LA 只能进入主表；消融表
  （PF→L→LA）必须继续使用同一协议下从零训练的 L 与 LA，否则 L→LA 一步混入热启动效应。
  若主表采用热启动 LA，论文需说明主表 LA 与消融表 LA 的初始化不同。
- 论文披露：采纳实验 2 时，`experiments.tex` 中 “trained from scratch” 必须改为
  “initialized from the trained PhaseFormer checkpoint of the same seed”，并说明总训练预算
  （PF 训练 + 微调 epoch 数）；该文字改动用 `\blue{}` 标出。

### 实验 2 预注册目标

- LA-warm 相对同 seed、同协议 PF 的 3-seed 均值 MSE 在 ≥ 22/28 个 setting 上改善；
- LA-warm 在 ≥ 4/7 个数据集上数据集平均 MSE 优于 Golden；

## 4. Bad case 分析（HOW_TO_DO_RESEARCH 要求）

两个实验完成后，对 LA 相对同协议 PF 退化最大的 5 个 setting，以及相对 Golden 差距最大的
3 个数据集，读取 seed-2021 checkpoint 的逐样本误差：按窗口 MSE 排序取 top-1% 窗口，统计
其输入周期均值漂移幅度、level 门值与 q 的范数，判断退化来自 level 分支误触发、modulation
放大还是骨干本身。分析记录写入结果文档。本轮是训练协议与初始化实验，不是新模型设想，
因此不触发 `/experiment-and-error-analysis` Skill，但保留同等的样本级证据。

## 5. 代码改动（已实现，2026-10-11；单元测试 `tests/test_vldbj_proto.py` 14 项 + 原有 10 项通过）

现有 `scripts/vldbj_m1/model.py` 与 `run.py` 不改动（保证已完成的 M1 结果可按原代码复现），
新增两个模块：

1. `scripts/vldbj_m1/protocol.py`：
   - `CONFERENCE`：28 setting 的会议**有效**配置（结构、batch、学习率、epochs、patience、
     Huber δ、attention dropout/window、位置编码）。review 补充发现：会议脚本的
     `PhaseFormerConfig` 硬编码 `phase_attn_dropout=0.1`、`phase_attn_window=None`、
     `phase_use_pos_embed=True`（ff14c16 的 ETTh1 脚本从字典读取 dropout 与位置编码），
     因此 Traffic-96 字典中的 `window=12, pos_embed=False` 实际未生效；E14 使用了
     `window=12, pos_embed=False`，会议协议改回有效值。单元测试从 git 中的会议脚本
     重新推导该表并逐项比对，同时校验脚本 blob 哈希。
   - `conference_source(e14_source)`：结构字段必须与会议一致（否则报错），只覆盖训练字段与
     上述有效字段，记录 `protocol_changes`，重算 `config_hash`。seed 仍为 2021–2023。
   - `FINAL_HEAD/FINAL_SHARED_GATE/FINAL_VARIANTS`：论文最终配置（pilot 选择 + confirm round
     的 8 个切换），已与服务器上的 `selection.json`/`tune_selection.json` 核对一致。
   - `golden()`：从金标准文档解析 28 个 Golden。
2. `scripts/vldbj_m1/proto.py`（控制器，阶段 `stage0/decide/main/warm-probe/warm-select/
   warm-main/collect/all`）：
   - `pf` 臂：PhaseFormer 采用同 seed L/LA 的骨干随机抽样（扩展模块先构造，直接构造的
     PhaseFormer 抽样会不同），使 PF→L→LA 三步共享同一初始化。
   - 每个 run 同一次训练记录 best-val 与 last-epoch 两组 val/test（LA 另含 modulation-off），
     保存 `last.pt` 与逐窗口 test MSE；Stage 0 另记录逐 epoch test 轨迹（仅诊断，测试前后
     保存/恢复全部 RNG 状态，训练不受影响）。
   - 热启动：骨干从 PF checkpoint 严格加载（缺失键只允许是扩展参数，sha256 校验），
     `gate_init=1e-3`、调制出口零初始化，保留最终配置的 width 与扩展学习率乘子；启动断言
     初始 val MSE 与 PF val MSE 相对差 ≤ 0.5%；骨干/扩展学习率分组。
   - `decide`、`warm-select` 按 §2/§3 的预注册规则写不可变 JSON（记录全部候选数值与
     test-set selection 标注）；`collect` 输出 `results.csv` 与 `summary.json`。
   - `all`：stage0 → decide → main 与 warm-probe 同队列 → warm-select → warm-main → collect；
     未采纳会议协议时跳过 main，热启动改用 E14 checkpoint（`attempts/*/checkpoints/best.ckpt`）。

## 6. 服务器执行（遵守 REMOTE_SERVER.md）

- 新建冻结 clone `~/niuyiming/PhaseFormer_vldbj_proto_20261011`（bundle 上行，
  `reset --hard FETCH_HEAD`），运行期间不同步代码；`VLDBJ_E14_ROOT=~/niuyiming/PhaseFormer`；
  输出写 `research_runs/vldbj_proto_v1/`。启动命令（获批后）：
  `python scripts/vldbj_m1/proto.py --stage all --scratch-dir research_runs/vldbj_proto_v1 --slots-per-gpu 3`。
- 服务器当前 8 张 A800 空闲。单 trainer GPU 利用率低，每卡并发 3 个 job（24 并发）。
- 规模与耗时估计（以 E14 实测 epoch 时间按 batch 换算，Stage 0 后修正）：

| 阶段 | 新 run 数 | 估计 GPU 时 | 估计墙钟 |
|---|---:|---:|---:|
| Stage 0（PF，seed 2021） | 28 | ~20 | ~2–3 h（Electricity/ETTm1 最长） |
| Stage 1（PF/L/LA × 3 seed） | 224 | ~160 | ~7–9 h |
| Stage 2a（LA-warm 探测） | 56 | ~35 | ~2–3 h |
| Stage 2b（LA-warm，seeds 2022/2023） | 56 | ~35 | ~2–3 h |
| 合计 | 364 | ~250 | ~13–17 h |

  Stage 2a 只依赖 Stage 0 的 seed-2021 PF checkpoint，可与 Stage 1 并行。

## 7. 产出与记录

- 结果文档 `docs/PhaseFormer_VLDBJ_LA_protocol_warmstart_results.md`：每个 stage 的全部
  run（含未被选中的）、每 setting 3-seed 均值 ± std、相对 Golden 和同协议 PF 的变化、
  checkpoint 规则与 `backbone_lr_mult` 的选择轨迹（标注 test-set selection）、bad case 分析。
- `docs/agent-log.md` 追加每个阶段的启动、完成和决策。
- 论文：只有在用户看过结果并确认采纳后才改主表/消融表；PhaseFormer 主表行仍为 Golden。

## 8. 用户决定与待确认事项（2026-10-11）

1. Stage 0 启动：**暂不批准**，计划需先修订；修订后再次确认才可启动。
2. 阶段推进：Stage 0 一旦获批并满足 §2 采纳规则，按本文件预注册规则自动进入 Stage 1 与
   实验 2，各阶段完成后汇报。
3. 实验 2 只跑 LA-warm，不跑 L-warm（见 §3 Stage 2b 的后果说明）。
