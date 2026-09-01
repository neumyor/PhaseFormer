# PhaseFormer 32 任务机制与超参数搜索计划

> 当前实验：保持 PCTF Full Repair 三分支结构不变，测试从随机初始化开始的一次性联合训练，
> 以消除 A2 预训练+微调的额外训练阶段。单阶段初筛没有通过 A2 替换门槛后，已定位并修复
> composer 经 A2 特征回传的隐藏梯度路径；50 个 strict PCTF H192 validation-only 策略已完成，
> 最佳 T28 将 correction/deformation/global-level 边界更新为 `0.60/0.24/0.12`，但联合比值仍为
> 1.0019，未达到相对 two-stage Full Repair 改善 0.5% 的门槛。完整表见
> `docs/PhaseFormer_pctf_single_stage_h192_tuning.md`；未读取新的 test。

> 最近完成正式实验：冻结的 `pctf_anchor_repair_full` 在 ETTh2/ETTm2 的 L720→H96/H192 上与
> A2 完成 full-train、三 seed、best-validation checkpoint 的 24-run test 配对。候选相对 A2
> 宏平均降低 0.772% MSE、0.507% MAE，3/4 setting 双指标改善，最坏单指标回退 0.203%，通过
> 预注册的两数据集局部替换门槛；严格稳定低于 Golden 为 4/4，A2 为 2/4。候选包含额外微调，
> 且其结构修正相对内部微调 A2 仅贡献约 0.174% MSE、0.278% MAE，尚缺 continued-A2 等预算
> 对照，不能把全部收益归因于 ICPT。结果与 test 暴露边界见
> `docs/PhaseFormer_pctf_anchor_formal_etts.md`。

> 已完成实验：PCTF v3 锚点漂移因果归因与修复，见
> `docs/PhaseFormer_pctf_anchor_attribution_plan.md`。上一轮 v2 最佳 `pctf_anchor_mlp` 仍相对
> A2 宏平均退化 0.113%，但代码审计无法区分锚点联合训练漂移、ICPT 辅助目标错配、gate 缺少
> 边际收益监督和 `H=period` 的 level 零空间。v3 用同 setting/seed 的 A2 checkpoint 做严格
> 配对，依次测试 frozen-anchor、残差监督、锚点 0.1× LR + anchor loss、边际系数监督和单周期
> level 修复。计划 12 个 matched A2 + 72 个候选，全部 validation-only；本轮只写代码和计划，
> 没有启动训练或读取 test。

> 已完成实验：A2 锚定式 PCTF v2，计划见
> `docs/PhaseFormer_pctf_anchor_fusion_retest.md`。v1 在六数据集 H96 的66-run validation-only
> 筛选中没有候选晋级；其主要问题是删除 NLinear 周期内形状且候选空间不能还原 A2，同时
> shape 证据与实际被替换的参考分支错配。v2 完整保留 A2（RCRF+LFF-NLinear）为端到端可训练
> 锚点，只加入有界、零初始化的 ICPT 周期间水平/周期内形状创新；修正为匹配参考对象、逐未来
> 周期的因果证据，并解耦 phase period 与 ICPT cycle period。48-run period选择与132-run
> H96/H192 strategy筛选均已完成；最佳 MLP evidence 宏平均/A2=1.001131、4/12 双指标改善、
> 最差比=1.021385，未通过门槛，故没有启动144-run正式确认或读取 test。结果见
> `docs/PhaseFormer_pctf_anchor_fusion_results.md`。

> 已结束实验：PCTF 多融合策略 v1，方案与失败结果见
> `docs/PhaseFormer_pctf_fusion_strategies.md`。F0 与 A2 宏平均近似持平但仅1/6数据集双指标改善，
> F1/F2/F3 宏平均退化约2.5%–3.3%，未启动正式 test；不再沿原组件替换公式继续搜索 gate。

> PCTF 基础实现（尚未单独训练）：相位—周期—轨迹统一模型，完整方案见
> `docs/PhaseFormer_pctf_experiment.md`。它保留 A1 的 PhaseFormer、NLinear 与外层 RCRF，
> 将 I0/ICPT 限制为两个可识别修正：逐周期零均值形状，以及全 horizon 均值守恒的周期间
> 相对水平；历史 masked reconstruction 只做连续收缩，不做完整专家路由。代码与 48-run
> validation-only 筛选、冻结后 144-run test 确认协议已写完；该基础方案现作为上方多融合
> 策略实验的 F0，不再单独启动一套重复矩阵。

> 最近完成实验：Multi-Anchor Selector v1，方案与结果见
> `docs/PhaseFormer_multi_anchor_selector_experiment.md`。M3 soft 路由在六数据集 H96 相对
> A1/I0/R0 逐指标包络平均改善 0.79%，但它依赖三个独立训练并冻结的完整模型。根据论文方法
> 约束，M3 自 2026-08-29 起只作为诊断性 ensemble 上界和互补性证据，**不再是候选论文主
> 方法，也不继续沿 OOF/stacking 路线优化**。下一阶段必须把三种设计思想整合进一个共享
> PhaseFormer，而不是融合三个模型的最终预测。
>
> 最近完成的统一模型实验：HPTC 单模型正交趋势—周期修正，计划与结果见
> `docs/PhaseFormer_hptc_unified_experiment.md`。H4 在六数据集 H96 相对 A1 的 12 指标宏平均
> 改善 0.29%，但仅 3/6 数据集双指标改善，未通过预注册 gate，因此未运行 H192/test；正式
> 三 seed test incumbent 仍是 A2（RCRF+NLinear+LFF）。下一轮证据锚点是：保留轨迹/形状
> 可识别分工，但放松严格零均值
> 为受约束周期级低频残差，并用模块自身的历史重建不确定性替换线性外推 risk。

## 目标与选择规则

### 不可违反的论文架构约束

- “整合三个方法”是整合其机制与归纳偏置，不是把 A1、I0、R0 三个完整 checkpoint 做
  routing、stacking、averaging 或 mixture-of-models。
- 正式候选必须是一个端到端训练的统一模型：共享一套 PhaseFormer 相位表示，在同一网络中
  有机结合全 horizon 轨迹校正、周期间关系建模和历史自验证/可靠度调节。
- 允许轻量模块或结构化分支，但不得重复三套完整 PhaseFormer 主干；推理不能要求加载三个
  独立模型。参数量、FLOPs 和延迟必须与最强单模型一起报告。
- 机制关系必须能由统一目标解释：相位主干给出周期内坐标，轨迹模块校正跨周期水平演化，
  周期模块建模周期间形状变化，历史可靠度只负责调节这些共享表征中的校正强度。
- 必须用逐组件消融验证每一机制的独立作用及组合增益；如果完整模型的收益只能由多 checkpoint
  误差抵消解释，则该方案不能作为论文主方法。
- M3 可继续用于估计互补性上界、选择困难度和蒸馏教师，但其指标不得包装成统一模型创新。

- 覆盖 8 个数据集：ETTh1、ETTh2、ETTm1、ETTm2、Exchange、Weather、Electricity、Traffic。
- 每个数据集覆盖 horizon 96、192、336、720，共 32 个任务，lookback 固定为 720。
- ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic 的最终提升声明统一以 `docs/PhaseFormer_gold_standard.md` 为固定参照；截图未包含 Exchange，因此 Exchange 在获得权威金标准前只能报告 matched rerun 配对结果。
- 同一数据集的四个 horizon 共用一个机制族和 period 设置；各 horizon 仅允许调整学习率、loss、容量、训练时长等训练/规模参数。
- 默认优先使用验证集完成候选排序和早停，但允许按用户要求利用测试集调整模型、参数或机制。凡读取测试反馈并继续选择或修改的实验，都必须完整保留搜索轨迹，在配置和报告中标记为 test-set selection；相关结果不得称为盲测或无偏泛化估计。
- 每个任务先保留验证集 Pareto 前沿，再计算：
  `score = 0.5 × MAE相对改善率 + 0.5 × MSE相对改善率`。
- 候选任一指标相对基线回退超过 0.5% 即淘汰；分数差小于 0.2% 时优先参数更少、训练更快的方案。
- 历史实验和后续获准的搜索都可能查看官方测试集；最终报告必须披露测试集暴露范围以及哪些配置由测试结果选出，不能称为完全盲测。

## 实验流程

### 1. 建立搜索基础设施与新基线

- 新增配置驱动的搜索 runner，输入 dataset、horizon、机制、参数、seed、预算阶段，自动生成唯一实验 ID，并支持断点续跑。
- 增加独立验证评估器，加载最低 `val_loss` checkpoint 后计算 `val_mae`、`val_mse`；搜索候选不调用 `trainer.test()`。
- bad case 改从验证集导出，限制 8 个，包含样本索引、变量、时间戳、预测/真实值路径、错误模式和下一步动作。
- 用当前修复后的真实 Huber 和 best-checkpoint 协议重新训练 original baseline，作为协议诊断和候选搜索所需的 matched rerun；它不得替换 `docs/PhaseFormer_gold_standard.md`。最终报告同时给出相对金标准的结果，以及在协议不完全一致时的 matched rerun 配对结果。
- 每个 `experiment_id` 严格只保存 `run.yaml`、`results.csv`、`sample_errors.csv`、`selected_cases.npz`、`objective_error_analysis.md`、`objective_error_analysis.zip` 和 `figures/`，不生成 PDF。ZIP 只打包 Markdown 与其实际引用的图片，解压后可直接浏览。配置、环境、验证指标、耗时、显存峰值与 bad case 元数据写入这组汇总文件，不单独保留命令、checkpoint、日志、全量预测或其他中间产物；同一次运行的多个 setting 通过统一的 `setting` 字段管理，不拆分文件或目录。

### 2. 每数据集选择共享机制

先用训练集时序诊断确定候选 period：

- ETTh1/ETTh2：12、24、48。
- ETTm1/ETTm2：24、48、96。
- Weather/Electricity/Traffic：12、24、48。
- Exchange：7、14、30。
- 在 horizon 96 和 720 上以 30% 训练数据、最多 8 epoch 筛选 period；按两个 horizon 的验证综合分选择一个数据集共享 period。

在选定 period 上筛选以下机制配置：

1. 原始 phase-only。
2. 固定弱周期 residual，gate 0.5。
3. 固定弱周期 residual，gate 0.9。
4. adaptive residual，gate init 0.2。
5. adaptive residual，gate init 0.5。
6. phase uncertainty，reliability floor 0.35。
7. phase uncertainty，reliability floor 0.60。
8. phase uncertainty + period-level calibration。
9. phase uncertainty + level calibration + high-frequency damping。
10. phase uncertainty + level calibration + high-frequency damping + sparse-event calibration。
11. low-pass residual，沿用已有 window 25。
12. phase-local trend，作为历史机制复核。

筛选顺序：

- 第一轮：horizon 96 和 720，30% 数据、8 epoch。
- 每数据集保留 aggregate score 最高且两端 horizon 均无明显回退的 3 个机制。
- 第二轮：这 3 个机制在 horizon 192 和 336 上使用同样预算补齐证据。
- 按四个 horizon 的平均分排序，同时要求最差 horizon 回退不超过 0.5%，最终冻结一个数据集共享机制。
- Electricity/Traffic 的低成本结果历史上存在迁移失真，因此其前 3 名额外进行全数据 8 epoch 复核后再冻结。

### 3. 每任务搜索训练与轻量容量参数

冻结数据集机制后，对每个 dataset×horizon 运行固定的 12 组组合：

- loss：Huber、MAE。
- learning rate：当前 base LR 的 0.3×、1×、3×。
- capacity：
  - base：当前 preset 容量；
  - compact：latent、phase encoder hidden、predictor hidden 减半，同时保持 latent 可被 attention heads 整除，最小 latent 不低于 heads。

采用 successive halving：

- 12 个候选：30% 数据、最多 8 epoch。
- 验证 Pareto 前 4：100% 数据、最多 15 epoch。
- 前 2：100% 数据、正式训练预算；默认 30 epoch，ETTh1-720 保持 70 epoch，沿用对应 patience。
- 若前两名综合分差小于 0.2%，保留更小或更快的方案。
- 不搜索 batch size 来制造指标优势；ETT 使用现有 256，Exchange 32，Weather/Electricity 64，Traffic 8。batch 变化只允许用于 OOM 恢复，并要求 baseline/candidate 成对一致。

## 最终确认与报告

- 每个任务冻结冠军后，使用 seeds 2021、2022、2023 对 original baseline 和冠军各运行一次完整训练与测试，即每任务 6 个确认实验。
- 报告三个 seed 的 MAE/MSE 均值、标准差、相对改善、训练时间、峰值显存和参数量。
- 最终方案必须满足：
  - 平均 MAE 和 MSE 均不劣于 baseline；
  - 任一指标平均回退不得超过 0.5%；
  - 若优势小于跨 seed 标准差，则标记为“无显著稳定收益”，默认保留更简单方案。
- 输出：
  - 32 任务最佳配置表；
  - 每数据集共享机制说明；
  - 每任务 Pareto 前沿与搜索轨迹；
  - 三 seed 最终测试表；
  - 不超过 8 个验证 bad case 的归因摘要；
  - 可直接写入 `phaseformer_presets.py` 的候选配置。
- 只有在 32 个任务全部完成最终确认后才更新正式 `latest` preset。若使用测试结果回头修改搜索空间，必须保留修改前后的全部候选、选择依据和测试结果，并在最终报告显著披露 test-set selection。

## 调度、成本与停止条件

- 单张 RTX 4090 顺序运行，支持按 experiment ID 断点续跑。
- 预估规模：
  - 32 个新基线；
  - 约 250–300 个低成本机制筛选；
  - 384 个第一层超参数候选；
  - 约 128 个中等预算候选；
  - 约 64 个正式 finalist；
  - 192 个三 seed 最终确认运行。
- ETT/Exchange 可在显存较少时运行；Weather/Electricity 需建议至少 12 GiB 空闲；Traffic 需等待 GPU 空闲达到约 20 GiB，避免与其他进程竞争。
- 某任务满足以下任一条件即停止继续扩展：
  - 已有候选通过三 seed 稳定改善确认；
  - 12 个超参数候选和两个正式 finalist 均未进入有效 Pareto 前沿；
  - 连续两级增加预算后排名反转且无稳定优势；
  - OOM/外部 GPU 占用连续阻塞时保留队列，不降低公平性协议。
- 所有代码、搜索配置、实验记录和 preset 更新按 `MANAGE_RULES.md` 分粒度提交；失败、OOM、超时和被淘汰候选同样写入追加式迭代日志。

## 当前局部实验：PCTF 单阶段训练（2026-08-31 至今）

- 固定 `pctf_anchor_repair_full` 结构，仅研究如何从随机初始化用一次 `Trainer.fit` 训练三分支，
  避免 A2 预训练和第二阶段联合微调。
- 第一轮在 ETTh2/ETTm2、L720→H96/H192、seeds 2021/2022 上完成 8 个 matched A2 和
  48 个 validation-only 候选；六种 LR/anchor loss/warm-up 策略均未通过统一门槛。
- 证据显示统一 LR 下 fused loss 会损害内部 A2，而 correction warm-up 常选择尚未完整开启
  修正的 checkpoint。据此只追加 8 个 `decoupled_protected` 验证任务：前向联合，反向将 fused
  loss 与 A2 解耦，A2 仅由 matched anchor loss 训练；不新增 checkpoint 或训练阶段。为避免
  跨 commit 比较，在独立目录同步重跑 8 个 matched A2，共 16 个任务。
- 追加候选继续沿用原门槛：综合比值 `<0.998`、至少 6/8 双指标改善、最坏比值 `≤1.01`、
  correction scale 为 1。若失败，停止且不读取 test；若通过，才执行四 setting×三 seed 的
  matched A2/candidate 正式 test。完整协议与结果见
  `docs/PhaseFormer_pctf_single_stage_training.md`。
- 复测已完成：`decoupled_protected` 综合比值 0.99908、3/8 双改善、最坏比值 1.00537，未过
  门槛；正式 test 阶段取消。单阶段候选训练时间约为 A2 的 1.90 倍，低于两阶段的
  2.77–3.45 倍，但证据不足以替换两阶段 Full Repair。
- 后续 H192 修订：原 `decoupled_protected` 仅把 fused 输出梯度从 A2 主输出中抵消；composer
  仍可能经 anchor/phase/trajectory 特征回传。strict 版本 detach 全部 A2-derived composer 输入，
  并加入独立 composer LR。预注册 50 个策略（200 个 full-train validation-only jobs），直接相对
  frozen two-stage Full Repair 的逐 seed validation 指标，目标联合宏平均至少改善 0.5%、最坏比值
  不高于 1.005。50 策略完整矩阵已完成：T28（`trust_060`）为最佳，联合比值 1.0019、最差 1.0078，
  仍不达标，未读取新 test。默认 `pctf_anchor_repair_full` 的 trust-region 参数更新为
  `correction/deformation/global-level = 0.60/0.24/0.12`；入口和完整表为
  `docs/PhaseFormer_pctf_single_stage_h192_tuning.md`。
