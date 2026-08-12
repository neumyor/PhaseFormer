# PhaseFormer 32 任务机制与超参数搜索计划

## 目标与选择规则

- 覆盖 8 个数据集：ETTh1、ETTh2、ETTm1、ETTm2、Exchange、Weather、Electricity、Traffic。
- 每个数据集覆盖 horizon 96、192、336、720，共 32 个任务，lookback 固定为 720。
- ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic 的最终提升声明统一以 `docs/PHASEFORMER_GOLD_STANDARD.md` 为固定参照；截图未包含 Exchange，因此 Exchange 在获得权威金标准前只能报告 matched rerun 配对结果。
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
- 用当前修复后的真实 Huber 和 best-checkpoint 协议重新训练 original baseline，作为协议诊断和候选搜索所需的 matched rerun；它不得替换 `docs/PHASEFORMER_GOLD_STANDARD.md`。最终报告同时给出相对金标准的结果，以及在协议不完全一致时的 matched rerun 配对结果。
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
