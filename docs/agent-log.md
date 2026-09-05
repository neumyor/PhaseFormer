# Agent Maintenance Log

## 2026-09-05 — 汇总并核验全部预测分歧成分的路由指标

- 新增 `scripts/write_asymmetric_case_all_metrics.py`，在生成一张三数据集×七成分的 Baseline-full、X-A、Only-A validation MSE/MAE 总表前，逐一核验 45 个真实 run 的协议、模式、有限指标、checkpoint 存在性及同成分 X-A/Only-A 参数一致性。输出写至 prediction-divergence 目录的 `ALL_COMPONENT_ROUTE_VALIDATION_METRICS.md`。

## 2026-09-05 — 用 ETTh1 慢趋势重训更新 EMA/Holt 可视化来源

- `export_asymmetric_joint_route_cases.py` 对 ETTh1 的 `causal_ema`、`holt_local_linear` 改为只解析已完成的慢趋势 `e30` 原始训练 run（排除 e1 smoke）；Weather、ETTm1 仍使用原交付 checkpoint。随后重建 joint-route 图、manifest、聚合表和提取参数审计；旧目录先移动至可恢复的 `/tmp` 备份。

## 2026-09-05 — 固定 SSA 参数并审计预测分歧案例的提取参数

- `ssa_low_frequency` 在 ETTh1、Weather、ETTm1 均冻结为 `W=144, retained-rank=2, candidate-rank=12, Pmin=144 steps`；映射显式写入 `scripts/probe_ssa_low_frequency_trend.py`，即便当前三者数值一致也避免依赖隐式默认值。
- 新增 `scripts/audit_asymmetric_case_extraction_params.py`，从 `asymmetric_prediction_divergence_cases` 真正用于图像生成的 X-A/Only-A checkpoint 配置反查并验证参数、L/H/seed/mode、63条 manifest 与63张图。它写入同目录 `EXTRACTION_PARAMETERS.md`；该审计明确现有 ETTh1 EMA/Holt 图仍是旧交付参数 α=.024、β=.006，而非新慢参数重训。

## 2026-09-05 — 低频 SSA 趋势提取与固定样本可视化

- 新增 `ssa_low_frequency`：将每个 `(sample, channel)` 的 720 步历史嵌入 144×577 轨迹矩阵，进行 SVD 与 Hankel 对角平均；在前12个 SSA 重构分量中按周期不少于144步的频谱能量占比选择两个相加，最后端点锚定。该固定低频筛选刻意不把 ETT 的24/96步主周期当趋势；它不是普通的“直接取最大奇异值”SSA。
- 新增 validation-only 探针 `scripts/probe_ssa_low_frequency_trend.py`：在 ETTh1、Weather、ETTm1 各两个既有 channel-0 固定样本绘制历史、SSA 趋势层级、当前慢速 Causal-EMA 趋势层级及后续96步 GT；GT 仅作图，不进入提取或参数选择。输出严格六文件审计目录与命令待验证后记录。

## 2026-09-04 — A6 trend-filter 非对称 Weak Residual 探针

- 新增冻结的趋势滤波趋势性成分 `A6=trend_filter`，定义为一阶 trend filtering：
  `min_f .5||X-f||²+λ||D²f||₁`，端点锚定 `A=f-f[-1]`，其中
  `λ=100·std(X)·(1 hour/Δt)²`；ETTh1/Weather 的 `Δt=1h`、ETTm1 的 `Δt=.25h`。
- 训练路径使用固定 256 步 GPU 批量、强凸加速的 Chambolle--Pock 近似，避免每个 forward 的 CPU ADMM；PhaseFormer
  继续看完整 X，NLinear 通过完整-X 共享 RevIN stats 接收 X-A 或 Only-A。新增启动器
  `scripts/run_weak_residual_asymmetric_trend_filter.py`；正式原始训练工件写到
  `research_runs/weak_residual_asymmetric_trend_filter_h96_scratch/`，最终六文件审计包将写到
  `research_runs/weak_residual_asymmetric_trend_filter_h96_audit/`。待完成：GPU 数值近似抽查、六个完整训练与审计。
- 冒烟训练首次暴露 `PhaseFormer` 未复制新配置字段，已补齐构造路径并新增 trend-filter candidate forward
  覆盖；失败的独立 smoke 目录仅含不完整尝试，未作为正式证据或复用。

## 2026-09-04 — ETTm1 H96 五趋势成分全流程及样本级审计

- 在 RTX 4090 / raft（CUDA 12.1）完成 ETTm1、L720→H96、seed2021 的 Baseline-full 与五个 `X-A`
  residual 分支条件共6次完整训练；30 epoch上限、Huber、最低 validation-loss checkpoint，未读取 test。
  所有任务完成且无失败，训练记录位于 `research_runs/weak_residual_asymmetric_ettm1_h96_scratch/`。
- 分析器的 dataset 白名单补充 ETTm1，并将审计打包器泛化为 Weather/ETTm1。最终包
  `research_runs/weak_residual_asymmetric_ettm1_h96_audit/` 严格含六个审计文件和 `figures/`：6个模型结果、
  57,125 条 channel-0 validation paired-error 行、五成分各10个最大正 MAE 差且起点间隔≥96步案例、共50图；
  Markdown/ZIP的50个图片引用已核验。
- channel-0 validation 基线 MAE/MSE=0.4949/0.4916。相对变化：CycleLevels +0.53%/-0.61%，
  RecentLinear -2.30%/-3.99%，GlobalLinear -2.41%/-4.05%，LocalSmooth +0.16%/-1.04%，
  MultiScaleSmooth +0.38%/-0.38%。这是单seed、validation观察；最大正差案例由排序规则产生，不能替代
  总体平均方向或多seed确认。

## 2026-09-04 — Weather H96 五趋势成分全流程及样本级审计

- 使用 experiment-and-error-analysis 流程，在 RTX 4090 / raft（CUDA 12.1）完成 Weather、L720→H96、
  seed2021 的 Baseline-full 加五个 `X-A` residual 分支条件的 6 次完整训练；30 epoch 上限、Huber、
  lowest validation-loss checkpoint，未读取 test。训练原始可恢复记录位于
  `research_runs/weak_residual_asymmetric_weather_h96_scratch/`。
- 新增 Weather 范围入口与 `scripts/package_weather_asymmetric_component_cases.py`。最终审计包
  `research_runs/weak_residual_asymmetric_weather_h96_audit/` 严格只含六个审计文件及 `figures/`，汇总五个
  成分的 6 个模型结果、25,875 条 channel-0 validation-origin paired error 行和各成分 10 个最大正 MAE
  差案例（共50图）；ZIP 与 Markdown 的50个图引用逐项核验。
- channel 0 validation 聚合结果相对 baseline（MAE/MSE=0.2208/0.0938）：CycleLevels -14.18%/-24.59%、
  RecentLinear -5.70%/-12.22%、GlobalLinear -6.37%/-12.22%、LocalSmooth -4.21%/-7.09%、
  MultiScaleSmooth +0.75%/-1.37%。这些是单 seed validation 观察；且各成分的案例按最大正误差差挑选，
  不应与总体平均方向混淆。

## 2026-09-04 — 修正五成分案例图的测试模型标签

- 用户审阅 RecentLinear sample 936 图时发现图中预测差异不如汇总 MAE 直观。人工复核确认：两预测的差异是
  跨多个峰谷约 0.2--0.5 的系统性偏低，而不是单次大幅分叉；该样本的 +0.2243 MAE 是 96 步平均误差差。
- 同时发现绘图图例将所有候选错误固定标为 `Asymmetric-A1`，实际预测数值和筛选均正确，但 RecentLinear、
  GlobalLinear、LocalSmooth、MultiScaleSmooth 的显示标签会误导阅读。已修正为动态 `Asymmetric-<component>`
  并重新导出五个案例包的全部 50 张图与 ZIP。
- 图中 RecentLinear 的 `X-A` 历史可出现远大于原序列幅度的长斜坡，源自“最近96步 OLS斜率外推到完整720步”
  的冻结定义。这是该成分干预强度/分布偏移风险，不能将其最大个例直接解释为 NLinear 对自然近期趋势的
  干净因果依赖；若继续推进 A2，需要先重新冻结较局部的平滑趋势定义并成对重训。

## 2026-09-04 — ETTh1 五类趋势成分的最大误差差异案例导出

- 用户将案例选择目标改为“两个预测模型误差差异最大”，而不是共同高误差。`analyze_weak_residual_asymmetric_cases.py`
  现按 `Asymmetric-A channel-0 MAE − Baseline-full channel-0 MAE` 从大到小选取 10 个 validation origin，
  并要求起点间隔至少 96 步；因此每张图展示的是该成分被 NLinear 遮蔽后最明显的正向退化案例。
- 在 ETTh1-H96、seed2021、完整 validation 的 channel 0 上完成五个成分各 10 个案例、合计 50 张图。
  五个独立可审计包在 `research_runs/weak_residual_asymmetric_etth1_h96_component_gap_cases/` 下，每个包均有
  `run.yaml/results.csv/sample_errors.csv/selected_cases.npz/objective_error_analysis.md/.zip/figures`；未读取 test。
- 最大单样本 MAE 差异分别为：CycleLevels +0.1498（sample 812）、RecentLinear +0.2243（936）、
  GlobalLinear +0.0991（587）、LocalSmooth +0.1271（810）、MultiScaleSmooth +0.0282（1597）。该案例筛选
  是用于定位机制敏感状态，不能替代全 validation 的平均效应或显著性判断。

## 2026-09-04 — ETTh1 A1 高误差 validation 案例审计

- 新增 `scripts/analyze_weak_residual_asymmetric_cases.py`，从本轮完成的 best-validation checkpoint
  重建 ETTh1-H96 Baseline-full 与 Asymmetric-A1（CycleLevels），只读取 validation split 的 channel 0。
  脚本对全部 2,785 个 origin 计算逐样本 MAE/MSE，交替抽取两个模型的高 MAE 样本，并要求起点间隔至少
  96 步，避免十张图只是高度重叠的滑窗。
- 完成并人工检查 10 张图：每张上图展示完整 720 步 `X` 和 NLinear 实际可见的 `X-A1`，下图展示最后
  192 步历史、96 步真值、Baseline-full 与 Asymmetric-A1 预测。审计包位于
  `research_runs/weak_residual_asymmetric_etth1_h96_a1_cases/`，含完整逐样本 CSV、选例数组、报告、图和 ZIP；
  未读取 test。
- Channel-0 全 validation 平均：A1 条件 MAE -0.21%、MSE +0.46% 相对 baseline，说明全变量汇总中的
  A1 退化不能直接外推为单一 channel-0 上的平均 MAE 退化。10 个困难且非重叠案例同时包含 A1 恶化
  （例如 sample 810: +0.1385 MAE）和改善（sample 2487: -0.1420），支持后续按时序状态而非只按
  平均指标分析。

## 2026-09-04 — Weak Residual 非对称趋势性成分发现阶段

- 在 `weak_residual_exploration` 分支将五个冻结候选写入
  `docs/Weak_residual_asymmetric_component_plan.md`：cycle-levels、recent-linear、global-linear、
  local Gaussian-smoothed trend、multi-scale Gaussian-smoothed trend。所有成分逐样本逐变量提取且
  末点锚定为零；A4/A5 均基于 Gaussian smoothing，不使用二次曲率拟合。
- 新增 `src/models/asymmetric_trend_components.py`，并为 `RevIN` 加入共享统计量归一化。PhaseFormer
  的相位路径保持完整 `X`；只有 shared NLinear weak-residual 路径读取 `X-A`，且使用完整 `X` 的同一
  RevIN 统计量。flag-off 时复用原 `X_norm`，保持历史 weak-residual 前向数值等价。
- 新增 `scripts/run_weak_residual_asymmetric_trend.py`：在 ETTh1/ETTh2/ETTm1/ETTm2/Weather × H96/H192
  × seed 2021 上顺序运行 10 个 Baseline-full 与 50 个 asymmetric-A validation-only full training，支持
  `--resume`。运行与监控记录固定在
  `research_runs/weak_residual_asymmetric_trend_discovery/`，确认 test 在 validation 选定成分后才启动。
- 校验：raft 环境运行
  `python -m unittest tests.test_asymmetric_trend_components tests.test_presets_and_loss -v`，14 项通过；
  launcher 的 cycle-levels 20-job dry-run 通过。GPU 为 RTX 4090（24 GiB）；仓库规则列出的 py310
  解释器在该主机不存在，故按既有约定使用 raft。

## 2026-09-04 — 冻结 Weak Residual 非对称输入实验设计

- 新增 `docs/Weak_residual_asymmetric_component_plan.md`：PhaseFormer 路径始终读取完整 X，NLinear
  residual 路径读取 `X-A`，首个 A 固定为 D3 cycle-levels；范围为五数据集×H96/H192×seed2021。
- 用户明确排除 sham/matched-control，文档相应限定可作的增量价值结论。新增强制共享 RevIN 约束：完整
  X 只估计一次统计量，`X-A` 必须使用同一统计量标准化和同一统计量反归一化，禁止分支独立归一化。

## 2026-09-03 — 补齐 D3 recent-linear 的完整证据链

- 主线叙事新增 D3 recent-linear 专节：给出末值锚定 OLS 提取公式、remove-trained 恢复结果、D4 全
  validation 的 B-only/A-only-anchor 冻结结果、M1/M2 NLinear-only 反事实及其CI、D5 低成本复核与
  D7 内部路径关联。
- 明确该对象是强共同依赖且增强可恢复的成分，不能作为“原版未用”候选；D7 的关联强度也说明最终缺陷
  更集中于跨周期水平状态而非广义近期线性趋势。

## 2026-09-03 — 在主线报告补齐 D2 全窗口结果

- 主线叙事的 D2 小节新增末尾24/48/96/192步直接置零的完整表：分别给出 remove-trained 恢复损失，
  以及 D5 full-trained→tail-zero 的冻结和 NLinear-only 分支反事实结果。
- 记录了窗口长度单调增加的共同依赖与增强恢复现象，并明确 M0 的强即时损失排除了 D2 作为“原版未用、
  增强在用”候选；未改动训练、评估或既有数据产物。

## 2026-09-03 — 在主线报告补齐 D1 全频率结果

- 主线叙事的 D1 小节新增六个训练期固定频率（96、48、32、24、677.647、205.714步）的完整表：分别
  报告 Gaussian-notch remove-trained 恢复损失与 D5 full-trained→notch 冻结/分支反事实结果。
- 明确两种表的回答对象不同，并记录全部六频率均未满足“M0近零、增强分支显著使用”的候选门槛；未改动
  训练、评估或既有数据产物。

## 2026-09-03 — 补全叙事报告的成分提取公式

- 在结构缺陷研究叙事新增“全部已测试成分/关系的提取步骤与公式”附录，覆盖 H1/H3/H4、C1--C7、
  D1--D3、D4/D5 分支反事实、D6 结构扰动和 D7 内部路径描述量。
- 附录明确区分加性分解、几何/频域变换、置零和关系扰动，注明末值锚定、训练拟合边界及每类实验可
  支持的解释范围，避免将非加性扰动误称为可重构成分。

## 2026-09-03 — 整理 PhaseFormer 结构缺陷研究叙事

- 新增 `docs/PhaseFormer_structural_defect_research_narrative.md`，将H1--H4、C1--C7、D1--D7的实验
  统一为可审计科研主线：输入盲区强假设被系统否定，证据收敛为 phase-only 对非平稳跨周期水平状态的
  建模不足，以及全时间轴校正路径对该残差的互补修正。
- 文档显式分开可支持的 claim、被反事实否定的说法、单seed/validation-only边界和下一步结构化状态
  校正头的参数量匹配验证要求；未改变训练代码或既有实验结果。

## 2026-09-03 — D7 内部路径诊断完成：锁定周期水平状态缺陷

- `raft` CUDA 完成512-origin完整输入诊断（约7秒）：M1/M2 NLinear correction 与 phase residual 对齐
  0.703/0.791，说明分支在系统性修正而非随机扰动。
- 修正收益与周期水平波动相关性最高（+0.490/+0.534），最后周期水平偏移次之（+0.483/+0.488）；
  预固定六特征连续五折 OOF R²=0.205/0.299。结合D3--D6，结论修订为 PhaseFormer 对该状态**建模不足**，
  而非完全不使用。新增D7报告并同步入全景汇总；后续应验证结构化轻量状态校正头及参数量匹配控制。

## 2026-09-03 — D7 内部路径诊断预注册

- 在 D1--D6 未发现输入候选后，新增低成本 D7：完整输入上直接量化 phase path 残差、NLinear correction
  和融合收益，并以六个预固定时序描述量做连续五折 OOF 探针；无训练、无 test。

## 2026-09-03 — D6 结构关系冻结筛查完成：未发现目标方向

- `raft` CUDA 完成512-origin validation-only D6（约7秒）：周期顺序反转、phase去同步、相邻 phase
  pair交换；无训练、无 test，且三种扰动均保持最后输入点。M1/M2 NLinear-only 分支重组最大回放误差
  小于 `1e-6`。
- 周期顺序与 phase同步对 M0 强烈重要（MAE +70.33/+61.71%），M1/M2及其NLinear branch也显著依赖，
  仅整体稍更可恢复；pair交换的 M0/M1/M2 为 +0.77/+0.08/+0.10%，方向相反。故不扩展这三个关系到
  高耗时重训，建议下一轮转为模型内部路径/残差表示诊断。详细数据、边界和命令写入新增 D6 报告，并
  同步进全景汇总。

## 2026-09-03 — D6 结构关系冻结筛查预注册

- D5 后不再扩展当前 D1/D2/D3 数值成分库，新增 D6：测试 phase folding 与全时间轴映射对时间关系的
  不同利用。三个 endpoint-preserving 扰动分别为早期周期顺序反转、phase-wise 周期去同步、相邻 phase
  pair 交换；均明确其保留统计量与破坏关系。
- 新增 `StructuralRelationBank`、D6 512-origin validation-only runner 和计划；仍无训练、无 test，
  并沿用 M1/M2 仅替换 NLinear branch 的可回放反事实。

## 2026-09-03 — D5 广泛冻结利用验证完成：15项均非目标候选

- 在 `raft` CUDA 上完成 D5 的512个时间均匀 validation origins 筛查（约9秒）：D1六个 Gaussian notch、
  D2四个尾部置零、D3五个末值锚定轨迹；全程无训练、无 test。M1/M2 均额外执行仅替换 NLinear branch
  的反事实，最大 fusion replay 误差 `3.82e-6`。
- 15项没有任何一项显示“M0近零而增强 NLinear 分支显著为正”。D1-32/D1-24 的 M0 效应最小
  （+0.59/+0.37%），但所有增强与分支效应也同样很小；其余13项的 M0 即时依赖均超过1%。
- D2 近期原始观测、D1主要频率、D3轨迹均表明 NLinear branch 会使用成分，但原版也有显著即时依赖。
  因而冻结该库，不进行高耗时多 seed 重训；完整表与后续“结构关系”候选方向记录在新增 D5 报告，并同步
  入全景汇总。

## 2026-09-03 — D5 广泛冻结利用验证预注册

- 基于 D4 的分支/恢复区分，新增 `scripts/run_d5_broad_frozen_utilisation.py` 与 D5 计划：固定复用三个
  ETTm1-H192 full checkpoint，在 validation-only 上一次性筛查当前定义的 D1六个频率、D2四个尾部窗口、
  D3五个末值锚定轨迹。
- 每个条件只做 full→remove 冻结前向与 M1/M2 的“固定 phase+gate、仅替换 NLinear branch”反事实；
  无重训、无 test。由于15项的全 validation 前向超过单次执行时限，发现阶段固定为时间均匀的512个
  origins；只有满足目标方向的项才允许以完整 validation 复核。预注册的判断明确禁止把较小 remove
  损失误读成 NLinear 不使用该成分。

## 2026-09-03 — D4 互补信息冻结诊断完成

- 在 `raft` CUDA 环境完成 ETTm1/L720/H192/seed2021 的 validation-only D4 冻结诊断（约22秒）：不训练、
  不读取 test。运行产物位于 `research_runs/d4_complementary_frozen_probe_control/`，包含协议、配对样本
  效应与全量聚合 CSV。
- 对 `recent_linear` 与 `cycle_levels` 分别比较 `X`、`X-A`、`repeat(last(X))+A`；M1/M2 另固定 full
  phase/gate，仅替换 NLinear branch。所有 fusion replay 最大误差小于 `4.8e-6`。
- `recent_linear` 的 M0/M1/M2 删除后 MAE 分别 +140.1/+201.0/+172.4%，故 M0 明显在用它；
  `cycle_levels` 为 +51.8/+46.0/+44.8%，但 M1/M2 的 NLinear-only 反事实仍显著变差（+33.3/+30.1%）。
  结论是分支实际使用与 remove-trained 的恢复能力不可混同，不能把已有鲁棒性结果表述为“增强分支
  不依赖被删成分”。完整边界、数值与命令记录在新增 D4 报告，并同步入全景汇总。

## 2026-09-03 — D4 互补信息冻结诊断准备

- 针对既有 D3 的 `recent_linear` 与 `cycle_levels`，新增低成本、validation-only 的冻结诊断脚本
  `scripts/run_d4_complementary_frozen_probe.py`：比较完整输入 `X`、其余历史 `X-A`，以及保留末值锚点
  的 A-充分性视图 `repeat(last(X))+A`；不重训、不读取 test。
- `ComplementaryTrajectoryBank` 明确把后者定义为“充分性 probe”而不是代数互补，避免因移除 NLinear
  的末值 persistence anchor 而错误归因。脚本对 M1/M2 固定 full-input 的 phase 输出与融合权重、仅替换
  NLinear 分支输出，并验证重组能精确回放实际融合输出。
- 修改 `PhaseFormer` 使静态 gate 的 weak-residual 模式与 RCRF 模式一样暴露最近一次相位/残差预测，
  仅供冻结归因读取；预测路径、参数与训练损失不变。待运行 CUDA validation 后再记录数值结论。

## 2026-09-02 — 预注册 PhaseFormer 输入成分 H1/H3/H4 消融

- 新增 `docs/PhaseFormer_input_component_H1_H3_H4_plan.md`，冻结 H1 同相位残差、H3 近期漂移、
  H4 相位漂移的提取公式，以及 `full/half_A/minus_A/sham` 四输入定义。
- 计划比较 `original`、`weak_residual`、`rcrf_nlinear_plain`，同时包含从头重训与固定 checkpoint
  干预、PhaseFormer residual probe、RCRF 分支/gate 反事实拆解、block bootstrap、程序化 bad-case
  选择和严格 QC。正式完整矩阵计划覆盖8数据集×4 horizon×3 seed；三个假设共享 full run。
- `EXPERIMENT_SEARCH_PLAN.md` 顶部已将本实验登记为当前用户指定任务。专项文档中的结果表全部
  留空；本次没有实现提取器、运行训练或读取 test。
- 文档验证：`git diff --check` 通过；已复核 H1/H3 的精确重构与末值保持定义、H4 小数 shift
  估计/能量审计、2880-run 计数及最终白名单产物结构相互一致。

## 2026-09-02 — 新增纯 RCRF + NLinear 因果消融机制

- 新增 `rcrf_nlinear_plain` preset：原始 PhaseFormer 相位主干 + `WeakPeriodResidualHead`
  （NLinear-style）+ RCRF，固定 `alpha_0=0.5`、`s_0=2`、`s_max=4`；不启用
  uncertainty shrinkage、period-level calibration 或 high-frequency damping。
- 该机制用于将 RCRF 的独立贡献与 golden-combo 额外相位模块区分；它是诊断/消融对照，尚无
  正式实验结果，不应表述为已验证的 incumbent。
- 修改：`src/models/phaseformer_presets.py`、`tests/test_presets_and_loss.py`、
  `docs/README.md`。验证：`.venv/bin/python -m py_compile
  src/models/phaseformer_presets.py tests/test_presets_and_loss.py` 通过；
  `.venv/bin/python -m unittest tests.test_presets_and_loss -v`，10 passed。

## 2026-09-02 — 仓库 docs 清理：只保留四种模型结构

- 用户确认当前版本（`474524f`）为最新并已提交后，要求仓库代码清理：只保留原始 PhaseFormer、
  PhaseFormer + NLinear + RCRF（LFF 编码与无编码两版）、当前最佳 strict-T28 四类结构；对
  docs 重新整理并删除冗余。用户选择 **git 硬删除**、范围 **只整理 docs**（不动 `src/` 与
  `scripts/`）。
- 删除 21 个冗余实验家族 docs + `PhaseFormer_M3_figures/`（共 24 文件，`0dd544e`）：
  TriAxis、M3/multi-anchor、HPTC、ICPT 周期间 transformer 头、纯相位/动态相位/残差拓扑、
  PCTF v1/v2 早期谱系。删除前已做交叉引用检查：保留文档不指向被删文件；仅历史记录
  （`agent-log.md` 旧条目、根目录 `EXPERIMENT_SEARCH_PLAN.md`）残留引用，按历史文档惯例保留。
  例外保留：`periodic_residual_next_stage.md`（top5 五模型矩阵的完整 3-seed 附录，承载
  K2/K3 数据）与 `PhaseFormer_pctf_anchor_formal_etts.md`（two-stage Full Repair/A2 正式
  测试，是 strict-T28 主计划明示的参照基线）。
- 新增 `docs/README.md` 作为四结构复现索引：结构→mechanism（`original`/`rcrf_pe_lff`/
  `gold_combo_reliability_s2`/`pctf_anchor_repair_strict_t28`）→权威结果文档→参数组合→
  复现命令。K4 每数据集 cycle/trust-region 表与 configs 一致（ETTh1 `u_lr020`、ETTh2 C、
  ETTm1 `w_aux01`、ETTm2 C、Weather W）。
- 机制映射已逐行核实：`rcrf_pe_lff` = `gold_combo_reliability_s2` +
  `use_periodic_residual_pe=True`（type `lff`）；K4 以 K2(A2) 为锚点。
- 验证：`git status --short` 干净；删除与 README/agent-log 分两次提交。脚本唯一引用被删文件
  的是已退役 M3 分析脚本 `analyze_m3_vs_original.py`（指向 M3_figures 输出目录），不影响
  保留结构复现。

## 2026-09-01 — ETTh1/ETTm1 单 seed Golden 定向搜索自动化

- 用户将目标扩展为 ETTh1、ETTm1 的 H96/H192 均至少超过 Golden 0.5%，允许使用极端参数且只用
  seed=2021。新增 `scripts/run_strict_t28_golden_hunt.py` 与
  `docs/PhaseFormer_strict_t28_ett_golden_hunt.md`。
- runner 以 test-set selection 方式搜索 cycle、off/C/W/X trust region、Huber/MAE 与 0.3/1/3 LR；
  每条命令最多自动重试三次，`--resume` 复用已完成运行，结果 CSV 以完整配置 key 去重。不得把该
  搜索的冠军称为盲测结果。

## 2026-09-01 — ETTh1 Strict-T28 重调参预注册

- 用户认为 T28-W 可能不适合 ETTh1，并要求调参以尝试超过 Golden。新增
  `docs/PhaseFormer_strict_t28_etth1_retune.md`：固定模型结构，按数据集而非 horizon 共同筛选
  cycle=24/48、C/M/S/W trust region 与 Huber/MAE（16 个配置）。先运行 32 个 validation-only
  低成本任务，再以 8 个全数据 validation 任务冻结唯一候选，最后才做 6 个用户授权的 test。
- 已知 A2 自身在 ETTh1 H96/H192 也弱于 Golden，因此本轮把损失函数纳入搜索；不承诺该小空间一定
  能达到 Golden。任何 Stage C 后按 test 改参的行为必须披露为 test-set selection。

## 2026-09-01 — 用户指定 Strict-T28 ETTh1 正式 test：未超过 Golden

- 用户明确要求对 strict 单阶段 T28 做 ETTh1 test，故在尚未完成全数据集 trust-region 筛选前，固定
  `cycle_period=48`、W/T28 边界 `0.60/0.24/0.12`，运行 H96/H192 × seeds 2021/2022/2023 的六个
  full-train、best-validation checkpoint、single-test job。所有任务在 RTX 4090 CUDA 上完成。
- H96：`0.366890±0.002813 MSE / 0.395406±0.001830 MAE`，相对 Golden `+2.198% / +3.510%`；
  H192：`0.400422±0.002279 / 0.415671±0.001225`，相对 Golden `+0.862% / +2.889%`。均为退化，
  三个 seed 无一双指标胜出。
- 这次 test 由用户要求直接读取，不构成参数选择；以后若按其数值修改 ETTh1 配置，必须披露为
  test-set selection。结果与逐 seed 明细写入 `docs/PhaseFormer_strict_t28_etth1_test.md`；临时
  checkpoint 和 metrics 留在 gitignore 的 `research_runs/pctf_strict_t28_etth1_formal_v1/`。

## 2026-09-01 — Strict-T28 全数据集 Golden 计划与周期探测

- 新增 `pctf_anchor_repair_strict_t28` preset，将 T28 的完整单阶段训练约束与 trust region 一并冻结：
  A2-derived composer 输入 stop-gradient、anchor/fusion 梯度解耦、anchor/composer LR 均为 1、无
  correction warm-up，边界为 `0.60/0.24/0.12`。这避免把仅更新边界的 two-stage Full Repair
  preset 误当成 T28。
- 新增 `docs/PhaseFormer_strict_t28_global_golden_plan.md`：先按数据集冻结 cycle period 与四档
  trust region（C/M/S/W），再做跨 horizon validation 确认，最后在 28 个有 Golden 的 task 做 3-seed
  test；同一数据集不按 horizon 选不同机制或参数。
- 完成 ETTm2 的 30%/8 epoch/seed 2021 CUDA validation-only 周期探测。cycle 48 在 H96/H336 的四项
  原始指标均略优于 24，但相对联合分数只差 0.087%，低于预注册 0.2% 阈值，故按复杂度 tie-break
  冻结 ETTm2 `cycle_period=24`。Traffic H96/cycle12 因外部 CUDA 进程占用约 19.1 GiB 后 OOM，未计入
  结果，待 GPU 空闲后补跑。
- 验证：`.venv/bin/python -m py_compile src/models/phaseformer_presets.py`；
  `.venv/bin/python -m pytest tests/test_anchored_phase_cycle_fusion.py -k strict_t28_preset -q`（1 passed）；
  ETTm2 H336/cycle48 的 30%/1 epoch CUDA smoke 产生有限 validation 指标且不读取 test。

## 2026-09-01 — T28 trust-region 参数冻结

- 用户完成 50 策略 H192 validation-only 搜索后提供结果：T28 `trust_060` 为联合宏平均最佳（相对
  two-stage Full Repair 为 1.0019，T00 为 1.0022），但最差配对仍为 1.0078，未达到预注册的
  `≤0.995`/`≤1.005` 门槛，未读取新的 test。
- 将 `pctf_anchor_repair_full` 默认 correction trust region 更新为 T28：
  `anchored_pctf_correction_max=0.60`、`anchored_pctf_deformation_max=0.24`、
  `anchored_pctf_global_level_max=0.12`。其余 Full Repair 默认训练参数保持不变。
- 搜索 runner 显式保留旧 T00 的 `0.25/0.10/0.05`，保证已完成的 T00–T49 比较可复现，不受新默认
  参数影响。详表写入 `docs/PhaseFormer_pctf_single_stage_h192_tuning.md`。

## 2026-09-01 — 单阶段 PCTF H192 调参扩展与 smoke

- 将严格单阶段 PCTF 的 H192 搜索从 10 个扩展为 50 个预注册策略：composer 学习率、shape/level/gate
  辅助监督、修正 trust region、非对称子空间监督、固定组合与收敛预算；正式矩阵为 ETTh2/ETTm2 ×
  seeds 2021/2022，共 200 个 full-train、validation-only 任务。
- 修复 runner 使用不存在 stage 的问题，改为训练入口支持的 `finalist`；新增 smoke 专用目录并让正式
  汇总只接受 `percent=100` 记录，避免 smoke 与正式训练重复 key。
- 严格梯度隔离新增 `anchored_pctf_detach_composer_inputs`：融合器读取 A2-derived 输入的 detached
  数值，A2 只接受 anchor loss；composer 支持独立学习率比例。单元测试验证无 composer→A2 梯度路径。
- 完成 4 个 RTX 4090 CUDA smoke（T00/T49 × ETTh2/ETTm2，30% train、1 epoch、无 test）；5%
  初始 smoke 因 ETTh2 不足一个 batch 失败，已修复为 30%。完整计划和命令见
  `docs/PhaseFormer_pctf_single_stage_h192_tuning.md`。

## 2026-08-29 — 导出正式 test 前五模型的公平比较

- 新增 `docs/PhaseFormer_top5_test_models.md`。只使用周期互补实验中同协议的 288-run 正式
  矩阵：L720、H96/H192、full-train、三 seed、best-validation checkpoint、一次 test。
- 按 12 setting、24 个 test 指标相对 A2 的宏平均比选择 I0、D1、D2、A2、A1；在开头逐
  setting 给出 MSE/MAE 均值并标出五模型最优，随后简述每个模型的结构、优势和失败边界。
- 明确排除 validation-only 的 HPTC/TriAxis 与不符合单模型论文约束的 M3，避免不公平混排；
  同时披露历史 test 暴露和当前只覆盖六数据集 H96/H192，不能解释为完全盲测的最终排名。
- 同步澄清活动计划中的 incumbent 口径：HPTC-H4 只在 validation 上相对 A1 配对，正式三 seed
  test 的统一模型 incumbent 仍是 A2（RCRF+NLinear+LFF）。

## 2026-08-29 — HPTC H96 调参与样本审计：未通过扩展门槛

- 完成 H0–H4 在 ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity 的 30 个 validation-only
  run：L720→H96、P24、30% train、8 epoch、seed 2021、Huber；所有 test 字段为空。
- H4（beta init 0.25、rolling risk scale 0.5）最好，相对 A1 的 12 指标宏平均比值 0.997098、
  最差 1.003407，在 ETTh1/ETTm2/Electricity 双指标改善。但双改善只有 3/6，预注册 gate
  失败，未运行 H192；H4 相对 A1/I0/R0 逐指标包络仍平均退化 1.47%。
- 回放六数据集 1,121,992 个样本×通道，显著改善/退化占比 13.38%/8.31%。Electricity 四段
  horizon 均改善；Weather 的退化组却获得比改善组更低的 rolling risk，表明代理置信失配；
  ETTm1 和远期 ETT 出现正负修正抵消。
- `scripts/analyze_hptc_unified.py` 生成严格审计目录 `research_runs/hptc_unified_v1/`，包含
  90 个程序化去重案例、7 张中文图和字节校验 ZIP；回放指标最大差 3.05e-6。float32 周期均值
  残差最大 2.15e-6，高于预注册 1e-6 阈值，已明确记录为数值检查未通过。
- H4 平均 95,964 参数（A1 为 72,803，+31.8%），配对 GPU 前向耗时约为 A1 的
  2.06–2.78 倍。大型 CSV、图片、ZIP、checkpoint 均留在 `.gitignore` 下，不提交。
- 决策：淘汰 HPTC v1；后续若继续，优先测试受守恒约束的低频周期水平残差，以及 ICPT 自身
  masked reconstruction uncertainty，禁止回到多完整模型 ensemble。

## 2026-08-29 — HPTC 单 checkpoint 有机整合：实现与预注册

- 基于既有 A1/I0/R0/M3 结果提出 HPTC：共享 PhaseFormer 负责相位，NLinear 独占未来周期
  水平/轨迹，ICPT 只建模逐周期零均值形状，rolling history evidence 只连续收缩形状修正，
  不选择完整专家。最终仍由 RCRF 做相位可靠度融合。
- 新增 `HierarchicalTrendCycleResidualHead` 与五个只改变 `β`/risk scale 的预注册配置；全模型
  端到端训练且只生成一个 checkpoint。ICPT 构造使用 forked RNG，保证 paired seed 下共享
  PhaseFormer 主干与 A1 逐参数同初始化。
- 预注册六数据集 L720/H96、30% train、8 epoch、seed 2021、Huber validation-only 搜索；
  H192 是否执行由固定 gate 决定，计划见 `docs/PhaseFormer_hptc_unified_experiment.md`。
- 验证：196 项仓库测试通过；四个 horizon 前向、零均值正交约束、三组件梯度、rolling 样本
  响应和单 checkpoint preset 均通过。ETTm2 5%/1 epoch GPU smoke 完成，96,066 参数、
  peak 393.4 MiB、test 字段为空。

## 2026-08-29 — 论文方法约束：停止多完整模型 ensemble 路线

- 用户明确纠正“整合”的含义：需要把 A1/I0/R0 的设计思想在一个模型中有机结合，而不是
  训练、冻结三个完整模型后在预测层做路由。指标提升不能凌驾于方法逻辑、创新性和可发表性。
- M3 从论文候选降级为诊断性 ensemble 上界、互补性证据或潜在蒸馏教师；不再继续优化
  shadow/full anchor、OOF stacking 或三 checkpoint 路由，也不得把其结果包装成统一模型贡献。
- 后续正式候选必须共享一套 PhaseFormer 相位主干，端到端联合全 horizon 轨迹校正、周期间
  关系和历史可靠度调节；允许轻量结构分支，但推理时只能加载一个模型，并必须报告相对最强
  单模型的参数量、FLOPs、延迟及逐组件消融。
- 该约束已写入 `EXPERIMENT_SEARCH_PLAN.md` 的“不可违反的论文架构约束”，并在 M3 草稿
  开头显著标为历史诊断方案，防止后续轮次再次把 ensemble 当作目标方法。

## 2026-08-29 — M3 multi-anchor independent paper draft

- 新增 `docs/PhaseFormer_M3_multi_anchor_paper_draft.md`，将已完成的 M3 实验整理为可独立
  阅读的中文论文草稿，而不是实验日志摘要。
- 草稿包含相位—周期互补动机、完整模型多锚点定义、24%→30% 时间外影子校准、16 维
  结构特征、周期级 soft 路由公式、训练目标、M0–M3 消融、六数据集 H96 结果、
  1,121,992 个 sample×channel 分析、局限性和下一步正式验证要求。
- 明确披露当前仅为 30% train、单 seed、H96 validation 机制筛选；Stage-A gate 失败，未运行
  H192/test/Golden，不能表述为全局最优或正式 SOTA。

## 2026-08-27 — Periodic-residual next-stage 288-run formal matrix completed

- 完成预注册 288-run 矩阵（12 setting × 8 mode × 3 seed；ETTh1/ETTh2/ETTm1/
  ETTm2/Weather/Electricity × horizon 96/192，lookback 720、period 24；
  full-train、best-val checkpoint、单次 test 读取）。4 张 GPU 并行（
  `scripts/_gpu_periodic_residual_runner.py` 按命令轮转分片），全部 run 正常完成，
  无缺失/重复 key。
- 汇总器生成 `research_runs/periodic_residual_next_stage_v1/formal_summary.csv` 与
  `decision_summary.json`；结果回填至
  `docs/PhaseFormer_periodic_residual_next_stage.md` §3.2/§3.3，结论写入 §4。
- 机制诊断（`scripts/collect_mechanism_diagnostics.py`，seed 2021、best.ckpt 前向）
  输出 `mechanism_diagnostics.csv`：D1 内容检索熵随样本变化未塌缩但 gate 只在
  Electricity 打开；D2 内层周期 gate 持续偏低；D3 路由按数据集选周期（ETTh→P24、
  ETTm1→P96、Weather→P12）但 correction gate 几乎恒为 0。
- 结论：**没有候选满足替换 A2 的统一门槛**。I0（`rcrf_icpt_none`）达到 8/12
  双指标改善（宏平均 0.9969，Weather/Electricity 稳定超 Golden），但 ETTh2-96
  MSE 回退 +6.5% 被挡在门槛外；D1/D2/D3 均在 ±0.6% 内、机制 gate 收敛到零。
  先前“ICPT 系统性弱于 NLinear”的结论只在 ETTh2 上成立。原始 checkpoint 与
  metrics 保留在被 `.gitignore` 忽略的 `research_runs/periodic_residual_next_stage_v1/`。

## 2026-08-27 — ICPT ETTh2/ETTm2 formal test rerun

- 按 full-train、best-validation checkpoint、single test read 协议完成
  ETTh2-720 与 ETTm2-96 的 `RCRF+NLinear`、旧 ICPT decoder、full-horizon ICPT，
  共 18 个 seed/model runs；GPU 为 RTX 4090。
- 汇总结果写入 `docs/PhaseFormer_icpt_test_results.md`；原始 checkpoint 与运行产物
  保留在被 `.gitignore` 忽略的 `research_runs/icpt_etth2_ettm2_full_20260827/`。
- 结论：full-horizon ICPT 两个 setting 均优于旧 decoder，但未稳定超过
  RCRF+NLinear 或固定 Golden。

## 2026-08-26 — Merge experiment plans and results into closed-loop experiment files

- 根据用户要求，将动态相位、Pure Phase、残差拓扑、Golden 组合和周期位置编码路线整理为“一条实验路线一个文件”，每个文件统一包含：设想、整体计划、实现与结果、最终结论。
- 新增：`docs/PhaseFormer_dynamic_phase_experiment.md`、`docs/PhaseFormer_pure_phase_experiment.md`、`docs/PhaseFormer_residual_topology_experiment.md`、`docs/PhaseFormer_gold_combo_experiment.md`、`docs/PhaseFormer_periodic_residual_pe_experiment.md`。
- `intercycle patch residual` 按用户要求未纳入；原始 plan/results 文件保留为审计来源。
- 验证：静态检查新增文档结构与 git diff；未重新运行训练实验。

## 2026-08-25 — Golden combo stability experiment (gold_combo_stability_v1)

Running `docs/PhaseFormer_gold_combo_plan.md` end-to-end (user authorized full
run on all 4 GPUs). Implementation committed `a5f0b1f` (RCRF module +
`gold_combo_*` preset modes), tooling `7694579` (analyze/fill scripts).

- **RCRF** (`ReliabilityCoupledResidualFusion`): reliability
  `r = Var_l(mean_k x) / (Var_l(mean_k x) + mean_l Var_k x + eps)` computed from
  the **pre-shrinkage** phase series; sensitivity `s = s_max·tanh(s_raw)` with
  `s_raw` initialized at `atanh(s0/s_max)` (s0=0 ⇒ α=0.5 constant = fixed-gate
  warm start); `alpha = sigmoid(logit(α₀) + s·(1−r))`, sample×channel.
- **Stage A** (validation-only, 30% data, 8 epochs, seed 2021, 18/18 runs;
  `test_mse/test_mae` empty = no test loader; unique config hashes). 6-ratio
  score: s2 0.80473 < adaptive 0.80720 < s0 0.80739 < fixed 0.80827.
  **Frozen candidate: `gold_combo_reliability_s2`** (selection source
  `validation_only`, test not read before freeze). record:
  `research_runs/gold_combo_screen_runs/freeze_record.json`.
- **Stage B** complete (27/27, all 4 GPUs): original/latest/frozen × 3 settings
  × seeds 2021/2022/2023. Frozen candidate `gold_combo_reliability_s2`:
  - ETTh2-720: 3-seed mean MSE 0.394228±0.005051, MAE 0.429443±0.002123 —
    **stable**, above Golden (0.402/0.436) +1.93%/+1.50%.
  - ETTm2-96: MSE 0.159755±0.000180, MAE 0.245331±0.000280 — **stable**, above
    Golden (0.163/0.256) +1.99%/+4.17%; also beats `latest` both metrics.
  - Electricity-336: all 3 seeds below Golden (MSE 0.162954/0.164409/0.164977,
    MAE 0.253420/0.254921/0.255533) but MSE mean+std 0.16516 crosses Golden 0.165
    by a rounding-level margin → NOT a stable gain per plan; slight regression vs
    `latest` (+0.47%/+0.51%). 3-seed mean vs Golden is −0.54%/−0.92% (improvement).
  - **Cross-dataset success criterion MET** (2/3 stable + remaining ≤1% regression
    vs Golden). Honest caveat recorded: no rounding-level margin claimed as stable.
- RCRF activity (r-α corr ≈ −1.0 across all 9 setting×seed): ETTh2 r=0.193→α≈0.77-0.81
  (sens 1.65-1.86); ETTm2 r=0.019→α≈0.87 (sens 1.97-2.01, low-reliability leans
  residual); Electricity r=0.772→α≈0.31 (sens 0.78-0.94, high-reliability leans
  phase — the mechanism behind the small regression vs `latest`).
- Smoke (3 settings) validated finite val loss, best.ckpt, validation-only
  isolation. Unit tests green (incl. 15 RCRF + gold_combo preset tests).
- Audit package `research_runs/gold_combo_stability_v1/` complete + validated:
  six-file protocol + figures/ (18 referenced PNGs, ZIP byte-identical), npz 2.2MB
  (269 aligned selected cells), sample_errors.csv per-cell 704MB (gitignored).
  Tables filled `docs/PhaseFormer_gold_combo_experiment_tables.md`; results doc
  `docs/PhaseFormer_gold_combo_results.md`. Committed via SSH over 443.

## 2026-08-24 — Pure Phase Modeling (phase-only forecasting, no residual)

Implemented 4 warm-start pure-phase modules (commits 1653cd1, 00f09dc) and ran
the next-stage plan (`docs/PhaseFormer_pure_phase_plan.md`)
at full budget: MultiScalePhase (period-axis long view, zeta gate), PhaseDeformation
(rate+stretch -> cumsum displacement warp), PhaseGraph (circular message passing),
TrajectoryDecoder (per-slot polynomial over the future axis). 7 modes registered
(multiscale_phase / phase_deformation / phase_geo / phase_graph / predictor_mlp /
trajectory_decoder / pure_full). Report:
`docs/PhaseFormer_pure_phase_results.md`.

- **Result (61/70 runs; 9 missing — Traffic h720 trajectory_decoder+pure_full,
  ETTh1 h720 all 7; user stopped the run mid-batch-2)**:
  - representation/evolution/interaction modules are parity with original:
    avg ΔMSE multiscale +0.53%, deformation −0.09%, phase_geo −0.16%,
    phase_graph −0.10%, predictor_mlp +0.03% (no consistent wins).
  - **TrajectoryDecoder is catastrophic** on 3/5 datasets (ETTm1 +90.5%/+71.8%,
    Electricity +26%, Traffic h336 +59.4%); mild improvement only on ETTh1/ETTh2.
    Analysis: it makes output smoother (−5.4% |dy|) but destroys phase peak
    alignment (peak_shift 3.67 vs 3.24). pure_full inherits the failure
    (avg +33.5%; best single result ETTh2 h720 −4.2%).
  - Deformation field learned compression (s≈0.67) but cumulative displacement
    <0.1 slot — numerically near-inactive. Multiscale zeta gate IS open
    (mean|ζ|≈0.17, 99% dims) but no MSE benefit.
  - **Conclusion: the "adaptive phase geometry" narrative is not supported** —
    pure-phase gains ≤±0.5% and the trajectory decoder dominates negatively.
- Artifacts: `research_runs/pure_phase_summary.csv`, `research_runs/pure_phase_analysis/`
  (4 CSVs + figures/), per-run `research_runs/dyn_phase_full/dynphase_*_<mode>_*/`.

## 2026-08-12 — Reliability-aware Adaptive Phase Evolution (RAPE)

New mechanism (67bb537): compose the adaptive phase warp + amplitude
calibration with a per-sample, per-channel ReliabilityGate. The gate
g=sigmoid(MLP(history volatility, linear slope, same-slot phase instability,
adaptation magnitude)) fuses `h~ = g*h_adapted + (1-g)*h_identity`, letting the
model fall back to the original fixed-grid phase prior on stable strong-period
windows. Zero-init gate -> g=0.5 at construction; warp+amp are identity then,
so the fused output equals the identity phase for any g (warm start). Mutually
exclusive with phase_align/phase_warp/phase_amp_calib, constructed last.
37/37 tests pass. Audit set in `research_runs/phase_rape_full/` (six files +
figures). Reuses `scripts/analyze_experiment.py`, extended with reliability-gate
activity + configurable report labels.

- Stage A (30%/8ep, val-only, 10 settings x original/warp/amp_calib/rape):
  rape improves Weather h192 (−6.45%) and slightly mitigates the ETTm1
  amp_calib regression; near-neutral elsewhere.
- Stage B (full budget, seed 2021, test eval, `research_runs/phase_rape_runs/`,
  10 settings, paired original + phase_rape):
  - dMAE improves on 6/10 (ETTh1 96/192, ETTh2 96, ETTm1 96/192, Weather 192);
    dMSE improves on 5/10 (ETTh1 96/192, ETTm1 192, Weather 96/192).
  - **Weather h192 beats the gold standard on both metrics** (dMSE +0.41%,
    dMAE +0.00% at 4-decimal precision; marginal, single-seed). ETTh1 h192
    beats gold on MSE (+1.66%) but not MAE (−0.84%); dMSE −3.36% is the largest
    improvement seen across all mechanisms so far.
  - Regressions: ETTm2 96 (+1.13/+1.97), ETTh2 192 (+1.07/+1.60), ETTm2 192
    (+0.91/+0.20), Weather 96 (+0.26/−0.65).
  - vs amp_calib (no gate, prior round): the gate helps ETTh1 96/192 (dMSE
    −0.01 vs +0.82; −3.36 vs −1.69) and Weather h192 (−0.90 vs −0.72), but is
    neutral-to-worse on ETTh2 192, ETTm1 192, ETTm2 192.
  - Reliability gate activity (mean g over test): high on 8/10 settings
    (0.70-0.92), lowest on ETTm2 192 (0.42) and Weather 96 (0.61). The gate
    mostly commits to the adapted representation rather than selectively
    falling back to the original phase prior; the "reliability-aware
    selection" is only weakly realized.
  - Training cost: candidate ~1.5-2.9x slower than original (Weather h192
    2120s vs 745s; ETTh1 96 146s vs 83s).
- Conclusion: no stable cross-task gain; two genuinely positive settings
  (ETTh1 h192 MSE, Weather h192 dual-metric gold beat) both improve over the
  no-gate amp_calib, but the benefit is dataset-dependent and within
  single-seed spread. Mechanism stays flag-gated and out of `_LATEST_POLICY`;
  the gate is not a reliable cross-task fix.

## 2026-08-12 — Phase-conditioned Amplitude Calibration

New mechanism (4afc634): phase-conditioned amplitude calibration builds on the
adaptive phase warp representation. `src/models/phase_amp_calib.py`
(`PhaseAmpCalibration`, flag `use_phase_amp_calib`) predicts per phase slot a
scale `alpha_l` and shift `beta_l` from the phase-slot position and per-slot
statistics of the phase history (mean/std/abs-mean/last period/linear trend),
then applies `h'[l,k] = alpha_l*h[l,k] + beta_l` broadcast over the period axis.
Zero-init final layer warm-starts at identity (alpha=1, beta=0). Module
constructed last so flag-off keeps baseline initialization; `phase_amp_calib`
ablation mode = `phase_warp` + `use_phase_amp_calib`. 31/31 tests pass. Audit
set in `research_runs/phase_amp_full/` (six files + figures). Reusable analysis
tool added as `scripts/analyze_experiment.py` (validated against phase_warp_full).

- Stage A (30%/8ep, val-only, `research_runs/phase_amp_screen/`, 10 settings x
  original/warp/amp_calib): dataset-dependent. amp_calib improves Weather
  (h192 dMAE −4.76%, h96 −1.83%) and mildly ETTh1/ETTh2 96; regresses ETTm1
  (h96 +2.78% MAE/+5.17% MSE) and mildly ETTm2.
- Stage B (full budget, seed 2021, test eval, `research_runs/phase_amp_runs/`,
  10 settings, paired original + phase_amp_calib):
  - dMAE improves on 6/10 (ETTh1 192 −0.13, ETTh2 96 −1.38, ETTm1 96 −0.54,
    ETTm1 192 −0.50, Weather 96 −0.03, Weather 192 −0.46); dMSE improves on 6/10.
  - Regressions: ETTm2 96 (+1.83/+2.01), ETTh2 192 (+0.62/+0.81), ETTh1 96
    (+0.09/+0.82), ETTm2 192 (+0.59/−0.34).
  - **No setting beats the gold standard on both MSE and MAE.** Weather 192
    beats gold on MSE (+0.25%) but not MAE (−0.07%); Weather 96 beats gold on
    MAE (+0.20%) but not MSE (−0.51%).
  - The screen's strong Weather signal (−4.76% at h192) collapsed to −0.46% at
    full budget; the ETTm1 screen regression inverted to slight improvement.
  - Calibration activity (mean |alpha−1| over test): most active ETTh1 (~0.79)
    and Weather 192 (~0.77), near-inactive ETTm2 192 (0.08); high activity with
    no net gain. beta small (<0.35). max_scale=2.0 permits alpha<0 (sign-flip),
    and the old log-alpha diagnostic nans showed it does occur.
  - Training cost: candidate ~1.7–2x slower than original (ETTm1 96 576s vs
    292s; Weather h192 1509s vs 751s; ETTh1 96 138s vs 82s).
  - Sample-level (per-cell delta_mae): ETTm2 96 42.6% cells improve (57.4%
    regress, net +0.00475), ETTh2 96 59.5% improve (net −0.00476); no dominant
    structural signature across groups beyond the aggregate sign.
- Conclusion: no stable cross-task gain, consistent with the phase_align and
  phase_warp explorations — the fixed phase grid is not the bottleneck on this
  grid, and adding a per-slot amplitude branch costs ~2x training for no net
  benefit. Mechanism stays flag-gated and out of `_LATEST_POLICY`. Diagnostic
  hook fixed to |alpha−1| (a820c2a) because log alpha nans when alpha≤0.

## 2026-08-12 — Simplified report archive validation

- Reduced ZIP validation to three practical checks: successful extraction,
  presence of the Markdown and referenced figures, and valid relative image
  links after extraction.
- Replaced the three detailed archive validation flags with one
  `archive_checked` status.

## 2026-08-12 — Portable Markdown report bundle

- Replaced the experiment PDF artifact with `objective_error_analysis.zip`.
- Required the archive to contain only the byte-identical Markdown report and
  the exact `figures/` images it references, using portable relative paths.
- Added ZIP integrity, path-safety, member-whitelist, byte-equivalence, and
  extracted-link validation; prohibited PDF generation.
- Updated the research guide and active experiment plan to use the same
  six-file Markdown-plus-ZIP contract.

## 2026-08-12 — Strict multi-setting experiment artifact layout

- Tightened `experiment-and-error-analysis` so every experiment directory has
  exactly six audit files plus one `figures/` directory.
- Prohibited retained checkpoints, command files, environment snapshots, logs,
  full predictions, temporary files, and per-setting output files inside an
  experiment directory.
- Required all settings from one run to share `run.yaml`, `results.csv`,
  `sample_errors.csv`, `selected_cases.npz`, and one Markdown/PDF report pair,
  with an explicit `setting` identifier in every applicable artifact.
- Updated the repository research guide and active search plan to use the same
  strict whitelist.
- Validation: checked Skill metadata, setting coverage requirements, directory
  whitelist language, repository references, whitespace, and the staged diff.

## 2026-08-11 — Adaptive Phase Warping exploration

Follow-up to Phase Alignment (2ab472b, 3b805d4, 08c74e4): replace the bounded
per-token phase correction with a monotonic, data-driven phase warp. A speed
field from `[value, time marks]` defines a normalized cumulative-sum map from
time-in-cycle to continuous phase (phi[0]=0, phi[L-1]=L-1), expressing
per-stage compression/stretch while preserving order; uniform speed reduces to
the identity grid (warm start). `use_phase_warp` flag, mutually exclusive with
`use_phase_align`, module constructed last. 26/26 tests pass. Audit set per
`experiment-and-error-analysis` skill in `research_runs/phase_warp_full/`.

- Stage A (30%/8ep, val-only): same sign pattern as Phase Alignment — 192
  horizons slightly positive (ETTm1 192 +0.54, Weather 192 +0.50), ETTm1 96 and
  Weather 96 eliminated.
- Report regenerated 2026-08-12 per the updated `experiment-and-error-analysis`
  skill contract: audit set in `research_runs/phase_warp_full/` is now exactly
  the six files (run.yaml, results.csv, sample_errors.csv, selected_cases.npz,
  objective_error_analysis.md, objective_error_analysis.zip) plus `figures/`
  over all 10 settings (single sample_errors.csv / selected_cases.npz with
  `setting` identifiers; ZIP = Markdown + referenced figures, byte-identical;
  PDF removed). Raw training runs preserved under `research_runs/phase_warp_runs/`.
- Stage B (full budget, seed 2021, test): no stable cross-task gain. vs matched
  original — clearly negative ETTm2 96 (dMSE -2.38%), mild positive on 192-horizon
  tasks (ETTm1 192, ETTm2 192, Weather 192). Weather 192 is the only task beating
  the gold standard on both metrics (dMSE +0.17%, dMAE +0.21%), within single-seed
  noise. Result mirrors Phase Alignment, consistent with screening.
- Sample-level (Weather 192, ETTm2 96): Weather 192 54.1% of cells improve (net
  -0.0018 delta_mae), improvement concentrated in later horizon segments and NOT
  from peak/std alignment (peak closer 1/10, std closer 0/10); ETTm2 96 53.1%
  regress (net +0.0032), regression cases show peak farther from truth in 8/10.
- Conclusion: no significant stable gain; mechanism flag-gated and out of
  `_LATEST_POLICY`. Same verdict as Phase Alignment — the fixed phase grid is not
  the bottleneck on this diagnostic grid.

## 2026-08-11 — Adaptive Phase Alignment exploration

New mechanism (b2d06ba, d1d2be1, 626b0f2): replace the fixed `time % period_len`
phase assignment with a learned continuous phase per time point. A small MLP
(`src/models/phase_align.py`, `PhaseAlignment`) maps `[RevIN value, time-mark]`
to a residual delta from the position-in-cycle; input evidence is soft-scattered
onto the two neighbouring phase slots via linear interpolation (k=2). Output
grid stays fixed, so reconstruction is unchanged. Flag-gated
(`use_phase_align`), module constructed last in `__init__` so toggling the flag
does not shift shared-module initialization; flag-off path byte-identical.
`x_mark_enc` (previously unused) now feeds the estimator; must `.float()` because
training passes it as float64.

- Tests: `tests/test_phase_align.py` (forward shape, zero-delta identity,
  flag-on@init ≈ flag-off, plumbing, mark-dim fallback). 20/20 pass.
- Stage A (30% data / 8 ep, paired same-budget original, val-only): 6/10 tasks
  slightly positive (+0.02..+0.43), 4/10 negative; 3 eliminated (ETTm1 96
  −0.81, ETTm2 96 −0.41, Weather 96 −2.43).
- Stage B (full budget, seed 2021, test eval, `research_runs/phase_align_full/`):
  no task beats the gold standard on both MSE and MAE (matched original reruns
  themselves sit 0.5-5% above gold). vs matched original: ETTm1 192 is the only
  clear dual-metric gain (MSE −1.26%, MAE −0.77%); ETTh2 96 (−1.13/−0.84) and
  ETTm2 96 (−1.34/−0.72) clearly regress; the rest are neutral or mixed. No
  cross-task stable direction; horizon split leans positive at 192, negative at 96.
- Estimator activity diagnostic (mean |delta| on test, of 24 slots): ETTm1 192
  0.108, ETTm2 96 0.140, Weather 96 0.038 — active but tiny (<1% of the cycle);
  the model finds little benefit in deviating from the fixed phase grid.
- Bad cases: worst-sample MSE roughly unchanged; ETTm1 192 and Weather 96 top
  cases improve slightly.
- Conclusion: no significant stable gain (advantage < single-seed spread, per
  `EXPERIMENT_SEARCH_PLAN.md`). Mechanism stays flag-gated and out of
  `_LATEST_POLICY`; treated as an exploration without a clear positive signal.

## 2026-08-11 — Cross-agent experiment analysis skill

- Added the project-level `experiment-and-error-analysis` Skill under
  `.claude/skills/`, with a Codex-compatible entry under `.agents/skills/`.
- Added native repository entry rules for both Codex (`AGENTS.md`) and Claude
  Code (`CLAUDE.md`) with identical trigger boundaries.
- Renamed the shared maintenance policy to `MANAGE_RULES.md` and updated all
  repository references.
- Integrated the Skill into `HOW_TO_DO_RESEARCH.md` and explicitly allowed
  test-set-driven model/configuration selection when the complete search trail
  is retained and the resulting reports disclose test-set selection.
- Validation: checked Skill metadata and structure, link resolution, all
  repository references, whitespace, and the staged diff. No model code or
  experiment results changed.

## 2026-08-11 — Original PhaseFormer gold standard

- Transcribed the user-provided paper Table 5 screenshot into
  `docs/PhaseFormer_gold_standard.md`.
- Recorded 28 original PhaseFormer results covering ETTh1, ETTh2, ETTm1,
  ETTm2, Weather, Electricity, and Traffic at horizons 96, 192, 336, and 720,
  with input length 720 and explicit MSE/MAE column ordering.
- Defined the fixed comparison formula, dual-metric claim rule, matched-rerun
  distinction, and update authority. Exchange remains intentionally unset
  because it is absent from the supplied source image.
- Updated `MANAGE_RULES.md`, `HOW_TO_DO_RESEARCH.md`, and
  `EXPERIMENT_SEARCH_PLAN.md` so future improvement claims use this fixed gold
  standard instead of silently replacing it with a retrained baseline.
- Validation: manually cross-checked all 28 rows against the source image and
  verified the Markdown table contains 7 datasets × 4 horizons with both
  metrics. No training or model behavior changed.

## 2026-07-26 — Training protocol and maintainability repair

- Fixed the `ett_all` train/validation/test dataset selection condition.
- Made the effective loss name authoritative and retained legacy Huber flags
  only as compatibility metadata.
- Changed official and research runners to evaluate the lowest-validation-loss
  checkpoint and use that same model for bad-case export.
- Replaced the standalone Traffic training loop with the shared preset runner,
  removing per-epoch access to the test set.
- Moved PhaseFormer weak-period and phase-adaptation helpers into
  `src/models/phase_adapters.py` while preserving public imports and state-dict
  keys.
- Added a uv project definition with separate core, development, and GIFT-Eval
  dependency groups.
- Validation commands:
  - `uv run pytest -q` — 7 passed.
  - `uv run python -m compileall -q config src scripts run_*.py`.
  - All seven official dataset entry points completed `--help` smoke checks.
  - `smoke_best_checkpoint_protocol_20260726b` completed a two-epoch GPU
    training/test cycle and restored `checkpoints/best.ckpt` before evaluation.
- Environment: NVIDIA GeForce RTX 4090; PyTorch 2.7.1+cu126; CUDA 12.6.
- Protocol compatibility: historical benchmark files used last-epoch weights.
  New best-checkpoint results require matched original/latest reruns and must
  not be compared directly with those historical metrics.
- Completed matched best-checkpoint regressions:
  - ETTm2 96: MAE -3.85%, MSE -5.82%.
  - ETTh2 720: MAE -4.75%, MSE -4.33%.
  - Exchange 96: MAE -13.27%, MSE -16.93%.
  - Weather 96: MAE -4.45%, MSE -1.85%.
  - Electricity 336: MAE -2.28%, MSE -2.31%.
- Traffic 96 batch64 was blocked by another process occupying 18.9 GiB GPU
  memory. The official batch8 setting entered training successfully but was
  stopped because completing both 30-epoch runs under contention was
  impractically slow. No Traffic metric is claimed from these incomplete runs.

## 2026-08-10 — Weak-residual branch refactor and cleanup

- Branch renamed `phaseformer-weather-electricity-presets` → `weak-residual-phaseformer`
  (confirmed independent from `main`, which removed the weak/adaptive residual line).
- Extracted the shared training protocol into `src/training/runner.py`
  (`build_logger`, `build_trainer`, `restore_best_checkpoint`); refactored the
  four previously duplicated Trainer assemblies (`run_ett_latest.py`,
  `scripts/benchmark_phaseformer_suite.py`, `scripts/research_weather_weak.py`,
  `scripts/search_phaseformer.py`) to use it. Best-checkpoint restore now has a
  single implementation.
- Converted the 37-branch `get_latest_overrides` if-ladder into a declarative
  `_LATEST_POLICY` table keyed by `(dataset, horizon)` with a per-dataset
  full-horizon fallback and the original guardrail default. Verified
  behaviorally identical for all 32 dataset×horizon tasks; added
  `LatestPolicyTableTests` in `tests/test_presets_and_loss.py`.
- Unified dataset entry: `run_ett_latest.py --datasets` runs multiple datasets;
  thin `run_*.py` wrappers unchanged. `run_all_experiments.py` marked
  deprecated (superseded by `scripts/run/*.sh` + benchmark suite).
- Archived 18 unused `src/models/layers/*` legacy modules to
  `archive/layers_legacy/` (the active model only imports
  `SelfAttention_Family.py`), with an explaining README.
- Archived `iteration_brief.md` / `iteration_log.md` to `docs/archive/` and
  repointed references in `MANAGE_RULES.md` / `HOW_TO_DO_RESEARCH.md` to the archived
  paths, clarifying the current active plan/log are `EXPERIMENT_SEARCH_PLAN.md`
  and `docs/agent-log.md`.
- Removed tracked `.DS_Store` files and added `.DS_Store` to `.gitignore`.
- Environment note: sandbox lacks the repo's locked deps (torch/lightning), so
  verification was static (AST parse + behavioral-equivalence simulation for the
  presets table). Full `uv run pytest` and a GPU smoke run should be executed in
  the real `raft`/`py310` environment to confirm runtime equivalence.

## 2026-08-11 — Weather 192 weak-period mechanism exploration

Follow-up to the original-vs-latest benchmark: the current `_LATEST_POLICY`
table has no entry for (Weather, 192), so `latest` falls back to the original
guardrail. Question: which weak-period mechanisms are actually useful for
Weather 720→192? Ran a validation-isolated search following
`EXPERIMENT_SEARCH_PLAN.md` (val-only until a frozen winner).

### Protocol

- Entry point: `scripts/search_phaseformer.py` (fixed a startup import-order bug
  — `from src...` ran before the `sys.path.insert`, so the script failed with
  `ModuleNotFoundError: No module named 'src'` when invoked directly).
- Stages: period screen → mechanism screen (30% / 8ep) → full-budget confirm
  (100% / 30ep, seeds 2021+2022) → 3-seed test was truncated by user decision
  after the 2-seed confirm proved stable.
- All runs: val-only (no test read during search), loss=huber, lr 0.001,
  batch 64, period search {12, 24, 48}.

### Results (val, period 48)

| run | seeds | avg val_MAE | avg val_MSE | dMAE% | dMSE% |
|---|---|---|---|---|---|
| **channel_residual** (gate 0.5) | 2 | 0.29925 | 0.43405 | **−4.93** | **−4.39** |
| channel_adaptive (channel head + adaptive gate) | 1 | 0.30001 | 0.43241 | −4.69 | −4.75 |
| phase_stack (uncert+level+hifreq+sparse) | 2 | 0.31090 | 0.45132 | −1.23 | −0.59 |
| adaptive_g02 (shared head + adaptive gate) | 1 | 0.31405 | 0.44393 | −0.23 | −2.22 |
| original | 2 | 0.31477 | 0.45400 | — | — |

### Findings

- **Period 48 wins** for Weather 192 (val MAE 0.341 vs 0.346 / 0.363 for 12 / 24).
  Note Weather 96's enhanced preset uses period 12 — the optimal cycle length
  differs across horizons.
- **Channel-wise weak-period residual head is the only robust winner**, stable
  across seeds (0.2993 / 0.2992). It extrapolates a per-channel centered
  trajectory + persistence anchor, gated at 0.5.
- **Adaptive gate adds nothing** on top of the fixed channel head (channel vs
  channel+adaptive ≈ equal); on the shared head it is a mild regression.
- **Phase adapters (uncertainty/level/hifreq/sparse) give only ~−1%** here, far
  below the Weather-96 preset's benefit — their effect does not transfer to the
  192-horizon setting.
- time_mark and phase_local_trend are clearly negative / no-op for this task.

### Artifacts

- Search runner output: `research_runs/weather192_explore/runs/` (per-experiment
  `metrics.csv`, `config.json`, best checkpoint).
- Logs: `~/.claude/jobs/eee0ff88/tmp/full/` and `tmp/final/`.
- The 3-seed `--evaluate-test` confirm round was launched then stopped by user
  request ("无需三个seed了，可以结束了"); no test-set numbers were produced.
  Test-set validation of the channel-residual winner is still outstanding.

### Open question

Whether to promote a channel_residual entry for (Weather, 192) into
`_LATEST_POLICY` — and whether the same mechanism helps Weather 336/720, which
currently also fall back to the original guardrail.

## 2026-08-12 — Compress experiment analysis Skill

- Condensed `.claude/skills/experiment-and-error-analysis/SKILL.md` from 300
  to 168 lines while retaining its experiment protocol, six-file artifact
  whitelist, unified multi-setting schema, test-set-selection disclosure,
  programmatic case selection, objective reporting, and Markdown/figure ZIP.
- Simplified repeated validation language into four required checks, consistent
  with the existing lightweight-validation requirement.
- Validation: Skill schema passed `quick_validate.py`; measured at 2,159
  `o200k_base` tokens and 2,577 `cl100k_base` tokens.

## 2026-08-24 — Residual topology plan and implementation

- Added `docs/PhaseFormer_residual_topology_plan.md` as the experiment anchor.
  It preregisters R0 original, R1 full-forecast convex output residual, R2
  zero-initialized additive output correction, R3 one-shot latent long skip,
  R4 layer-wise latent injection, and R5 R2+R4 hybrid across four representative
  settings.
- Implemented the residual primitives and PhaseFormer wiring, registered all five
  candidate modes in presets/search, and added the resumable
  `scripts/run_residual_topology.py` scheduler with validation-screen/full-confirm
  stages and matched-delta summaries.
- Preserved comparison fairness by constructing the R1 control head after all
  shared modules so feature flags do not shift shared RNG initialization. R2--R5
  are exact zero-initialized warm starts; the residual master switch disables all
  new paths.
- Verification: Python compilation passed; the complete suite passed `90/90`;
  Stage A dry-run produced 24 commands and the frozen-candidate Stage B example
  produced four commands. Tests cover forward shapes, finite values, exact shared
  initialization, zero-init equivalence, gradients, optimizer movement, one-layer
  R3/R4 equivalence, multi-layer depth, 321-channel input, and summary arithmetic.
- Per the revised user scope, no training was launched, no test split was read,
  and no experimental result or error-analysis package was generated.

## 2026-08-24 — Residual topology experiments executed (Stage A + Stage B)

- Executed the plan end-to-end on 4× A100-40GB (multi-GPU via
  `CUDA_VISIBLE_DEVICES`). **Stage A**: 24 validation-screen runs
  (`search_phaseformer.py --stage mechanism_screen_1`, 30% data, ≤8 epochs,
  no `--evaluate-test`). **Stage B**: 12 full-budget confirm runs
  (`benchmark_phaseformer_suite.py`, 100% data, ≤30 epochs, val early stop +
  best ckpt, test metrics). Tests passed 90/90 before launch.
- **R3≡R4 equivalence verified numerically** on ETTh2-h720 (1 layer): identical
  val_mae=0.66184554, val_mse=0.82789717, params=734 → implementation correct.
- **Stage A freeze** (score = 0.5·ΔMAE% + 0.5·ΔMSE%): R1 convex (15.55) and R2
  additive (13.59) → R0+R1+R2 advanced; all candidates 4/4 settings both-metric
  improvement, no regression.
- **Stage B result (test, positive = improvement)**: residual output fusion is
  cross-setting inconsistent — ETTh2-h720 **strong** (R1 ΔMAE +5.75/ΔMSE +7.66,
  R2 +5.69/+7.56), Electricity **mild** (+0.81/+1.57, +0.41/+1.32), ETTh1/ETTm1
  neutral-to-slightly-negative (R1 −0.75/−0.06, −0.19/−0.83). Reproduces the
  prior dynamic-phase finding exactly. R1 ≥ R2 on 3/4 settings → H2 ("additive
  correction beats convex fusion") **not supported**; R3/R4/R5 provide no
  additional benefit.
- Judgment call disclosed: plan gated Electricity behind "前三项通过且仍有正向
  信号"; borderline, but ran it (extra ~1 GPU·h) to complete all 4 planned
  settings, consistent with prior full-budget residual evidence.
- Single-seed only; **`_LATEST_POLICY` not updated**. No champion topology.
- Artifacts: `research_runs/residual_topology_screen_runs/` (24 metrics.csv +
  `screen_summary.csv` + `stage_a_selection_notes.md`), `research_runs/
  residual_topology_full_runs/` (12 metrics.csv + per-setting `*_summary.csv` +
  `full_summary.csv`). Report: `docs/PhaseFormer_residual_topology_results.md`.
- Plan §4 (sample-level error analysis package at `research_runs/
  residual_topology_v1/`) was **not produced** — see report; flag if needed.

## 2026-08-25 — Output-residual layerwise variants (A1/A2) screened and confirmed

- Completed the output×depth design-space cell the first round left open: R1/R2
  had only single-point output fusion; added **A1** `residual_output_layerwise_convex`
  (R1 convex fusion applied at each routing depth) and **A2**
  `residual_output_layerwise_additive` (R2 additive correction at each depth).
  Implemented via `PhaseSlotResidualHead` (zero-init Linear(seq_len→P) in the
  phase-slot domain (B,C,24,30); `anchor=True` = convex/persistence, `anchor=False`
  = additive/warm-start), intermediate gates shape (1,enc_in,1,1), constructed only
  for `phase_layers−1` intermediate depths. 1-layer ⇒ A1≡R1, A2≡R2 exactly.
- Tests extended (90→99/99): module broadcast/anchor tests; one-layer reduction to
  parent; multilayer warm-start (A2 == original); closed-gate A1 == R1; gate/head
  receive gradients; master-switch disable. Feature-flag init isolation preserved.
- **Stage A** (validation, 8 added runs): A1 ≥ R1 on all settings (avg 15.72 vs
  15.55), A2 < R2 (13.42 vs 13.59). Strict freeze top-2 = A1+R1; per user request
  to compare both layerwise forms, sent **A1+A2** to Stage B (deviation disclosed
  in `stage_a_selection_notes.md`).
- **Stage B** (test, 8 runs, 20 total with reused originals): **layerwise does NOT
  transfer** — all multilayer settings A1 ≤ R1 and A2 ≤ R2 except A2@Electricity
  (+0.59/+1.83 vs R2 +0.41/+1.32). Test-set avg score R1 1.75 > R2 1.53 > A1 1.38 >
  A2 1.31. **Stage A validation signal reversed on test** (A1≥R1 on val vs A1<R1 on
  test everywhere) — a clean screen-vs-confirm divergence, consistent with the
  single-seed / validation-not-guarantee protocol caveat.
- 1-layer degeneracy verified numerically (ETTh2 A1≡R1, A2≡R2 byte-identical
  metrics). All deltas recomputed from on-disk `*_summary.csv` and match
  `full_summary.csv`. Report updated with §3.2 four-form comparison and H6.
  Conclusion unchanged: single-point output convex fusion (R1) remains the
  correct insertion point; layerwise cascade not adopted. `_LATEST_POLICY` not
  updated (single seed).

## 2026-08-25 — ETTm2 RCRF sample-level analysis

- Ran a matched ETTm2-h96 comparison of ordinary PhaseFormer versus
  `gold_combo_reliability_s2` with lookback 720, batch 256, MAE loss, lr 3e-4,
  best-validation checkpoints, and seeds 2021/2022/2023. The raw runs are under
  `research_runs/ettm2_rcrf_sample_raw/`.
- RCRF improved every seed. Mean test MSE changed 0.167989 → 0.159761 (4.90%);
  mean test MAE changed 0.256186 → 0.245333 (4.24%). These are matched-rerun
  deltas, not replacements for `docs/PhaseFormer_gold_standard.md`.
- Added `scripts/analyze_ettm2_rcrf_samples.py` to reconstruct all six
  checkpoints and export sample×channel errors, phase/residual branch outputs,
  reliability `r`, gate `alpha`, dataset statistics, deterministic categories,
  non-overlapping Top-K cases, and Chinese matplotlib figures.
- Operational “significant stable improvement” means all three seeds improve
  and mean relative window MAE improves by at least 10%: 2,035/11,425 windows
  (17.81%). It is explicitly not a statistical-significance claim. Net
  regression occurs on 2,697 windows (23.61%).
- Version-controlled user-facing report:
  `docs/ETTm2_RCRF_sample_analysis/ETTm2_RCRF_sample_analysis.md`. The 11
  generated figures and portable ZIP remain local-only under ignored paths.
  Canonical six-file audit package:
  `research_runs/ettm2_rcrf_sample_analysis_v1/`. The report opens with a
  plain-language evidence summary: strong improvements overrepresent drift
  windows (38.38% vs 28.14% among net regressions), while net regressions
  overrepresent high-volatility windows (21.65% vs 11.99%); nearly identical
  alpha values in both groups identify gate saturation/discrimination as the
  next mechanism to test.
- Validation passed: 54 relevant unit tests; six checkpoint metrics reproduced
  within 1e-5; exported branches and gates reconstruct the final RCRF output
  within 2e-5; 239,925 sample-error rows were re-aggregated; Top-K, setting
  coverage, Chinese glyph rendering, Markdown references, directory whitelist,
  and byte-identical ZIP members were checked.
- Corrected the ETT dataset roots in `src/dataset/data_info.py` from the absent
  `resources/all_datasets/ETT-small` directory to the repository's actual
  `resources/all_datasets/ETT` directory. No model architecture or default
  hyperparameter was changed.

## 2026-08-26 — Periodic position encoding for the RCRF residual branch

- Implemented a flag-isolated `PeriodPositionEncodedResidualHead`: a shared
  NLinear delta is blended with a position-similarity periodic retrieval delta
  before the unchanged outer RCRF. Added seven controlled PE presets: ST-Informer,
  single-cycle, fixed harmonics, Traffic hybrid, Time2Vec, learnable Fourier
  features (LFF), and calendar cycles. RoPE was excluded because NLinear has no
  query/key and adding attention would confound the architecture comparison.
- Stage A completed 24 validation-only screens (30% data, at most 8 epochs,
  seed 2021, no test read). LFF froze first with six-ratio mean `0.9995488` and
  worst `1.0003643`; Time2Vec was second. Stage B completed all 18 current-RCRF
  versus LFF runs across ETTh2-720, ETTm2-96, Electricity-336 and three seeds.
- Mean MSE/MAE current RCRF→LFF: ETTh2 `0.394228/0.429443 →
  0.393591/0.428967` (+0.162%/+0.111%); ETTm2 `0.159762/0.245333 →
  0.159678/0.245196` (+0.052%/+0.056%); Electricity `0.164114/0.254625 →
  0.164260/0.254876` (−0.089%/−0.099%). The pre-registered cross-dataset
  effectiveness rule passes, but LFF is not a universal RCRF improvement.
- Relative to fixed Golden, LFF is stably better on ETTh2 and ETTm2. Across all
  18 dataset×seed×metric cells, 17 are below Golden; Electricity seed-2022 MSE
  `0.165042` is the sole exception versus `0.165`.
- Canonical audit `research_runs/periodic_residual_pe_v1/` contains 5,028,081
  sample×channel rows, 270 programmatically selected cases, 44 Chinese
  matplotlib figures and the exact ZIP whitelist. All 18 checkpoints reproduced
  logged metrics within 1e-5; setting/case/CSV/NPZ/report/ZIP validation passed.
- Environment fallback: base conda, Python 3.13.5, torch 2.7.1+cu126, RTX 4090;
  the documented py310 path was absent. Results doc:
  `docs/PhaseFormer_periodic_residual_pe_results.md`.

## 2026-08-26 — Generated asset history cleanup

- Rewrote the branch commits after `be8a22e` to remove the ETTm2 report's 11
  generated PNGs and ZIP from version control while preserving them locally.
- Added ignore rules for `docs/**/figures/` and `docs/*.zip`; reports, code,
  numeric results, and experiment conclusions are unchanged.

## 2026-08-26 — ICPT periodic residual follow-up plan (design only)

- Closed the NLinear+periodic-PE round and designed its successor without
  implementing or running experiments. The proposed Inter-Cycle Patch
  Transformer (ICPT) treats each complete `P=24` cycle as a token, models
  cycle-to-cycle motif evolution, and replaces only the NLinear residual head;
  the current PhaseFormer phase path and outer RCRF equation stay fixed.
- The complementarity claim is structural and pre-registered: PhaseFormer
  summarizes the same-phase axis of the cycle matrix, whereas ICPT embeds each
  complete-cycle row and models the inter-cycle axis. Controls include last-cycle
  repetition, CycleNet-style recurrent template, ICPT without PE, ICPT-only,
  fixed fusion, non-period-aligned patches, no anchor, and no attention.
- Planned a validation-only screen of nine PE variants plus no-PE: fixed/learned
  absolute, Time2Vec, RoPE, relative bias, ALiBi, LFF, absolute+relative, and
  calendar. Calendar is ranked separately because it consumes real timestamp
  information. A frozen index-PE must beat ICPT-none, not only NLinear.
- Formal confirmation covers six datasets/settings, three seeds, matched current
  RCRF and fixed Golden comparisons, resource accounting, internal attention/
  gate diagnostics and programmatic sample errors. Pre-registered adoption
  requires at least 4/6 settings to improve both mean metrics, all remaining
  regressions ≤0.5%, and at least 4/6 settings to stably beat Golden before any
  optional 28-task expansion.
- The design, validation gates, executed results, and stop decision were later
  consolidated into `docs/PhaseFormer_intercycle_patch_residual_experiment.md`.
  No code, checkpoint, validation metric or test metric was produced in this
  design-only step.

## 2026-08-26 — ICPT periodic residual experiment: Stage 0 pass, Stage A gate failure

Executed the pre-registered ICPT plan
(`docs/PhaseFormer_intercycle_patch_residual_experiment.md`) under full-GPU
authorization. Implementation committed `372a5af` (ICPT module, PE variants,
PhaseFormer wiring), presets/runner `086f241`, GPU parallel runner + analyzer
`bca8909`.

- **Stage 0**: `pytest tests/ -q` all green (124 existing + 15 new ICPT tests);
  P0–P9 PE forward/backward finite with gradients; flag-off paths untouched.
- **Stage A** (architecture screen, validation-only, 30% data, ≤8 epochs, seed
  2021): 16 runs over 4 settings × {A2 gold_combo, A3 repeat-last-cycle,
  A4 CycleNet-style, A5 ICPT-none} on GPUs 0/1. Metrics in
  `research_runs/phaseformer_icpt_pe_screen/screen_summary.csv`.
- **A5 vs A2 gate** (8 ratios = 4 settings × MSE/MAE): mean **1.137**, worst
  **1.278**; only ETTh2-720 improves both metrics (0.960/0.973). Gate failed —
  neither mean<1 nor ≥3/4 settings both-metric improve holds.
- **Architecture diagnosis**: A3 RepeatLastCycle (≈0.7–4.7K params) is near
  parity only on ETTh2-720, regresses 15–60% elsewhere; A4 CycleNet
  (≈ A2 param count) is numerically within 1.3% of A2 on all 4 settings, with
  no statistical claim from the single seed; A5 ICPT (24.7K–28.2K params, far smaller than NLinear) beats A2 only
  on ETTh2-720, regresses 7–28% on the other three.
- **Decision per plan §13**: Stage A architecture gate failed → **ICPT main line
  stopped**; no PE freeze, no Stage B/C/D. `freeze_record.json` written with
  `stage_a_passed: false`; test set was never read.
- Plan doc updated: tables 9.1/9.2 filled with actuals, 9.3–9.8 marked 不适用,
  §7 B/C/D sections marked 未运行, status header reflects the stop.

## 2026-08-27 — GitHub SSH-over-443 route documented

- Verified pull/push route: GitHub's `ssh.github.com:443` through the local
  SOCKS5 proxy at `127.0.0.1:7897`.
- Added reusable temporary `core.sshCommand` examples to `AGENTS.md`; the
  commands leave the configured remote URL unchanged.

## 2026-08-27 — ICPT report consolidation and result review

- Consolidated the ICPT plan and filled Stage A results into
  `docs/PhaseFormer_intercycle_patch_residual_experiment.md`, following the
  repository's four-section closed-loop report format.
- Recomputed the reported A5-vs-A2 percentage changes: ETTh2-720 improves
  3.98%/2.67% MSE/MAE, while ETTm2, Electricity, and Weather regress by
  7.35%–27.77%. The pre-registered Stage A failure decision is unchanged.
- Clarified that Stage B/C/D and formal Golden comparison were not run, so the
  experiment neither ranks position encodings nor supports a Golden-beating
  claim. The locally generated screen CSV is absent from the current checkout,
  which limits independent run-level re-aggregation.

## 2026-08-27 — ICPT full-horizon head experiment preregistration

- Started a new, separately identified ICPT experiment at
  `docs/PhaseFormer_icpt_horizon_head_experiment.md`; the stopped decoder-based
  ICPT result remains unchanged.
- Replaced future-query decoding in the candidate with an ordered flattened
  full-horizon head and restored last-value centering/anchoring. With
  `d_model=24`, the `30×24→H` prediction matrix matches NLinear's `720→H`
  matrix size; the cycle encoder is the only additional capacity.
- Pre-registered a validation-only four-setting screen of none plus eight index
  position encodings and a separately ranked calendar encoding. All encodings
  will run; no-position is an ablation rather than a gate that blocks PE tests.
- Formal three-seed test and Golden comparison are allowed only after a frozen
  candidate beats the matched NLinear validation gate.

## 2026-08-27 — ICPT full-horizon head experiment: validation gate failure

- Implemented the ordered `30×24→H` full-horizon ICPT head, last-value
  centering/anchoring, cycle-anchor control, and nine index/calendar position
  variants. The legacy decoder remains the default flag-off path.
- Stage 0 passed: 146 repository tests, finite forward/backward for every
  candidate, exact zero-init last-value persistence, history-only calendar
  invariance, and two ETTm2 5%/1-epoch GPU smoke runs. The full-horizon matrix
  matches NLinear's `720→H`; total residual-head overhead ranges from 8.07% at
  H=96 to 1.08% at H=720.
- Stage A completed all 48 validation-only runs on ETTh2-720, ETTm2-96,
  Electricity-336, and Weather-336 (seed 2021, 30% train, at most 8 epochs),
  with no test loader and no OOM. All candidates improved both metrics only on
  ETTh2.
- `sincos_relative` had the best eight-ratio mean versus matched RCRF-NLinear
  at 0.999544, but its worst ratio was 1.041909 and it improved both metrics in
  only 1/4 settings. Calendar also failed (mean 1.002364, worst 1.042364).
  Consequently no candidate was frozen and formal three-seed testing was not
  run.
- Relative to the stopped decoder ICPT, the new no-PE head recovered roughly
  18.4%/12.6% MSE/MAE on ETTm2, 13.6%/5.2% on Electricity, and 20.2%/16.7%
  on Weather. This validates the head/anchor diagnosis but not stable
  superiority over NLinear. Full results and the stop decision are in
  `docs/PhaseFormer_icpt_horizon_head_experiment.md`.

## 2026-08-27 — Periodic-complementary residual next-stage preregistration

- Pre-registered three NLinear-preserving residual directions: content-aware
  phase-template-error memory, dual-reliability LFF routing, and an adaptive
  12/24/48/96 multi-period bank.
- The new plan re-evaluates both decoder and full-horizon no-PE ICPT without an
  early architecture gate. All eight matched modes must cover ETTh1/ETTh2/
  ETTm1/ETTm2/Weather/Electricity at lookback 720 and horizons 96/192, with
  three seeds: 288 formal runs in total.
- The plan discloses prior ETTh2/ETTm2 test exposure, freezes all candidates
  before further tests, and uses RCRF+NLinear+LFF as the primary incumbent.
  Protocol, success rules and empty result tables are in
  `docs/PhaseFormer_periodic_residual_next_stage.md`.

## 2026-08-27 — Periodic-complementary residual candidates implemented

- Implemented `PhaseErrorPeriodicMemoryHead`,
  `DualReliabilityPeriodicFusion`, and `AdaptiveMultiPeriodResidualHead` in
  `src/models/periodic_residual_experts.py`. D1/D3 start exactly as NLinear;
  D2 preserves the old LFF component outputs but replaces its global blend with
  sample/channel residual-cycle reliability.
- Added isolated presets `rcrf_phase_error_memory`,
  `rcrf_dual_reliability_lff`, and `rcrf_multiperiod`; existing NLinear, LFF
  and both ICPT paths remain unchanged by default.
- Added a formal runner/summarizer that expands the frozen six-dataset,
  96/192, three-seed matrix to 36 commands and 288 model runs. Summarization
  refuses incomplete/duplicate matrices and computes sample std, A2 ratios,
  stable-Golden counts and the pre-registered replacement gate.
- Verification: 160 repository unit tests passed; full PhaseFormer forwards at
  both horizons, actual `720→192` finite backward, exact NLinear warm starts,
  normalized/sample-varying diagnostics, all dataset presets, dry-run count and
  synthetic summarization were checked. No training/test experiment was run.
  Code commit: `d1ab49e`.

## 2026-08-28 — TriAxis 自验证三专家实验在 validation 门槛停止

- 实现 PhaseFormer/NLinear/旧 decoder ICPT 三个原子专家与单一历史路由器；T0 固定均匀，T1
  使用结构统计，T2 使用历史内伪预测风险。推理路由不读取 future value、future mark 或专家预测。
- T2 训练目标加入专家辅助损失 0.2 和 oracle 路由 KL 0.1；旧 preset flag-off state dict 不变。
- 验证：168 项单元测试通过；ETTh2、ETTm2、Weather、Electricity 的 L720→H96、seed 2021、
  30% train、8 epoch validation-only 共 20 个 run 完成。
- T2 的 8 指标宏平均比值 1.0005、最差 1.0426，只在 2/4 setting 双指标改善；T0/T1 也失败。
  按预注册规则停止，不读取 test，不更新 A1/RCRF+NLinear incumbent。
- 三专家逐点 oracle 宏平均改善 47.80%，但实际路由命中率只有 34.54%–39.27%，说明瓶颈是
  历史代理风险与未来专家 regret 的错配，而不是专家完全缺乏互补性。
- 审计产物：`research_runs/triaxis_self_validating_v1/`；实现 commit `e313ee4`。原始 checkpoint
  和训练日志只保留在被忽略的 scratch 目录，不加入版本控制。

## 2026-08-28 — TriAxis v2 多截点滚动校准仍在 validation 门槛停止

- 修正 v1 的单截点代理错配：对最近四个历史目标周期按未来 1–4 个周期的相同 lead 做
  rolling-origin 回测，输出三专家风险及跨 origin 方差。R0 只把证据作为特征，R1 强制低风险
  单调先验，R2 再加周期级 soft-oracle KL。实现 commit `d7ecc7f`。
- Stage 0：174 项仓库测试通过；新增 H96/H192 shape、严格历史因果、线性/周期回测、风险单调、
  不确定性收缩、梯度和完整 PhaseFormer forward 测试；ETTm2 5%/1 epoch GPU smoke 通过。
- Stage A：ETTh2/ETTm2/Weather/Electricity，L720→H96、P24、seed 2021、30% train、最多
  8 epoch、validation-only，完成 R0/R1/R2 共 12 个新 run，并复用 A1/I0/T2-v1 配对结果。
- R0/R1/R2 的 8 指标宏平均比值分别为 0.992243/0.999310/1.007830，最差比值分别为
  1.026184/1.015926/1.042114；双指标改善为 2/4、2/4、1/4，全部未通过预注册 gate。
  R0 改善 Weather 和 Electricity，也改善 ETTh2 MAE，但 ETTm2 MSE/MAE 回退 2.62%/1.42%。
- 结论：多截点等 horizon 特征相对 T2-v1 有效，但伪风险排序不够可靠；强制风险单调和周期级
  路由监督都使宏平均更差。按规则停止，未访问 test，A1/RCRF+NLinear incumbent 不变。
- 三专家 validation 优势：ETTm2 的轨迹专家四个 24 步段都第一，领先第二名 10.7%–29.8%；
  ETTh2 的周期间专家在 1–24 领先 23.9%，且高 lag-24/低形状创新区间胜率显著提高；Weather
  和 Electricity 的较远区间更多由相位专家占优。共得到 48 个满足 n、lift 和 bootstrap CI
  约束的优势区间，但 R0 的滚动风险首选命中率仅约 30.7%–41.8%。
- 审计：`scripts/analyze_triaxis_rolling_calibration.py` 在 validation 上复算 A1/T2-v1/R0 指标，
  误差均 `<1e-5`；本地 `research_runs/triaxis_rolling_calibration_v2/` 含 1,022,522 条
  sample×channel 记录、9 个程序化去重案例、7 张中文图和已校验 ZIP。该目录被忽略，不提交
  426 MiB 的样本 CSV 或图片；代码与数值结论写入仓库文档。
- 关键命令：`python scripts/search_phaseformer.py ... --mechanism
  <triaxis_rolling_features|triaxis_rolling_prior|triaxis_rolling_calibrated> --lookback 720 --horizon 96
  --percent 30 --max-epochs 8 --seed 2021 --loss huber`；审计命令：
  `python scripts/analyze_triaxis_rolling_calibration.py`（RTX 4090，torch 2.4.1+cu121）。

## 2026-08-29 — M3 相对原始 PhaseFormer 的成功/失败样本审计

- 在查看配对预测前固定判据：sample×channel 相对 MSE ≤-10% 且 MAE 同时下降为成功，
  相对 MSE ≥+10% 且 MAE 同时上升为失败；案例按绝对 MSE 差排序，并以 96 个窗口去重。
- 补跑 ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity 的同协议 original：L720/H96、
  30% train、8 epoch、seed 2021、Huber、validation only。该 matched rerun 只用于协议内
  诊断，不替代 Golden。
- M3 在六个 validation setting 的 MSE/MAE 均低于 original；MSE 相对变化为 -40.62%、
  -36.41%、-19.52%、-13.24%、-8.04%、-5.32%。块长 96、1000 次 block bootstrap 的
  MSE 区间均低于 0，但 M3 已经由同一 validation 选择，不能解释为独立测试显著性。
- 回放 1,121,992 个样本×通道，成功 38.00%、失败 15.29%。强周期+近期漂移组的六数据集
  宏平均 MSE 为 -14.35%、成功率 60.27%，但“其他”组也为 -13.12%；全部输入特征的成功/
  失败宏平均 |SMD|≤0.095，未形成可靠逐样本适用域。
- 事后未来水平迁移 SMD 为 -0.199、未来 lag-24 相关为 +0.106，提示未预见状态切换和稀疏
  假周期是失败边界，但不能做因果解释。论文加入三张中文图和六个程序化极端案例；图片共约
  636 KiB，大型 CSV/checkpoint 留在忽略目录。
- 代码与协议：`scripts/analyze_m3_vs_original.py`、
  `docs/PhaseFormer_M3_vs_original_analysis_protocol.md`；本地严格审计：
  `research_runs/m3_vs_original_phaseformer_v1/`。回放聚合指标与日志最大绝对差
  `1.21e-5`（阈值 `2e-5`），test 字段均为空。

## 2026-08-29 — PCTF 相位—周期—轨迹统一模型实现（等待实验确认）

- 根据正式 test 中 A1/A2 的轨迹稳定性、I0 的跨周期优势与 ETTh2 最坏回退，以及 HPTC 的
  validation 失败边界，实现单 checkpoint 的 PCTF；未采用三个完整模型 routing/ensemble。
- NLinear 预测完整轨迹；no-PE ICPT 仅贡献 `ICPT-NLinear` 的逐周期零均值形状，以及全
  horizon 均值守恒的周期间相对水平。两个修正相互正交，per-cycle gate 后再次投影，确保
  实际输出仍由 NLinear 独占 horizon-wide 绝对水平。
- 提供 shape-only、level-only、dual-fixed、masked-absolute、masked-regret 五个 preset；
  masked evidence 用最近两个严格历史伪起点，只连续收缩修正，不做专家选择或置信度反传。
- 预注册六数据集 H96 validation-only 48-run 筛选；只有通过 A2 平均/覆盖/最坏回退和三参考
  模型包络门槛后，才允许冻结冠军进入 H96/H192、三 seed、144-run 正式 test。汇总器检测到
  validation 结果中存在 test 数值会拒绝继续。
- 完整计划和空结果表在 `docs/PhaseFormer_pctf_experiment.md`。本轮只完成实现、结构测试与
  dry-run 校验；全仓 208 tests 和 187 subtests 通过。没有启动训练或读取新 test 结果，等待
  用户确认公式和实验范围。

## 2026-08-29 — PCTF 多融合策略代码与实验协议

- 固定一个 PhaseFormer、一个 NLinear 和一个 no-PE ICPT，实现分量标量/逐周期融合、单调
  历史证据、证据 MLP、相位模板调制七个新 preset；完整预测均匀平均和 Softmax 仅为不可晋级
  负对照，不是论文候选。
- F1/F2 将预测正交分为 horizon 绝对均值、周期间零均值水平和周期内零均值形状；NLinear
  独占绝对均值。F3 以24个可微 circular shift、受限幅度和零均值形变让 ICPT 调制 PhaseFormer
  模板。A1 高频校准对新模型改为融合前只处理相位分量，旧 preset 路径不变。
- 实验 runner 预注册六数据集 H96、A1/A2/I0+八融合策略的66-run validation-only筛选；论文
  候选还必须优于两个负对照包络，才可冻结进入H96/H192、三seed、144-run正式确认。
- 验证：223 tests和229 subtests通过；七策略有限值forward/backward、结构约束、单调方向、
  严格历史因果、共享初始化、负对照不可晋级、test泄漏拒绝及66/144 dry-run均通过。ETTm2-H96
  参数量为96,063–96,335，对比A1的72,905。没有启动训练或读取新的test结果。
- 方案、公式、命令及空结果表：`docs/PhaseFormer_pctf_fusion_strategies.md`；runner：
  `scripts/run_pctf_fusion_strategies.py`。

## 2026-08-29 — PCTF v1 失败诊断与 A2 锚定式 v2 修复

- v1 的66-run validation 结果表明 F1/F2/F3 相对 A2 宏平均退化约2.5%–3.3%。代码诊断确认
  旧分量公式删除 NLinear 周期内形状，任何 gate 都不能还原 A2；F2 又用 ICPT-vs-NLinear
  shape regret 控制 ICPT-vs-PhaseFormer shape，证据对象不匹配。checkpoint 审计显示大多数
  gate 仍接近初值；环境审计另发现55次 CUDA、11次 CPU，F0 的亚千分位差异不可作提升结论。
- 新增单 checkpoint、端到端 `AnchoredPhaseCycleFusionComposer`：完整保留 A2 的 PhaseFormer、
  LFF-NLinear、RCRF 和输出校准，只添加 `L_C-L_T` 与 `S_C-S_P` 两个正交创新。系数使用
  有界 tanh 且严格零初始化，因此同 seed 候选的全部 A2 state tensor 和初始输出逐点等于
  独立 A2；不是多 checkpoint ensemble，也不冻结锚点。
- 修复历史证据：多个严格因果、horizon-matched rolling origins 分别比较 ICPT-vs-phase
  template 的 shape 和 ICPT-vs-LFF trajectory 的 level，保留有符号 log regret，并输出逐未来
  周期置信度。PhaseFormer period 固定24，ICPT period 独立；period96 会因果截取720输入的
  最近672步完整周期。
- 零校正首批没有 ICPT 主损失梯度，因此加入只训练 ICPT 的 shape/level 组件辅助损失；
  validation 和 checkpoint 仍只按最终预测选择。新增 scalar/cycle、单调证据、证据 MLP、
  phase modulation 五个论文候选及 shape-only/level-only 消融。
- 新 runner 预注册48-run period选择、132-run H96/H192 strategy筛选和通过门槛后的144-run
  三seed test；训练命令强制 CUDA，汇总器拒绝混合硬件/软件、选择阶段 test 泄漏、非零 A2
  identity、重复或缺失矩阵。
- 验证：全仓 `241 passed`，另有 `250 subtests passed`；七种策略完整 forward、A2 exact
  identity、period24/48/96、正交/均值约束、严格因果、辅助梯度、48/132/144 dry-run 和合成
  汇总均通过。仅运行代码测试与 dry-run，未启动训练或读取新 validation/test。
- 完整计划与空表：`docs/PhaseFormer_pctf_anchor_fusion_retest.md`；runner：
  `scripts/run_pctf_anchor_fusion_retest.py`。
## 2026-08-30 — 执行 PCTF v2 复测（阶段性）

- Stage P 完成 48/48 个 validation-only runs，冻结周期：ETTh1/ETTh2=48，ETTm1=48，ETTm2=96，Weather=24，Electricity=12。
- Stage S 完成 111/132 个 validation-only runs；因长任务会话产生孤儿进程并触发 CUDA OOM/落盘竞态，剩余矩阵未闭合。
- 未执行 Stage F，未读取 test；不能据此声明候选优于 A2 或 Golden。
- 结果说明：[docs/PhaseFormer_pctf_anchor_fusion_results.md](PhaseFormer_pctf_anchor_fusion_results.md)。
## 2026-08-30 — 完成 PCTF v2 Stage S 复测

- Stage S 已完成 132/132 个 validation-only runs；环境统一为 RTX 4090、PyTorch 2.7.1+cu126、CUDA 12.6、Lightning 2.6.5。
- 所有论文候选均未通过联合门槛。最佳为 `pctf_anchor_mlp`：宏平均/A2=1.001131，4/12 个 setting 双指标改善，最差比=1.021385，参考包络比=1.007364。
- MLP 相对 component-cycle 的嵌套对照为 0.999924，9/12 个 setting 双指标改善，但最差比=1.010829，仍不稳定。
- 按预注册协议阻断 Stage F；没有读取 test。详情见 `docs/PhaseFormer_pctf_anchor_fusion_results.md`。

## 2026-08-30 — PCTF v3 锚点漂移归因与修复代码（未运行实验）

- 将上一轮失败拆成四个可证伪来源：联合训练导致内部 A2 漂移、ICPT 绝对目标与增量职责错配、
  evidence gate 缺少边际收益监督、`H=period` 时 horizon-centered level 恒为零。
- 新增五个诊断/修复 preset：冻结锚点绝对目标、冻结锚点残差目标、锚点安全联合残差、边际
  gate 监督和完整单周期 level 修复。论文候选仍是单 checkpoint 的端到端联合模型；冻结模式
  仅用于归因，不作为 ensemble 或最终方法。
- `scripts/search_phaseformer.py` 支持从 matched A2 checkpoint 严格子集初始化、按需冻结锚点，
  并记录内部 anchor 误差、fused/anchor 比、修正 RMS 与 gate/真实收益相关性。
- 新增 `scripts/run_pctf_anchor_attribution.py`：6 settings×2 seeds 的12个 A2 与72个候选，强制
  CUDA、validation-only，汇总时拒绝 test 泄漏、环境混用、缺失/重复和初始锚点不一致。
- 计划、公式、判据和待填表见 `docs/PhaseFormer_pctf_anchor_attribution_plan.md`。本轮按用户要求
  没有启动训练或读取新 validation/test；只执行语法检查、dry-run 和单元测试。

## 2026-08-30 — PCTF v3 实验启动受 GPU 不可用阻塞

- 按计划尝试启动 v3 validation-only 矩阵前，沙盒与提权环境的 `nvidia-smi` 均无法取得设备：
  前者为 driver communication failure，后者为 `No devices were found`。
- `research_runs/pctf_anchor_attribution_v3/` 尚不存在，没有任何部分结果可整理；未改用 CPU，
  因为协议要求所有配对任务强制 CUDA，改用 CPU 会破坏公平性。
- 待 RTX 4090/CUDA 恢复后，按 `docs/PhaseFormer_pctf_anchor_attribution_plan.md` 的
  `anchors → candidates → summarize` 顺序续跑；结果表目前仍全部待填。

## 2026-08-30 — PCTF v3 复测完成

- RTX 4090 恢复后完成 12 个 A2 锚点与 72 个候选 validation-only 运行；补跑了唯一中断的
  Electricity H96 / seed 2022 / repair_full 任务。
- 汇总结果写入 `research_runs/pctf_anchor_attribution_v3/`，冻结控制的数值等价判定改用 `1e-6`
  容差，以覆盖浮点归约误差。
- 详细分析见 `docs/PhaseFormer_pctf_anchor_attribution_results.md`。候选宏观收益约 0.64%，但
  最差设置退化约 1.25%，未通过预正式门槛，未读取 test 指标。

## 2026-08-30 — 预注册 PCTF ETTh2/ETTm2 正式 test

- 用户明确授权将 validation 冻结的 `pctf_anchor_repair_full` 与 A2 在 ETTh2/ETTm2、L720、
  H96/H192 上进行 full-train、三 seed test，并与固定 Golden 同表比较。
- 新增 `scripts/run_pctf_anchor_formal_etts.py`：先训练12个 matched A2，再由对应 checkpoint
  初始化12个候选；全部最多30 epoch、Huber、best-validation checkpoint、强制 CUDA。
- 预注册局部门槛、额外微调成本和后续 test-set selection 边界记录在
  `docs/PhaseFormer_pctf_anchor_formal_etts.md`；提交代码后再启动正式实验。

## 2026-08-30 — PCTF Full Repair 正式 test 完成

- 在实验冻结提交 `c8b61c4` 上完成 ETTh2/ETTm2、L720→H96/H192、三 seed 的 12 个 A2 与
  12 个 Full Repair 正式 test；全部为同一 RTX 4090/CUDA/软件环境，候选均从同 setting/seed
  的 A2 best-validation checkpoint 精确初始化。
- Full Repair 相对 A2 宏平均降低 0.772% MSE、0.507% MAE，3/4 setting 双指标改善，最坏为
  ETTm2-H192 MSE 回退 0.203%；通过预注册的两数据集局部替换门槛。严格稳定低于 Golden 的
  setting 数为候选 4/4、A2 2/4。
- 归因审计显示，候选内部继续训练后的 A2 相对原始 A2 平均改善约 0.599% MSE、0.230% MAE；
  ICPT 融合相对内部 A2 再改善约 0.174% MSE、0.278% MAE。结果证明完整流程有效，但额外训练
  与结构贡献尚未由 continued-A2 等预算对照完全分离。
- 完整结果、成本和 test-set selection 边界见
  `docs/PhaseFormer_pctf_anchor_formal_etts.md`。

## 2026-08-31 — 预注册 PCTF 单阶段联合训练

- 用户要求取消“A2 预训练→Full Repair 微调”的两阶段流程。结构保持不变，新增 checkpoint
  持久化 correction warm-up：所有分支从 epoch 0 同时训练，只有 ICPT 对最终输出的影响在前
  5 epoch 从 0 平滑升至 1；不加载中间 A2 checkpoint。
- 预注册六个训练策略，分离 A2 主干 0.1×/1.0× 学习率、0/0.25/1.0 锚点保护损失和 warm-up。
  先运行 ETTh2/ETTm2、H96/H192、两 seed 的56-run validation-only 筛选；通过门槛后才运行
  24-run 三 seed 正式 test。
- GPU smoke test 已在 RTX 4090 上通过：单次训练无初始化 checkpoint，epoch 0 correction
  scale=0，内部 A2 与最终输出严格一致；未读取 test。协议和待填表见
  `docs/PhaseFormer_pctf_single_stage_training.md`。

## 2026-09-01 — Strict T28 Golden 搜索验收覆盖修复

- `scripts/verify_strict_t28_golden_goal.py` 现纳入校准精修 ledger；因此四阶段任一候选都必须以同一
  dataset 内共享配置同时通过 H96/H192 的 MSE、MAE 四项门槛，才可报告目标达成。
- 同步补充 `docs/PhaseFormer_strict_t28_ett_golden_hunt.md` 的第四阶段定义和验收规则；已使用
  conda `raft` Python 完成 `py_compile` 与 `git diff --check`，未读取训练中实验的新结果。

## 2026-09-02 — Strict T28 horizon 定向精修

- 共享配置的 broad、参数、loss、校准四阶段已穷尽，严格共享验收未通过。汇总显示明确的 horizon 分化：
  ETTm1-H96 已有单项通过，而 H192 的最佳 MSE 仅差 0.132pp；ETTh1 两 horizon 的主要瓶颈均为 MAE。
- 新增可恢复的 `scripts/run_strict_t28_golden_horizon_refinement.py` 和独立四-setting 验收器。它只解除
  “同一超参数同时覆盖 H96/H192”的附加约束，不改变 strict-T28 拓扑，并将每项完整训练/test 选择轨迹写入
  单一 horizon ledger。计划与 test-set-selection 边界已写入
  `docs/PhaseFormer_strict_t28_ett_golden_hunt.md`。
- 已在 conda `raft` 下完成两个脚本的 `py_compile`、三组非空 dry-run、空网格短路与 diff 检查；尚未启动
  第五阶段训练。

## 2026-09-02 — Strict T28 最优共享配置长 horizon 扩展

- 用户要求把每个数据集当前最优的共享配置扩展至 H336/H720；固定 ETTh1 `u_lr020` 与 ETTm1
  `w_aux01`，不在长 horizon 重调参数。
- 新增 `scripts/run_strict_t28_best_long_horizons.py`，对四个 setting 完整训练并一次性读取 test，三次
  自动重试、`--resume` 和紧凑 CSV ledger 均已配置。协议、Golden 参照、待填表和复现命令见
  `docs/PhaseFormer_strict_t28_best_long_horizons.md`。
- 已在 conda `raft` 下通过 `py_compile`、4-command dry-run 和 `git diff --check`；随后启动持久 GPU 服务。
- 四项完整训练均完成：ETTh1-H720 以 `0.41424/0.44185` 相对 Golden 改善 3.888%/1.810%；ETTh1-H336
  仅 MSE 改善（MAE +0.456%），ETTm1-H336/H720 均仅 MAE 改善。原始精确数值在该实验的 CSV ledger，汇总表已填入
  `docs/PhaseFormer_strict_t28_best_long_horizons.md`。

## 2026-08-31 — PCTF 单阶段第一轮筛选与梯度解耦复测

- 在提交 `7cb64cc`、RTX 4090 上完成 8 个 matched A2 和 48 个 candidate 的 validation-only
  筛选；输出位于 ignored 目录 `research_runs/pctf_single_stage_training_v1/`，未读取 test。
- 六种策略全部未过门槛。`legacy_safe` 联合比 0.99875 但最坏退化 1.36%；统一 LR 中最好的
  `uniform_protected` 联合比 0.99919、内部 A2/A2 为 1.00142，说明修正能够改善内部锚点，
  但 fused loss 同时把锚点拉离独立 A2。warm-up 多次选中 correction scale<1 的 checkpoint。
- 据此实现 `decoupled_protected`：同一次前向和训练中，融合预测数值不变，fused loss 只更新
  ICPT/融合器，A2 仅由权重 1.0 的 anchor loss 更新。新增配置、runner 策略与梯度作用域测试；
  targeted 测试 21 passed / 31 subtests。为保证同 commit 配对，独立重跑 8 个 A2 和 8 个候选，
  再按原门槛决策；runner 的 `--policies` 参数用于限定该可复现复测，不改变模型配置。

## 2026-08-31 — PCTF 单阶段梯度解耦复测完成

- 在提交 `5bf0534`、RTX 4090 上完成独立目录中的 8 个 matched A2 与 8 个
  `decoupled_protected` validation-only 任务；汇总确认 test 指标未读取。
- 候选 MSE/A2=0.99914、MAE/A2=0.99902、联合比=0.99908、最坏比=1.00537、3/8 双改善，
  未通过原门槛，按协议不启动正式 test。内部 A2/A2=1.00066，fused/内部A2=0.99842；融合
  修正本身 8/8 改善 MAE、7/8 改善 MSE，但 best-fused 与 best-anchor epoch 不同步仍造成回退。
- 单阶段平均训练 22.98 秒，为 matched A2 的 1.90 倍；比历史两阶段 2.77–3.45 倍节省约
  32%–45%，但稳定精度不足。若强制一次训练，保留梯度解耦+1.0× LR+1.0 anchor loss 作为
  当前最合理配方；当前正式最佳仍是两阶段 Full Repair。完整表见
  `docs/PhaseFormer_pctf_single_stage_training.md`。

## 2026-09-02 — H1/H3/H4 输入成分干预流程实现与校验

- 实现 `src/dataset/input_component_ablation.py`：在 train-fitted scaling 后、模型 RevIN 前，仅对
  `seq_x` 执行 H1 同相位跨周期残差、H3 近期局部趋势和 H4 相位漂移的 `full/half_A/minus_A/sham`
  干预；稳定 seed 不依赖 Python hash，目标与时间标记不进入变换函数。
- 数据入口、单 run 搜索、2880-run Track R 去重矩阵、full-checkpoint 发现与 Track F 固定权重评估、
  配对 moving-block bootstrap、RCRF 相位/NLinear 分支诊断和 sham-adjusted interaction 汇总均已接通。
  正式阈值与复现命令见 `docs/PhaseFormer_input_component_H1_H3_H4_plan.md`。
- 校验：`.venv/bin/python -m pytest tests/ -q` 为 276 passed、262 subtests passed；新增合成测试覆盖
  H1/H3 重构与端点、H4 已知位移恢复/相位方差降低/实值能量守恒/不可辨识回退、确定性、目标隔离和
  RCRF 分支重建。`py_compile` 与 `git diff --check` 通过。
- 真实数据校验：ETTm2-H96 的 10 个输入条件均能生成 `(720,7)` 历史且 `(96,7)` target 与时间标记
  逐元素不变；三个假设的非 full 干预均非零且最后观测最大误差不超过 `4.45e-16`。CPU 环境完成
  `rcrf_nlinear_plain`、H4 `minus_A`、5% train、1 epoch 的 validation-only smoke（train 1597，
  val smoke-limit 32，未构造 test loader），并完成 original 的受限 test/frozen/summarizer 链路。
- 所有 smoke 产物位于 `/tmp`，不是正式 benchmark，不支持任何 H1/H3/H4 有效性结论。当前 `.venv`
  的 PyTorch 为 CPU build（尽管机器存在 RTX 4090）；正式 2880-run 矩阵前必须切换 CUDA 环境并
  记录单 run 成本，且 `--max-eval-samples/--max-samples` 必须保持默认 0。

## 2026-09-02 — H1/H3/H4 正式测试分离、CI 与 RCRF 反事实补全

- 按复核结论保留 H3 极小非零分量和 H4 固定 Nyquist 的实现，只把实验文档修正为真实算法边界；
  文档现明确披露此前32-window test smoke、Nyquist 约定及 residual probe/样本报告未自动化。
- Track R 训练入口改为严格 validation-only；新增独立的 retrained checkpoint test 与2592-run
  非 full 矩阵驱动。288个共享 `none/full` 结果由 Track F 一次生成并复用，避免重复读取同一
  checkpoint×input-condition。
- fixed/retrained 两条轨道均保存逐 origin MSE 与 MAE；汇总器计算 MSE/MAE 的绝对及相对效应
  moving-block CI，并联合审计 frozen/retrain 两轨、每轨288个 setting 和每 setting 10个条件。
- `rcrf_nlinear_plain` 的固定权重评估已流式实现四类反事实：variant branches+full gate、variant
  gate+full branches、phase-only variant、NLinear-only variant；每个条件强制 fused 重建误差
  `<2e-5`，不保存全量巨大分支张量。
- 正式入口默认要求 CUDA、完整 checkpoint 数量、100% train、零样本上限、源训练结果无 test；
  smoke/CPU/不完整汇总必须显式 opt-in，已有结果默认拒绝覆盖。ETTm2-H96 的128-window RCRF
  smoke 验证四类反事实最大重建误差不超过 `4.77e-7`，MSE/MAE 相对 CI 均成功生成；该结果不用于
  效果结论。完整测试为280 passed、262 subtests passed。

## 2026-09-02 — H1/H3/H4 正式矩阵启动

- 在 commit `963a0f7`、主机 RTX 4090、`/home/wangjing/miniconda3/envs/raft`（PyTorch
  2.4.1+cu121、PyTorch Lightning 2.5.6）上启动完整 2880-run validation-only Track R；
  输出目录为 ignored 的 `research_runs/input_components_h134_scratch/`，使用 `--resume`。
- Track R 完成后自动串行启动 288-run Track F frozen 评估，以及 2592-run non-full Track R
  retrained test 评估；三阶段命令及参数与实验文档一致，正式运行未启用 smoke 或 CPU 选项。
- 实验主进程 PID 为 `150545`，标准输出/错误暂存于 `research_runs/input_components_h134_control/`；
  独立监控进程 PID 为 `162217`，使用 `sleep 1800` 每 30 分钟记录三阶段已完成文件数、GPU
  利用率和主进程状态至同一控制目录。当前仅有启动阶段计数，尚无正式效果结论。

## 2026-09-02 — 暂停正式矩阵并调整优先级

- 按用户要求终止 Track R 主进程组和监控进程组，保留已生成的 validation-only checkpoint 与指标；
  当前没有 Track F 或 retrained test 结果被写入。
- 将主日志、进度日志和监控启动记录从 `/tmp` 移至
  `research_runs/input_components_h134_control/`，后续实验相关日志不得写入 `/tmp`。
- 在三个矩阵驱动脚本中加入默认的 priority-first 调度：先运行单个 seed `2021`、
  `horizon=192`；正式优先阶段命令覆盖8数据集×3模型×10条件，共240个 Track R 任务。通过
  validation 审计后再用 `--resume` 扩展其余 horizon 和 seed；如需旧顺序，显式使用
  `--no-priority-first`。
- 校验：三个脚本的 `--help`、优先阶段命令计数、Python 编译检查和 `git diff --check` 通过；
  尚未恢复实验，也未形成效果结论。

## 2026-09-02 — 实验文档修订为“决策范围优先”（v1.1）并恢复 D0 Track R

- 按用户要求把 `docs/PhaseFormer_input_component_H1_H3_H4_plan.md` 从“优先阶段只读
  validation 前置”改为“决策范围优先（v1.1）”：`horizon=192, seed=2021`（记 D0，8 数据集 × 3
  模型 × 10 输入条件 = 240 个 Track R）先完整走完 Track R(validation) → validation 审计 →
  Track F（24 个 full 锚点 ×10）→ retrained test（216 非 full）→ D0 汇总，先形成单 seed
  h192 结论（provisional）；其余 horizon×seed 作为 D1（2640 个 Track R）在 D0 结论形成后按
  相同冻结协议补跑并入三 seed 宏平均。test 唯一单元 456（D0）+ 5016（D1）= 5472。
- 主要修改文件：`docs/PhaseFormer_input_component_H1_H3_H4_plan.md`（状态块、§2.1、§7.2
  Stage 3a/3a-F/3b/3b-F、§7.3 D0/D1 命令与“实现说明”、§8.2/§8.3 provisional 语义、新增
  §13.0 D0 表、§14 D0 优先冻结说明）。本修订不改任何提取公式、模型、超参、QC 阈值或判定门槛。
- 实验恢复：并行分阶段启动器 `scripts/run_input_components_parallel.py`
  （`--gpus 2,3 --jobs-per-gpu 4 --max-stage 3`，supervisor PID 记录于
  `research_runs/input_components_h134_control/parallel_supervisor.pid`）在 GPU2/3 恢复 D0
  Track R；产物目录（gitignored）：`research_runs/input_components_h134_scratch/runs/*/metrics.csv`、
  控制目录 `research_runs/input_components_h134_control/`（done.tsv、supervisor.json、jobs/）。
- 校验：D0 范围 validation 审计（无 test 泄漏、config_hash 无重复、健康度正常）已覆盖已完成
  run；只读，不形成效果结论。尚未读取 test。
- 已知后续（必做项，须在 D0 Track R 收尾前落地，见文档 §7.3“实现说明”）：为
  `run_input_component_frozen_matrix.py`、`run_input_component_retrained_test_matrix.py` 与
  `summarize_input_component_ablation.py` 增加 `--horizons/--seeds`（或 `--scope d0|all`）
  范围过滤，`expected-count` 由范围推导，D0 下游产物写独立 `*_d0` 目录。

## 2026-09-02 — D0 范围过滤落地到三个下游 runner

- 为决策范围（D0/D1/全矩阵）给 `scripts/run_input_component_frozen_matrix.py`、
  `scripts/run_input_component_retrained_test_matrix.py` 与
  `scripts/summarize_input_component_ablation.py` 增加 `--horizons/--seeds` 范围过滤；
  `--expected-count` / `--expected-settings-per-track` 由范围自动推导（共享
  `scripts/run_input_component_ablation.py` 新增的 `parse_scope()` 与
  `expected_full_anchors()`：D0=24 锚点/216 retrained，全矩阵默认=288/2592）。
  重复 checkpoint 检测仍保持全局（源内任何重复都拒绝）；完整性/无泄漏/percent 门只校验范围内
  行，范围外未完成的 D1 条件不会阻塞 D0 读取。不带过滤参数即为全矩阵，行为与原来一致。
- 同步修订实验文档 §7.3：D0 汇总命令改为 `--horizons 192 --seeds 2021`；“实现说明”改为已落地
  措辞。
- 校验：四脚本 py_compile、模块 import、`git diff --check` 通过；对真实
  `research_runs/input_components_h134_scratch` 验证——D0 冻结 smoke 列出 19/24 个 full 锚点
  （Track R 仍在跑）、正式门按范围推导报 `expected 24, found 19`；retrained D0 门先由完整性检查
  正确拦截（Traffic 条件未收尾）；默认无参数门仍为 288；`--horizons 192,999` 被 parse_scope
  拒绝。未读取 test。

## 2026-09-02 — v1.2 修订落地：Traffic 剔除 + D0→D1 串行编排（自动恢复）

- 数据范围修订（v1.2）：为加快结论产出，将 **Traffic** 从执行与评估矩阵剔除（不再调度训练、
  Track F / retrained test / 汇总全部跳过）。执行范围收敛为 7 个数据集（ETTh1、ETTh2、ETTm1、
  ETTm2、Exchange、Weather、Electricity）。实现方式为新增 `--datasets` 白名单过滤而非改
  `DATASETS` 常量：给 `scripts/run_input_component_ablation.py`（`parse_dataset_scope()` +
  `expected_full_anchors(..., datasets=None)`）、`run_input_component_frozen_matrix.py`、
  `run_input_component_retrained_test_matrix.py`、`summarize_input_component_ablation.py`
  全部加 `--datasets`，范围期望计数（锚点/retrained）由白名单自动推导，故 Traffic 可随时加回。
  已完成的 6 个 Traffic D0 run 与进行中的 Traffic 训练就此搁置（其 run dir 与 done.tsv keys
  被 scope 过滤忽略，不删产物），不参与任何 test 读。v1.2 修订仅为范围/资源管理决定，不改任何
  提取公式、模型、超参、QC 阈值或判定门槛。
- 编排决策改为**串行**（因两个调度器各自激活 GPU 后永久占满 slots、绝不 yield，不能安全共享
  同一 GPU）：先跑完 D0 Track R（210 validation-only runs，7 数据集 h192×seed2021）→ 调度器以
  `--max-stage 1` 停在 stage1 → D0 下游编排器独占 GPU2/3 跑 audit→Track F(21 锚点)→
  retrained(189)→汇总 → provisional 结论 → 再恢复 D1 训练（`--max-stage 3` 全矩阵）。
- 新增 `scripts/run_d0_downstream.py`（563 行）：D0 下游自动编排器，状态机
  `wait_3a → track_f → retrained → summarize → done`（失败态 *_failed）。wait_3a 复用
  frozen/retrained 两个矩阵 runner 的 print-only 门做就绪探测（rc=0 才推进，否则 60s 再探）；
  audit 复用 discover() 权威 coverage/leak/percent/dup 门 + 锚点唯一性检查；track_f/retrained
  阶段用 `Dispatcher` 做 GPU 占用与崩溃安全重试（子进程 --resume）；summarize 成功且无 err 后
  `relaunch_supervisor()`（`--resume-supervisor-argv`）重启 D1 supervisor 再落 done。
- 实际操作（control 目录 `research_runs/input_components_h134_control/`，gitignored）：终止原
  `--max-stage 3` 调度器进程组（-575623，确认进程组清空、GPU2/3 空闲）；以 `trackr_d0_argv.json`
  （`--max-stage 1`、7 数据集、jobs-per-gpu 4、min-free 5000）重启 D0 supervisor →
  **PID 607551**（stage 1: 210 jobs, 180 done, 30 Weather pending，8 个 Weather run 已上
  GPU2/3）；后台启动编排器（`orchestrator_argv.json`，--probe-sec 60，
  --resume-supervisor-argv=trackr_d1_argv.json）→ **PID 609856**，d0_state=wait_3a（frozen/
  retrained 门 rc=2 = 未收尾，60s 周期重探）。编排器逻辑已在启动前核对：Weather 30 未完成期间
  只会停在 wait_3a，不会提前推进。
- 校验：五个改动脚本 + 新编排器 py_compile/模块 import 通过；plan doc 状态块与 §2.1 计数已更新
  为 v1.2（7 数据集：D0 210/21/189；D1 全 2520/252/2268）。未读取 test。

## 2026-09-03 — 修复 D0 audit 的 retrained 计数口径（210 vs 189 误报）

- 现象：D0 Track R 全部 210 完成后，编排器在 wait_3a→audit 停在终端态 `audit_failed`：
  `full_anchors=21 ✓` 但 `retrained_checkpoints=210 ≠ expected 189`，管道停摆。
- 根因：`run_d0_audit` 用 `len(retrained_discover(...))` 作为 retrained 计数，但该 discover
  （`run_input_component_retrained_test_matrix.py`）作为完整性门返回**全部条件行**（含 21 个
  none/full 锚点），其自身 main() 在计数前会先滤掉 none/full（只读 9 个干预变体）。audit 少了
  这一步过滤，故 210 被拿去比 189。先前 6 数据集 probe 显示 162/189 是因为当时 6×27=162 个非
  full 行恰好等于过滤后的期望，掩盖了该口径问题；7 数据集全完成后 210>189 才暴露。
- 修复：`run_d0_downstream.py` 的 `run_d0_audit` 在计数前加与 runner main() 相同的
  `~(input_hypothesis=='none' & input_variant=='full')` 过滤，再比 `expected_retrained`。
- 校验：直接对真实 `research_runs/input_components_h134_scratch` 调 `run_d0_audit(...)` →
  `PASSED: True`（full_anchors=21、retrained=189、anchors_per_setting 1/1、anchor_path_dups=0）。
  `input_components_h134_frozen_d0/` 与 `input_components_h134_retrained_test_d0/` 均空（0 文件），
  无半成品需清理。状态文件 d0_state.json 重置为 wait_3a，编排器按原 argv 重新 detach 启动。
  未读取 test。

## 2026-09-03 — D0 阶段性汇报文档（H1/H3/H4）

- 新增 `docs/PhaseFormer_input_component_H1_H3_H4_stage_report_D0.md`：把已完成的 D0 全链路
  （Track R 210 → audit → Track F 210 读 → retrained 189 → 汇总 420 行）整理为对计划文档的阶段
  性汇报；所有数值由 `result_summary_d0.csv` 长表按 §8.1 定义重算（Δ=变体相对自身 full 基线的
  宏平均，Interaction=逐 setting 配对差），一律标注 `provisional (seed2021 only)`。
- 关键修订：先前口径把 frozen 的 h1-sham 之类数字当成「~1.72」小量，实为存储列=小数（0.6875
  =68.7%）；本汇报按真实百分比重算。D0 宏观事实：M0 对 H1/H3/H4 的 minus_A 等效门槛全不成立
  （retrain 2.1–84%，frozen 9.9–86%）；三模型对输入扰动普遍极敏感且 frozen 侧 `sham ≥ minus_A`
  几乎处处成立、retrain 侧 H3/H4 `sham≈minus_A`——D0 provisional 判定：三假设均达不到
  Strong/Partial，证据形态 OOD/confounded + 近 null 混合，不宣告 Rejected，等 D1 三 seed。
- 遗留标记（写入文档 §8，不在本次静默改计划）：① summarize 的 aggregate interaction 列口径与
  §8.1 不符（frozen H1 minus M1−M0 长表重算 +2.4 pp vs 该列 +36.1 pp），D1 汇总前需修；② 计划
  §7.2/§7.3/§13.0 正文仍是 v1.1 的 8 数据集/24 锚点/216/456 计数，与 v1.2 实际（7/21/189/399）
  不一致；③ selection_source 列约 55% 空。

## 2026-09-03 — ETTm1-H192 输入盲区候选发现方案

- 新增 `docs/PhaseFormer_input_candidate_discovery_ETTm1_H192_plan.md`，把后续工作独立为新的
  validation-only 候选发现实验；范围固定 ETTm1、h192、seed2021，不读取新的 test，也不回写
  已经发生 test exposure 的 H1/H3/H4 D0 结论。
- 候选库改为与 PhaseFormer 的24步 phase folding/低维投影和 NLinear 完整时间轴差异直接对应的
  六类方向：96步日周期增量、672步周内低频、非24整齐频带、周期边界连续性、周期间幅度包络、
  平滑相位速度。
- 筛选先做连续序列上的 train-fitted 分解和严格 real/sham 分布 QC，再以512个 validation origins
  做冻结低剂量筛选、PF残差 cross-fitted ridge probe 与 RCRF 四类分支反事实；至多3个进入全
  validation，至多2个进入重训。
- 从零开始的训练上限为21 runs（3个 full 锚点 + 最多18个候选重训），所有日志与监控只写
  `research_runs/`。若没有候选通过 sham-adjusted Interaction 与分支证据门槛，明确报告未找到，
  不按排名强行选择。当前仅完成方案，尚未实现或运行。

## 2026-09-03 — 候选发现方案加入近程依赖并冻结确认协议

- 将“输入尾部近期创新”登记为 C7：先用仅在 train 拟合的因果一步预测器生成连续创新，再对每个
  输入窗口最后24步施加固定余弦支撑，避免把粗暴删除尾部造成的断点误认成近程信息；sham 使用
  train 内按时刻、前缀状态和波动匹配的连续残差块，保留跨变量同步与块内顺序。
- C7 的主终点预注册为预测步1–24的 MSE/MAE，另报25–48、49–96、97–192；只有效应集中在近程且
  随预测距离总体不增强，才能解释为近程依赖。该设计与 `WeakPeriodResidualHead` 的最后值锚定和
  完整时间轴线性映射直接对应。
- 按用户决策保留 `remove_025/remove_050/sham_025/sham_050` 主干预；候选仅在 validation 发现并
  冻结，之后一次性读取 ETTm1-H192 test 确认。因该 test 已被旧 D0 暴露，结论必须标为
  `test-set-exposed confirmation`，不能称为盲测；两个候选同时确认时使用 Holm 校正。
- “增强分支正在利用 A”的必要机制证据改为损失反事实：固定 full 输入下的 PhaseFormer 输出与
  fusion gate，只替换成干预输入下的 NLinear 输出，要求重组预测的 MSE/MAE 显著上升；仅有分支
  数值敏感或 gate 变化不算实际利用。筛选条件读由6候选75次更新为7候选87次。当前仍只修订方案，
  未实现、未训练、未为本方案读取 test。

## 2026-09-03 — ETTm1-H192 C1--C7 候选发现：S1 早停

- 新增独立连续候选层 `src/dataset/input_candidate_discovery.py` 与冻结 runner
  `scripts/run_input_candidate_discovery_frozen.py`：所有 C1--C6 在连续、按训练集 scaler 缩放的 ETTm1
  序列上构造后切窗；C7 先通过 train-fitted 因果三滞后预测器得到连续创新，再对每个 origin 固定支撑
  到最后24步。S1 runner 同时输出全窗口及1–24/25–48/49–96/97–192指标、样本级配对误差、移动块CI和
  RCRF 四类重组（含固定 full phase/gate、只替换 NLinear 的损失反事实）。
- 环境：`/home/wangjing/miniconda3/envs/raft/bin/python`（torch 2.4.1+cu121），RTX 4090。完成三项
  ETTm1-H192 seed2021 full-input 锚点训练（original/weak_residual/rcrf_nlinear_plain，各30 epoch），
  然后运行 S1a 512 origins（7×4×3+3=87 条件读）和 S1b 全 validation 11,329 origins（C2/C3/C7）。
  本实验没有读取 test。
- 结果：C2 出现显著的 sham 更有害模式；C3 对 weak_residual 的差异不足预注册效应和CI门；C7 的近程
  响应不满足“PhaseFormer等效不敏感、增强模型更依赖”的方向。无候选通过S1，依 §6 早停，未重训候选、
  未读取 ETTm1-H192 test。
- 用 `scripts/package_input_candidate_discovery_s1.py` 生成严格审计包
  `research_runs/input_candidate_discovery_ettm1_h192_v1/`（6文件+figures，123条完整搜索结果、11,329条
  样本级诊断行、15个程序化案例和引用图片 ZIP）。已验证目录白名单、结果/案例对齐、Markdown图片和 ZIP
  原件字节一致；scratch checkpoint、日志与监控保留在 `research_runs/..._scratch`/`..._control`（gitignore）。

## 2026-09-03 — D1 频谱周期与 D2 近期创新 remove-only 筛查

- 按用户要求新增 `SpectralRemoveBank` 和 `scripts/run_d1_d2_remove_screen.py`；该轮没有 sham。D1 先只在
  ETTm1 train（34,560步）上聚合多通道 periodogram，再固定6个峰（96、48、32、24、677.647、205.714步），
  用 train-fitted 连续谐波回归删除。D2 用既有 train-fitted 因果创新，分别完整删除窗口末尾24/48/96/192步。
- 用已完成的三项 full anchor 在全 validation 做冻结评估，未读 test、未训练新模型。D1-1（96步日周期）
  对三模型影响最大（MAE +13.28% original / +14.31% weak / +13.67% RCRF）；48步次之，约678步和206步
  接近零。D2 删除长度越长，三者均单调退化，但 original 在四种长度均不低于增强模型的敏感度。
- 结论限定为 remove 敏感性：96步周期和近期创新是共同有用的信息；没有出现原版忽略而增强模型依赖的
  明确模式。原始 CSV 与协议位于 `research_runs/input_candidate_discovery_ettm1_h192_v1_scratch/d1_d2_remove/`，
  GPU日志位于对应 control 目录，均不提交。

## 2026-09-03 — D1/D2 remove-trained 训练阶段对照

- 按用户修正，新增 `d1`/`d2` 数据入口与 `scripts/run_d1_d2_retrained_remove.py`：每个条件在训练、
  validation 均移除相同 A、目标不变，再从头训练 original/weak_residual/rcrf_nlinear_plain；不再用冻结
  删除的即时反应代替训练后利用判断。新增汇总脚本 `scripts/summarize_d1_d2_retrained_remove.py`。
- 在 RTX 4090、`/home/wangjing/miniconda3/envs/raft/bin/python`（raft）完成 ETTm1 H192 seed2021 的
  30/30 个30-epoch上限任务，未读取 test。主结果写在计划§11，原始运行、逐任务日志和汇总均位于
  `research_runs/d1_d2_retrained_remove_{scratch,control}/`（gitignore）。
- 结果：D1-96 令三模型的重训后 MAE 均明显恶化（original +17.83%、weak +14.23%、RCRF +16.27%），
  属于共同关键日周期；D2-24/48/96/192 中增强模型的损失均低于原版。没有稳定的“原版相对不利用、增强
  模型更依赖”交互；D1-48 的 weak +0.47pp 单点效应不足以成为候选。

## 2026-09-03 — 按用户定义重跑 D1/D2（高斯陷波 / 直接置零）

- 用户否定上一轮的连续谐波回归与创新残差定义；`GaussianNotchBank` 现对每个720步标准化历史作 rFFT
  Gaussian notch（目标频率 `1/P`，`sigma=1/720`，DC保留），`TailZeroBank` 直接将末尾24/48/96/192步的
  所有标准化输入设为零。`search_phaseformer.py` 和 launcher 记录 D1 sigma，旧结果明确保留为历史而不混用。
- 两个1 epoch、5% train GPU smoke（D1-96 和 D2-24）通过；RTX 4090 / raft 上完成新的30/30个 ETTm1-H192
  seed2021、30 epoch上限、validation-only任务。运行、日志和汇总位于
  `research_runs/d1_d2_gaussian_tailzero_{scratch,control}/`，未读 test。
- 新结果写入候选发现计划§12：D2 置零使 MAE 随尾长从原版 +6.53% 增至 +33.52%，但两种增强模型在四个长度
  都有更小损失；D1 除约678步的 +0.20/+0.29pp 微小单点外，增强也更可恢复。因此仍未出现可用的原版盲区候选。

## 2026-09-03 — D3 全时间轴轨迹成分筛查

- 新增 `TrajectoryComponentBank` 与 `scripts/run_d3_trajectory_retrained_remove.py`。五个 remove-only、
  末值锚定候选为全局线性趋势、最近96步线性趋势、24步周期级水平轨迹、按phase的跨周期漂移、周期幅度
  包络；每个都改变历史而保留最后输入值。汇总脚本现可按输出目录自动验证/汇总 D1、D2 或 D3 完整网格。
- 先修复 D3 dataset 参数转发（使用已传递的 `input_period_len`）；语法检查、5个提取器不变末值锚点单测、
  1 epoch/5% GPU smoke 与既有13项消融测试均通过。RTX 4090 / raft 完成新的15/15个 ETTm1-H192、
  seed2021、30 epoch上限任务；仅读取 validation。
- 结果记录于计划§13、原始产物在 `research_runs/d3_trajectory_remove_{scratch,control}/`：五个成分中原版
  的 MAE 损失始终更大，最显著为 recent-linear +7.20% vs weak +1.65% / RCRF +2.81%，cycle-levels
  +5.11% vs +1.95% / +1.89%，cycle-amplitude +3.16% vs +0.98% / +1.03%。该批候选不支持原版盲区假设。

## 2026-09-03 — 输入成分利用问题的全证据汇总

- 新增 `docs/PhaseFormer_input_component_evidence_summary.md`，统一整理 H1/H3/H4 D0、C1--C7 冻结候选
  发现、已弃用 D1/D2 定义、当前 D1/D2 高斯陷波/尾部置零，以及 D3 末值锚定轨迹重训的设定、处理、结果
  与结论边界。
- 文档明确区分 frozen 即时依赖、remove-trained 恢复能力与 NLinear 分支实际利用；当前证据只支持增强
  模型对成分缺失有更强的替代/恢复能力，尚不能证明其不依赖相关成分，也没有找到“原版忽略、增强实际
  使用”的成分。给出 D2-192、D3-recent-linear、D3-cycle-levels 的2×2冻结/重训/分支反事实后续方案。

## 2026-09-04 — Weak-residual 非对称趋势输入的三数据集全通道案例统一

- 新增 `scripts/analyze_asymmetric_multichannel_cases.py` 与内存安全的
  `scripts/finalize_asymmetric_multichannel_audit.py`。后者从既有 validation 预测的完整样本表流式挑选案例，
  不重训也不读取 test：案例单位从单一 channel 0 改为 `validation origin × channel`。
- 在 ETTh1、Weather、ETTm1 的 H96、seed2021、L720 设置中，基线的 PhaseFormer/NLinear 都看完整 X；
  非对称候选仅将 NLinear 的输入改为 `X-A`，且二者共享完整 X 的 RevIN 统计。每个数据集×趋势成分
  （cycle-levels/recent-linear/global-linear/smooth-local/smooth-multiscale）均保留候选相对基线 MAE 最大的
  5 个退化和 5 个改善案例；同一组全部十个 origin 相隔至少96步。
- 输出严格审计包 `research_runs/asymmetric_trend_multichannel_three_dataset_audit/`：18条模型结果、
  1,040,725 条样本×通道误差行、150 个选择案例及其图、Markdown 和可携带 ZIP。以 raft Python + RTX 4090
  只做既有 checkpoint 的 validation 推理和出图；语法检查及目录白名单、案例数/去重、图片引用、ZIP 原件
  一致性校验均通过。全通道分布显示每个数据集×成分组合都含有正负两类样本影响，因此这些结果用于定位
  条件性行为模式，不能单独证明某趋势成分被某分支稳定利用或忽略。

## 2026-09-04 — Weak-residual only-A 趋势信息补充实验（进行中）

- 按用户的反向输入约束，新增 `weak_residual_asymmetric_input_mode=component_only`：PhaseFormer 保持完整
  `X`，NLinear 残差支路仅接收端点锚定的 A，并继续复用完整 X 的 RevIN 统计；缺省
  `minus_component` 保留此前 `X-A` 路由。新增三数据集 H96 launcher，并在计划中冻结 five-A、
  seed2021、30-epoch 上限、validation-only 协议和同设置既有 Baseline-full 对照。
- RTX 4090/raft 上的 ETTh1 cycle-levels、1 epoch 冒烟训练通过（71,295 参数、验证链路正常，未读 test）。
  随后已启动 ETTh1/Weather/ETTm1 × 五个 A 的15项完整训练；原始输出写入
  `research_runs/weak_residual_asymmetric_only_trend_three_dataset_h96_scratch/`。汇总结果待全部 checkpoint
  完成后再生成，不能把中途指标作为结论。

## 2026-09-04 — X-A 预测曲线分歧案例导出（待 only-A 训练结束后执行）

- 按用户的新筛选口径新增 `scripts/export_asymmetric_prediction_divergence_cases.py`。单位为完整 validation
  的 `origin × channel`，排序量固定为未来96步 `mean(|prediction_asymmetric - prediction_baseline|)`，不读取、
  不参与排序、也不显示 ground truth。每个 ETTh1/Weather/ETTm1 × 五个 A 导出10个最大分歧且 origin 相隔
  至少96步的案例。
- 每个 `research_runs/asymmetric_prediction_divergence_cases/<dataset>/<component>/` 子目录将保存选例 CSV、
  数组和案例图；图依次显示完整历史 X、提取的 A 轨迹、Baseline-full 与 Asymmetric X-A 的预测曲线。为避免
  与进行中的 only-A 全量训练争抢 RTX 4090，本次只完成静态校验，待训练释放 GPU 后再执行只读推理导出。
- 用户随后将案例范围收紧为固定 channel 0、每个数据集×成分仅3个最大预测曲线分歧案例；脚本默认值已
  同步修改，排序公式和“不使用/不显示 GT”的约束不变。
- 用户要求在 only-A 完成后同步导出两种路由。导出器现显式支持 `minus_component` 与 `component_only`，
  并将分别写入 `research_runs/asymmetric_prediction_divergence_cases/X_minus_A/` 和 `.../Only_A/`；二者均
  复用同一批此前 Baseline-full checkpoint，只有候选 checkpoint 与图例标签不同。
- `scripts/run_after_only_trend_exports.sh` 已作为低优先级后处理队列启动：每60秒检查 only-A 的15项状态，
  只在全部 completed 后串行进行两套 GPU 推理导出；若训练非正常停止则显式退出而不生成不完整结果。队列日志
  位于 `research_runs/weak_residual_asymmetric_only_trend_three_dataset_h96_control/export_after_training.log`。
- only-A 训练现已15/15完成。后台等待会话未保留日志，故在确认 GPU 空闲后直接顺序执行两套只读导出；
  `X_minus_A/` 与 `Only_A/` 均经程序校验为15个 dataset×component 目录、每目录3条 channel-0 记录和3张图，
  各45个案例、合计90张图。抽查确认图的三面板为 X、A、两条预测，GT 未被使用或显示。
- 用户澄清：GT 只应排除在筛选指标外，而必须显示在图中。已修正绘图与数组导出并重生成两套90张图：第三
  面板现含黑色 GT、蓝色 Baseline 和红色候选预测；`forecast_curve_mad` 的候选排序公式完全不变，仍只用
  两条预测曲线。此前“GT 未显示”的描述作废。

## 2026-09-04 — 趋势滤波平滑尺度的 validation 诊断

- 新增 `scripts/probe_trend_filter_smoothing.py`。该工具不训练模型、不读取 test；它严格复用数据加载器的
  training-scaler 和 validation 窗口定义，在 ETTh1、Weather、ETTm1 各固定两个 channel-0、L=720 窗口上，
  对比连续 72 步线性样条趋势与一阶趋势滤波趋势。所有成分均末点锚定。
- 比较的冻结规则为 `lambda = kappa * sample_std * (1 hour / sample_interval)^2`，`kappa={25,100,400}`；因此
  ETTm1 的离散 lambda 在同一 kappa 下为小时级数据的16倍，属于采样间隔换算而非按数据集调参。诊断包位于
  `research_runs/trend_filter_parameter_probe/`，包含六张图、参数表和可携带 ZIP，且只含规定的审计文件与
  `figures/`。
- 人工检查六张 validation 图：`kappa=25` 在 ETTh1/ETTm1 仍保留显著局部周期起伏，`kappa=400` 在多个样本中
  近似全局漂移；`kappa=100` 保留中尺度趋势转折而未跟随主周期，故作为未来趋势滤波 A 候选的暂定统一尺度。
  该结论只证明提取尺度的视觉合理性，不构成预测提升、分支利用或因果结论。

## 2026-09-04 — 单侧局部趋势候选的实现与图形筛查

- 在 `src/models/asymmetric_trend_components.py` 新增 `causal_ema`、`causal_local_linear` 与
  `holt_local_linear`。三者均逐样本逐变量提取、无右侧 padding，并以 `A[L-1]=0` 末点锚定；新增单测确认
  shape/锚定、缩放等变性、flag-off 等价性，以及三种单侧提取器在消除共同锚定平移后的前缀相对轨迹不依赖未来点。
- 新增 `scripts/probe_causal_trend_components.py`，在 ETTh1、Weather、ETTm1 各两个固定 channel-0 validation
  窗口上直接使用与训练相同的256步趋势滤波近似，叠图比较四种成分。审计包为
  `research_runs/causal_trend_component_visual_probe/`，含六张图、统计量、Markdown 与 ZIP；未训练模型、未读 test。
- 图形检查：三种单侧方法消除了 A4/A5 的末端 replicate-padding 问题；但在 ETTh1/ETTm1 当前冻结尺度下均保留
  较明显主周期，尤其局部线性与 Holt。它们暂不进入 X-A/Only-A 训练，除非先重新定义目标时间尺度并独立复核。

## 2026-09-04 — 单侧趋势参数的频谱泄漏约束

- 用户指出“无 padding 伪影”不足以证明趋势纯度。对三个数据集八个固定 validation 历史窗口（仅输入、无标签）
  的平均 periodogram 及参数网格，新增/冻结筛选规则：趋势在输入主周期及相邻 bin 的能量比例必须不高于 0.10，
  再在合格项中取最大更新增益，避免后验按预测指标调参。
- ETTh1 的最强主峰为24步；EMA/Holt 的 `alpha=.024`、`beta=.006` 满足泄漏约束。ETTm1 的主峰集中在约90--103步；
  `alpha=.006`、`beta=.0015` 满足约束。Weather 没有同样尖锐的短周期峰，暂不套用机械的72步抑制规则。
- 单侧局部线性在 ETTm1 从72到720步窗口、多个带宽的网格中最低泄漏仍约0.19，未达到趋势纯度阈值；因此不能以
  “调大窗口”包装为趋势候选，当前不进入 X-A/Only-A 重训。提取器已支持显式参数传入，以便仅对通过该约束的
  EMA/Holt 候选进行后续冻结与训练。

## 2026-09-04 — 频谱约束后的 EMA/Holt 可视复核

- 按用户要求从后续候选与新版图中排除 `causal_local_linear`，并更新
  `scripts/probe_causal_trend_components.py`：仅绘制 A6 trend-filter、频谱约束 EMA 和频谱约束 Holt。
- 新审计包位于 `research_runs/causal_trend_component_spectral_probe/`。ETTh1/Weather 固定
  `alpha=.024,beta=.006`，ETTm1 固定 `alpha=.006,beta=.0015`；Weather 参数为保守的小时级设置，非预测指标选择。
  六个固定 validation 样本图、目录白名单及 ZIP 原件一致性均已校验，且未读 test、未训练模型。
- 图形复核确认：ETTh1 24步与 ETTm1 约96步主周期不再主导 EMA/Holt 曲线；这一结论限于目标频带的抑制，不可表述
  为模型预测收益或分支利用证据。

## 2026-09-04 — 三趋势成分非对称输入实验准备

- 新增 `docs/Weak_residual_three_trend_components_experiment_plan.md` 与
  `scripts/run_weak_residual_trend_comparison.py`。计划冻结 trend-filter、频谱约束 causal-EMA、频谱约束 Holt，
  在 ETTh1/Weather/ETTm1、L720→H96、seed2021、validation-only 下执行 X-A/Only-A 共18项 candidate 训练，
  并复用已有同协议 Baseline-full。
- PhaseFormer 与 preset 配置现可显式传递 EMA/Holt/单侧局部线性参数到提取器；单测覆盖三种候选的真实 model
  forward。当前 `nvidia-smi` 无法连接 GPU driver，故仅完成 CPU 静态/forward 与 launcher dry-run 校验；CUDA
  1-epoch smoke 被如实保留为待办，未启动完整训练。

## 2026-09-04 — 三成分实际输入频谱验收与 A6 收敛修正

- 使用 raft 对 ETTh1/Weather/ETTm1 的16个固定 channel-0 validation 历史窗口直接执行真实提取器；验收量为主周期
  及相邻频点的 `trend_power/input_power <= .10`。ETTh1 的24步峰：A6 `.034`、EMA `.009`、Holt `.009`；ETTm1 的
  约96步峰：EMA `.024`、Holt `.029`。三者均抑制主周期。
- 初始 A6 的 ETTm1 256步 Chambolle--Pock 近似泄漏 `.976`，不合格；kappa 增大无效，表明问题是固定迭代尚未收敛。
  4096步时泄漏降至 `.056`，故新三成分实验 launcher 对 ETTm1 冻结4096步、ETTh1/Weather 保持256步。未把256步的
  ETTm1 A6 当作已验证趋势成分。
- GPU driver 当前不可用（raft 的 `torch.cuda.is_available()=False`、NVML 初始化失败），故CUDA smoke仍无法执行；
  在实际GPU烟雾测试确认4096步的时间/显存前，不启动完整18项训练。

## 2026-09-04 — 两种单侧趋势成分实验交付物归档

- 按用户提供的 `weak_residual_trend_2comp_3ds_experiment.tar.gz` 解压并原样归档到
  `research_runs/weak_residual_trend_2comp_3ds_experiment_scratch/`。输入压缩包经路径安全检查，不含绝对路径或 `..` 路径穿越项；原压缩包未删除。
- 归档内容为 `causal_ema` 和 `holt_local_linear` 在 ETTh1、ETTm1、Weather、L=720→H=96、seed=2021、validation-only 上的完整原始交付物：3 个 Baseline-full 与 12 个 X-A/Only-A 候选的 checkpoint、日志、预测、代码快照、汇总结果和 channel-0 预测差异图。
- 该目录明确标为 scratch，因为它含 checkpoint 和全量中间产物；其中的 `analysis/audit/` 缺少规范审计根所必需的 `sample_errors.csv`（README 说明该文件约119 MB且未随包提供），故不得将其标为符合六文件白名单的正式审计目录。未重算或改写任何实验指标。

## 2026-09-04 — 交付 checkpoint 的预测分歧案例重新导出

- 扩展 `scripts/export_asymmetric_prediction_divergence_cases.py`，使其可读取归档交付物的
  `checkpoints/<dataset>_h96_seed2021/<component>-<route>/` 布局，并从候选 `config.json` 复用精确的趋势提取超参数；普通本地 `runs/` 布局保持兼容。
- Baseline-full 使用当前已有 checkpoint：ETTh1 来自 `weak_residual_asymmetric_trend_discovery`、Weather 来自
  `weak_residual_asymmetric_weather_h96_scratch`、ETTm1 来自 `weak_residual_asymmetric_ettm1_h96_scratch`；候选只使用用户交付的 `causal_ema` 与 `holt_local_linear` checkpoint。三者均为 L=720→H=96、seed=2021 的 validation 配置。
- 重新推理并导出 channel 0 的案例；排序仅使用 Baseline 与候选预测曲线在96步上的 MAD，GT 不参与排序但在图中显示。每个 dataset×component×route 保留3个、原点间隔至少96的案例。结果合并至既有 `research_runs/asymmetric_prediction_divergence_cases/X_minus_A/` 与 `.../Only_A/`，各新增18张图；各路由的 `imported_2trend_current_baselines_manifest.csv` 记录新选择，未覆盖此前五种成分的产物。

## 2026-09-04 — 统一 X-A / Only-A 样本图

- 用户要求将所有样本图统一为 dataset×component 各一张。新增
  `scripts/export_asymmetric_joint_route_cases.py`：对 channel 0，在每个成分下按
  `.5 * (MAD(X-A, Baseline) + MAD(Only-A, Baseline))` 选择一个 validation origin；GT 不参与选择。
- 统一图均为两行：第一行叠绘完整 history X 与候选提取的 A，标题注明 dataset、validation origin、channel、L=720、H=96；第二行叠绘 GT、Baseline-full、X-A、Only-A，并在标题写出三者的样本 MAE/MSE 和按 MSE 的最佳者。
- 通过 raft 在 ETTh1、Weather、ETTm1 和7种成分（原五种、causal_ema、holt_local_linear）上重新推理，产生21张图及根目录 `manifest.csv`。输出位于 `research_runs/asymmetric_prediction_divergence_cases/<dataset>/<component>/`。旧版按路由拆分的 `X_minus_A/`、`Only_A/` 产物未直接删除，已移动至 `/tmp/asymmetric_prediction_divergence_cases_*_previous/`，可恢复。

## 2026-09-04 — 统一样本图的去重重导出与最终校验

- 发现初版联合选择使用“两个路由相对 Baseline 的平均分歧”，会被同一异常窗口主导，多个成分因此复用相同 origin，不适合作为逐成分图册。选择规则修正为直接最大化 `MAD(X-A prediction, Only-A prediction)`；在每个数据集内，七个成分的已选 origin 两两至少相隔96步。
- 重导出的21张图全部位于 `research_runs/asymmetric_prediction_divergence_cases/<dataset>/<component>/`。最终程序校验通过：`manifest.csv` 恰有21行，ETTh1/Weather/ETTm1 各7行、每行对应文件存在、同数据集 origin 满足96步间隔；21张 PNG 有21个不同 SHA-256。人工抽查 ETTm1/smooth_local 的两行图，确认标题与四条预测曲线均正确。
- 被替换的首版统一图未直接删除，已移动至 `/tmp/asymmetric_prediction_divergence_cases_joint_previous/`，可恢复。

## 2026-09-04 — 每个成分的三例 X-A / Only-A 最大分歧图

- 用户澄清每个 dataset×component 需要三个、而非一个样本。`export_asymmetric_joint_route_cases.py` 已改为在每个 dataset×component 内按 `MAD(X-A prediction, Only-A prediction)` 降序选择3个 channel-0 validation origin，并要求这三个 origin 两两相隔至少96步；GT 始终不参与选择。
- 同时修复上一版绘图循环错误复用最后一个候选预测数组的问题；现在绘制和保存的 X-A/Only-A 预测均从当前 component 的候选数组读取。
- 重新导出63张两行图（3 datasets×7 components×3 cases）至 `research_runs/asymmetric_prediction_divergence_cases/<dataset>/<component>/`。最终校验通过：根 manifest 恰有63行，每 dataset×component 均为rank 1--3且满足间隔，三个 `selected_cases.npz` 预测数组逐项重算的 X-A/Only-A MAD 与 manifest 精确一致，63张PNG有63个不同 SHA-256。被替换的21图版本移至 `/tmp/asymmetric_prediction_divergence_cases_joint_21_previous/`，可恢复。

## 2026-09-04 — 统一案例筛选与可视化的独立 GPU 复核

- 对63个案例从 checkpoint 独立重跑完整 validation：每个 dataset×component 的 manifest origin 均精确等于按 channel-0 的 `mean_t |X-A prediction - Only-A prediction|` 降序、并在同组内执行96步间隔后的前三项；GT 未进入该排序。
- 两路候选的趋势提取超参数逐组件一致。每个 `selected_cases.npz` 的 Baseline、X-A、Only-A 预测与独立重跑逐元素一致（`atol=1e-6`）；A 与同一历史窗口按同一GPU提取路径重算一致（最大绝对差 `2.38e-6`，为float32舍入）。
- 静态检查确认图的第一行是 full X 与 A，标题包含 dataset/origin/channel/L720/H96；第二行是 GT、Baseline、X-A、Only-A，标题列出三者 MAE/MSE，并按最小 MSE 标注最佳者。因此当前筛选及可视化逻辑符合用户指定口径。

## 2026-09-04 — X-A / Only-A / Baseline 聚合差异表

- 新增 `scripts/summarize_asymmetric_component_metrics.py`，从当前图册实际使用的 Baseline-full、X-A、Only-A checkpoint 的 `metrics.csv` 生成 validation 聚合表；执行前校验 dataset、L=720、H=96、percent=100 与 seed=2021。
- 结果写入 `research_runs/asymmetric_prediction_divergence_cases/XA_OnlyA_Baseline_validation_comparison.md`。按 ETTh1、Weather、ETTm1 分表列出7种成分的 MSE/MAE 及 X-A、Only-A 相对 Baseline-full 的绝对/百分比差。差值定义为候选减 Baseline，负值为更好；文档明确它不能和按预测分歧选择的局部案例误差混读。

## 2026-09-05 — ETTh1 单侧平滑参数的周期泄漏调试

- 使用已导出的 ETTh1 channel-0 origin 853、1978、2533 历史窗口，对 `causal_ema` 与 `holt_local_linear` 比较当前 `alpha=.024`（Holt `beta=.006`）和更慢的 `.006/.0015`、`.003/.00075`。叠图与24步谐波回归幅度表位于 `research_runs/causal_ema_holt_etth1_parameter_debug_scratch/`；不训练模型、不读取 test。
- 当前参数在三个样本的24步谐波幅度增益为 EMA `.097--.106`、Holt `.097--.109`，因而虽不主导频谱，时域图上仍有明显周期波纹。`.006/.0015` 降至约 `.023--.026`，`.003/.00075` 降至约 `.011--.012`；图形确认周期纹波随之明显消失。
- 结论仅限提取纯度：ETTh1 的当前参数过快，不能把 EMA/Holt 表述为严格纯趋势；`.006/.0015` 是保留较慢轨迹的候选折中，`.003/.00075` 更严格但更接近全局慢漂移。尚未用新参数重训，故不得将其推断为预测收益。

## 2026-09-05 — 启动 ETTh1 慢趋势参数重训练

- 将 `scripts/run_weak_residual_trend_comparison.py` 的 ETTh1 参数更新为 `causal_ema α=.006`、`holt_local_linear α=.006, β=.0015`，与已完成的周期泄漏调试结论一致；Weather/ETTm1 参数保持不变。
- 本轮正式范围限定为 ETTh1、L=720、H=96、seed=2021、validation-only 的 causal EMA/Holt 两成分×X-A/Only-A 四项训练；原始日志和 checkpoint 写入 `research_runs/weak_residual_etth1_slow_causal_trend_h96_scratch/`。
- 训练前必须使用 raft 环境完成 CUDA smoke 和参数/频谱校验；本条记录对应配置修正，最终指标、审计报告及案例图待训练完成后补充。
- 轻量校验：launcher dry-run 正确列出4项；四组 overrides 均显示上述慢参数；脚本通过 `py_compile`。首次随机张量检查误用了函数参数名，已按实际接口 `causal_ema_alpha`/`holt_level_alpha` 修正 smoke 命令，未启动错误配置训练。
