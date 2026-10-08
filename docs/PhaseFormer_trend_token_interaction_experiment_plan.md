# PhaseFormer-L low-rank interaction matrix

> **Correction**: the first `trend_token_interaction_20261006` implementation
> used a new Conv1d history encoder and therefore did not test PhaseFormer-L.
> That matrix is invalid for the intended hypothesis. The corrected arms below
> reuse `PooledLowRankWeakPeriodResidualHead` and insert attention only at its
> learned rank-`r` latent.

## Purpose

## 当前阶段：固定静态融合的非 attention token 交互（2026-10-08）

用户将输出 Gate 与 latent 交互分开研究。本阶段固定已经验证的静态水平/形状
双分量融合，独立探索原低秩 trend token 与 phase token 的小宽度消息。
用户最新要求由单个 agent 完成代码核验、修复、smoke 与正式启动，不再使用子智能体。
此前并行 attention/static 实验按用户要求停止，保持停止状态，不续跑。

### 固定五臂与信息路径

令 t=E(x-x_last) 为原 rank-r latent，Z 为 phase 主干完成 routing 后、最终 predictor
前的表示；z_bar=mean_phase_slot(Z)，不是时间/周期轴平均。k=min(2,r,d)。
消息矩阵均无 bias，B 零初始化；不增大 r，不增加 attention、norm、activation、
动态 Gate 或损失项，不 detach 消息来源。

| arm | 交互 | 额外参数 |
|---|---|---|
| b0 | 无 latent 交互 | 0 |
| pt_add | t'=t+B V z_bar，Z'=Z | k(d+r) |
| pt_correct | t'=t+B(V z_bar-Ct)，Z'=Z | k(d+2r) |
| tp_add | t'=t，Z'_p=Z_p+B A t | k(r+d) |
| tp_mod | t'=t，Z'_p=Z_p+Z_p*(B A t) | k(r+d) |

PT 更新在原 D 解码前；TP 更新在原 phase predictor 前，广播到 phase slots。
PT-correct 的线性自项可吸收进可训练 E/D，比较反映参数化/优化与 phase 信息作用，
不能据此独称 Kalman 或可识别纠错机制。phase-slot 均值对 slot 重排不变，因此
PT 不以 roll/reverse 为有效信息控制。TP-add 对共享线性 predictor 提供共同 slot
轨迹，TP-mod 改变各 slot 的特征响应；两者表达能力不同。

### 训练与决策协议

- ETTh1/ETTm2/Weather × H96/H720 × seeds2021/2022，12 settings ×5 arms=60次。
- L720、P24、r=H/8、pool_factor1、smooth_ratio0、原 E/D、last-value anchor、
  phase 主干与 RevIN 不变；每个 setting 沿用已完成 level study manifest 的 LR、
  batch、容量、Huber 和 scheduler，来源配置仅提供超参数，不加载来源权重。
- 全部随机初始化、全参数联合训练，同 seed 公共未训练 state 完全匹配、数据顺序
  匹配；静态 level/shape Gate 均从0.2开始，30轮上限、patience8，最低 val Huber
  checkpoint。禁止预训练/冻结；本阶段只评估完整 validation，不读 test。
- 先8项针对性测试，再两个 setting（ETTh1-H96-S2021、Weather-H720-S2021）×5臂
  smoke：1轮、4 train batches、2 val batches，随后完整 validation 审计。
  零 D 和零 B 会使梯度分步启动，记录最初8个优化步，检查消息所有参数最终获得
  非零有限梯度，不要求第0步全部非零。smoke 结果不能作为效能结论。
- 候选进入后续复核须平均/中位相对MSE均<0、至少8/12 MSE胜出、至少4/6任务
  双seed平均胜出、平均相对MAE<=0；重复严重尾部退化阻止默认升级。
  同容量上下文置零、双向组合、k搜索、第三seed或跨任务确认仅在审计后另行登记，
  不由此60-run controller自动触发，不按dataset/horizon拼接赢家。
- 比较参数、训练耗时/显存、同GPU串行随机次序 latency、分支/融合/oracle、
  native/off消息干预、水平/形状、horizon三段、峰谷、p95和top1%误差。
  off只诊断同一训练checkpoint；oracle仅事后读label，不可部署。
- 全量sample×channel排名，每候选每setting各类Top3 MAE high error/regression/
  improvement，同类同channel起点至少相隔H；报告/ZIP必须经过白名单和重算校验。
  validation已有选择暴露；历史test查看不因本轮未读test而消失，不能声称盲测或
  Golden优越。静态基线与历史结果匹配属于复现，不是独立seed证据。

### 隔离与启动

专用入口 `scripts/lowrank_token_interaction/{model,run,analyze}.py` 使用私有
construct/restore，仅替换新建模型实例的 branch，不改旧模型、factory、runner
或全局注册。测试位于 `tests/test_lowrank_token_interaction.py`。
远程使用单独 checkout；原实验目录保持不变。manifest 冻结commit/source hashes，
controller/worker/setting审计加锁；成功结果校验后跳过，失败目录保留，修复须新
attempt目录；活跃源码不得原地替换。进程停止传递到子进程组并回收。

```bash
python scripts/lowrank_token_interaction/run.py \
  --source-manifest <completed-level-scratch>/manifest.json \
  --scratch-dir research_runs/lowrank_token_interaction_20261008_scratch \
  --audit-dir research_runs/lowrank_token_interaction_20261008 \
  --gpus 0,1,2,3,4,5,6,7
```

smoke 使用独立 `_smoke_scratch`/`_smoke` 路径，补 `--smoke --epochs 1
--settings ETTh1:96:2021,Weather:720:2021`。canonical仅六文件加figures，
训练checkpoint/日志/manifest单独放ignored scratch。当前代码已实现，远程数值
验证、smoke和正式启动待完成，尚无新机制效果结论。

> **Protocol correction (2026-10-07, user clarification):** the research target
> is architecture comparison with training from scratch, not adaptation of a
> pretrained i0. Audit of the corrected stage-1 remote configs found 210/210
> with an empty `init_checkpoint`, an eight-epoch budget and completed metrics.
> In contrast, all 54 valid stage-2 runs explicitly load trained stage-1 i0
> weights; all 42 stage-3 runs explicitly load stage-2 tp_pre weights. Their
> existing metrics remain valid for those warm-start protocols, but cannot
> establish the best architecture under training from scratch. In particular,
> tp_pre's normalization comparison, rejection of selective/null mechanisms,
> and centered_frozen's advantage over joint adaptation must not be promoted
> to from-scratch architecture decisions. Earlier "best model" wording is
> withdrawn to that extent. Subsequent architecture tests must initialize and
> train all branches jointly from scratch, use matched training/selection
> budgets and seeds, and audit that no trained checkpoint was loaded. Keep
> prior warm-start evidence separate as diagnostics; no new runs are launched
> by this audit.

This round tests whether the learned trend token should only correct the final
forecast, read the phase tokens, write back to the phase tokens, or participate
in repeated/joint token interaction. The existing low-rank PhaseFormer-L
results establish the capacity settings used here; this round changes only the
interaction graph around the legacy residual latent.

## 新第二阶段：六个从头训练的研究方案（2026-10-07）

**状态（2026-10-08）：168次初筛和96次长预算训练均完成、训练失败为0；
样本审计、报告、图包、本地同步与独立复核全部完成。** 用户将范围收敛为三种动态输出 Gate，加上
原第二阶段从实验现象提出的三个交互修改，共六个研究假设。此前七类 Gate 计划
由本节取代，不继续扩展。下文旧 Stage 2/3 保留为 warm-start 历史证据。
六个假设不等于六次训练；必要基线、机制消融和重复 seed 单列，不算新增研究方向。

### 六个方案与最小判别对照

| ID | 方案 | 要验证的假设 | 主要对照 |
|---|---|---|---|
| H1 | 保留原表示的残差交互：`z'=z+alpha*F(LN(z),LN(s))`，alpha零初始化；P->T和T->P分别测试 | 从训练开始保留原表示并学习消息强度，是否优于post-norm更新 | 每个方向保留post、pre、scaled-pre，拆开归一化位置与缩放的贡献 |
| H2 | 可拒绝的P->T读取：加入最终消息为零的null选项 | 强制读取不可靠相位是否有害，允许trend拒绝是否更好 | 与H1的pt_scaled比较，其他条件不变 |
| H3 | 按相位选择的T->P写入：`p_j'=p_j+alpha*g_j*v(t)` | 不同phase是否需要不同的消息接收量 | 与H1的tp_scaled广播比较；共享消息方向，只改变接收量 |
| G1 | 相位可靠度动态输出Gate（原G1） | 历史周期重复程度能否预测两分支的相对价值 | 无交互i0的静态输出Gate |
| G2 | 预测分歧感知动态输出Gate（原G4） | token状态与两路预测的水平/斜率/振幅分歧，能否帮助决定融合比例 | 静态i0；若有收益，补同容量分歧输入置零消融 |
| G3 | 周期均值/形状分别动态融合（原G7） | 两条分支可能分别擅长水平与形状，是否需要两个融合系数 | 静态i0及两个静态分量Gate |

选择三种Gate覆盖三个互补问题：历史可靠度、预测分歧、融合分量。历史统计独立
Gate、token-only独立Gate、预测距离Gate和相位输出Gate不进入本轮。H3虽然也有
门控，但作用在token消息上，不能与三种输出融合Gate混淆。原第三阶段的内容修正
不包含在本轮六个方案中。不预先组合六项改动，也不根据旧续训结果跳过H1--H3。

### 共同模型和训练协议

- 保留原低秩`E(x-x_last)->t->D(t)+x_last`、单个rank-r token，r=H/8，
  pool_factor=1、smooth_ratio=0、L=720、P=24、原phase主干与RevIN。
- 全部从随机初始化开始、全模型联合训练，不加载训练好的i0/tp_pre，不冻结
  原分支。同seed可复制**未训练**公共参数以匹配起点，保存初始哈希并匹配数据顺序。
- H1--H3使用静态输出融合；G1--G3使用无token交互的i0，先独立验证各自贡献。
- Gate只读取历史、当前latent与模型预测，不读未来标签。读取latent/预测时统一
  stop-gradient；两分支仍从最终预测损失获得梯度，这不等于冻结原模型。
- 所有输出Gate初始alpha=0.2；MLP隐层宽度16、输出层权重零初始化，初始保持
  同一静态融合。记录参数量与成本，不将共同初始化控制视为已证实的性能机制。
- ETTh1、ETTm2、Weather × H96/H720 × seeds2021/2022，共12个设置；8GPU调度。
  初筛全训练集、8 epochs，继承对应dataset/horizon的既有容量、LR、batch、Huber
  和scheduler，不为每个方案分别调参。正式执行前冻结完整arm清单及运行数量。
- 有信号的方案和对应基线重新从头训练，统一最多30 epochs、patience=8，按最低
  validation Huber选择checkpoint，不从8轮模型接着训练。短预算未胜出不能直接
  判机制无效，对仍明显学习的候选登记后作等预算复核。
- 比较逐setting相对MSE/MAE、跨seed方向及尾部退化；不按数据集拼接赢家。
  validation已反复使用，属于探索性证据；本阶段不读test、不宣称Golden超越。
- 新的无LayerNorm对照、梯度路由对照或跨机制组合仅在具体归因需要时另行登记，
  不作为本次六方案之外自动扩展的研究方向。

### 三种Gate的具体定义

G1：`alpha=sigmoid(b_c+s*(1-r))`，r为跨相位模板方差除以模板方差加同相位
跨周期方差，分母加eps；s=4*tanh(s_raw)，初始0。允许学习负s，不强制假设
周期弱就一定应该更多使用trend。

G2：`alpha=sigmoid(b_c+MLP(features))`。features包括LN(t)、t的均值/标准差、
phase tokens的跨相位均值/标准差，以及两路预测的均值、斜率、振幅和差值RMS。
每个样本×变量一个alpha，预测窗口内共享。分歧不等于谁更准确，必须通过实验
验证；无交互i0不存在交互注意力，不引入attention entropy。

G3：用与G2相同的可用上下文生成alpha_level与alpha_shape，每样本×变量各一个。
按每个完整24步预测周期计算`mu_b=mean_phase(y_b)`和`s_b=y_b-mu_b`，融合为：
`y=(1-a_level)*mu_P+a_level*mu_T+(1-a_shape)*s_P+a_shape*s_T`。
本轮H均整除24。不预设trend负责水平；两个Gate可以自行学习。这不是逐点两条
完整预测的凸组合，需检查超出两路预测区间的比例及峰值误差。两个静态分量Gate
对照用于区分分量自由度与样本动态性的收益。

### 必要校验与样本分析

实现后先二次代码审查与逐arm smoke，验证shape/梯度、train模式、未加载训练
权重、共同初始参数、纯融合初始输出一致、null无bias泄漏、G3分量重建及标签
不可达，再正式运行。固定权重的干预只用于诊断，不能替代从头训练对照。

记录每样本×变量的两路及融合误差、Gate均值/波动/饱和率、分歧和分段误差。
用训练集估计的按变量平均Gate替换动态Gate，并在同变量非重叠样本间打乱Gate，
检验样本对应是否有用。H3另做phase错位和平均接收量干预；G3分别固定水平与
形状Gate，检验真实分工。用未来标签得到的oracle融合只作事后诊断，不进入训练。
程序化筛选高误差、退化、改善案例，检查水平、形状、突变和峰谷，不只挑旧anchor。

实验工件遵循六文件加figures的统一目录；训练日志/checkpoint另放忽略目录。
保存所有尝试与失败、运行成本和选择规则。

**执行清单**：`scripts/run_lowrank_six.py` 实现14个固定配置：i0、pt_post、pt_pre、
pt_scaled、tp_post、tp_pre、tp_scaled、pt_null、tp_selective、g1_reliability、
g2_disagreement、g2_without_forecasts、g3_components、components_static。共12个
settings、168次8轮从头训练。将G2等参数去预测特征对照提前纳入，避免仅在胜出后
补测造成不完整归因。所有公共初始权重哈希按setting匹配；初始化期间禁止torch.load，
替换未来标签后预测必须不变；每个训练batch检查全模型train模式与requires_grad。

正式入口：`python scripts/run_lowrank_six.py --source-dir
research_runs/phaseformer_l_interaction_20261006 --scratch-dir
research_runs/lowrank_six_20261007_scratch --audit-dir research_runs/lowrank_six_20261007`。
source-dir只提供配置，不加载旧checkpoint。代码包含自动全validation配对审计，
每个arm/setting选每类Top3（至多9条）并保存可复算案例。推理时Gate可替换为训练集
平均或同变量、预测窗不重叠样本的Gate；H2/H3包含no-null、H3包含接收量均值/错位。
G3另测试固定水平与形状Gate。训练前必须完成所有14个arm的smoke和审计闭环。

### 从头训练结果与长预算确认（2026-10-08）

初筛 `lowrank_six_20261007`：168/168，14配置、12设置、8轮上限；
审计通过，11,044,600条样本比较、1,404条选例、84张图。
训练代码 `a30494f`；既有报告保留，不混入旧warm-start结果。

随后在 `lowrank_six_20261007_long` 完成96/96次随机初始化训练，最多30轮、
patience=8，实际10--30轮；原数据比例100%，ETT batch=256、Weather batch=64，
其余超参数继承初筛。配置为i0、G2、G3、components_static、tp_pre、tp_post、
tp_pre_g2、tp_pre_g3，仍为同12设置，未读test。训练代码 `f8fc8b2`，
耗时合计7.113 GPU任务小时（各任务训练墙钟时间相加，非设备利用率计量）。
选择G2/G3和tp_pre来自初筛信号；组合用于检验交互和动态融合收益能否叠加。
**该8配置矩阵此前记录在对话与manifest，本节在启动后补记，不能宣称事前注册。**
组合没有引入新机制，但其结果同样经过validation选择，属于探索性证据。

长预算首次分析在Weather-H96两个seed的G3重构断言处失败。模型先在RevIN
归一化空间融合再还原，而审计先还原两分支再融合，浮点运算顺序产生最大
7.25e-5误差。`fa02c14`通过只读forward hook捕获归一化分支，所有Gate干预
按模型顺序计算，原Gate重放要求bitwise一致。45项测试通过；原失败日志和
10个已完成分析分片保留于scratch，全12设置重新审计，不重训模型、不放宽断言。
修复后96行结果、6,311,200条比较、756条选例、48张图通过审计。

下表为setting等权candidate-minus-control差值，负数更好：

| 比较 | 8轮ΔMSE | 30轮上限ΔMSE | 30轮上限ΔMAE | 长预算MSE胜出 |
|---|---:|---:|---:|---:|
| tp_pre - tp_post | -0.004485 | -0.004385 | -0.000359 | 9/12 |
| G2 - i0 | -0.001599 | -0.000926 | -0.000062 | 6/12 |
| G3 - i0 | -0.002216 | -0.001808 | -0.000759 | 6/12 |
| G3 - components_static | -0.001774 | -0.001076 | +0.000162 | 6/12 |
| G3 - G2 | -0.000618 | -0.000881 | -0.000697 | 9/12 |
| components_static - i0 | -0.000443 | -0.000732 | -0.000922 | 10/12 |
| tp_pre - i0 | -0.003193 | -0.002681 | +0.000293 | 7/12 |
| tp_pre_g2 - tp_pre | 未测 | +0.000805 | -0.000597 | 5/12 |
| tp_pre_g3 - tp_pre | 未测 | -0.000121 | -0.000671 | 7/12 |
| tp_pre_g3 - G3 | 未测 | -0.000994 | +0.000381 | 6/12 |

**六项判定（仅限已测设置和预算）：**

- H1部分支持：T->P pre-norm相对post-norm的收益在长预算复现；不能将收益归因于
  零初始化缩放。初筛tp_scaled-tp_pre为+0.002256 MSE，pt_scaled-pt_pre为
  +0.000032；P->T pre-post仅5/12胜出。scale及P->T未进入长预算确认。
- H2未获得稳定支持：pt_null-pt_scaled初筛ΔMSE=-0.000201，6/12胜出，
  setting等权相对MSE反而+0.045%。关闭null会平均变差，说明已有模型使用了它，
  不能替代架构比较。仅有短预算结论，不否定更长训练下的可能性。
- H3目前不支持：tp_selective-tp_scaled初筛ΔMSE=+0.000563、ΔMAE=+0.000333，
  5/12胜出。phase错位使10/12设置变差，但平均接收量干预仅4/12变差；
  phase依赖与广播控制相比更有效是不同命题。未做长预算复核。
- G1动态可靠度假设未获支持：对i0有小幅短预算改善（ΔMSE=-0.000784，7/12），
  但同变量Gate的样本标准差均值仅0.00136，换训练集均值反而平均改善MSE约0.000429。
  大部分收益不能直接归因于样本可靠度匹配，未做长预算复核。
- G2部分支持：长预算相对i0均值改善缩小且仅6/12胜出；均值Gate替换使9/12
  变差（平均+0.003802 MSE），打乱使11/12变差（+0.011218），支持已训练模型
  使用样本上下文。预测分歧本身只有短预算等参数去特征对照（ΔMSE=-0.000291、
  ΔMAE=+0.000203）；长预算未包含该控制，不能独立确认分歧输入贡献。
- G3部分支持：优于G2的均值与9/12方向较稳定，但对静态双分量控制仅6/12胜出，
  MAE略退化。分量分开有具体样本支持；两个Gate都必须动态的命题不成立于当前证据。

**激活与具体样本：**长预算G3固定水平Gate平均ΔMSE=+0.003616，9/12变差；
固定形状Gate=-0.000125，仅4/12变差。tp_pre_g3相应为+0.003006和-0.000339。
G3平均level/shape Gate为0.208/0.468，6.10%的预测点超出两分支逐点区间；
该非凸逐点结果来自周期分量重组，不自动等于异常或收益。

- Weather-H96-S2022，sample1702/channel14，G3相对静态双分量控制：
  MSE 76.3642->15.3934，Gate=(0.0131,0.9971)。本模型trend分支周期均值
  误差25560.59，但形状误差10.4022，比phase的29.2491小；G3组合后水平/
  形状误差为4.9684/10.4250。固定形状Gate使MSE升至26.7399，说明在这一
  具体样本中动态形状仍有价值，不能从均值结果认定它处处无用。
- Weather-H96-S2021，sample1980/channel14，G3 MSE 0.1501->2.9085；
  真值均值-0.0850，预测均值1.5541，周期水平误差2.7142主导退化。
  使用该模型两分支的事后oracle双Gate仍只有1.3358，差于独立静态控制，
  说明此例问题不只是Gate数值，还涉及联合训练后的分支预测。
- 加入tp_pre后，上述1980样本MSE降至0.2642，但Weather-H96-S2022的
  sample1956/channel14由G3的3.9333变为12.6234。组合并未系统消除水平误差。

**模型选择与后续启发：**没有单一设计同时支配MSE、MAE、跨seed稳定性与成本。
tp_pre_g3平均MSE最低（0.581470 vs i0 0.584271），8/12胜出，但相对G3的
setting等权相对MSE反而+0.101%，均值收益受高误差设置影响。components_static
平均MAE最低（0.441097 vs i0 0.442019），MSE在10/12改善且仅多7/21参数，
是更保守的简洁候选。G3仍是三个动态Gate中最值得保留的候选，不能称为普遍最佳。
下一步可提出动态level+静态shape控制、水平误差条件下的Gate校准与分支训练诊断，
但不在本轮自动新增实验。两seed、6任务、重叠窗口及重复使用validation不能支撑
统计显著性、盲测泛化或Golden超越；样本干预不等于从头训练消融。

## 下一阶段：静态双分量基线与动态水平融合（2026-10-08）

**状态：96/96正式训练及48项冻结checkpoint test评估完成、0失败；49项测试、
validation/test完整样本审计及本地同步复核已通过，本阶段已完成。** 用户接受将静态双分量融合
保留为下一阶段研究基线，进一步验证动态水平Gate及其与T->P pre-norm的组合。
本阶段仅支持实验内小幅配对收益，不代表Golden或盲测泛化提升。旧Stage 3是续训
历史，本节是新的从头训练阶段，两者不可混用。

### 研究问题与模型边界

1. 静态双分量融合的优势在匹配重复实验中能否复现？
2. 只让水平Gate动态变化，能否优于静态双分量融合，并保住完整G3的收益？
3. T->P pre-norm在双分量融合下是否仍有价值，和动态水平选择是互补还是重复？
4. 若动态融合退化，是Gate选择失误，还是联合训练后分支能力下降？

保持原phase主干、RevIN、`E(x-x_last)->t->D(t)+x_last`、单个rank-r token，
L=720、P=24、r=H/8、pool_factor=1、smooth_ratio=0。pre-norm仍使用已有
共享消息广播，不增加零初始化残差缩放、null、选择性写入或内容修正。

按每个24步未来周期拆分两路预测为`mu_b + s_b`，其中`mean_phase(s_b)=0`：

`y = (1-a_level)*mu_P + a_level*mu_T + (1-a_shape)*s_P + a_shape*s_T`。

- 静态双分量：两个系数均为每变量可学习常量。
- 动态水平/静态形状：`a_level(x,c)=sigmoid(b_level,c+MLP(features(x,c)))`，
  `a_shape(c)=sigmoid(b_shape,c)`；动态系数每样本/变量一个，预测窗内共享，
  各周期均值本身仍分别计算。形状Gate静态不意味着形状预测静态。
- 完整G3：水平、形状两个Gate均动态，作为复杂度和极端样本保护的必要对照。

新模式复用既有G3的特征、16维MLP、stop-gradient和初始化：只读当前历史、
latent和两分支预测；初始两个Gate均为0.2，动态输出层为零；两分支通过最终
融合损失联合学习。MLP上下文范围先保持不变，避免同时更换Gate输入。
pre-norm条件下读取phase表示的位置与已有tp_pre_g3一致并写入manifest。
所有分量融合与干预在RevIN归一化空间执行后再还原，重放要求逐位一致。

### 第一轮固定矩阵：8配置 x 12设置 = 96次

| ID | token交互 | 输出融合 | 作用 |
|---|---|---|---|
| R0 / i0 | 无 | 原单静态Gate | 原架构参照 |
| R1 / tp_pre | T->P pre-norm | 原单静态Gate | 原交互参照 |
| B0 / components_static | 无 | 水平静态、形状静态 | 下一阶段主要基线 |
| B1 / tp_pre_components_static | T->P pre-norm | 水平静态、形状静态 | 未验证的基础组合 |
| L0 / components_level_dynamic | 无 | 水平动态、形状静态 | 核心简化假设 |
| L1 / tp_pre_level_dynamic | T->P pre-norm | 水平动态、形状静态 | 核心交互组合 |
| D0 / g3_components | 无 | 水平动态、形状动态 | 完整G3对照 |
| D1 / tp_pre_g3 | T->P pre-norm | 水平动态、形状动态 | 完整组合对照 |

ETTh1、ETTm2、Weather x H96/H720 x seeds2021/2022，共12设置；全训练集，
最多30 epochs、patience=8、最低validation Huber选checkpoint。所有配置重新
随机初始化、全模型联合训练，不加载旧权重、不冻结分支；共同未训练权重哈希
及数据顺序按setting匹配。继承上一轮LR、batch、Huber和scheduler，无逐方案
调参；记录30轮上限是否仍在学习，不把未收敛当作机制否定。8GPU仅用于调度。
上一轮结果用于复现核对，不替换此次完整配对矩阵。smoke不计入96次正式训练。

主要配对提前固定：B0-R0、B1-R1确认分量拆分；L0-B0、L1-B1检验水平动态性；
L0-D0、L1-D1检验能否简化形状Gate；B1-B0、L1-L0、D1-D0检验交互贡献。
同时报告`(MSE_L1-MSE_B1)-(MSE_L0-MSE_B0)`的逐setting差值，判断交互是否
改变动态水平的边际收益；该差值是描述性归因，不自动构成统计显著性。

### 动态性归因与分支诊断

第一轮训练前校验每个arm的shape、全模型梯度/模式、无预训练加载、公共初始
参数、融合初始等价、标签不可达、分量重建和原Gate重放。新增组合分别匹配
相同交互的单Gate参照，不能要求有交互模型初始等于无交互模型。

对L0/L1做固定权重干预：用训练集每变量平均水平Gate替换；在同变量、预测窗
不重叠样本间打乱水平Gate，至少5个固定随机种子。D0/D1分别固定水平、形状。
报告每个设置差值、Gate饱和率和超出两分支逐点区间比例。干预只解释已有模型，
不能替代从头训练消融，也不能据此永久否定动态形状。

若L0或L1达到下述初步性能入围条件，补其对应的同容量、上下文置零训练控制，保持MLP
和所有可学习参数，仅将整个特征输入置零；从头训练相同12设置和预算，每个
入围交互条件12次，最多24次。它检验样本上下文相对额外参数/优化路径的价值，
不独立证明某一预测分歧特征有效。若核心候选未入围，停止扩展并记录失败原因。

逐样本/变量拆分周期水平MSE与形状MSE，数值检查其和等于总MSE（周期内形状
零均值）。分别记录本模型phase、trend分支误差、融合误差、预测近/中/远段误差、
峰谷误差、p95样本MSE及最差1%样本平均MSE，不能只看总体平均。

事后oracle固定在同一模型两分支预测上，分别计算水平/形状最优凸融合误差，
且在模型实际归一化空间计算后还原。比较oracle与本模型融合及B0/B1，区分
Gate regret和分支可达误差；oracle差于独立基线提示分支能力问题，不构成其
训练原因的证明。本轮先诊断，不自动新增辅助损失、冻结训练或梯度路由方案。

程序化筛选改善、高误差和退化案例，每个候选/setting至多9条，选例由完整排名
产生并去重。另对上一轮Weather 1702/c14、1980/c14、1956/c14按原seed固定追踪，
这些历史锚点只作复现诊断，不能取代新选例或作为入围条件。验证是否保住动态
形状曾修复的样本，以及动态水平是否减少系统水平偏差。

### 入围、扩展与停止规则

选择使用setting等权相对MSE为主指标，并报告中位数、MAE、seed方向及尾部。
不依据原始MSE均值的少数高误差任务选择，不按dataset/horizon拼接赢家。
以下是事先约定的工程筛选门槛，不是统计显著性标准：

- L0/L1对相同交互的B0/B1：平均及中位相对MSE均小于0，至少8/12设置改善，
  至少4/6个dataset-horizon任务的两seed平均改善，平均相对MAE不退化。
- 若某任务两个seed的MSE和MAE均退化超过1%，或p95/最差1%出现两个seed一致
  退化超过5%，列为阻止默认升级的问题；即使均值入围也只能保留为探索候选。
- 正式升级还须由干预显示跨样本匹配有实际价值，并由同容量上下文置零训练控制确认。
  达标但较D0/D1损失关键形状收益时，保留完整G3为备选，不宣称简化成功。
- B1仅在相对B0具有重复收益且成本可接受时保留；L1须优于B1及L0才支持组合
  互补。若收益接近且方向不稳，优先静态双分量或无交互的较简单方案。

第一轮后冻结统一候选和参数，补独立seed2023及跨任务确认：原三个dataset的
H96/H720 x seed2023为6个新增设置；ETTh2、ETTm1 x H96/H720 x 三seed为12个
设置，共18个。固定保留B0和一个入围候选，若需解释交互再加入B1及另一候选，
合计2--4配置、36--72次新训练；所有对照同预算。不因新增结果自动逐数据集调参。
这些validation未被保证未曾查看，仍称跨任务复核，不称盲测。H192先不进入固定
矩阵；若另行扩大，先补记理由、完整配置、任务数与预算。

正式训练预算：96次核心实验；有信号才增加12--24次容量控制；入围后增加
36--72次复核，全部执行的上限192次。模型未达标时可在96次后结束。本轮不预设
一次完成所有条件分支；各阶段开始前冻结manifest、代码commit和输出路径。
成本参考上一轮96次约7.113 GPU任务小时，仅供初始估算；不把它当作墙钟承诺，
补seed/数据集的训练和审计成本以实测为准。

机制和超参数冻结后，才一次性评估全部已登记保留模型及对应B0的test，测试结果
不反馈到本阶段选择。此前已查看test的任务须披露，不能称全新盲测；若依据test
再改模型，另开实验并记录test-set selection。相对Golden的正式声明仅在任务和
协议与`docs/PhaseFormer_gold_standard.md`一致时成立，否则只报告内部配对收益。

### 实施与验证记录（2026-10-08）

新增模型/训练/审计实现提交`08f4117`；运行溯源和seed模板记录提交`dc78bd3`。
`0ee87a4`允许后续复核显式以components_static自对照，不强制额外训练i0，
确保2--4配置复核预算可执行。正式矩阵manifest和训练代码冻结于`0ee87a4`。
`LowRankDynamicFusion`增加`components_level_dynamic`及同容量上下文置零模式，
默认六方案配置清单保持原行为；`--study level`启用本轮8配置和30轮预算。
静态双分量/pre-norm、动态水平/pre-norm均使用已有模型插入点。
Gate读取交互后的phase Z，与已有tp_pre_g3一致；形状系数按变量静态学习。

远程time环境（torch2.6.0+cu124、Lightning2.6.5、numpy1.26.4、A800）通过
`python -m unittest discover -s tests -p 'test_lowrank*.py'`的48项测试。
ETTh1-H96-S2021、Weather-H720-S2021的8配置各做两个训练/验证batch的smoke，
16/16完成、0失败；每个setting的公共初始化哈希一致，未加载训练权重，
新增融合初始输出与相同交互参照逐位相等，全模型可训练。

`lowrank_level_fusion_smoke`完成全validation审计：920,528条sample/channel/arm
记录、126条程序选例，结果/排名/NPZ/报告/ZIP/六文件目录均通过校验；
Weather-H720的动态水平组合图已人工检查。smoke不计入正式训练预算、不用于
模型效果判断。所有Gate重放按模型原顺序逐位复现，水平和形状MSE和式经校验。

正式入口：`python scripts/run_lowrank_six.py --study level --source-dir
research_runs/phaseformer_l_interaction_20261006 --scratch-dir
research_runs/lowrank_level_fusion_next_scratch --audit-dir
research_runs/lowrank_level_fusion_next --epochs 30`。使用全8GPU，训练后自动运行
完整审计；具体manifest在启动前冻结。同容量控制通过显式`--arms`登记；
后续seed2023使用显式`--allow-seed-template`从同dataset/horizon的seed2021
继承超参数，只改变随机种子并记录来源，不加载其checkpoint。
后续精简复核使用`--baseline-reference components_static`及登记的`--arms`。
第三seed/跨任务18设置的配置模板与数据均已检查存在。正式控制器PID48342，
首次检查96任务中2完成、8运行、86待运行、0失败，训练指标已持续写入。
代码同步时一次fetch提前于传输完成而失败；待传输完成并核对SHA256一致后，
用增量bundle重新同步成功，没有启动错误版本训练或改动旧产物。

已有每小时heartbeat已更新并关联当前会话，监测本轮目录、修复实际故障、避免
重复启动，依据上述条件继续控制/复核及最终评估，全部授权工作完成后暂停。

### Validation决策与最终评估冻结（2026-10-08，test前）

96/96训练完成、0失败；12/12完整分析通过，6,311,200条记录、756条选例。
L0-B0平均/中位相对MSE为+0.203639%/+0.167988%，5/12胜出，2/6任务平均
改善，平均相对MAE+0.152349%。L1-B1相应+0.441762%/+0.470778%，4/12胜出，
2/6任务改善，MAE+0.078565%。两者均未达到事先入围门槛，因此不执行上下文
置零控制及第三seed/跨任务扩展，不追加训练。

B0-R0的MSE在10/12设置改善，平均相对MSE/MAE为-0.122076%/-0.198574%。
B1-R1在11/12改善MSE，但B1-B0仅6/12改善，平均相对MSE-0.128600%、
MAE+0.106115%；不能认定pre-norm组合全面更优。B0保留为统一默认研究候选，
B1只保留为MSE权衡候选。此决策只使用validation，并在查看本轮test前冻结；
同seed同协议的精确复现是流程可复现证据，不能当作新增独立seed证据。

最终test固定评估R0、R1、B0、B1的既有validation-Huber最佳checkpoint，
12个同dataset/horizon/seed设置，共48个checkpoint推理，无新增训练、无调参。
R0/R1用于对应的单Gate参照；默认B0与权衡B1的角色不根据test结果重新选择。
采用`scripts/evaluate_lowrank_level_test.py`，使用
`research_runs/lowrank_level_fusion_next_test_scratch`临时目录和
`research_runs/lowrank_level_fusion_next_test`严格六文件审计目录，加载原训练
目录中登记并核验SHA256的checkpoint；test样本定位单独标记，不混入validation。
这些任务有历史test查看，不称全新盲测；本阶段无test-set selection。
相对原始PhaseFormer Golden的提升声明仍受原任务/训练/评估协议可比性限制。

### 最终结论与冻结test结果（2026-10-08）

结论：保留B0静态双分量、无token交互作为统一研究基线；B1只保留为历史登记的
MSE权衡候选，不默认加入pre-norm。L0/L1均未入围，不执行上下文置零控制或
第三seed/跨任务扩展。正式训练合计96次、0失败，约8.177 GPU任务小时；test
只对已登记权重做48次推理，没有训练、调参或test反馈选择。

以下变化为setting等权相对误差，负号代表改善；R0/R1是匹配重跑参照，非Golden。

| 比较 | validation相对MSE | test相对MSE | test相对MAE | test MSE胜出 |
|---|---:|---:|---:|---:|
| B0-R0 | -0.122076% | -0.136905% | -0.157427% | 9/12 |
| B1-R1 | -0.141343% | -0.147327% | -0.143509% | 12/12 |
| B1-B0 | -0.128600% | +0.091565% | +0.117127% | 6/12 |

静态分量拆分在两种交互条件下均有小幅配对收益，但收益约0.1%，不能包装为
显著创新或统计显著性。B0的test均值MSE/MAE为0.293325/0.333211；仅用于
描述这12设置，不用原始均值替代setting等权选择。每任务两seed均值如下：

| 任务 | B0 test MSE | B0 test MAE | B1 test MSE | B1 test MAE | Golden MSE/MAE |
|---|---:|---:|---:|---:|---|
| ETTh1 H96 | 0.365508 | 0.398784 | 0.370124 | 0.404319 | 0.359/0.382 |
| ETTh1 H720 | 0.425970 | 0.449376 | 0.423400 | 0.447410 | 0.431/0.450 |
| ETTm2 H96 | 0.160794 | 0.252036 | 0.161362 | 0.252550 | 0.163/0.256 |
| ETTm2 H720 | 0.348397 | 0.377249 | 0.347818 | 0.376562 | 0.351/0.379 |
| Weather H96 | 0.146032 | 0.193777 | 0.145719 | 0.193030 | 0.148/0.195 |
| Weather H720 | 0.313250 | 0.328044 | 0.312919 | 0.328335 | 0.309/0.332 |

Golden列仅作固定数值参照：本轮已明确匹配lookback/horizon，但原论文完整训练
协议未被证明一致，不作可直接比较的Golden超越声明。任务有既有test查看，
本阶段在test前冻结且没有test-set selection，不称全新盲测。

诊断支持的边界：

- B0相对R0的validation形状MSE在12/12设置降低，平均-0.240014%；水平误差
  仅5/12降低。validation p95/最差1%平均+0.178259%/+0.237526%，test为
  +0.063462%/-0.495114%，因此整体收益不等于统一尾部保护。
- L0-B0、L1-B1的平均相对MSE为+0.203639%/+0.441762%；L0、L1的固定水平
  Gate干预平均增加本模型MSE约0.565911%/0.734968%，打乱增加1.510979%/
  1.703037%。样本匹配对动态模型有价值，但没有超过独立训练静态架构。
- 两动态模型的事后分量oracle在12/12设置低于各自静态基线，平均相对约
  -25.240784%/-24.199917%。提示融合选择仍有空间；oracle读取未来真值，
  不能证明这些系数可由历史学到，不能排除分支质量问题或证明联合训练退化原因。
- L0对D0仅1/12胜出，L1对D1仅3/12；不能宣称静态形状成功简化完整G3。
  Weather-H96-S2022 1702/c14的B0/L0/D0 MSE为76.364170/33.272280/15.393433，
  L0/D0形状MSE为22.539290/10.425045；S2021 1980/c14却是B0/L0/D0
  0.150080/0.054468/2.908488。历史锚点只作诊断，不用于选择或替代完整排名。
- B1-B0在validation/test的平均形状MSE均退化（+0.569223%/+0.561062%），
  水平MSE改善（-0.365193%/-0.165057%）。pre-norm有误差分量权衡，没有证实
  与动态水平互补，也未形成一致默认收益。

与上一轮相同seed/协议的60条MSE/MAE逐项完全复现，只是流程复现证据，非新增
独立seed。仅i0、B0的ETTh1-H96-S2022达到30轮上限；两者最佳validation Huber
均在epoch23（从0计数）。有效early-stopping patience由trainer固定为8，来源
配置中的旧字段不改变该callback。结论适用于当前预算，不保证所有机制已收敛。

validation证据目录`research_runs/lowrank_level_fusion_next`：6,311,200条记录、
756条选例、84张图；test目录`research_runs/lowrank_level_fusion_next_test`：
4,925,984条记录、324条选例、36张图。均为六文件加figures，全部候选搜索结果
保留，报告第11节登记结论和按唯一案例计数的可测量模式；ZIP仅含报告和引用图。
`scripts/finalize_lowrank_level_audit.py`按已冻结validation选择补齐决策、有效预算、
历史复现核验及ZIP重建。下一步研究应针对可从历史预测的可靠性及分支误差，另行
提出可检验设想；本阶段不自行新增机制、辅助损失、冻结分支或扩大训练。

两目录的六文件及全部引用图已同步本地，逐文件SHA256与远程一致；完整样本表
分别为`49429c067b6226e8b8f3b27e847666cfafb0c319b8a429803260e53bc503ca8c`与
`b637234f468ce4ee824e5605c1c4c5fa7e6854f1ebdf5cf53172ffe8556c1c0c`。
本地独立重算全部144项汇总MSE/MAE及水平/形状MSE、1080条案例指标；抽查
validation Weather-H96-S2022的L0/D0与test同任务B0的三类Top3排序，均符合
H间隔规则。YAML/CSV/NPZ设置一致，ZIP白名单及字节一致性检查通过。

### 交付与实施范围

已在现有lowrank_fusion和六方案入口补齐动态水平模式、静态双分量/pre-norm
组合及对应训练/审计配置。最终工件使用`research_runs/lowrank_level_fusion_next`及独立确认实验ID，
遵守六文件加figures规则，保存全部试验、配置与选择轨迹。最终回答：静态分量
收益可复现；动态水平不值得本轮升级；形状动态性简化未成立；交互互补未成立；
融合选择存在oracle空间，但分支能力和训练原因仍不能作确定归因。条件扩展未执行。

## 并行双向交互与静态双分量融合（2026-10-08，前瞻登记）

**状态：用户于2026-10-08明确要求停止；远程控制器、训练及其子进程已终止，
GPU全部释放，不自动续跑。30/36训练已有完整result，6项中断；正式样本审计和
test尚未开始，不能将本阶段标为完整审计通过。**

用户要求关注并行P<->T与静态双分量融合，已授权实现和测试。假设是：原并行
双向交互在单Gate下的水平/形状权重冲突，能否通过分别学习两个静态系数缓解。
这是对已有机制组合的检验，不预设新方法成立；不混用8轮历史或warm-start结果。

固定三个配置：B0=`components_static`（无交互、静态双分量），Q0=`parallel_post`
（并行双向、原单静态Gate），Q1=`parallel_components_static`（并行双向、静态
双分量）。主要比较Q1-B0检验交互价值，Q1-Q0检验分量拆分，Q0-B0仅作结构权衡。
不追加i0，因为本阶段目标是相对已冻结B0的升级；B0本轮重新训练，不加载旧权重。

两方向均读取交互前原状态：T'=LN(T+F_PT(T,P))、P'=LN(P+F_TP(P,T))，
T'进入原rank-r decoder，P'进入原phase predictor，彼此不顺序读取更新后的状态。
沿用历史parallel的post-norm，不同时试pre-norm、缩放、null或选择性写入。
T->P只有单个trend key/value，仍是广播对照；P->T可选择phase slots。
Gate为每变量两个静态sigmoid系数，初始化0.2，在RevIN归一化空间按24步周期
融合水平/零均值形状。保留L720、P24、r=H/8、pool_factor1、smooth_ratio0及原主干。

先对ETTh1-H96-S2021、Weather-H720-S2021做三配置两batch smoke并完整验证审计。
通过后运行ETTh1、ETTm2、Weather x H96/H720 x seeds2021/2022，12设置x3配置
=36次从头联合训练，上限30轮、patience8、最低validation Huber checkpoint。
使用此前同任务超参数与全数据，不按arm调参；共同未训练权重和数据顺序匹配，
不得加载训练checkpoint或冻结原分支。smoke不计入36次。此有限矩阵旨在同当前
研究基线直接配对；不自动扩展第三seed、数据集或新增机制。

validation的setting等权平均/中位相对MSE均<0、至少8/12设置和4/6任务改善、
平均相对MAE不退化才支持默认升级。任务两seed均退化MSE/MAE>1%，或p95/
最差1%两seed均退化>5%时阻止默认升级。即使通过，也只作为两seed探索性候选，
不宣称显著性。Q1-Q0独立报告，不用相对Q0的改善替代相对B0的升级要求。

样本审计：每周期水平/形状MSE、两分支和oracle误差、近中远段、峰谷误差、p95
及最差1%；每arm/setting按MAE三类Top3并按同channel/H间隔去重。固定权重关闭
两个方向、仅保留P->T、仅保留T->P，只作诊断，不替代重新训练单向对照。
双向Q1是否真正改善两分支由结果检验，不把消息变化或oracle空间当作因果证明。

在validation冻结统一结论后，一次性评估已登记B0/Q0/Q1的36个checkpoint test，
不根据test改变本轮选择。既有test查看及Golden完整协议可比性限制继续披露。
输出`research_runs/lowrank_parallel_static`及独立`lowrank_parallel_static_test`，
分别严格六文件加figures；checkpoint/日志在同名scratch目录，旧成功结果不覆盖。
入口`run_lowrank_six.py --study parallel --baseline-reference components_static`；
manifest、训练代码提交与预算启动前冻结。完成后同步本地、提交计划/log并暂停
既有监测。本节只登记方案，效果结论待真实实验与审计。

### 用户停止时的部分结果（2026-10-08）

实现/登记提交f6db615，test冻结与结论工具40d456e，顺序控制器f65f369。
远程53项lowrank单元测试通过；两设置三arm的6项两batch smoke全部完成，
完整smoke样本审计通过。smoke只证明流程可运行，不能作为效果证据。
正式训练冻结于f65f369，停止时B0完成12/12、Q0完成10/12、Q1完成8/12。
Q1四个Weather设置未完成；Q0缺Weather-H720-S2022与Weather-H96-S2021。
6项中断不记为训练故障，已有result/checkpoint/日志保留，自动重启被用户撤销。

Q1-B0仅使用已完成的ETTh1/ETTm2共8个配对设置，setting等权相对MSE/MAE为
+1.033810%/+1.112077%，两指标均仅2/8设置改善。按任务两seed相对变化均值：

| 任务 | Q1-B0相对MSE | Q1-B0相对MAE |
|---|---:|---:|
| ETTh1 H720 | -1.456579% | -0.447454% |
| ETTh1 H96 | +2.952234% | +2.908181% |
| ETTm2 H720 | +1.459104% | +0.949034% |
| ETTm2 H96 | +1.180481% | +1.038548% |

ETTh1-H96两seed的MSE和MAE都退化>1%，已触发事先默认升级阻止项。
Q1已有6个设置MSE退化，剩余4个即便全改善也至多6/12胜出，达不到8/12门槛。
因此本轮不支持将并行双向静态分量升级为默认；仍保留B0。不能据此推断Weather
结果或完整矩阵平均值，不能据此否定所有双向交互、pre-norm或其他预算。

同8设置Q1-Q0的相对MSE/MAE为-0.207665%/-0.170258%，MSE胜5/8、MAE胜6/8。
说明静态分量拆分能小幅缓解已测双向单Gate误差，但未抵消交互相对B0的退化；
尚无样本级因果归因。B0本轮12项与上一轮同seed validation MSE/MAE精确一致，
仅为流程复现证据。Q0-B0在其10个配对中MSE/MAE为+1.032172%/+0.797636%，
MSE胜3/10；覆盖不同，不与8设置数字混当完整矩阵比较。

部分结果同步到`research_runs/lowrank_parallel_static_scratch/partial_results.csv`
和`partial_summary.json`；这是scratch中止快照，不伪装为Skill六文件完成工件。
正式run.yaml标为incomplete，pipeline/status标为stopped_by_user；原每小时自动
任务保持PAUSED。本次遵从停止指令，不继续训练、样本推理、补test或扩展实验。

## Controlled mechanisms (historical stage 1)

The arrow denotes information flow. `P` is the routed phase-token sequence and
`T` is one per-channel rank-`r` residual latent, with `r=H/8`.

| Arm | Interaction | Definition |
|---|---|---|
| `phaseformer_l_i0_late_fusion` | none | Legacy rank-`r` latent is decoded independently and fused at the output |
| `phaseformer_l_i1_phase_to_trend` | `P -> T` | Rank-`r` latent queries phase tokens before the unchanged decoder |
| `phaseformer_l_i2_trend_to_phase` | `T -> P` | Phase tokens receive the rank-`r` state before the phase predictor |
| `phaseformer_l_i3_parallel_bidir` | parallel `P <-> T` | Both directions read the original states and update in parallel |
| `phaseformer_l_i6_alternating` | two-round alternating | Two sequential `P -> T -> P` rounds |
| `phaseformer_l_i7_joint` | joint attention | Full self-attention over `[T, P_1, ..., P_M]` |

The phase-to-trend path preserves the original low-rank residual behavior. The
trend-to-phase single-token path is intentionally retained as a broadcast
control: with one key/value token its attention weights are necessarily one.
The joint and alternating arms test whether the phase path benefits from a
state update rather than only from a final residual forecast.

## Frozen protocol

- Data: ETTh1, ETTh2, ETTm1, ETTm2, Weather.
- Lookback: 720; phase period: 24.
- Horizons: 96, 192, 720.
- Seeds: 2021, 2022.
- Loss: Huber; epochs: 8; lowest validation-loss checkpoint.
- Residual head: `PooledLowRankWeakPeriodResidualHead`, `pool_factor=1`,
  `rank=H/8`, `smooth_ratio=0`; the latent is `encoder(pooled(x-x_last))`.
- Readout: unchanged legacy `decoder(r -> H)` plus the `x_last` anchor.
- Fusion gate: static per-channel sigmoid, initialized at 0.2.
- Learning rate: the existing per-dataset/per-horizon base preset; no new grid.
- The primary matrix is validation-only. Test is not read for mechanism or
  hyperparameter selection.

The corrected matrix contains the six requested arms plus a matched `phase_only` control:
`5 datasets × 3 horizons × 2 seeds × 7 arms = 210 jobs`.

## Comparability and reporting

Every arm uses the same pooled-low-rank head, rank, gate initialization, loss,
data split, and seed. Interaction modules are constructed only for the arms that
use them. The report must include parameter counts and wall-clock time because
joint/alternating attention has additional capacity and compute.

The result is an architecture screen. Any later test evaluation must be a
separate, predeclared stage after validation results are frozen; it must not
replace this matrix or be described as an unbiased benchmark.

## Stage 2: Identity Residual and Selective Interaction

User-authorized on 2026-10-07. Implement and test three hypotheses: preserve the
rank latent with a pre-norm residual; let the trend reject phase messages with a
zero-valued null; let each phase choose between the trend and a zero message.
One rank-r trend token, the original low-rank encoder/decoder, phase backbone,
and static output fusion are retained. No Conv1d trend encoder is introduced.

Pilot settings: ETTh1-H96, Weather-H720, ETTm2-H720, each seed 2021/2022.
All arms start from the same completed stage-1 i0 checkpoint within a setting.
All trainable parameters are updated for four additional epochs with a fresh
Adam optimizer and the source learning rate/scheduler, batch size and Huber
loss. The source config and checkpoint hash are saved. Lowest validation-loss
checkpoint after training is used for every arm (no candidate-only initial
checkpoint fallback). This is matched fine-tuning, not a from-scratch benchmark.

| Arm | Update | Purpose |
|---|---|---|
| i0_continue | no interaction | Control for extra training |
| pt_post / tp_post | legacy LN(z + message) | Matched legacy update controls |
| pt_pre / tp_pre | z + F(LN(z), LN(source)) | Normalization-position controls |
| pt_scaled / tp_scaled | z + alpha F(LN(z), LN(source)) | Proposal 1, alpha starts at zero |
| pt_null | pt_scaled with zero-valued null competitor | Proposal 2 |
| tp_selective | tp_scaled with per-phase trend/null choice | Proposal 3 |

This gives 54 runs. First complete the 42 baseline/update-control jobs, then
the 12 routing jobs; do not combine null read and selective write in this pilot.
Pre-norm arms share legacy attention projections. Null mass scales all output
biases too, making a fully rejected message exactly zero. Scaled arms preserve
the full i0 output exactly at initialization. The source weight loader rejects
unexpected keys or missing common weights, and a real validation batch checks
initial identity before each scaled-arm run. The seed is reset immediately
before fit to align data order independently of constructor RNG consumption.

Second verification: source diff review, configuration/full-output equivalence,
null rejection with nonzero biases, selective-query gradients, residual-scale
gradient startup, broadcast control, and explicit attention vs MHA equivalence.
Then run all nine arms through a two-train-batch/two-validation-batch integration
smoke before launching the complete pilot on the eight available A800 GPUs.

Canonical output: `research_runs/lowrank_interaction_stage2_20261007_v2/`, containing
the six audit files and referenced figures only. Checkpoints/logs live in the
separate ignored `research_runs/lowrank_interaction_stage2_20261007_v2_scratch/`.
Entrypoint: `scripts/run_lowrank_interaction_stage2.py --source-dir
research_runs/phaseformer_l_interaction_20261006 --scratch-dir
research_runs/lowrank_interaction_stage2_20261007_v2_scratch --audit-dir
research_runs/lowrank_interaction_stage2_20261007_v2 --gpus 0,1,2,3,4,5,6,7`.

The first stage-2 batch without the v2 suffix is INVALID: the initialization
audit left the model in eval mode before fit (22 result files at interruption).
Code `0f14e9e` explicitly calls train() after the audit and adds a per-training-
batch assertion that all modules are in training mode. All nine integration
smokes were repeated successfully before the v2 launch. Earlier smoke results
establish pipeline behavior only and cannot establish training efficacy.

Post-training paired analysis ranks sample/channel MAE, selects top five high
baseline error, regression and improvement cases with a within-channel start
distance of at least H per category, and saves forecasts only for selected cases.
Every fixed candidate is retained and compared to i0_continue; pt_null is also
compared to pt_scaled and tp_selective to tp_scaled. Capture null acceptance,
within-sample phase-gate spread, raw/effective message norms, and same-checkpoint
message-disabled predictions. Gate/entropy correlations are descriptive, not
causal claims. Setting coverage, error means, ranking, NPZ predictions, figure
references and ZIP bytes must be verified before reporting completion.
No test loader or hyperparameter search is permitted in this pilot. No automatic
full-matrix expansion is planned before assessing pilot evidence.

### Stage 2 Completed Results (2026-10-07)

The valid v2 matrix completed all 54 runs with zero failures on code `0f14e9e`.
The paired audit passed: 3,050,768 sample/channel comparison rows, 720 selected
cases and 24 figures. All six canonical files were downloaded and their SHA256
hashes match the remote originals. The local directory whitelist, all 54 result
rows, setting coverage and report/ZIP byte consistency were independently checked.
The earlier unsuffixed eval-mode batch remains invalid and is excluded below.

Setting-equal candidate-minus-`i0_continue` validation differences (negative is
better; wins count any negative MSE difference, including very small differences):

| Arm | Mean delta MAE | Mean delta MSE | MSE wins |
|---|---:|---:|---:|
| pt_post | +0.009651 | +0.016683 | 2/6 |
| pt_pre | +0.000289 | +0.000410 | 2/6 |
| pt_scaled | -0.000140 | +0.000167 | 3/6 |
| tp_post | +0.000551 | +0.001539 | 2/6 |
| tp_pre | -0.000270 | -0.000606 | 4/6 |
| tp_scaled | -0.000225 | -0.000379 | 5/6 |
| pt_null | +0.000174 | +0.000515 | 2/6 |
| tp_selective | +0.000458 | +0.000869 | 1/6 |

**Proposal decisions**:

- Retain pre-norm T->P as the strongest direction for further investigation.
  `tp_pre - tp_post` has mean delta MSE -0.002145 and improves all six settings.
  However, `tp_scaled - tp_pre` is +0.000227, with only 2/6 improvements: the
  evidence does not establish an additional average benefit from zero-initialized
  residual scaling. In P->T, pre-norm largely removes the ETTh1 warm-start penalty
  (seed 2021 MSE: i0 0.672340, post 0.721411, pre 0.672266), but does not provide
  a consistent gain across settings. This is evidence about the update design in
  matched fine-tuning, not a general claim that post-norm is always harmful.
- Do not expand null read in its current form. `pt_null - pt_scaled` is
  +0.000314 MAE / +0.000348 MSE; only 2/6 MSE improvements, both tiny ETTm2
  differences. The rejection option has no demonstrated incremental benefit.
- Do not expand selective write in its current form. `tp_selective - tp_scaled`
  is +0.000683 MAE / +0.001249 MSE, with 0/6 improvements. Phase-dependent
  receipt alone did not improve upon the corresponding broadcast control.

**Activation and sample evidence**:

- P->T null real-phase acceptance is 0.951--0.963 on ETT, close to the 24/25
  acceptance of equal phase/null scores. Weather acceptance is 0.459/0.670,
  showing more rejection, but both seeds are worse than `pt_scaled` in MSE.
  Rejection magnitude is not established as a measure of reliability.
- Selective-write phase-gate standard deviation is 0.071/0.120 on ETTm2, but
  only 0.0071/0.0077 on Weather. Thus ETTm2 shows phase differentiation while
  Weather remains nearly a common receipt magnitude. ETTm2 effective update
  norms are 0.466/0.554, and same-checkpoint enabled-minus-disabled MSE is
  +0.001131/+0.000479. Weather's corresponding MSE is -0.006392/-0.005379,
  yet it still loses to the trained broadcast control. Reliance on a message
  within one checkpoint does not establish superiority over another model.
- The programmatically selected Weather-H720 seed2021 null-read regression
  `1910,c0` has acceptance approximately 1 and delta MSE +2.35810 versus i0;
  disabling its message at fixed weights reduces MSE by 3.0895. The improvement
  `2111,c14` has acceptance 0.025 and delta MSE -1.28771, but the enabled-minus-
  disabled difference is only -0.00037. Its improvement versus i0 cannot be
  attributed to the current message alone.
- Weather-H720 seed2021 selective-write regression `2577,c15` has acceptance
  approximately 1, delta MSE +4.04697 versus i0 and enabled-minus-disabled MSE
  +3.19742. ETTm2-H720 seed2022 regression `7538,c4` instead has acceptance
  0.063 and delta MSE +0.25369, while enabling the message improves MSE by
  0.04355. These cases rule out a simple reading of low acceptance as successful
  protection or of every regression as direct harm from the current message.

These observations motivate keeping the identity bypass and studying message
content before adding more routing freedom. They do not identify a proven causal
failure mechanism. The pilot used four extra training epochs from selected i0
checkpoints and reused validation for checkpoint selection and analysis. It is
not a from-scratch comparison, a full ETT/Weather benchmark, a significance test,
or evidence of improvement over Golden/test results. No automatic expansion is
launched; i0 remains the established low-rank baseline.

Canonical report and portable image bundle:
`research_runs/lowrank_interaction_stage2_20261007_v2/objective_error_analysis.md`
and `objective_error_analysis.zip` in the same directory. Source checkpoints,
invalid-run evidence and v2 training records remain outside the canonical audit
directory for reproducibility; they are not committed to Git.

### Frozen-checkpoint mechanism audit (2026-10-07)

The user requests finer sample/activation analysis and reassessment of the
research hypotheses. No training or test evaluation is added. Use the valid v2
checkpoints only. `scripts/audit_lowrank_stage2_mechanisms.py` evaluates all eight
interaction arms on 128 equally spaced validation windows per setting, with all
channels. This cohort is selected independently of errors and is not a new full
validation efficacy estimate. Adjacent windows may overlap.

Ten detailed cases are selected from the existing audit's rank-one MAE cases:
ETTh1-H96 both seeds, pt_post regression; Weather-H720 seed2021, pt_null and
tp_selective regression/improvement; ETTm2-H720 seed2022, tp_selective
regression/improvement; Weather-H720 seed2022, tp_scaled regression/improvement.
The same windows/channels are also evaluated across all eight arms to prevent
confusing differences between samples with differences between mechanisms.

Frozen interventions: message bypass; 0.25/0.5/2 times message scale; removal of
null with real-key renormalization; uniform phase read with original acceptance;
constant per-sample mean phase gate; a 12-slot cyclic shift of gates preserving
their histogram; and separate post-norm message-only versus norm-only paths.
Native and bypass implementations must match the original model at every batch.
ETTh1 source-i0 insertion checks additionally separate the initial normalization
shock from later model adaptation. Use highest float32 precision, documenting
the difference from the original medium-precision efficacy evaluation.

Measure both latent changes and output corrections. For d=y_on-y_off and
e=y_true-y_off, check the exact identity delta MSE=mean(d^2)-2 mean(d*e).
The label-dependent optimum along d is a diagnostic upper bound, never an
inference-time gate or an authorized model selection rule. Full-validation
gate/error correlations are reused from the previous audit and reported as
observational. Hold weights fixed, and qualify all activation-distribution shift
and overlapping-window limitations.

Output: `research_runs/lowrank_stage2_mechanism_audit_20261007/`, in the canonical
six-file format with ten figures. Analysis intermediates use a separate ignored
directory. Existing source weights and evidence are not changed.

#### Mechanism audit outcome

Completed 72,912 sample/channel/arm records and ten six-panel figures. All six
settings have native-output max_abs=0 against the original forward. Off-path,
gate-mean preservation and linear P->T half-scale identities passed. The final
report adds a Chinese hypothesis review and next-step rationale in sections 9/10;
this editorial interpretation is separate from the script-generated numeric
sections. No training or test evaluation was performed.

- **Identity bypass is supported in this warm-start context.** At source-i0
  insertion on ETTh1 seed2021 `485,c5`, off/message-only/norm-only/post-norm MSE
  is 1.357256/1.358235/4.341222/4.339780. Seed2022 `533,c5` reproduces the same
  pattern (1.221824/1.235505/4.502841/4.503758). This directly identifies LN as
  the dominant insertion perturbation for those samples, not as a universal
  explanation of every trained-model difference. The additional efficacy of
  zero-initialized scale remains unproven against the pre-norm-only control.
- **Attention receipt is not a calibrated reliability measure.** Weather
  seed2021 null-read `1910,c0` accepts nearly all mass and has output correction
  cosine -0.959, native/off MSE 4.340156/1.250325. Conversely ETTm2 seed2022
  selective-write `7538,c4` accepts only 0.0632, but native/off/no-null MSE is
  0.587408/0.630951/0.143215. Rejection can suppress a useful correction.
  Removing rejection is not universally beneficial: its diagnostic-cohort
  effect differs across settings, so these examples do not justify removing
  all gates or claiming that no rejection mechanism could work.
- **Useful phase alignment is not demonstrated.** Constant mean receipt has
  lower diagnostic-cohort MSE in all six settings; a 12-slot gate shift has
  lower MSE in five. Many differences are tiny. The current update matrix over
  phase slots has rank at most one, delta_p_j=alpha*g_j*v(t): its selectivity
  varies magnitude, not direction. It should not be described as phase-specific
  message content.
- **Magnitude control alone is insufficient.** Wrong-direction examples need
  different content or rejection, while `7538,c4` and `8142,c3` improve when
  more of their current correction is allowed. Large token deltas and small
  deltas can both help or hurt. Output-space alignment and energy are more
  informative than raw latent norms, but require future labels for diagnosis.
- **Training adaptation must be distinguished from current message efficacy.**
  Weather selective-write `1955,c7` has i0/native/off MSE
  1.012050/0.215311/0.214392. The large cross-model improvement is not evidence
  that the current message helps. An off checkpoint also differs from a model
  trained without interaction, so this is not a full causal decomposition.

Next steps, not yet implemented: first compare frozen-i0 interaction-only
learning with matched joint fine-tuning. Then test phase-conditioned message
content using a small bilinear/Hadamard interaction from the same single rank-r
trend token, with a centered-phase variant and a matched uncentered control.
If reliability gating is revisited, use historical hindcast errors/disagreement
to assess expected correction benefit, and validate calibration on held-out
time intervals. Consider output-correction energy rather than latent norm for
amplitude limits; retain separate direction/scale ablations. No oracle scale
derived from the future labels may enter inference. These are new hypotheses,
not conclusions that the proposed changes will improve forecasting.

## Stage 3: Phase-conditioned content on tp_pre (2026-10-07)

User authorizes implementation and tests of the preceding content hypothesis.
Use the valid stage-2 `tp_pre` checkpoint for each of the six existing settings:
ETTh1-H96, ETTm2-H720, Weather-H720, seeds 2021/2022. Keep original pooled-lowrank
E/D, one rank-r trend token, r=H/8, phase backbone, pre-norm broadcast and static
output fusion. No Conv1d, additional trend tokens, null gate, output norm penalty
or additional rejection mechanism is introduced in this stage.

Add u_j=Wo[Wt(LN(t))*Wp(LN(p_j))], width 8, and update p_j'=p_j+m0+beta*u_j.
The centered candidate uses u_j-mean_k(u_k). beta starts at zero so the complete
initial output exactly equals source tp_pre. Both conditional variants have
identical parameters. The trend-only capacity control replaces LN(p_j) with
adaptive pooling of LN(t) to the phase width, broadcast to every phase; it uses
the same projection shapes and parameter count without unused parameters.
Context and output projections omit biases that would cancel under centering.
This control matches capacity but does not uniquely match every optimization
property; trend-only performance must be reported, not assumed inferior.

Seven trained arms per setting: tp_pre_continue; trend_frozen, phase_frozen,
centered_frozen; trend_joint, phase_joint, centered_joint. Frozen arms train only
the new content module while all original weights/buffers remain fixed and all
original modules stay in eval mode (including dropout). Joint arms train all
parameters in normal train mode. All start from the same tp_pre checkpoint,
receive four epochs with fresh Adam, source LR/scheduler, batch size, loss and
data order seed, and select minimum validation Huber after training. Source
tp_pre is also evaluated as the frozen baseline. Frozen and joint comparisons
are reported separately; a joint gain is not attributed to the new message
unless fixed-weight message ablations support that narrower claim.

Total: 42 training jobs plus six unchanged source evaluations. Complete baseline
and frozen diagnostics before joint candidates. Nine existing stage-2 mechanism
tests and new identity/gradient/centering/capacity/alignment checks precede seven
arm training smokes. Each job checks source SHA256, exact common-weight loading,
full-model initial identity, training mode, and (frozen arms) final bitwise
equality of every source state tensor. No automatic hyperparameter sweep.

Hypotheses and failure conditions:

1. Phase-conditioned content should improve over matched tp_pre continuation
   and over equal-parameter trend-only expansion, across seeds/settings. Without
   both comparisons, added capacity cannot be excluded.
2. Centering should add value over the uncentered conditional arm. Latent zero
   mean is checked explicitly but is not assumed to preserve output mean.
3. Frozen-arm gains establish that the new message can help without modifying
   old weights; joint-only gains are labeled adaptation-dependent.
4. At fixed trained weights, disabling just the new content, or cyclically
   shifting/reversing its phase context, should worsen predictions if the new
   content and its alignment matter. The trend-only control must be invariant
   to phase permutations. Keep the old tp_pre broadcast enabled in every such
   diagnostic. Off-path frozen predictions must equal source tp_pre.
5. Revisit Weather-H720 seed2021 1910,c0 and 2577,c15 as prespecified anchor cases:
   reduced harm on the former must not silently erase benefit on the latter.
   Also inspect 1955,c7 and ETTm2-H720 seed2022 7538,c4 / 8142,c3. These targeted
   checks do not replace programmatic top-five MAE regression/improvement/high
   baseline error selection per setting/arm.

Full validation inference records paired MAE/MSE, horizon thirds, content raw
and effective update norms, phase mean/spread, output-correction energy and
alignment, and enabled/off/phase-permuted predictions. The same validation set
has already influenced source model selection and hypotheses; this is an
exploratory matched continuation, not a blind or from-scratch benchmark. No test
or Golden superiority claim, statistical significance claim, or output-space
constraint experiment is authorized by this fixed matrix. Output constraints
remain a separately testable follow-up if needed.

Entrypoint: `scripts/run_lowrank_stage3.py --source-scratch
research_runs/lowrank_interaction_stage2_20261007_v2_scratch --scratch-dir
research_runs/lowrank_stage3_20261007_scratch --audit-dir
research_runs/lowrank_stage3_20261007 --gpus 0,1,2,3,4,5,6,7`.
Canonical output has the six audit files and referenced figures only. Training
records/checkpoints are kept separately in the ignored scratch directory.

### Stage 3 completed: hypothesis decisions

All 42 training jobs completed with zero formal failures on `a8b2026`.
Full-validation audit passed: 2,669,422 paired rows, 665 selected case records
(including 35 prespecified anchor records), 24 figures. Source and continued
tp_pre references yield 48 aggregate rows. Thirteen unit checks, all seven
training smokes, initial full-output identity, frozen-state equality and final
off=source/permutation/centering invariants passed. `93fae7c` adds symmetric
activation heatmap colors and selected-case update direction diversity; it does
not change training or inference results.

Setting-equal deltas, negative better. Frozen arms compare against source tp_pre;
joint arms compare against matched tp_pre_continue:

| Arm | Mean delta MAE | Mean delta MSE | MSE wins |
|---|---:|---:|---:|
| trend_frozen | +0.000777 | +0.000536 | 3/6 |
| phase_frozen | +0.000670 | +0.000688 | 1/6 |
| centered_frozen | -0.000222 | -0.000290 | 5/6 |
| trend_joint | +0.000866 | +0.000533 | 2/6 |
| phase_joint | +0.000311 | +0.000392 | 2/6 |
| centered_joint | -0.000034 | +0.000205 | 2/6 |

The continued baseline itself has mean delta MSE +0.001139 versus source tp_pre
(2/6 wins). centered_joint is +0.001344 versus source. Thus neither additional
training nor unfreezing is assumed beneficial. Results are a four-epoch pilot,
not a claim that joint training can never work.

1. **Centering: supported in this protocol.** centered_frozen improves over
   phase_frozen by -0.000978 mean MSE, 6/6 settings; centered_joint improves over
   phase_joint by -0.000187, 4/6. Uncentered phase content often mainly changes
   the common phase component: Weather frozen mean-token norm is 0.206546 vs
   total per-token norm 0.206647 in seed2021, and 0.117665 vs 0.117706 in seed2022.
2. **Phase content beats capacity: partially supported, not universal.**
   centered_frozen minus trend_frozen is -0.000826 on average, but only 3/6 wins.
   Uncentered phase_frozen is worse than trend_frozen on average (+0.000152).
   Added phase dependence alone is insufficient.
3. **Independent useful correction: supported in 5/6 settings.** Frozen off
   exactly equals source tp_pre throughout validation, and old state tensors
   are unchanged. Its improvements therefore occur without old-weight
   adaptation. ETTh1 seed2022 has a small regression (+0.000013 MSE).
4. **Phase alignment: supported for the useful centered corrections.** Both
   rolling and reversing phase context worsen all five improving frozen
   settings. ETTm2 seed2022: native-minus-off -0.001188, roll-minus-native
   +0.002931, reverse-minus-native +0.002702. The non-improving ETTh1 seed2022
   also fails this alignment check. In joint models, native beats off and both
   permutations in all six settings, yet whole-model MSE fails to beat matched
   continuation on average. Useful message and useful joint training differ.
5. **Targeted error repair: partial and small.** For Weather seed2021, frozen
   centered source->candidate MSE is 2.095243->2.095060 at 1910,c0,
   3.336931->3.322840 at 2577,c15 and 0.236906->0.224013 at 1955,c7.
   The first anchor's main level error is not solved. ETTm2 seed2022 7538,c4
   improves 0.365061->0.363035, whereas 8142,c3 worsens 0.744187->0.747560.

**Mechanistic refinement:** all six source predictors are shared linear maps,
with no phase alignment module; RevIN and fusion are channelwise affine. For
these exact settings, centered latent updates imply zero summed correction over
the 24 phases of each future cycle in real arithmetic. This is more specific
than the earlier general warning for nonlinear predictors. A highest-precision
check on eight validation windows per setting gives maximum absolute cycle-mean
change 3.97e-8--7.45e-8. The medium-precision efficacy runs can show small rounding
departures. The learned addition is therefore primarily a cycle-shape correction
preserving the old cycle mean, not a solution for common level/trend errors.

Available multi-direction capacity is not yet proved necessary. Selected-case
participation rank (sum sigma^2)^2/sum sigma^4 averages approximately 1.00--1.39
across settings; many updates remain dominated by one direction. A future
centered signed-scalar control and a centered phase-only control are needed to
separate the benefits of centering, direction diversity and trend conditioning.
These controls are recommendations, not new experiments launched in this stage.

**Decision:** retain centered_frozen as a small, attributable refinement of the
trained tp_pre checkpoint, while retaining tp_pre as its baseline. It adds 169
parameters on ETTh1 and 857 on ETTm2/Weather. Do not adopt joint unfreezing or claim
a general benchmark win. Canonical evidence is
`research_runs/lowrank_stage3_20261007/`; selection reused validation, no test
was read, and the original PhaseFormer-L rank-r encoder/decoder remains intact.
