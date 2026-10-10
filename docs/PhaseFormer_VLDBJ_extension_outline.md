# PhaseFormer 期刊扩展稿（VLDBJ 特刊）大纲与五天执行计划

记录日期：2026-10-10。状态：**大纲与计划，未启动任何新实验**。实验部分须用户确认第8节决策后执行。

## 0. 定位与硬约束

### 0.1 本稿是什么

**本稿是 ICLR 2026 PhaseFormer（Niu, Deng, Tong；arXiv:2510.04134）的期刊扩展版，不是独立的后续工作。**

- 正文主体（问题、phase 视角、架构、主实验、效率、消融）沿用 ICLR 版。
- 扩展部分回答 ICLR 版第6节自述的局限：PhaseFormer 假设周期局部稳定，“may fail to capture meaningful dynamics”于不规则周期；未来工作为“非平稳与复杂漂移”。
- 扩展模型是 **ICLR PhaseFormer 加两个轻量插件**，而不是新架构：phase 主干、路由层、共享预测头保持 ICLR 设计。
- 措辞上一律写“we extend / we revisit / in the conference version [ICLR] we showed …”，不写“we propose a new model”。

### 0.2 投稿约束

| 项目 | 内容 | 来源 |
|---|---|---|
| 特刊 | Spatio-Temporal Data Management and Analytics: Machine Learning-based Solutions and Beyond；forecasting 为列明主题，强调效率与可扩展性 | [CFP](https://link.springer.com/journal/778/updates/27853170) |
| 截稿 | **2026-10-15**（时区未在摘要中写明，按北京时间 10-15 白天提交为安全线，须核实） | 同上 |
| 后续 | 一审 2026-12-15；修改稿 2027-02-28；终审 2027-05-31 | 同上 |
| 扩展规则 | 与已发表会议版重叠时须有 **≥30% significant new content**，并**另附说明信**逐项解释新增内容 | [VLDBJ 投稿规则](https://www.vldb.org/vldb_journal/index.php/34-author-guidelines/submission-rules) |
| 格式 | VLDBJ 双栏 LaTeX 模板；初稿可交 PDF | [Springer 作者须知](https://link.springer.com/journal/778/submission-guidelines) |
| 版权 | 复用 ICLR 已发表图表需确认版权许可（ICLR 为 OpenReview/CC 许可，须核实） | 同上 |
| 可复现 | CFP 要求公开代码与数据 | CFP |

### 0.3 仓库规则约束

- 所有“提升”声明以 [Golden](PhaseFormer_gold_standard.md)（ICLR Table 5 原值）为准；单指标改善须写明。
- 依据 test 做过的选择（含选定 M1 本身）标注为 test-set selection，不称盲测或无偏泛化估计。

## 1. 扩展叙事（一段话版本）

> ICLR 版证明了 phase token 对周期**形状**的跨周期变化稳定且低维（ICLR Thm 1），这是 PhaseFormer 以约1k参数达到 SOTA 的原因。期刊版追问其对偶面：phase 表示**不能**承载什么？我们证明，一个跨周期的**水平**漂移会以共同平移进入所有 phase token，原架构可以表示它，但最后一层 LayerNorm 把预测水平限制在一个可从权重读出的包络内，并且水平与形状争用同一预算。包络外的部分只沿少数固定 horizon 轮廓作用，因此可以由一个带锚点的低秩线性分支补全（中间形态 PhaseFormer-L）。水平交出之后，剩余的缺口是**随状态变化的形状振幅**（经典的乘性季节性）。我们用一个零初始化、经中心化的乘性调制把它补上，并证明它对每个未来周期均值的写入恒为0，因此不破坏水平侧的任何保证。最终模型仍是 ICLR PhaseFormer，只多出一个低秩水平分支与一个约200参数的调制器。

叙事阶梯：

| 层级 | 问题 | 机制 | 理论 | 位置 |
|---|---|---|---|---|
| ICLR | 形状如何跨周期复现？ | phase token + router | ICLR Thm 1（phase 子空间稳定） | 保留 |
| 扩展-1 | 水平能走多远？ | 原架构分析 | 能力定理、包络定理 | 新增，源自 0930 |
| 扩展-2 | 包络外怎么补？ | 带锚点低秩水平分支（PhaseFormer-L，中间形态） | 低秩互补定理 | 新增，源自 0930 |
| 扩展-3 | 水平交出后还缺什么？ | 中心化乘性调制（M1） | 水平不变的形状调制定理（待证） | 新增 |

## 2. 论文结构与内容来源映射

标记：**[I]** ICLR 原文保留（可精简改写）；**[I+]** ICLR 内容扩写；**[N]** 新增。比例为目标篇幅估计，新增合计约35–40%，留出 30% 底线余量。

| 节 | 内容 | 来源 | 目标占比 |
|---|---|---|---:|
| Abstract | 重写：phase 视角 + 扩展问题 + 最终模型 + 结果 | I+ | — |
| 1 Introduction | ICLR 动机（patch 低效）保留；新增一段“phase 表示的边界”与扩展贡献列表；**明确声明为会议版扩展并列出差异** | I+ | 6% |
| 2 Related Work | ICLR 原文；新增：非平稳与 RevIN 类方法、趋势/季节分解、乘性季节性（Holt–Winters）、2025–26 新基线 | I+ | 6% |
| 3 Motivation：phase vs patch | ICLR Insight 1/2、ICLR Thm 1 | I | 8% |
| 4 PhaseFormer 架构 | ICLR 4.1–4.3：预处理、Embedding、Cross-Phase Routing、Predictor、复杂度 | I | 10% |
| **5 Beyond stable cycles：水平与振幅** | 5.1 记号与水平/形状误差分解；5.2 能力定理；5.3 包络定理及标度无关推论；5.4 低秩互补定理；5.5 水平不变的形状调制定理（新）；5.6 与 ICLR Thm 1 的衔接 | N | 15% |
| **6 扩展模型** | 6.1 低秩水平分支（含锚点、rank 选择）；6.2 静态水平/形状双分量融合；6.3 中心化乘性调制；6.4 退化性质与复杂度（相对 ICLR 的增量） | N | 7% |
| 7 Experiments | 7.1 设置（ICLR 原文+扩展说明）；7.2 主表（ICLR Table 1 加扩展模型行）；7.3 效率（ICLR Table 4 口径加扩展模型）；7.4 ICLR 消融（router 数、routing 变体） | I+ | 18% |
| **7.5–7.8 扩展实验** | 7.5 逐步消融阶梯；7.6 理论验证（包络、交接、谱与模式，源自 0930）；7.7 M1 机制验证（第6节辅助验证）；7.8 案例 | N | 14% |
| 8 Discussion & Limitations | ICLR 局限 + 新局限（Traffic/Electricity 上分支作用小、空间维度未建模、调制信号来源未确定等） | I+ | 3% |
| 9 Conclusion | 重写 | I+ | 2% |
| Appendix | ICLR A.1–A.6 保留；新增：新定理证明、逐 horizon 扩展表、逐 seed 结果、复现与审计信息 | I+N | — |

说明信（Summary of new content）草案要点见第9节。

## 3. 理论部分规划

### 3.1 保留

- ICLR Thm 1（Phase Tokenization Stability）及附录 A.6 的引理与 Thm 2/3，原样保留。

### 3.2 新增（源自 [minipaper_0930](minipaper_0930.md) 第2–5节，需大幅压缩并重新编号）

| 期刊编号（暂定） | 0930 原编号 | 内容 | 处理 |
|---|---|---|---|
| Thm 2 | Thm 1 | 结构性水平能力：共同平移精确穿过 router attention；可逼近形状+随机游走水平模型的贝叶斯预测 | 正文给结论与直觉，构造证明放附录 |
| Thm 3 | Thm 2 | 最终 LayerNorm 水平包络 `I_φ`、标度无关需求、误差下界、水平–形状竞争 | 正文保留 (b)(c)(d) 与推论 2.1 |
| Thm 4 | Thm 3 | 低秩线性互补：输出秩≤轮廓数、读出为 recency kernel、联合训练中锚定秩一分支可精确接管水平 | 正文保留 (a)(c)，(b)(d) 视篇幅 |
| **Thm 5（新，待证）** | — | 水平不变的形状调制：见 3.3 | 新写 |

注意 0930 的 Thm 2(a) 每相位槽椭球与 (d) 竞争对 M1 的 phase 输出不再成立（调制量 q 不受 LN 约束），正文须写明 Thm 3 的 (b)(c) 仍对 M1 成立而 (a)(d) 被 M1 有意放宽。

### 3.3 Thm 5 规划（今明两天推导）

- **(a) 周期均值不变性**：共享线性预测头、完整周期重排条件下，`Σ_p (Z_p − z̄) ⊙ q = 0`，故调制对每个未来周期均值的直接贡献为0；推论：Thm 3(b) 水平包络对 M1 的 phase 路径原样成立，Thm 4 的水平互补分析可整段继承。已有6个设置的数值核验（最大差 5.96e-8–1.44e-7）。难度低。
- **(b) 能力**：生成模型 `X_kp = ℓ_k + (1+α ℓ_k) s_p + e_kp` 下，贝叶斯形状预测为 `(1+α ℓ̂) s_p`；若 t 线性编码 ℓ̂，k=1 的 M1 可精确实现。难度中。
- **(c) 振幅包络（可能只能得到弱版本）**：无调制时，phase 输出落在 Thm 3(a) 的有界椭球内，形状振幅有上界且与水平争用预算；trend 分支对历史线性、在融合中只能加性贡献。需处理 MLP 的近似能力，若无法证明不可能性，则只陈述“有界”而不声称“不可表示”。
- 仿真验证见辅助验证 V1。

### 3.4 与 ICLR Thm 1 的衔接（待核实）

ICLR Thm 1 将周期模式变化建模为 `X′ = XSᵀ + R`。跨周期水平漂移在折叠坐标下是一个秩一加性扰动（对所有 phase token 加同一向量），在 ICLR 模型中只能归入 R，于是 ICLR 的稳定性界随该扰动增大而退化。若矩阵方向核对无误，这一段可作为“ICLR 理论在何处停止保证 → 期刊理论从此处接手”的桥。须对照 ICLR 附录 A.6 的记号核实后再写。

## 4. 扩展模型（M1）规划

- 组成：ICLR PhaseFormer 主干 + rank-r 带锚点低秩水平分支 + 静态水平/形状双分量融合 + 中心化乘性调制（`Z′_p = Z_p + (Z_p − z̄) ⊙ (B·A·t)`，k=min(2,r,d)，B 零初始化）。设计理念见 [M1 设计文档](PhaseFormer_M1_design_rationale.md)。
- 退化性质：q=0 时等于 B0；融合权重取0时只剩 phase 路径。须如实说明静态权重初始为0.2，不是初始即等于 ICLR 模型。
- **命名待定**（第8节 D5）。
- **用户决定（2026-10-10）**：严守 ICLR“低参数、高效率”的 claim。M1 一律使用**压缩到很小的分支**，全量主表也用这个版本；rank=H/8 的低秩头不进入期刊稿主表，只作为中间形态的历史证据出现在分析章节。

### 4.1 压缩版 M1 的规格（理论驱动，训练前固定）

理论依据：
- Lemma 1.1 / 0930 式(9)：形状加随机游走水平模型的贝叶斯水平预测是**周期均值向量**的加权和 `b^T r̄`，所以分支只需读周期均值。
- Thm 4(a)（0930 Thm 3）：最优线性修正的输出秩不超过 horizon 轮廓数 m（位移、倾斜、曲率）。
- 闭式 capture 实测（[rank 容量报告](PhaseFormer_rank_capacity_and_data_property_report.md) 表A）：r=1 保留 65.5–86.2%，r=2 保留 81.6–96.7%，**r=3 保留 88.1–99.3%**（val 口径，满分辨率输入）。

候选规格（参数按代码公式计算；fusion 与主干另计）：

| 候选 | 分支输入 | 秩 | 解码器 | 分支参数 | H96 | H720 | 代码 |
|---|---|---:|---|---|---:|---:|---|
| S1 | pool=24，30 个周期均值（减去末值） | 3 | 可学习 `Linear(3→H)` | 93+4H | 477 | 2,973 | 现有 `PooledLowRankWeakPeriodResidualHead`，无需新代码 |
| S2 | 同上 | 3 | 固定低频轮廓基（Legendre 0–2 阶，即位移/倾斜/曲率），可学习 3×3 混合与偏置 | 约 105，**与 H 无关** | ~105 | ~105 | 需新写约 30 行 |
| S3 | pool=8，90 个均值 | 3 | 可学习 | 277+4H | 661 | 3,157 | 现有代码 |
| 对照：PhaseFormer-L | pool=1 | H/8 | 可学习 | (721)r+(r+1)H | 9,900 | 130,410 | — |

另加两部分：
- M1 调制：`k(r+d)`，k=2、r=3，d=8 时为 22 参数。
- 静态水平/形状融合：当前实现**每变量** 2 个参数，共 2C（ETT 14、Weather 42、Electricity 642、Traffic 1,724）。在 Traffic 上会超过分支本身，见 D8。

S2 与 Thm 4 一一对应：读周期均值，写固定轮廓。它的参数与 horizon 无关，最符合 phase 设计的精神；风险是丢掉 0930 中 ETTh2-720 占 15% 贡献的周期形状修正模态（理论上应由 phase 主干承担）。S1 无需新代码、风险最低。

**选择规则（预登记）**：在 4 个先导 setting（ETTh2-720、ETTm2-96、Weather-192、ETTh1-96）× seed 2021 上训练 S1/S2/S3，**只看 validation**，选 val MSE 等权平均最低者；若差距小于 0.5%，选参数最少者。选定后 28 setting 全部使用，不再更换。先导 setting 的 test 不读取。

### 4.2 主干必须与 ICLR 版一致（阻塞问题）

本地按仓库 preset 实例化 phase-only 主干并实测参数（`build_hyperparams(ds, H, 'no_residual')`，2026-10-10）：

| 数据集 | H96 | H192 | H336 | H720 |
|---|---:|---:|---:|---:|
| ETTh1 | 2,072 | 2,092 | 2,122 | **60,762** |
| ETTh2 | 1,716 | 1,752 | 1,806 | 718 |
| ETTm1 | 3,666 | 3,702 | 1,806 | 3,900 |
| ETTm2 | 3,666 | 1,752 | 1,806 | 1,950 |
| Weather | 5,616 | 3,702 | 3,756 | 3,900 |
| Electricity | 3,666 | **273,160** | 3,756 | **275,998** |
| Traffic | **37,154** | **273,160** | **273,934** | **275,998** |

ICLR 正文与 Table 4 报告大多数数据集约 1.156K、Weather 308（output 96）。仓库 preset 与此不一致，ETTh1-720、Electricity-192/720、Traffic 全部 horizon 相差 1–2 个数量级。期刊稿的主干必须是 **ICLR Table 5 实际使用的逐 setting 配置**，否则：
- 参数量 claim 无法对齐；
- 扩展模型相对 Golden 的差值会混入主干配置的差异。

需要作者提供 ICLR 逐 setting 配置（见 D9）。如果 ICLR Table 5 本身就用了这些大配置，正文的参数表必须按 horizon 如实报告。

## 5. 实验部分规划

### 5.1 保留 ICLR 实验（不重跑）

ICLR Table 1/5 主表与8个基线（PatchTST、Crossformer、TimeMixer、SparseTSF、TimeBase、FITS、FEDformer、iTransformer）、Table 4 效率、router 数消融、routing 变体消融、Traffic 案例、模型规模与输入长度分析、PCA 图。PhaseFormer 行即 Golden，不重训替换。

### 5.2 新增实验

| 编号 | 内容 | 证据来源 | 状态 |
|---|---|---|---|
| E-main | 主表加扩展模型行：7数据集×4 horizon，3 seed 均值，对比 Golden 与8个基线 | 五天内新跑（第6.1节） | 待跑 |
| E-ladder | 消融阶梯：PhaseFormer（Golden 与 E14 phase-only 复训）→ +低秩分支（E14 l_q1_8）→ +双分量融合（B0）→ +调制（M1） | E14 已有前两级；B0 需补跑 | 部分已有 |
| E-eff | 参数、FLOPs、延迟、峰值显存：ICLR PhaseFormer、B0、M1，按 ICLR Table 4 口径（input 720 / output 96）并补 H720 | 五天内只读测量 | 待做 |
| E-theory | 包络实测、水平交接、谱与模式、干预 | 0930 第7节、`research_runs/level_envelope_v2/` 等 | 已有，压缩复用 |
| E-M1 | M1 机制验证 | 第6.2节辅助验证 | 待做 |

## 6. 五天内 M1 实验计划

### 6.1 任务一：全量测试，对标 ICLR 大表

**范围**：7数据集 × {96,192,336,720} = 28 setting，与 Golden（ICLR Table 5）一致；seed 2021/2022/2023，与 ICLR 的3 seed 一致。

**训练量**：
- M1：28×3 = 84 次。已有 6 setting × 2 seed，但配置模板来自 level-study manifest，可能与 E14 不同（例如 ETTh1-H720 用 lr=1.5e-4、huber_delta=0.3），见第8节 D2。
- B0：同样 84 次，用于把 M1 模块的作用与融合方式分开；已有部分同上。
- 如果不重跑已有部分，新增各 72 次；如果统一重跑，各 84 次。

**登记规则**（训练前固定，不看 test 调整）：
1. 配置模板：主干取 ICLR Table 5 的逐 setting 配置（D9）；分支取第4.1节先导选定的压缩规格；融合为 `components_static`；加 M1 调制（width=2，零初始化）。训练超参沿用 E14 已登记值，不做新搜索。
2. 训练：最多30 epoch、patience 8、按最低 val Huber 选 checkpoint；共同初始化与 B0 配对。
3. 全部训练完成后统一读取一次 test；不按 test 换配置、换 seed 或挑数据集。
4. 每个 setting 自动记录周期均值不变性核验（V5）与 off 干预（V7）所需的输出。
5. 报告：3 seed 均值±标准差；相对 Golden 的 MSE/MAE 变化与改善率；双指标/单指标分类；也报告相对 E14 phase-only 与 l_q1_8 的配对差值。

**执行顺序与削减规则**（单次耗时须先在服务器核实）：
1. 24 个非 Traffic setting 的 M1 × 3 seed；
2. 同 24 个 setting 的 B0 × 3 seed；
3. Traffic 4 个 setting（batch=8，最慢）的 M1，然后是 B0；
4. 若截稿前 Traffic 未完成，正文主表标注“pending”，不用其他配置或单 seed 代替，修改阶段补齐。

**预期风险（提前写明）**：0930 主表中 ETTh1-96/192/336、ETTm1-192/720 即使经 test 搜索也未低于 Golden；当前6任务中 M1 的 MAE 均值劣于 Golden。主表必须如实报告，叙事不以“全面超越 ICLR 版”为前提（第8节 D1）。

### 6.2 任务二：辅助验证

按“支撑哪条主张、是否需要训练、失败时如何解读”登记。**优先级 P0 = 截稿前必须完成，P1 = 尽量完成，P2 = 修改阶段**。

| 编号 | 验证 | 支撑的主张 | 类型与成本 | 预登记的判读 | 优先级 |
|---|---|---|---|---|---|
| V1 | **合成数据验证**：在 `X = ℓ + (1+αℓ)s + e` 上，α∈{0, 0.25, 0.5, 1}，比较 phase-only、B0、M1 与贝叶斯预测；另做只有水平漂移的对照 | Thm 5(b)；Thm 4（分支补水平） | 小模型训练，数小时 | α=0 时 M1≈B0；M1 相对 B0 的优势随 α 增大；若无此趋势，Thm 5 的动机不成立 | P0 |
| V2 | **数据特征**：各数据集/通道在 RevIN 窗口坐标下的水平–振幅耦合指数（周期均值与周期内振幅的相关或回归斜率），按 train/val/test 分别统计 | “真实数据存在乘性季节性”；VLDB 的数据特征分析 | 纯数据统计，CPU，数小时 | 若耦合普遍很弱，需在论文中承认 M1 的适用面窄 | P0 |
| V3 | **振幅包络实测**：读 phase-only/B0/M1 checkpoint，测 phase 路径形状振幅的可达上界与各窗口的“振幅需求”，统计超出比例（类比 0930 第7.6节） | Thm 5(c) | 只读，复用 level_envelope 脚本思路 | 若几乎无窗口超出，振幅包络不是实际瓶颈，正文降级为“理论可能性” | P0 |
| V4 | **收益分层**（P5 的类比）：按 V2 耦合或 V3 需求的分位数，统计 M1 相对 B0 的逐窗口收益 | M1 收益集中在高耦合窗口 | 只读，需 6.1 完成 | 预测：收益集中在最高分位；若均匀或相反，如实报告 | P0 |
| V5 | 周期均值不变性：28 个 setting 全部核验；同时核验 Thm 3(b) 水平包络对 M1 的 phase 路径仍成立 | Thm 5(a) | 只读，随 6.1 自动完成 | 必须全部通过，否则是实现错误 | P0 |
| V6 | 水平交接：在 M1 与 B0 checkpoint 上复测 0930 的“主干让出水平、分支承载水平” | Thm 4(c) 在 M1 中仍成立 | 只读 | 交接被破坏则需解释 | P1 |
| V7 | 同 checkpoint 关闭调制（q=0） | M1 训练后是否依赖调制 | 只读 | 只辅助归因，不作独立基线 | P1 |
| V8 | **来源对照**：同参数量的 phase-only 调制（q 来自 phase 自身摘要）与时间错配的 t，6 任务×3 seed | 回应 C2 打平：收益是否来自 trend 信息 | 需训练，约36次 | 若对照与 M1 持平，正文必须写“调制信号来源未被确定” | P1 |
| V9 | 效率：参数、FLOPs、batch1/batch 延迟、峰值显存，与 ICLR Table 4 同口径 | VLDB 效率主张 | 只读，隔离 GPU | 如实报告相对 ICLR 的增量 | P0 |
| V10 | 压缩版与 rank=H/8 版 M1 在 6 任务上的配对对比（后者已有） | “按理论压缩不损失精度” | 6.1 完成后只读汇总 | 若压缩明显变差，正文如实报告精度–参数权衡 | P0 |

### 6.3 日程（北京时间）

| 日期 | 实验 | 写作 |
|---|---|---|
| 10-10（今） | 核实服务器与单次耗时；登记模板与 manifest；smoke；启动 6.1 第1批 | 确认大纲；向用户索取 ICLR LaTeX 源文件 |
| 10-11 | 6.1 持续运行；V1、V2、V9 | 推导 Thm 5(a)(b)；把 0930 理论压缩成第5节 |
| 10-12 | 6.1 收尾与 test 读取；V5、V3、V6、V7；GPU 空出后跑 V8 | 写第6节模型与第7.5–7.7节 |
| 10-13 | V4；汇总主表与消融阶梯；Traffic 视进度 | 整合主表，改写引言与摘要 |
| 10-14 | 补跑与审计 | 全文成稿，写说明信，整理 artifact |
| 10-15 | — | 校对并提交 |

## 7. 叙事对各种结果的应对

主张强度跟证据走，事先写好分支，避免看结果改故事：

| 结果 | 主表措辞 | 第5–7节措辞 |
|---|---|---|
| M1 在多数 setting 双指标优于 Golden，且 V4 显示收益集中在高耦合窗口 | “extended PhaseFormer improves the conference version on X/28” | Thm 5 有实证支持 |
| M1 平均略优、提升不稳定，V3/V4 支持 | “comparable accuracy with gains concentrated where amplitude coupling is strong” | 以分析为主，收益为条件性 |
| M1 与 B0 或 Golden 打平，V8 对照也打平 | 主张收窄为理论与诊断；M1 写为“保证不破坏水平的安全扩展” | 明确调制来源未确定 |
| M1 明显劣于 Golden | 须先与用户重议是否仍以 M1 为最终模型 | — |

## 8. 待用户决策

| 编号 | 决策 | 选项与影响 | 建议 |
|---|---|---|---|
| D1 | 主表结果不理想时是否仍投 | 理论驱动定位下，主表只需诚实；但 VLDBJ 审稿人会看对 ICLR 版的改进 | 投，按第7节分支措辞 |
| D2 | 已有 6 setting 的 M1 结果如何处理 | 压缩版规格不同，必须重跑 | 已解决：全部重跑，旧结果作为 rank=H/8 先导记录 |
| D3 | 参数量冲突怎么处理 | — | **已决定（2026-10-10）**：主表用压缩版 M1，规格按第4.1节先导选定 |
| D8 | 融合权重是否跨变量共享 | 每变量 2C（Traffic 1,724 参数）对比全局 2 个标量 | 建议共享为 2 个标量，加入第4.1节先导的一个因子；或保持每变量并如实计数 |
| D9 | ICLR 主干的逐 setting 配置 | 仓库 preset 与 ICLR 报告的约1k 参数不一致（第4.2节） | **阻塞**：请作者提供 ICLR Table 5 实际配置 |
| D4 | “空间”维度 | PhaseFormer 对通道独立建模；在 Traffic/Electricity 上水平分支作用小 | 定位为时空系统中的传感器流与边缘端轻量预测，局限中诚实界定 |
| D5 | 扩展模型命名 | 例如 PhaseFormer+、PhaseFormer-LS（Level–Shape） | 由作者决定 |
| D6 | 是否加入 2025–26 新基线 | 只引用其论文数值，不重跑；口径须一致 | 选 2–3 个引用数值 |
| D7 | ICLR LaTeX 源与图 | 写作需要 ICLR 源文件 | 请用户提供 |

## 9. 说明信（Summary of new content）草案

1. 新理论：phase 表示的水平能力定理、LayerNorm 水平包络定理、低秩互补定理、水平不变的形状调制定理（会议版只有 phase 稳定性定理）。
2. 新方法：在不改动会议版主干的前提下，增加带锚点的低秩水平分支与中心化乘性调制，并给出退化与复杂度分析。
3. 新实验：扩展模型在完整28 setting 上的结果；逐步消融；84个 checkpoint 上的包络实测与水平交接；谱、模式与干预分析；M1 的合成与真实数据机制验证；扩展的效率测量。
4. 扩展的相关工作与局限讨论。
5. 与会议版重叠的部分（第3、4节与第7.1–7.4节）逐节标明。

## 10. 来源

- ICLR 版：arXiv:2510.04134（PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting）
- 中间形态理论与证据：[minipaper_0930](minipaper_0930.md)
- M1 设计：[PhaseFormer_M1_design_rationale](PhaseFormer_M1_design_rationale.md)
- 交互实验结果：[PhaseFormer_interaction_results_record_20261010](PhaseFormer_interaction_results_record_20261010.md)
- E14 主比较：`research_runs/phaseformer_L_e14_main_v1/`
- Golden：[PhaseFormer_gold_standard](PhaseFormer_gold_standard.md)
